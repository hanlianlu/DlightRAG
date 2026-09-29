# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The offline rebuild restores fused drawing vectors through PostgreSQL LightRAG storages.

Restoration was only ever exercised against stubs, which hid that its surface check could
never pass on the storage-only surface the rebuild builds. This suite runs the real restore,
through ``_lightrag_surface`` and ``PGCorpusChunkStore``, on a dedicated scratch database.
"""

import hashlib
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from PIL import Image

from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    drop_scratch_database,
    skip_without_postgres,
)

pytestmark = [
    pytest.mark.integration,
    # The writer attach creates asyncpg pools bound to the loop they were born on.
    pytest.mark.asyncio(loop_scope="module"),
]

_MAINT_DB = "postgres"
_EXTENSIONS = ("vector", "pg_textsearch", "pg_trgm")
_WORKSPACE = "rr_it_ws"
_DIM = 8
# The fused vector the fake provider returns; distinct from any text embedding below.
_FUSED = [0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.75, -0.75]


def _kwargs(database: str) -> dict[str, Any]:
    return {**PG_CONN_KWARGS, "database": database}


async def _create_database(database: str) -> None:
    conn = await asyncpg.connect(**_kwargs(_MAINT_DB))
    try:
        await drop_scratch_database(conn, database)
        await conn.execute(f'CREATE DATABASE "{database}"')
    finally:
        await conn.close()
    db = await asyncpg.connect(**_kwargs(database))
    try:
        for extension in _EXTENSIONS:
            await db.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
    finally:
        await db.close()


def _embedding_func() -> Any:
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], *, context: str = "document") -> Any:
        import numpy as np

        values = []
        for text in texts:
            digest = hashlib.sha256(f"{context}:{text}".encode()).digest()
            values.append([((digest[i] / 255.0) * 2.0) - 1.0 for i in range(_DIM)])
        return np.array(values, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=_DIM,
        max_token_size=512,
        func=embed,
        model_name="rr-it-fake",
        supports_asymmetric=True,
    )


class _FusedEmbedder:
    """A multimodal provider that fuses every description-image pair into ``_FUSED``."""

    supports_images = True

    def __init__(self) -> None:
        self.fused: list[tuple[str, tuple[int, int]]] = []

    async def embed_index_fused(
        self, items: Sequence[tuple[str, Image.Image]]
    ) -> list[list[float]]:
        self.fused.extend((description, image.size) for description, image in items)
        return [list(_FUSED) for _ in items]

    async def embed_texts(self, texts: Sequence[str], *, context: str = "document") -> Any:
        raise AssertionError("restoration must fuse the drawing, never fall back to text")


@dataclass
class _Writer:
    config: Any
    settings: Any
    lightrag: Any


@pytest.fixture(scope="module")
async def writer() -> AsyncIterator[_Writer]:
    """One writer attach over a scratch database this module creates and drops."""
    await skip_without_postgres()
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.application.config import DlightragConfig, reset_config, set_config
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings
    from dlightrag.engine.rag.workspace.ports import CorpusRuntimeModels

    database = f"dlightrag_rebuild_restore_{uuid.uuid4().hex[:12]}"
    await _create_database(database)
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={"workspace": _WORKSPACE, "working_dir": "/tmp/rr_it_workdir"},
        storage={
            "postgres": {
                "host": str(PG_CONN_KWARGS["host"]),
                "port": int(PG_CONN_KWARGS["port"]),
                "user": str(PG_CONN_KWARGS["user"]),
                "password": str(PG_CONN_KWARGS["password"]),
                "database": database,
                "pool_min_size": 1,
                "pool_max_size": 3,
            }
        },
        models={
            "max_concurrency": 1,
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    provider="openai",
                    model="rr-it-fake-llm",
                    api_key="rr-it-fake-key",
                    timeout=30,
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="rr-it-fake",
                api_key="rr-it-fake-key",
                dim=_DIM,
                max_token_size=512,
                max_concurrency=1,
                batch_size=2,
                startup_probe=False,
            ),
            "rerank": {"enabled": False},
        },
        corpus={
            "ingestion": {"chunk_token_size": 128, "pipeline": {"max_concurrency": 1}},
            "retrieval": {"bm25_enabled": False},
        },
    )
    set_config(config)
    pg_pool.bind(config)
    lightrag: Any = None
    try:
        backend = build_pg_corpus_backend(config)
        await backend.maintenance.initialize(validate_only=False)
        settings = rag_settings(config)
        lightrag = backend.runtime.create(
            models=CorpusRuntimeModels(
                default_llm_func=lambda _prompt, **_kwargs: "{}",
                embedding_func=_embedding_func(),
                role_llm_configs=None,
            ),
            settings=settings,
        )
        await backend.runtime.attach(lightrag)
        yield _Writer(config=config, settings=settings, lightrag=lightrag)
    finally:
        if lightrag is not None:
            try:
                await lightrag.finalize_storages()
            except Exception:  # noqa: BLE001 - teardown must still drop the database
                pass
        await pg_pool.close()
        reset_config()
        await drop_database(database)


def _drawing_sidecar(root: Path) -> Path:
    """Write a LightRAG sidecar holding one analysed drawing and its image."""
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "report.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (128, 128), "navy").save(root / "images" / "chart.png")
    (root / "report.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "drawing-1": {
                        "path": "images/chart.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


async def test_rebuild_restores_a_fused_drawing_vector_in_postgres(
    writer: _Writer, tmp_path: Path
) -> None:
    from lightrag.base import DocStatus

    from dlightrag.adapters.postgres import rebuild_vdb
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
    from dlightrag.engine.ai.telemetry import NoopTelemetry
    from dlightrag.engine.rag.lightrag.stores import LightRAGStores

    lightrag = writer.lightrag
    doc_id = "doc-restore"
    drawing_id = f"{doc_id}-mm-drawing-000"
    text_id = f"{doc_id}-chunk-000"
    sidecar = _drawing_sidecar(tmp_path / "sidecar")
    chunks = {
        text_id: {"chunk_order_index": 0, "content": "Revenue grew in every quarter."},
        drawing_id: {"chunk_order_index": 1, "content": "A bar chart of quarterly revenue."},
    }
    now = datetime.now(UTC).isoformat()
    await lightrag.full_docs.upsert(
        {
            doc_id: {
                "content": "Revenue report",
                "file_path": "report.pdf",
                "sidecar_location": sidecar.as_uri(),
            }
        }
    )
    await lightrag.text_chunks.upsert(
        {
            chunk_id: {
                **chunk,
                "tokens": 6,
                "full_doc_id": doc_id,
                "file_path": "report.pdf",
            }
            for chunk_id, chunk in chunks.items()
        }
    )
    await lightrag.chunks_vdb.upsert(
        {
            chunk_id: {
                **chunk,
                "tokens": 6,
                "full_doc_id": doc_id,
                "file_path": "report.pdf",
            }
            for chunk_id, chunk in chunks.items()
        }
    )
    # The vector store buffers upserts; restoration reads and rewrites the table.
    await lightrag.chunks_vdb.index_done_callback()
    await lightrag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_list": list(chunks),
                "chunks_count": len(chunks),
                "content_summary": "Revenue report",
                "content_length": 14,
                "file_path": "report.pdf",
                "created_at": now,
                "updated_at": now,
            }
        }
    )
    before = await lightrag.chunks_vdb.get_vectors_by_ids([text_id, drawing_id])
    assert before[drawing_id] != pytest.approx(_FUSED)

    # The rebuild's own tool holds the storages its setup would open.
    tool = rebuild_vdb.DlightRAGRebuildTool(writer.config, embedding_func=lightrag.embedding_func)
    tool.graph = lightrag.chunk_entity_relation_graph
    tool.entities_vdb = lightrag.entities_vdb
    tool.relationships_vdb = lightrag.relationships_vdb
    tool.chunks_vdb = lightrag.chunks_vdb
    tool.text_chunks = lightrag.text_chunks
    tool.full_docs = lightrag.full_docs
    tool.doc_status = lightrag.doc_status
    surface = rebuild_vdb._lightrag_surface(tool)
    rebuild_vdb._verify_restoration_surface(surface)
    embedder = _FusedEmbedder()

    stats = await rebuild_vdb.restore_sidecar_image_vectors(
        workspace_id=_WORKSPACE,
        settings=writer.settings,
        lightrag=surface,
        stores=LightRAGStores(surface, chunk_store=PGCorpusChunkStore(surface)),
        multimodal_embedder=embedder,
        telemetry=NoopTelemetry(),
    )

    assert stats == {"processed_docs": 1, "skipped_docs": 0}
    # The drawing's VLM description is fused with its image; the text chunk is untouched.
    assert embedder.fused == [("A bar chart of quarterly revenue.", (128, 128))]
    after = await lightrag.chunks_vdb.get_vectors_by_ids([text_id, drawing_id])
    assert after[drawing_id] == pytest.approx(_FUSED, abs=1e-6)
    assert after[text_id] == pytest.approx(before[text_id], abs=1e-6)
