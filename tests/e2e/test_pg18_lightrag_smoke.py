# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opt-in PostgreSQL 18 + LightRAG main-path smoke tests.

Run with:
    DLIGHTRAG_RUN_E2E_PG18=1 uv run pytest tests/e2e -m e2e_pg18 -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dlightrag.config import set_config
from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter
from tests.e2e.pg18_harness import (
    RUN_E2E_ENV,
    e2e_enabled,
    fetch_pg_prereq_report,
    image_seed,
    install_fake_model_functions,
    make_e2e_config,
    make_workspace_name,
    pg_conn_kwargs_from_env,
    stable_vector,
)

pytestmark = [
    pytest.mark.e2e_pg18,
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not e2e_enabled(),
        reason=f"set {RUN_E2E_ENV}=1 to run PG18 E2E smoke tests",
    ),
]


@pytest.fixture
async def pg_conn():
    import asyncpg

    conn = await asyncpg.connect(**pg_conn_kwargs_from_env())
    try:
        yield conn
    finally:
        await conn.close()


async def test_pg18_extensions_and_preload_are_ready(pg_conn) -> None:
    report = await fetch_pg_prereq_report(pg_conn)

    assert report.server_major == 18, report.server_version
    assert report.missing_extensions == []
    assert report.missing_preload_libraries == []


async def test_unified_image_ingest_replace_and_filtered_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.service import RAGService
    from dlightrag.storage.pool import pg_pool

    conn_kwargs = pg_conn_kwargs_from_env()
    workspace = make_workspace_name()
    cfg = make_e2e_config(
        working_dir=tmp_path / "storage",
        workspace=workspace,
        conn_kwargs=conn_kwargs,
    )
    set_config(cfg)
    install_fake_model_functions(monkeypatch, dim=cfg.embedding.dim)

    service = await RAGService.create(config=cfg, enable_vlm=True)
    image_path = tmp_path / "native-image.png"
    Image.new("RGB", (16, 16), "green").save(image_path)

    try:
        first = await service.aingest(
            source_type="local",
            path=str(image_path),
            replace=True,
            title="PG18 E2E Image",
            metadata={"e2e_case": " pg18 "},
            metadata_policy="validate",
        )
        second = await service.aingest(
            source_type="local",
            path=str(image_path),
            replace=True,
            title="PG18 E2E Image",
            metadata={"e2e_case": " pg18 "},
            metadata_policy="validate",
        )

        doc_id = second["doc_id"]
        chunk_id = second["chunks"][0]
        assert first["doc_id"] == doc_id
        assert first["chunks"] == second["chunks"]

        metadata = await service.aget_metadata(doc_id)
        assert metadata["filename"] == image_path.name
        assert metadata["doc_title"] == "PG18 E2E Image"

        doc_ids = await service.asearch_metadata(MetadataFilter(custom={"e2e_case": "pg18"}))
        assert doc_ids == [doc_id]

        assert service._metadata_index is not None
        candidate_chunks = await metadata_retrieve(
            metadata_index=service._metadata_index,
            stores=service._lightrag_stores,
            filters=MetadataFilter(custom={"e2e_case": "pg18"}),
        )
        assert candidate_chunks == [chunk_id]

        assert service._bm25 is not None
        bm25_rows = await service._bm25.search(
            "Native image",
            candidate_ids=set(candidate_chunks),
            top_k=5,
        )
        assert any(row["chunk_id"] == chunk_id for row in bm25_rows)

        with Image.open(image_path) as image:
            query_embedding = stable_vector(image_seed(image), dim=cfg.embedding.dim)
        async with metadata_filter_scope({"missing-chunk"}):
            assert (
                await service._lightrag.chunks_vdb.query(
                    "",
                    top_k=5,
                    query_embedding=query_embedding,
                )
            ) == []

        async with metadata_filter_scope(set(candidate_chunks)):
            vector_rows = await service._lightrag.chunks_vdb.query(
                "",
                top_k=5,
                query_embedding=query_embedding,
            )
        assert any(row["id"] == chunk_id for row in vector_rows)

        assert (await service.aget_metadata(doc_id))["filename"] == image_path.name
    finally:
        if service._initialized:
            await service.areset(keep_files=False)
        await service.close()
        await pg_pool.close()
