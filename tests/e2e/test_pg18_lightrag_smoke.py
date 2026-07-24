# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opt-in PostgreSQL 18 + LightRAG main-path smoke tests.

Run with:
    DLIGHTRAG_RUN_E2E_PG18=1 uv run pytest tests/e2e -m e2e_pg18 -q
"""

from pathlib import Path

import pytest

from dlightrag.config import set_config
from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter
from tests.e2e.pg18_harness import (
    RUN_E2E_ENV,
    e2e_enabled,
    fetch_pg_prereq_report,
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


async def test_unified_text_ingest_replace_and_filtered_retrieval(
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

    service = await RAGService.acreate(config=cfg, enable_vlm=True)
    doc_path = tmp_path / "pg18-native-smoke.md"
    doc_text = (
        "# PG18 native smoke document\n\n"
        "This document proves metadata filtering, BM25 retrieval, vector storage, "
        "and replace semantics against the PostgreSQL 18 LightRAG path.\n"
    )
    doc_path.write_text(doc_text, encoding="utf-8")

    try:
        first = await service.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="PG18 E2E Document",
            metadata={"e2e_case": " pg18 "},
            metadata_policy="validate",
        )
        second = await service.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="PG18 E2E Document",
            metadata={"e2e_case": " pg18 "},
            metadata_policy="validate",
        )

        doc_id = second["doc_id"]
        chunk_id = second["chunks"][0]
        assert first["doc_id"] == doc_id
        assert first["chunks"] == second["chunks"]

        metadata = await service.aget_metadata(doc_id)
        assert metadata["filename"] == doc_path.name
        assert metadata["doc_title"] == "PG18 E2E Document"

        doc_ids = await service.asearch_metadata(MetadataFilter(custom={"e2e_case": "pg18"}))
        assert doc_ids == [doc_id]

        assert service._metadata_index is not None
        assert service._lightrag_stores is not None
        candidate_chunks = await metadata_retrieve(
            metadata_index=service._metadata_index,
            stores=service._lightrag_stores,
            filters=MetadataFilter(custom={"e2e_case": "pg18"}),
        )
        assert candidate_chunks == [chunk_id]

        assert service._bm25 is not None
        bm25_rows = await service._bm25.search(
            "PG18 native smoke document",
            candidate_ids=set(candidate_chunks),
            top_k=5,
        )
        assert any(row["chunk_id"] == chunk_id for row in bm25_rows)

        assert service._lightrag_stores is not None
        raw_chunks = await service._lightrag_stores.get_text_chunks([chunk_id])
        indexed_text = str(raw_chunks[0]["content"])
        query_embedding = stable_vector(f"document:{indexed_text}", dim=cfg.embedding.dim)
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

        assert (await service.aget_metadata(doc_id))["filename"] == doc_path.name
    finally:
        if service._initialized:
            await service.areset(keep_files=False)
        await service.aclose()
        await pg_pool.close()


async def test_reader_role_attaches_read_only_and_rejects_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader attaches to an existing schema read-only, serves reads, rejects writes.

    Writer and reader run sequentially in one process (fully closing between
    phases resets the process-wide LightRAG client and domain pool). This
    validates DlightRAG's reader code against a real PostgreSQL under a read-only
    session; physical streaming replication itself is an infrastructure
    guarantee and is out of scope for this test.
    """
    from dlightrag.config import reset_config, set_config
    from dlightrag.core.service import RAGService
    from dlightrag.storage.pool import pg_pool

    conn_kwargs = pg_conn_kwargs_from_env()
    workspace = make_workspace_name("reader")
    writer_cfg = make_e2e_config(
        working_dir=tmp_path / "storage",
        workspace=workspace,
        conn_kwargs=conn_kwargs,
    )
    set_config(writer_cfg)
    install_fake_model_functions(monkeypatch, dim=writer_cfg.embedding.dim)

    # ── Writer: provision schema + ingest ──────────────────────────────
    writer = await RAGService.acreate(config=writer_cfg)
    try:
        doc_path = tmp_path / "reader-smoke.md"
        doc_path.write_text(
            "# Reader smoke\n\nA replica reader attaches to the existing schema "
            "and serves stateless reads.\n",
            encoding="utf-8",
        )
        result = await writer.aingest(
            source_type="local",
            path=str(doc_path),
            replace=True,
            title="Reader Smoke",
            metadata={"e2e_case": "reader"},
            metadata_policy="validate",
        )
        doc_id = result["doc_id"]
        chunk_id = result["chunks"][0]
    finally:
        await writer.aclose()
        await pg_pool.close()
        reset_config()

    # ── Reader: read-only attach + retrieve + write rejection ──────────
    reader_cfg = writer_cfg.model_copy(update={"service_role": "reader"})
    set_config(reader_cfg)
    reader = await RAGService.acreate(config=reader_cfg)
    try:
        assert reader.config.is_reader

        retrieval = await reader.aretrieve(
            "reader attaches to the existing schema", top_k=5, chunk_top_k=5
        )
        chunk_ids = {c.get("chunk_id") for c in retrieval.contexts.get("chunks", [])}
        assert chunk_id in chunk_ids

        metadata = await reader.aget_metadata(doc_id)
        assert metadata["doc_title"] == "Reader Smoke"

        with pytest.raises(PermissionError):
            await reader.areset()
        with pytest.raises(PermissionError):
            await reader.aupdate_metadata(doc_id, {"note": "nope"})
        with pytest.raises(PermissionError):
            await reader.aingest(source_type="local", path=str(doc_path))
    finally:
        await reader.aclose()
        await pg_pool.close()
        reset_config()

    # ── Cleanup: remove the workspace via a writer ─────────────────────
    set_config(writer_cfg)
    cleanup = await RAGService.acreate(config=writer_cfg)
    try:
        await cleanup.areset(keep_files=False)
    finally:
        await cleanup.aclose()
        await pg_pool.close()
        reset_config()
