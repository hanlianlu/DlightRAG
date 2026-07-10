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
