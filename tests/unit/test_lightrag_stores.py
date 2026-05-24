# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG storage boundary adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.lightrag_stores import LightRAGStores


class FakeLightRAG:
    def __init__(self) -> None:
        self.chunks_vdb = object()
        self.text_chunks = object()
        self.full_docs = object()
        self.doc_status = object()
        self.entities_vdb = object()
        self.relationships_vdb = object()
        self.chunk_entity_relation_graph = object()
        self.full_entities = object()
        self.full_relations = object()
        self.entity_chunks = object()
        self.relation_chunks = object()
        self.llm_response_cache = object()

    def _build_global_config(self):
        return {"ok": True}


def test_lightrag_stores_validates_required_surfaces() -> None:
    fake = FakeLightRAG()
    stores = LightRAGStores(fake)

    assert stores.text_chunks is fake.text_chunks
    assert stores.build_global_config() == {"ok": True}


def test_lightrag_stores_reports_missing_surfaces() -> None:
    class Broken:
        chunks_vdb = object()

    with pytest.raises(RuntimeError, match="missing"):
        LightRAGStores(Broken())


async def test_upsert_chunks_with_vectors_requires_matching_dimension() -> None:
    stores = LightRAGStores(FakeLightRAG())

    with pytest.raises(ValueError, match="vector dimension"):
        await stores.upsert_chunks_with_vectors(
            {"chunk-1": {"content": "x", "full_doc_id": "doc-1"}},
            {"chunk-1": [0.1, 0.2]},
            embedding_dim=3,
            max_token_size=8192,
        )


async def test_upsert_chunks_with_vectors_appends_doc_status_chunks() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.executed: list[tuple] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            return await operation(self)

        async def executemany(self, sql, values) -> None:  # noqa: ANN001
            self.executed.append((sql, values))

    fake = FakeLightRAG()
    fake.text_chunks = AsyncMock()
    fake.chunks_vdb = SimpleNamespace(table_name="LIGHTRAG_VDB", db=FakeDB(), workspace="ws")
    fake.doc_status = AsyncMock()
    fake.doc_status.get_by_id.return_value = {
        "content_summary": "source doc",
        "content_length": 10,
        "chunks_count": 1,
        "status": "processed",
        "file_path": "/tmp/source.pdf",
        "chunks_list": ["text-1"],
        "metadata": {"source_kind": "document"},
        "content_hash": "sha256:abc",
        "created_at": "2026-05-25T00:00:00+00:00",
        "updated_at": "2026-05-25T00:00:00+00:00",
    }

    stores = LightRAGStores(fake)
    await stores.upsert_chunks_with_vectors(
        {
            "img-1": {
                "content": "direct image",
                "full_doc_id": "doc-1",
                "file_path": "/tmp/page.png",
            }
        },
        {"img-1": [0.1, 0.2, 0.3]},
        embedding_dim=3,
        max_token_size=8192,
    )

    fake.text_chunks.upsert.assert_awaited_once()
    fake.doc_status.upsert.assert_awaited_once()
    status_payload = fake.doc_status.upsert.await_args.args[0]["doc-1"]
    assert status_payload["chunks_list"] == ["text-1", "img-1"]
    assert status_payload["chunks_count"] == 2


async def test_chunk_ids_for_docs_reads_lightrag_text_chunks() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            return await operation(self)

        async def fetch(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            return [{"id": "chunk-a"}, {"id": "chunk-b"}]

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = LightRAGStores(fake)

    result = await stores.chunk_ids_for_docs(["doc-1", "doc-2"])

    assert result == ["chunk-a", "chunk-b"]
    assert db.fetch_args is not None
    assert db.fetch_args[1] == "ws"
    assert db.fetch_args[2] == ["doc-1", "doc-2"]
