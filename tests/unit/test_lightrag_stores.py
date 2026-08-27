# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG storage boundary adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
from dlightrag.engine.rag.lightrag.stores import LightRAGStores


class FakeLightRAG:
    def __init__(self) -> None:
        self.chunks_vdb = object()
        self.text_chunks = object()
        self.full_docs = object()
        self.doc_status = object()


def _stores(fake: FakeLightRAG) -> LightRAGStores:
    return LightRAGStores(fake, chunk_store=AsyncMock())


def test_lightrag_stores_validates_required_surfaces() -> None:
    fake = FakeLightRAG()
    stores = _stores(fake)

    assert stores.text_chunks is fake.text_chunks
    assert stores.full_docs is fake.full_docs


async def test_overwrite_chunk_vectors_requires_matching_dimension() -> None:
    stores = PGCorpusChunkStore(FakeLightRAG())

    with pytest.raises(ValueError, match="vector dimension"):
        await stores.overwrite_chunk_vectors(
            {"chunk-1": [0.1, 0.2]},
            embedding_dim=3,
        )


async def test_overwrite_chunk_vectors_updates_existing_rows_only() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.executed: list[tuple] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def executemany(self, sql, values) -> None:  # noqa: ANN001
            self.executed.append((sql, values))

    fake = FakeLightRAG()
    db = FakeDB()
    fake.chunks_vdb = SimpleNamespace(table_name="LIGHTRAG_DOC_CHUNKS", db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    await stores.overwrite_chunk_vectors(
        {"doc-1-mm-drawing-000": [0.1, 0.2, 0.3]},
        embedding_dim=3,
    )

    assert len(db.executed) == 1
    sql, values = db.executed[0]
    assert "UPDATE LIGHTRAG_DOC_CHUNKS" in sql
    assert "INSERT" not in sql
    assert values[0][0] == "ws"
    assert values[0][1] == "doc-1-mm-drawing-000"
    assert values[0][2] == [0.1, 0.2, 0.3]


async def test_overwrite_chunk_vectors_respects_batch_record_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.batches: list[list[tuple]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def executemany(self, sql, values) -> None:  # noqa: ANN001
            self.batches.append(list(values))

    monkeypatch.setattr(PGCorpusChunkStore, "_VECTOR_WRITE_MAX_RECORDS", 1)
    monkeypatch.setattr(PGCorpusChunkStore, "_VECTOR_WRITE_MAX_BYTES", 16_000_000)

    fake = FakeLightRAG()
    db = FakeDB()
    fake.chunks_vdb = SimpleNamespace(table_name="LIGHTRAG_VDB", db=db, workspace="ws")

    stores = PGCorpusChunkStore(fake)
    await stores.overwrite_chunk_vectors(
        {
            "img-1": [0.1, 0.2, 0.3],
            "img-2": [0.4, 0.5, 0.6],
        },
        embedding_dim=3,
    )

    assert len(db.batches) == 2
    assert [batch[0][1] for batch in db.batches] == ["img-1", "img-2"]


async def test_count_chunks_for_docs_counts_without_reading_ids() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchval(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            return 1470

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    result = await stores.count_chunks_for_docs(["doc-1", "doc-2"])

    assert result == 1470
    assert db.fetch_args is not None
    assert "count(*)" in db.fetch_args[0]
    assert db.fetch_args[1] == "ws"
    assert db.fetch_args[2] == ["doc-1", "doc-2"]


async def test_context_chunks_by_ids_formats_text_chunks() -> None:
    fake = FakeLightRAG()
    fake.text_chunks = AsyncMock()
    fake.text_chunks.get_by_ids.return_value = [
        {"content": "alpha", "file_path": "/tmp/a.pdf", "full_doc_id": "doc-a"},
        {"content": "beta", "file_path": "/tmp/b.pdf"},
    ]
    stores = _stores(fake)

    result = await stores.context_chunks_by_ids(["c1", "c2"])

    fake.text_chunks.get_by_ids.assert_awaited_once_with(["c1", "c2"])
    assert result == [
        {
            "chunk_id": "c1",
            "content": "alpha",
            "reference_id": "",
            "file_path": "/tmp/a.pdf",
            "full_doc_id": "doc-a",
        },
        {"chunk_id": "c2", "content": "beta", "reference_id": "", "file_path": "/tmp/b.pdf"},
    ]


async def test_fetch_chunk_contents_reads_lightrag_doc_chunks() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetch_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetch(self, *args):  # noqa: ANN002, ANN202
            self.fetch_args = args
            return [{"id": "chunk-a", "content": "hello"}]

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    result = await stores.fetch_chunk_contents(["chunk-a"])

    assert result == [{"id": "chunk-a", "content": "hello"}]
    assert db.fetch_args is not None
    assert "FROM LIGHTRAG_DOC_CHUNKS" in db.fetch_args[0]
    assert db.fetch_args[1] == "ws"
    assert db.fetch_args[2] == ["chunk-a"]


async def test_update_chunk_bm25_languages_uses_batch_update() -> None:
    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.execute_args: tuple | None = None

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def execute(self, *args):  # noqa: ANN002, ANN202
            self.execute_args = args

    fake = FakeLightRAG()
    db = FakeTextChunksDB()
    fake.text_chunks = SimpleNamespace(db=db, workspace="ws")
    stores = PGCorpusChunkStore(fake)

    await stores.update_chunk_bm25_languages({"chunk-a": "en", "chunk-b": "zh"})

    assert db.execute_args is not None
    sql = db.execute_args[0]
    assert "UPDATE LIGHTRAG_DOC_CHUNKS AS chunks" in sql
    assert "FROM UNNEST($2::text[], $3::text[])" in sql
    assert "dlightrag_bm25_language" in sql
    assert db.execute_args[1] == "ws"
    assert db.execute_args[2] == ["chunk-a", "chunk-b"]
    assert db.execute_args[3] == ["en", "zh"]
