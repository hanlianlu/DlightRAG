# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for strict metadata in-filtering context."""

from unittest.mock import AsyncMock

from dlightrag_rag.retrieval import MetadataScope
from dlightrag_rag.retrieval.filtering import (
    FilteredChunkStore,
    FilteredVectorStorage,
    _active_filter,
    metadata_filter_scope,
)

from dlightrag.adapters.postgres.corpus_vectors import PGFilteredVectorSearch


class _FakeDB:
    vector_index_type = "HNSW"

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[object, ...] = ()
        self.local_settings: list[str] = []

    async def _run_with_retry(self, operation):
        return await operation(self)

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, sql: str) -> None:
        self.local_settings.append(sql)

    async def fetch(self, sql: str, *params):
        self.sql = sql
        self.params = params
        return []


class _FakePGVectorStorage:
    table_name = "lightrag_vdb_chunks_test"
    workspace = "default"
    cosine_better_than_threshold = 0.3

    def __init__(self) -> None:
        self.db = _FakeDB()


_FakePGVectorStorage.__name__ = "PGVectorStorage"


async def test_empty_scope_is_active_filter() -> None:
    empty = MetadataScope(doc_ids=frozenset(), chunk_count=0)
    async with metadata_filter_scope(empty):
        assert _active_filter.get() == empty


async def test_none_scope_is_no_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None


async def test_filtered_query_uses_query_embedding_context() -> None:
    storage = type(
        "PGVectorStorage",
        (),
        {
            "table_name": "lightrag_vdb_chunks_test",
            "workspace": "default",
            "cosine_better_than_threshold": 0.3,
            "db": _FakeDB(),
        },
    )()
    embedding_func = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    filtered_search = AsyncMock()
    filtered_search.search.return_value = []
    wrapper = FilteredVectorStorage(
        original=storage,
        embedding_func=embedding_func,
        filtered_search=filtered_search,
    )

    async with metadata_filter_scope(MetadataScope(doc_ids=frozenset({"doc-1"}), chunk_count=3)):
        await wrapper.query("question", top_k=5)

    embedding_func.assert_awaited_once_with(["question"], context="query")


async def test_large_candidate_pg_search_places_distance_filter_outside_cte() -> None:
    storage = _FakePGVectorStorage()
    search = PGFilteredVectorSearch(storage, exact_threshold=1)

    await search.search(
        [0.1, 0.2, 0.3],
        scope=MetadataScope(doc_ids=frozenset({"doc-1", "doc-2"}), chunk_count=9_000),
        top_k=5,
    )

    assert storage.db.local_settings == [
        "SET LOCAL hnsw.iterative_scan = 'relaxed_order'",
        "SET LOCAL hnsw.max_scan_tuples = 20000",
    ]
    assert storage.db.sql is not None
    assert "WITH nearest_results AS MATERIALIZED" in storage.db.sql
    assert "FROM nearest_results" in storage.db.sql
    cte_sql, outer_sql = storage.db.sql.split("FROM nearest_results", maxsplit=1)
    assert "score > $4" not in cte_sql
    assert "score > $4" in outer_sql


class _FakeChunkKV:
    """Mimics PGKVStorage.get_by_ids: caller ordering preserved, None for misses."""

    workspace = "default"

    def __init__(self, rows: dict[str, dict[str, object]]) -> None:
        self._rows = rows
        self.requested: list[list[str]] = []

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, object] | None]:
        self.requested.append(list(ids))
        return [self._rows.get(chunk_id) for chunk_id in ids]


def _chunk(chunk_id: str, doc_id: str | None) -> dict[str, object]:
    return {"id": chunk_id, "content": chunk_id, "full_doc_id": doc_id}


async def test_chunk_store_passes_through_without_scope() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    store = FilteredChunkStore(original=kv)

    rows = await store.get_by_ids(["c1", "c2"])

    assert [r["id"] for r in rows if r is not None] == ["c1", "c2"]


async def test_chunk_store_nulls_out_of_scope_rows() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    store = FilteredChunkStore(original=kv)
    scope = MetadataScope(doc_ids=frozenset({"doc-1"}), chunk_count=1)

    async with metadata_filter_scope(scope) as stats:
        rows = await store.get_by_ids(["c1", "c2"])

    # Positional alignment is the storage contract both KG legs zip against.
    assert rows[0] is not None and rows[0]["id"] == "c1"
    assert rows[1] is None
    assert stats.kg_chunks_dropped == 1


async def test_chunk_store_drops_rows_without_document_attribution() -> None:
    kv = _FakeChunkKV({"c1": _chunk("c1", None)})
    store = FilteredChunkStore(original=kv)

    async with metadata_filter_scope(MetadataScope(doc_ids=frozenset({"doc-1"}), chunk_count=1)):
        rows = await store.get_by_ids(["c1"])

    assert rows == [None]


async def test_chunk_store_still_requests_every_id() -> None:
    """Filtering must not shorten the request: callers zip results against their ids."""
    kv = _FakeChunkKV({"c1": _chunk("c1", "doc-1"), "c2": _chunk("c2", "doc-2")})
    store = FilteredChunkStore(original=kv)

    async with metadata_filter_scope(MetadataScope(doc_ids=frozenset({"doc-1"}), chunk_count=1)):
        rows = await store.get_by_ids(["c1", "c2"])

    assert kv.requested == [["c1", "c2"]]
    assert len(rows) == 2


async def test_chunk_store_proxies_unknown_attributes() -> None:
    kv = _FakeChunkKV({})
    store = FilteredChunkStore(original=kv)

    assert store.workspace == "default"


async def test_stats_stay_zero_without_scope() -> None:
    async with metadata_filter_scope(None) as stats:
        assert stats.kg_chunks_dropped == 0
