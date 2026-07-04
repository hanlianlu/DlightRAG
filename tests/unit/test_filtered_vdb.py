# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for strict metadata in-filtering context."""

from unittest.mock import AsyncMock

from dlightrag.core.retrieval.filtered_vdb import _active_filter, metadata_filter_scope


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


async def test_empty_candidate_set_is_active_filter() -> None:
    async with metadata_filter_scope(set()):
        assert _active_filter.get() == set()


async def test_none_candidate_set_is_no_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None


async def test_large_candidate_pg_search_places_distance_filter_outside_cte() -> None:
    storage = _FakePGVectorStorage()
    wrapper = __import__(
        "dlightrag.core.retrieval.filtered_vdb",
        fromlist=["FilteredVectorStorage"],
    ).FilteredVectorStorage(
        original=storage,
        embedding_func=AsyncMock(),
        exact_threshold=1,
    )

    await wrapper._pg_filtered_search([0.1, 0.2, 0.3], {"c1", "c2"}, top_k=5)

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
