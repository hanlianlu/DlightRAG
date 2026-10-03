# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata scope rules that hold without a database.

These pin input refusals, the resolver's exact-then-contains decision, an empty
scope that skips its search, and the guarantee that no document-id array crosses
into Python. The predicates, legs and plans run against PostgreSQL in
tests/integration/test_metadata_scope_pg.py.
"""

import json
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.corpus_vectors import PGChunkVectorStore
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    metadata_match_conditions,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_scope import build_bounded_scope_probe
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope


def _scope(*, candidate_count: int, candidate_count_exact: bool = True) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=True,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


# ---------------------------------------------------------------------------
# Shared predicate builder
# ---------------------------------------------------------------------------


def test_invalid_filename_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="filename mode"):
        metadata_match_conditions(
            "ws",
            MetadataFilter(filename="x"),
            filename_mode="regex",
        )


def test_custom_key_collisions_resolve_deterministically_in_caller_order() -> None:
    filters = MetadataFilter(custom={"A": 1, "a": 2})

    conditions, params = metadata_match_conditions("ws", filters, filename_mode="exact")

    # The filter model folds keys once, mirroring the ingest normalization:
    # the later key wins, and the bound containment object carries it.
    assert json.loads(params[1]) == {"a": 2}


# ---------------------------------------------------------------------------
# Bounded scope preflight
# ---------------------------------------------------------------------------


def test_probe_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        build_bounded_scope_probe(
            "ws",
            MetadataFilter(filename="x"),
            filename_mode="exact",
            threshold=-1,
        )


async def test_scope_resolver_reports_empty_scope_when_no_document_matches() -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": False, "chunk_count": 0}

    lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
    db = FakeTextChunksDB()
    lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
    stores = PGCorpusChunkStore(lightrag, exact_threshold=8192)

    scope = await stores.resolve_scope(MetadataFilter(filename="missing.pdf"))

    assert bool(scope) is False
    assert scope.candidate_count == 0
    assert scope.candidate_count_exact is True
    assert len(db.fetches) == 2  # exact miss widens to contains, which also misses


async def test_scope_resolver_reports_at_threshold_and_sentinel_exactly() -> None:
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self, count: int) -> None:
            self._count = count
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": True, "chunk_count": self._count}

    for count, exact in ((3, True), (4, False)):
        lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
        db = FakeTextChunksDB(count)
        lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
        stores = PGCorpusChunkStore(lightrag, exact_threshold=3)

        scope = await stores.resolve_scope(MetadataFilter(file_extension="pdf"))

        assert scope.candidate_count == count
        assert scope.candidate_count_exact is exact
        assert len(db.fetches) == 1  # no filename: never widens


# ---------------------------------------------------------------------------
# Vector legs
# ---------------------------------------------------------------------------


class _FakeVectorDB:
    vector_index_type = "HNSW"

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: tuple[object, ...] = ()

    async def _run_with_retry(self, operation):
        return await operation(self)

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, sql: str) -> None:
        return None

    async def fetch(self, sql: str, *params):
        self.sql = sql
        self.params = params
        return []


def _vector_storage() -> Any:
    storage = type(
        "PGVectorStorage",
        (),
        {
            "table_name": "lightrag_vdb_chunks_test",
            "workspace": "default",
            "cosine_better_than_threshold": 0.3,
            "db": _FakeVectorDB(),
        },
    )()
    return storage


def test_filtered_vector_search_requires_postgres_capabilities() -> None:
    with pytest.raises(RuntimeError, match="cosine_better_than_threshold"):
        PGChunkVectorStore(type("IncompleteVectorStorage", (), {})())


async def test_scoped_vector_search_skips_an_empty_scope_entirely() -> None:
    storage = _vector_storage()
    search = PGChunkVectorStore(storage)

    scope = MetadataScope(
        filters=MetadataFilter(filename="missing.pdf"),
        filename_mode="exact",
        doc_exists=False,
        candidate_count=0,
        candidate_count_exact=True,
    )
    assert await search.search([0.1], scope=scope, top_k=3) == []
    assert storage.db.sql is None


async def test_retrieval_path_never_binds_a_document_id_array() -> None:
    """The complete doc-id match set must not cross the Python boundary.

    The graph chunk guard still binds the *already-bounded requested chunk ids*;
    nothing on the retrieval path binds a document-id array.
    """
    from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore

    class FakeTextChunksDB:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def _run_with_retry(self, operation, timing_label=None):  # noqa: ANN001, ANN202
            assert timing_label is None or isinstance(timing_label, str)
            return await operation(self)

        async def fetchrow(self, *args):  # noqa: ANN002, ANN202
            self.fetches.append(args)
            return {"doc_exists": True, "chunk_count": 2}

    lightrag: Any = type("L", (), {"chunks_vdb": object(), "text_chunks": object()})()
    db = FakeTextChunksDB()
    lightrag.text_chunks = type("T", (), {"db": db, "workspace": "ws"})()
    stores = PGCorpusChunkStore(lightrag)

    await stores.resolve_scope(MetadataFilter(filename="x.pdf"))

    probe_params = db.fetches[0][1:]
    assert all(not isinstance(param, list) for param in probe_params)

    storage = _vector_storage()
    await PGChunkVectorStore(storage).search(
        [0.1],
        scope=_scope(candidate_count=1),
        top_k=1,
    )
    assert all(not isinstance(param, list) for param in storage.db.params)
