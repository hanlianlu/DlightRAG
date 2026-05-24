# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL BM25 retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from dlightrag.core.retrieval.bm25 import BM25_INDEX, PostgresBM25, build_bm25_sql


def test_bm25_sql_filters_candidates() -> None:
    sql = build_bm25_sql(candidate_ids={"chunk-a"}, limit=20)

    assert "id = ANY" in sql
    assert "to_bm25query" in sql
    assert BM25_INDEX in sql


def test_bm25_sql_has_no_candidate_clause_when_unfiltered() -> None:
    sql = build_bm25_sql(candidate_ids=None, limit=20)

    assert "id = ANY" not in sql


async def test_bm25_search_empty_candidate_set_short_circuits() -> None:
    bm25 = PostgresBM25(pool=AsyncMock(), workspace="default")

    assert await bm25.search("query", candidate_ids=set()) == []


async def test_bm25_search_maps_rows() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {"id": "chunk-a", "content": "hello world", "file_path": "a.md", "score": 1.5}
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(pool=pool, workspace="default", top_k=3)

    rows = await bm25.search("hello", candidate_ids={"chunk-a"})

    assert rows == [
        {
            "chunk_id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "score": 1.5,
        }
    ]
