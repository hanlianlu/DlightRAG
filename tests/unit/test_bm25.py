# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL BM25 retrieval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.core.retrieval import bm25 as bm25_module
from dlightrag.core.retrieval.bm25 import (
    BM25_PROFILE_FALLBACK,
    BM25IndexOptions,
    BM25Profile,
    PostgresBM25,
    build_bm25_sql,
    detect_query_language,
)


def test_bm25_sql_filters_candidates() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_en",
        candidate_ids={"chunk-a"},
        limit=20,
    )

    assert "id = ANY" in sql
    assert "LIMIT $4" in sql
    assert "to_bm25query" in sql
    assert "idx_lightrag_doc_chunks_bm25_en" in sql


def test_bm25_sql_has_no_candidate_clause_when_unfiltered() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_simple",
        candidate_ids=None,
        limit=20,
    )

    assert "id = ANY" not in sql
    assert "LIMIT $3" in sql


def test_bm25_sql_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        build_bm25_sql(
            index_name="idx_lightrag_doc_chunks_bm25_simple",
            candidate_ids=None,
            limit=0,
        )


def test_bm25_index_options_render_pg_textsearch_with_tuning() -> None:
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    options = BM25IndexOptions(profile=profile, k1=1.4, b=0.65)

    assert options.create_index_sql() == (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65)"
    )


def test_bm25_index_options_render_qualified_text_config() -> None:
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",))
    options = BM25IndexOptions(profile=profile)

    assert "text_config='public.jiebacfg'" in options.create_index_sql()


def test_bm25_index_options_reject_unsafe_text_config() -> None:
    with pytest.raises(ValueError, match="unsafe BM25 text_config"):
        BM25Profile(name="en", text_config="english'; DROP TABLE x; --")


def test_bm25_index_options_match_real_pg_indexdef_format() -> None:
    profile = BM25Profile(name="simple", text_config="simple", fallback=True)
    options = BM25IndexOptions(profile=profile, k1=1.2, b=0.75)

    assert options.matches_indexdef(
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=simple, k1='1.2', b='0.75')"
    )


def test_query_language_detection_routes_plain_chinese() -> None:
    assert detect_query_language("现金流 风险 因素") == "zh"


def test_query_language_detection_routes_plain_english() -> None:
    assert detect_query_language("risk factors revenue guidance") == "en"


def test_query_language_detection_lets_lingua_handle_chinese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeDetector:
        def detect_language_of(self, query: str) -> SimpleNamespace:
            calls.append(query)
            return SimpleNamespace(name="CHINESE")

    monkeypatch.setattr(
        bm25_module,
        "_lingua_detector",
        lambda _language_codes: (FakeDetector(), {"CHINESE": "zh"}),
    )

    assert detect_query_language("现金流 风险 因素", supported_languages=("zh", "en")) == "zh"
    assert calls == ["现金流 风险 因素"]


def test_query_language_detection_uses_configured_lingua_language_set() -> None:
    assert (
        detect_query_language(
            "Wie hoch ist der Umsatz im letzten Quartal?",
            supported_languages=("de", "en"),
        )
        == "de"
    )


async def test_bm25_search_empty_candidate_set_short_circuits() -> None:
    bm25 = PostgresBM25(pool=AsyncMock(), workspace="default", profiles=[BM25_PROFILE_FALLBACK])

    assert await bm25.search("query", candidate_ids=set()) == []


async def test_bm25_search_maps_rows() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {"id": "chunk-a", "content": "hello world", "file_path": "a.md", "score": 1.5}
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        top_k=3,
        profiles=[BM25Profile(name="en", text_config="english", fallback=True)],
    )

    rows = await bm25.search("hello", candidate_ids={"chunk-a"})

    args = conn.fetch.await_args.args
    assert args[1] == "hello"
    assert args[2] == "default"
    assert args[3] == ["chunk-a"]
    assert args[4] == 3
    assert rows == [
        {
            "chunk_id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "bm25_profile": "en",
            "score": 1.5,
        }
    ]


async def test_bm25_ensure_index_rebuilds_when_options_change() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.2, b=0.75)",
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", fallback=True)
    bm25 = PostgresBM25(pool=pool, workspace="default", profiles=[profile])

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert executed == [
        "DROP INDEX IF EXISTS idx_lightrag_doc_chunks_bm25_en",
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65)",
    ]


async def test_bm25_ensure_index_keeps_matching_index() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.4, b=0.65)",
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", fallback=True)
    bm25 = PostgresBM25(pool=pool, workspace="default", profiles=[profile])

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    conn.execute.assert_not_awaited()


async def test_bm25_ensure_index_verifies_qualified_text_config_by_schema() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [1, None]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", fallback=True)
    bm25 = PostgresBM25(pool=pool, workspace="default", profiles=[profile])

    await bm25.ensure_indexes()

    first_fetch = conn.fetchval.await_args_list[0]
    assert first_fetch.args[1:] == ("public", "jiebacfg")
    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert executed == [
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_zh "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='public.jiebacfg', k1=1.2, b=0.75)"
    ]


async def test_bm25_routes_chinese_query_to_jieba_plus_simple_fallback() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "zh-hit", "content": "现金流", "file_path": "cn.md", "score": 2.0}],
        [{"id": "simple-hit", "content": "现金流", "file_path": "mix.md", "score": 1.0}],
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        profiles=[
            BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",)),
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ],
    )

    rows = await bm25.search("现金流", candidate_ids=None, top_k=5)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_zh" in fetched_sql[0]
    assert "idx_lightrag_doc_chunks_bm25_simple" in fetched_sql[1]
    assert len(conn.fetch.await_args_list) == 2
    assert {row["chunk_id"] for row in rows} == {"zh-hit", "simple-hit"}


async def test_bm25_routes_configured_language_to_matching_profile_plus_fallback() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "de-hit", "content": "Umsatz", "file_path": "de.md", "score": 2.0}],
        [{"id": "simple-hit", "content": "Umsatz", "file_path": "mix.md", "score": 1.0}],
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        profiles=[
            BM25Profile(name="de", text_config="german", languages=("de",)),
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ],
    )

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", candidate_ids=None)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_de" in fetched_sql[0]
    assert "idx_lightrag_doc_chunks_bm25_simple" in fetched_sql[1]
    assert len(conn.fetch.await_args_list) == 2
    assert {row["chunk_id"] for row in rows} == {"de-hit", "simple-hit"}


async def test_bm25_routes_region_language_tag_to_profile_plus_fallback() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "de-hit", "content": "Umsatz", "file_path": "de.md", "score": 2.0}],
        [{"id": "simple-hit", "content": "Umsatz", "file_path": "mix.md", "score": 1.0}],
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        profiles=[
            BM25Profile(name="de", text_config="german", languages=("de-DE",)),
            BM25Profile(name="en", text_config="english", languages=("en-US",)),
            BM25_PROFILE_FALLBACK,
        ],
    )

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", candidate_ids=None)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_de" in fetched_sql[0]
    assert "idx_lightrag_doc_chunks_bm25_simple" in fetched_sql[1]
    assert len(conn.fetch.await_args_list) == 2
    assert {row["chunk_id"] for row in rows} == {"de-hit", "simple-hit"}
