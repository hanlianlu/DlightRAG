# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL BM25 retrieval."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from dlightrag_rag.retrieval import MetadataScope

from dlightrag.core.retrieval import bm25 as bm25_module
from dlightrag.core.retrieval.bm25 import (
    BM25_LANGUAGE_COLUMN,
    BM25_PROFILE_FALLBACK,
    BM25IndexOptions,
    BM25Profile,
    PostgresBM25,
    build_bm25_sql,
    create_postgres_bm25,
    profiles_from_config,
    rebuild_postgres_bm25,
    required_postgres_extensions,
)


def _scope(*doc_ids: str, chunk_count: int = 12) -> MetadataScope:
    return MetadataScope(doc_ids=frozenset(doc_ids), chunk_count=chunk_count)


def test_bm25_sql_filters_candidates() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_en",
        scoped=True,
        limit=20,
        language="en",
    )

    assert "full_doc_id = ANY" in sql
    assert "LIMIT $4" in sql
    assert "to_bm25query" in sql
    assert "idx_lightrag_doc_chunks_bm25_en" in sql
    assert "dlightrag_bm25_language = 'en'" in sql


def test_bm25_sql_has_no_candidate_clause_when_unfiltered() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_simple",
        scoped=False,
        limit=20,
        language=None,
    )

    assert "full_doc_id = ANY" not in sql
    assert "LIMIT $3" in sql


def test_bm25_sql_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        build_bm25_sql(
            index_name="idx_lightrag_doc_chunks_bm25_simple",
            scoped=False,
            limit=0,
        )


def test_bm25_index_options_render_pg_textsearch_with_tuning() -> None:
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    options = BM25IndexOptions(profile=profile, k1=1.4, b=0.65)

    assert options.create_index_sql() == (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65) "
        "WHERE dlightrag_bm25_language = 'en'"
    )


def test_bm25_index_options_render_qualified_text_config() -> None:
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",))
    options = BM25IndexOptions(profile=profile)

    assert "text_config='public.jiebacfg'" in options.create_index_sql()


def test_bm25_language_profile_requires_one_language_bucket() -> None:
    with pytest.raises(ValueError, match="exactly one language"):
        BM25Profile(name="mixed", text_config="simple", languages=("de", "sv"))

    with pytest.raises(ValueError, match="exactly one language"):
        BM25Profile(name="empty", text_config="simple")


def test_bm25_fallback_profile_rejects_languages() -> None:
    with pytest.raises(ValueError, match="fallback profile must not declare languages"):
        BM25Profile(name="simple", text_config="simple", languages=("en",), fallback=True)


def test_bm25_profiles_from_config_preserves_runtime_fields() -> None:
    profiles = profiles_from_config(
        [
            SimpleNamespace(
                name="zh",
                text_config="public.jiebacfg",
                languages=["zh-CN"],
                fallback=False,
            ),
            SimpleNamespace(
                name="simple",
                text_config="simple",
                languages=[],
                fallback=True,
            ),
        ]
    )

    assert profiles == (
        BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",)),
        BM25_PROFILE_FALLBACK,
    )


def test_bm25_required_extensions_follow_profile_text_configs() -> None:
    assert required_postgres_extensions(
        [BM25Profile(name="en", text_config="english", languages=("en",))]
    ) == ("pg_textsearch",)
    assert required_postgres_extensions(
        [BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",))]
    ) == ("pg_textsearch", "pg_jieba")


async def test_create_postgres_bm25_returns_none_when_disabled() -> None:
    config = SimpleNamespace(bm25_enabled=False)

    assert await create_postgres_bm25(config, pool=object()) is None


@pytest.mark.parametrize("is_reader", [False, True])
async def test_create_postgres_bm25_provisions_for_service_role(
    monkeypatch: pytest.MonkeyPatch,
    is_reader: bool,
) -> None:
    from dlightrag.core.retrieval import bm25 as module

    instance = SimpleNamespace(ensure_indexes=AsyncMock(), verify_indexes=AsyncMock())
    constructor = MagicMock(return_value=instance)
    monkeypatch.setattr(module, "PostgresBM25", constructor)
    pool = object()
    profiles = (BM25_PROFILE_FALLBACK,)
    config = SimpleNamespace(
        bm25_enabled=True,
        is_reader=is_reader,
        workspace="research",
        bm25_profiles=[],
        bm25_k1=1.4,
        bm25_b=0.65,
    )

    result = await create_postgres_bm25(config, pool=pool, profiles=profiles)

    assert result is instance
    constructor.assert_called_once_with(
        pool=pool,
        workspace="research",
        profiles=profiles,
    )
    expected = instance.verify_indexes if is_reader else instance.ensure_indexes
    unexpected = instance.ensure_indexes if is_reader else instance.verify_indexes
    expected.assert_awaited_once_with(k1=1.4, b=0.65)
    unexpected.assert_not_awaited()


async def test_rebuild_postgres_bm25_provisions_then_relabels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.retrieval import bm25 as module

    retriever = SimpleNamespace(
        relabel_chunk_languages=AsyncMock(return_value={"processed_chunks": 3, "updated_chunks": 3})
    )
    create = AsyncMock(return_value=retriever)
    monkeypatch.setattr(module, "create_postgres_bm25", create)
    config = SimpleNamespace(bm25_enabled=True, is_reader=False)
    pool = object()

    stats = await rebuild_postgres_bm25(
        config,
        pool=pool,
        batch_size=25,
    )

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    create.assert_awaited_once_with(config, pool=pool)
    retriever.relabel_chunk_languages.assert_awaited_once_with(batch_size=25)


async def test_rebuild_postgres_bm25_skips_when_disabled() -> None:
    config = SimpleNamespace(bm25_enabled=False)

    assert await rebuild_postgres_bm25(config, pool=object()) == {
        "processed_chunks": 0,
        "updated_chunks": 0,
    }


async def test_rebuild_postgres_bm25_rejects_reader_role() -> None:
    config = SimpleNamespace(bm25_enabled=True, is_reader=True)

    with pytest.raises(RuntimeError, match="writer service role"):
        await rebuild_postgres_bm25(config, pool=object())


def test_bm25_index_options_reject_unsafe_text_config() -> None:
    with pytest.raises(ValueError, match="unsafe BM25 text_config"):
        BM25Profile(name="en", text_config="english'; DROP TABLE x; --")


def test_bm25_index_options_match_real_pg_indexdef_format() -> None:
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    options = BM25IndexOptions(profile=profile, k1=1.2, b=0.75)

    assert options.matches_indexdef(
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1='1.2', b='0.75') "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)"
    )


def test_bm25_index_options_keeps_simple_fallback_full_table() -> None:
    options = BM25IndexOptions(profile=BM25_PROFILE_FALLBACK)

    assert options.create_index_sql() == (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='simple', k1=1.2, b=0.75)"
    )


async def test_bm25_search_empty_candidate_set_short_circuits() -> None:
    bm25 = PostgresBM25(pool=AsyncMock(), workspace="default", profiles=[BM25_PROFILE_FALLBACK])

    assert await bm25.search("query", scope=MetadataScope(doc_ids=frozenset(), chunk_count=0)) == []


async def test_bm25_search_maps_rows() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {
            "id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "full_doc_id": "doc-a",
            "score": 1.5,
        }
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        top_k=3,
        profiles=[BM25Profile(name="en", text_config="english", fallback=True)],
    )

    rows = await bm25.search("hello", scope=_scope("doc-a"))

    args = conn.fetch.await_args.args
    assert args[1] == "hello"
    assert args[2] == "default"
    assert args[3] == ["doc-a"]
    assert args[4] == 3
    assert rows == [
        {
            "chunk_id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "full_doc_id": "doc-a",
            "bm25_profile": "en",
            "score": 1.5,
        }
    ]


async def test_bm25_relabels_existing_chunks_in_workspace_batches() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [
            {"id": "chunk-a", "content": "hello"},
            {"id": "chunk-b", "content": "现金流"},
        ],
        [{"id": "chunk-c", "content": "ambiguous"}],
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PostgresBM25(
        pool=pool,
        workspace="research",
        profiles=[
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ],
    )
    bm25._language_classifier = MagicMock()
    bm25._language_classifier.detect.side_effect = ["en", "simple", "simple"]

    stats = await bm25.relabel_chunk_languages(batch_size=2)

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    assert [call.args[1:] for call in conn.fetch.await_args_list] == [
        ("research", "", 2),
        ("research", "chunk-b", 2),
    ]
    assert "ORDER BY id" in conn.fetch.await_args_list[0].args[0]
    assert [call.args[1:] for call in conn.execute.await_args_list] == [
        ("research", ["chunk-a", "chunk-b"], ["en", "simple"]),
        ("research", ["chunk-c"], ["simple"]),
    ]
    assert "FROM UNNEST" in conn.execute.await_args_list[0].args[0]


async def test_bm25_relabel_rejects_non_positive_batch_size() -> None:
    bm25 = PostgresBM25(
        pool=AsyncMock(),
        workspace="default",
        profiles=[BM25_PROFILE_FALLBACK],
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        await bm25.relabel_chunk_languages(batch_size=0)


async def test_bm25_search_logs_profile_routing_and_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    with caplog.at_level(logging.INFO, logger="dlightrag.core.retrieval.bm25"):
        await bm25.search("hello", scope=_scope("doc-a"))

    assert "[BM25] search" in caplog.text
    assert "workspace=default" in caplog.text
    assert "query='hello'" in caplog.text
    assert "profiles=en" in caplog.text
    assert "candidate_scope=1" in caplog.text
    assert "returned=1" in caplog.text
    assert "top=chunk-a:en:1.500" in caplog.text


async def test_bm25_ensure_index_rebuilds_when_options_change() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.2, b=0.75) "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)",
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=simple, k1=1.4, b=0.65)",
    ]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        profiles=[profile, BM25_PROFILE_FALLBACK],
    )

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert "ALTER TABLE LIGHTRAG_DOC_CHUNKS ADD COLUMN IF NOT EXISTS " in executed[0]
    assert BM25_LANGUAGE_COLUMN in executed[0]
    assert "DROP INDEX IF EXISTS idx_lightrag_doc_chunks_bm25_en" in executed
    assert (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65) "
        "WHERE dlightrag_bm25_language = 'en'"
    ) in executed


async def test_bm25_ensure_index_keeps_matching_index() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.4, b=0.65) "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)",
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=simple, k1=1.4, b=0.65)",
    ]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    bm25 = PostgresBM25(
        pool=pool,
        workspace="default",
        profiles=[profile, BM25_PROFILE_FALLBACK],
    )

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert len(executed) == 3
    assert executed[0].startswith("ALTER TABLE LIGHTRAG_DOC_CHUNKS ADD COLUMN IF NOT EXISTS")
    assert executed[1].startswith(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_dlightrag_bm25_language"
    )
    assert executed[2].startswith(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_dlightrag_full_doc_id"
    )


async def test_bm25_ensure_index_verifies_qualified_text_config_by_schema() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [1, None]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", fallback=True)
    bm25 = PostgresBM25(pool=pool, workspace="default", profiles=[profile])

    await bm25.ensure_indexes()

    first_fetch = conn.fetchval.await_args_list[0]
    assert first_fetch.args[1:] == ("public", "jiebacfg")
    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_zh "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='public.jiebacfg', k1=1.2, b=0.75)"
    ) in executed


async def test_bm25_routes_chinese_query_to_jieba_profile_only() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "zh-hit", "content": "现金流", "file_path": "cn.md", "score": 2.0}],
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

    rows = await bm25.search("现金流", scope=None, top_k=5)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_zh" in fetched_sql[0]
    assert "dlightrag_bm25_language = 'zh'" in fetched_sql[0]
    assert len(conn.fetch.await_args_list) == 1
    assert {row["chunk_id"] for row in rows} == {"zh-hit"}


async def test_bm25_routes_configured_language_to_matching_profile_only() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "de-hit", "content": "Umsatz", "file_path": "de.md", "score": 2.0}],
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

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", scope=None)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_de" in fetched_sql[0]
    assert "dlightrag_bm25_language = 'de'" in fetched_sql[0]
    assert len(conn.fetch.await_args_list) == 1
    assert {row["chunk_id"] for row in rows} == {"de-hit"}


async def test_bm25_routes_region_language_tag_to_profile_only() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "de-hit", "content": "Umsatz", "file_path": "de.md", "score": 2.0}],
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

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", scope=None)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_de" in fetched_sql[0]
    assert len(conn.fetch.await_args_list) == 1
    assert {row["chunk_id"] for row in rows} == {"de-hit"}


async def test_bm25_routes_unknown_language_to_simple_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [{"id": "simple-hit", "content": "query", "file_path": "mix.md", "score": 1.0}],
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
    monkeypatch.setattr(
        bm25_module.BM25LanguageClassifier,
        "detect",
        lambda *_args, **_kwargs: "simple",
    )

    rows = await bm25.search("unsupported", scope=None)

    fetched_sql = [call.args[0] for call in conn.fetch.await_args_list]
    assert "idx_lightrag_doc_chunks_bm25_simple" in fetched_sql[0]
    assert "dlightrag_bm25_language" not in fetched_sql[0]
    assert len(conn.fetch.await_args_list) == 1
    assert {row["chunk_id"] for row in rows} == {"simple-hit"}
