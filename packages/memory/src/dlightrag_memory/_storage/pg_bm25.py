# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""pg_textsearch BM25 mechanics for the memory sparse leg.

A narrow port of the corpus BM25 knobs (same extension, same k1/b) kept
private to this package so the memory adapter never depends on
dlightrag.engine.rag. Like the corpus, each fact is labelled with its language
and indexed by a partial index under that language's text search
configuration, so every language keeps its own stopwords and stemming: a
positive score means the query and a fact share a content word.

Each index keeps one name per language whatever its configuration, so a process
started before an extension appeared keeps querying while the writer rebuilds
that index in place. A language whose configuration this database lacks (say
``public.jiebacfg`` without pg_jieba) is indexed under ``simple`` meanwhile.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import asyncpg

_logger = logging.getLogger(__name__)
_INDEX_PREFIX = "idx_dlightrag_memory_records_bm25"
_K1 = 1.2
_B = 0.75
_FALLBACK = "simple"

_LANGUAGE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_CONFIG = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?")
_WHITESPACE = re.compile(r"\s+")
_PREFIXED_INDEXES_SQL = (
    "SELECT indexname, indexdef FROM pg_indexes "
    "WHERE tablename = 'dlightrag_memory_records' AND indexname LIKE $1"
)
_HAS_TEXT_SEARCH_SQL = "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch')"
_CONFIGS_SQL = """
SELECT n.nspname || '.' || c.cfgname AS qualified, c.cfgname AS name
FROM pg_ts_config c
JOIN pg_namespace n ON n.oid = c.cfgnamespace
"""


def bm25_query_text(query: str) -> str:
    """The query text every memory index's tokenizer should see.

    pg_jieba indexes each whitespace run as a term (jaiminpan/pg_jieba#47), so a
    spaced query would match every record that contains a space. A full-width
    comma splits words exactly where the whitespace did: jieba drops it as a
    stopword, and PostgreSQL's default parser reads it as a separator.
    """
    return _WHITESPACE.sub("，", query)


def index_name(language: str) -> str:
    if not _LANGUAGE.fullmatch(language):
        raise ValueError(f"unsafe BM25 language: {language!r}")
    return f"{_INDEX_PREFIX}_{language}"


@dataclass(frozen=True)
class BM25IndexOptions:
    """One language's partial BM25 index over the memory table."""

    language: str
    text_config: str

    def __post_init__(self) -> None:
        index_name(self.language)
        if not _CONFIG.fullmatch(self.text_config):
            raise ValueError(f"unsafe BM25 text_config: {self.text_config!r}")

    @property
    def index_name(self) -> str:
        return index_name(self.language)

    def create_index_sql(self) -> str:
        return (
            f"CREATE INDEX {self.index_name} ON dlightrag_memory_records "
            f"USING bm25(body) WITH (text_config='{self.text_config}', k1={_K1:g}, b={_B:g}) "
            f"WHERE bm25_language = '{self.language}'"
        )

    def matches_indexdef(self, indexdef: str | None) -> bool:
        if not indexdef:
            return False
        normalized = re.sub(r"\s+", "", indexdef.lower().replace('"', "").replace("'", ""))
        config = self.text_config.lower()
        return (
            "usingbm25(body)" in normalized
            and (
                f"text_config={config}," in normalized
                or f"text_config={config}::regconfig," in normalized
            )
            and f"k1={_K1:g}" in normalized
            and f"b={_B:g}" in normalized
            and f"bm25_language={self.language}" in normalized
        )


async def install_text_search(conn: Any) -> None:
    """Create pg_textsearch, then pg_jieba beside it, where the server allows.

    Neither is a trusted extension, so a managed server without a superuser
    refuses them; the sparse leg then serves whatever the operator installed.
    Each statement runs in its own (sub)transaction, so a refusal leaves the
    connection usable.
    """
    for extension in ("pg_textsearch", "pg_jieba"):
        try:
            async with conn.transaction():
                await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
        except asyncpg.PostgresError as exc:
            _logger.warning("Profile Memory could not create %s: %s", extension, exc)
            return  # pg_jieba serves nothing here without pg_textsearch


async def served_indexes(
    conn: Any, text_configs: Mapping[str, str]
) -> tuple[BM25IndexOptions, ...]:
    """Each language's index under its configuration, or under ``simple`` while
    this database lacks it; none at all without pg_textsearch."""
    if not await conn.fetchval(_HAS_TEXT_SEARCH_SQL):
        return ()
    installed: set[str] = set()
    for row in await conn.fetch(_CONFIGS_SQL):
        installed.update((row["qualified"], row["name"]))
    return tuple(
        BM25IndexOptions(
            language=language,
            text_config=config if config in installed else text_configs[_FALLBACK],
        )
        for language, config in text_configs.items()
    )


async def ensure_bm25_indexes(conn: Any, text_configs: Mapping[str, str]) -> None:
    """Build or rebuild each language's BM25 index (writer only).

    A rebuild drops and recreates one index in a transaction, so a concurrent
    recall waits instead of finding no index. Every other index under the BM25
    prefix is dropped, so a retired name never keeps paying write cost.
    """
    options = await served_indexes(conn, text_configs)
    existing = {
        row["indexname"]: row["indexdef"]
        for row in await conn.fetch(_PREFIXED_INDEXES_SQL, f"{_INDEX_PREFIX}%")
    }
    for option in options:
        indexdef = existing.get(option.index_name)
        if option.matches_indexdef(indexdef):
            continue
        async with conn.transaction():
            if indexdef:
                await conn.execute(f"DROP INDEX IF EXISTS {option.index_name}")
            await conn.execute(option.create_index_sql())
    served = {option.index_name for option in options}
    for name in existing.keys() - served:
        await conn.execute(f'DROP INDEX IF EXISTS "{name}"')


async def served_bm25_languages(conn: Any, languages: Collection[str]) -> frozenset[str]:
    """The languages whose BM25 index this process may query; never DDL.

    Any configuration the writer built serves: a process never fails because
    an extension appeared after it started, or before the writer rebuilt.
    """
    existing = {
        row["indexname"]: row["indexdef"]
        for row in await conn.fetch(_PREFIXED_INDEXES_SQL, f"{_INDEX_PREFIX}%")
    }
    served = frozenset(language for language in languages if index_name(language) in existing)
    if not served:
        _logger.warning("Profile Memory matches facts without BM25: no index is built")
    elif fallback := sorted(
        language
        for language in served - {_FALLBACK}
        if "text_config=simple" in re.sub(r"[\s']", "", existing[index_name(language)].lower())
    ):
        _logger.warning(
            "Profile Memory BM25 indexes %s under simple: their configurations are missing",
            ", ".join(fallback),
        )
    return served


def build_bm25_sql(*, language: str, limit: int) -> str:
    """Rank one owner's active facts in one language through its BM25 index.

    The index's top-k scan stays near constant however many facts an owner
    keeps (19 ms at 10,000 facts, where scoring every fact took 360 ms), and it
    re-seeds until enough rows pass the owner filter. The language predicate
    selects that language's partial index and no other.
    """
    name = index_name(language)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    return (
        "SELECT owner_id, memory_id, kind, body, normalized_body, "  # noqa: S608 - checked names
        "origin_kind, origin_id, run_id, session_id, status, supersedes_id, "
        "embedding_fingerprint, "
        "created_at, updated_at, "
        f"-(body <@> to_bm25query($1, '{name}')) AS score "
        "FROM dlightrag_memory_records "
        "WHERE owner_id = $2 AND status = 'active' AND kind = 'fact' "
        f"AND bm25_language = '{language}' "
        f"ORDER BY body <@> to_bm25query($1, '{name}') "
        f"LIMIT {limit_value}"
    )


__all__ = [
    "BM25IndexOptions",
    "bm25_query_text",
    "build_bm25_sql",
    "ensure_bm25_indexes",
    "index_name",
    "install_text_search",
    "served_bm25_languages",
    "served_indexes",
]
