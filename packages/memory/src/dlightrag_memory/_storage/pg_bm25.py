# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""pg_textsearch BM25 mechanics for the memory sparse leg.

A narrow port of the corpus BM25 knobs (same extension, same k1/b) kept
private to this package so the memory adapter never depends on
dlightrag.engine.rag. One stopword-aware index serves the table:
``public.jiebacfg`` when pg_jieba is installed, which segments Chinese and drops
Chinese and English function words, else ``english``. A positive score means
the query and a fact share a content word, never just "the" or "的"; function
words of other languages still count.

The index keeps one name whatever its configuration, and queries read the same
text under either, so a process started before an extension appeared keeps
querying while the writer rebuilds the index in place.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import asyncpg

_logger = logging.getLogger(__name__)
_ENGLISH_CONFIG = "english"
_JIEBA_CONFIG = "public.jiebacfg"
INDEX_NAME = "idx_dlightrag_memory_records_bm25"
_K1 = 1.2
_B = 0.75

_WHITESPACE = re.compile(r"\s+")
_INDEXDEF_SQL = "SELECT indexdef FROM pg_indexes WHERE indexname = $1"
_PREFIXED_INDEXES_SQL = (
    "SELECT indexname FROM pg_indexes "
    "WHERE tablename = 'dlightrag_memory_records' AND indexname LIKE $1"
)
_SERVED_CONFIG_SQL = """
SELECT
    EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') AS bm25,
    EXISTS (
        SELECT 1
        FROM pg_ts_config c
        JOIN pg_namespace n ON n.oid = c.cfgnamespace
        WHERE n.nspname = 'public' AND c.cfgname = 'jiebacfg'
    ) AS jieba
"""


def bm25_query_text(query: str) -> str:
    """The query text the memory index's tokenizer should see.

    pg_jieba indexes each whitespace run as a term (jaiminpan/pg_jieba#47), so a
    spaced query would match every record that contains a space. A full-width
    comma splits words exactly where the whitespace did: jieba drops it as a
    stopword, and the english configuration's parser reads it as a separator.
    """
    return _WHITESPACE.sub("，", query)


@dataclass(frozen=True)
class BM25IndexOptions:
    """The memory-table BM25 index under one configuration."""

    text_config: str

    def create_index_sql(self) -> str:
        return (
            f"CREATE INDEX {INDEX_NAME} ON dlightrag_memory_records "
            f"USING bm25(body) WITH (text_config='{self.text_config}', k1={_K1:g}, b={_B:g})"
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


async def served_config(conn: Any) -> str | None:
    """jiebacfg, else english, else None when pg_textsearch is not installed."""
    row = await conn.fetchrow(_SERVED_CONFIG_SQL)
    if not row["bm25"]:
        return None
    return _JIEBA_CONFIG if row["jieba"] else _ENGLISH_CONFIG


async def ensure_bm25_index(conn: Any) -> None:
    """Build or rebuild the BM25 index for the served configuration (writer only).

    A rebuild drops and recreates the one index in a transaction, so a
    concurrent recall waits instead of finding no index. Every other index
    under the BM25 prefix is dropped, so a retired name never keeps paying
    write cost.
    """
    config = await served_config(conn)
    if config is None:
        return
    option = BM25IndexOptions(text_config=config)
    indexdef = await conn.fetchval(_INDEXDEF_SQL, INDEX_NAME)
    if not option.matches_indexdef(indexdef):
        async with conn.transaction():
            if indexdef:
                await conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
            await conn.execute(option.create_index_sql())
    for row in await conn.fetch(_PREFIXED_INDEXES_SQL, f"{INDEX_NAME}_%"):
        await conn.execute(f'DROP INDEX IF EXISTS "{row["indexname"]}"')


async def served_bm25_index(conn: Any) -> str | None:
    """The BM25 index this process may query, or None; never DDL.

    Any configuration the writer built serves: a process never fails because
    an extension appeared after it started, or before the writer rebuilt.
    """
    indexdef = await conn.fetchval(_INDEXDEF_SQL, INDEX_NAME)
    if not indexdef or "usingbm25(body)" not in re.sub(r"\s+", "", indexdef.lower()):
        _logger.warning("Profile Memory matches facts without BM25: no index is built")
        return None
    if "jiebacfg" not in indexdef:
        _logger.warning("Profile Memory BM25 runs on english: pg_jieba is missing")
    return INDEX_NAME


def build_bm25_sql(*, limit: int) -> str:
    """Rank one owner's active facts against ``$1``; non-matching facts score 0.

    Ordering by the score itself keeps the plan on the owner's rows whatever the
    table statistics say, instead of a BM25 index scan over every owner.
    """
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    return (
        "SELECT owner_id, memory_id, kind, body, normalized_body, "  # noqa: S608 - fixed name
        "origin_kind, origin_id, run_id, session_id, status, supersedes_id, "
        "embedding_fingerprint, "
        "created_at, updated_at, "
        f"-(body <@> to_bm25query($1, '{INDEX_NAME}')) AS score "
        "FROM dlightrag_memory_records "
        "WHERE owner_id = $2 AND status = 'active' AND kind = 'fact' "
        "ORDER BY score DESC "
        f"LIMIT {limit_value}"
    )


__all__ = [
    "INDEX_NAME",
    "BM25IndexOptions",
    "bm25_query_text",
    "build_bm25_sql",
    "ensure_bm25_index",
    "install_text_search",
    "served_bm25_index",
    "served_config",
]
