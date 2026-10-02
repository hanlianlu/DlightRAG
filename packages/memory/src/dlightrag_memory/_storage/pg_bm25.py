# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""pg_textsearch BM25 mechanics for the memory sparse leg.

A narrow port of the corpus BM25 knobs (same extension, same k1/b) kept
private to this package so the memory adapter never depends on
dlightrag.engine.rag. Two stopword-aware configs serve one table: ``english``
stems Latin text and ``public.jiebacfg`` segments Chinese. Queries hit both and
keep each record's best score, so a positive score means the query and the
record share a content word, never just "the" or "的".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_LATIN_CONFIG = "english"
_JIEBA_CONFIG = "public.jiebacfg"
_INDEX_PREFIX = "idx_dlightrag_memory_records_bm25"
_K1 = 1.2
_B = 0.75

_CONFIG_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?$")
_WHITESPACE = re.compile(r"\s+")
_PREFIXED_INDEXES_SQL = (
    "SELECT indexname FROM pg_indexes "
    "WHERE tablename = 'dlightrag_memory_records' AND indexname LIKE $1"
)


def bm25_query_text(text_config: str, query: str) -> str:
    """The query text one index's tokenizer should see.

    pg_jieba indexes each whitespace run as a term (jaiminpan/pg_jieba#47), so a
    spaced query would match every record that contains a space. A full-width
    comma is both a jieba separator and a jieba stopword: it splits words
    exactly where the whitespace did and adds no term.
    """
    return _WHITESPACE.sub("，", query) if text_config == _JIEBA_CONFIG else query


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def validate_text_config(text_config: str) -> str:
    value = str(text_config).strip()
    if not _CONFIG_NAME_RE.fullmatch(value):
        raise ValueError(f"unsafe BM25 text_config: {text_config!r}")
    return value


@dataclass(frozen=True)
class BM25IndexOptions:
    """One memory-table BM25 index with the corpus-tuned parameters."""

    index_name: str
    text_config: str
    k1: float = _K1
    b: float = _B

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        validate_text_config(self.text_config)

    def create_index_sql(self) -> str:
        config = validate_text_config(self.text_config)
        return (
            f"CREATE INDEX {self.index_name} ON dlightrag_memory_records "
            f"USING bm25(body) WITH (text_config='{config}', "
            f"k1={_format_float(self.k1)}, b={_format_float(self.b)})"
        )

    def matches_indexdef(self, indexdef: str | None) -> bool:
        if not indexdef:
            return False
        normalized = re.sub(r"\s+", "", indexdef.lower().replace('"', "").replace("'", ""))
        config = validate_text_config(self.text_config).lower()
        return (
            self.index_name.lower() in normalized
            and "usingbm25(body)" in normalized
            and (
                f"text_config={config}" in normalized
                or f"text_config={config}::regconfig" in normalized
            )
            and f"k1={_format_float(self.k1)}" in normalized
            and f"b={_format_float(self.b)}" in normalized
        )


def index_name(suffix: str) -> str:
    return _validate_index_name(f"{_INDEX_PREFIX}_{suffix}")


def _validate_index_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"unsafe index name: {name!r}")
    return name


def extension_bootstrap_sql() -> tuple[str, ...]:
    """The extensions the sparse leg needs, bootstrapped like root does."""
    return (
        "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
        "CREATE EXTENSION IF NOT EXISTS pg_jieba",
    )


async def text_configs_available(conn: Any) -> tuple[str, ...]:
    """Return the installed textsearch configs this adapter can serve.

    ``english`` is pg_catalog built-in; ``public.jiebacfg`` comes from the
    pg_jieba extension, matching the corpus BM25 profiles.
    """
    jieba = await conn.fetchval(
        """
        SELECT 1
        FROM pg_ts_config c
        JOIN pg_namespace n ON n.oid = c.cfgnamespace
        WHERE n.nspname = 'public' AND c.cfgname = 'jiebacfg'
        LIMIT 1
        """
    )
    return (_LATIN_CONFIG, _JIEBA_CONFIG) if jieba else (_LATIN_CONFIG,)


def desired_indexes(available: tuple[str, ...]) -> tuple[BM25IndexOptions, ...]:
    """The indexes to provision: english always, jieba when installed."""
    options = [BM25IndexOptions(index_name=index_name("english"), text_config=_LATIN_CONFIG)]
    if _JIEBA_CONFIG in available:
        options.append(BM25IndexOptions(index_name=index_name("jieba"), text_config=_JIEBA_CONFIG))
    return tuple(options)


def build_bm25_sql(*, index_name: str, limit: int) -> str:
    """Rank one owner's active facts against ``$1``; non-matching facts score 0."""
    safe_index = _validate_index_name(index_name)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    return (  # noqa: S608 - interpolates only the validated index name
        "SELECT owner_id, memory_id, kind, body, normalized_body, "  # noqa: S608
        "origin_kind, origin_id, run_id, session_id, status, supersedes_id, "
        "embedding_fingerprint, "
        "created_at, updated_at, "
        f"-(body <@> to_bm25query($1, '{safe_index}')) AS score "  # noqa: S608
        "FROM dlightrag_memory_records "
        "WHERE owner_id = $2 AND status = 'active' AND kind = 'fact' "
        f"ORDER BY body <@> to_bm25query($1, '{safe_index}') "  # noqa: S608
        "LIMIT " + str(limit_value)
    )


async def ensure_bm25_indexes(
    conn: Any,
    *,
    available: tuple[str, ...] | None = None,
    verify_only: bool = False,
) -> tuple[BM25IndexOptions, ...]:
    """Provision or validate the memory-table BM25 indexes.

    With ``verify_only`` this performs no DDL: readers load the served indexes
    and fail when a configured index is missing, matching the corpus verify
    path. Otherwise every other index under the BM25 prefix is dropped, so a
    retired config never keeps paying write cost.
    """
    installed = available if available is not None else await text_configs_available(conn)
    options = desired_indexes(installed)
    for option in options:
        indexdef = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1", option.index_name
        )
        if option.matches_indexdef(indexdef):
            continue
        if verify_only:
            raise RuntimeError(
                f"BM25 index {option.index_name} is missing or does not match configured "
                "options; initialize it on the writer first"
            )
        if indexdef:
            await conn.execute(f"DROP INDEX IF EXISTS {option.index_name}")
        await conn.execute(option.create_index_sql())
    if not verify_only:
        served = {option.index_name for option in options}
        for row in await conn.fetch(_PREFIXED_INDEXES_SQL, f"{_INDEX_PREFIX}_%"):
            if row["indexname"] not in served:
                await conn.execute(f"DROP INDEX IF EXISTS {_validate_index_name(row['indexname'])}")
    return options


__all__ = [
    "BM25IndexOptions",
    "bm25_query_text",
    "build_bm25_sql",
    "desired_indexes",
    "ensure_bm25_indexes",
    "extension_bootstrap_sql",
    "index_name",
    "text_configs_available",
    "validate_text_config",
]
