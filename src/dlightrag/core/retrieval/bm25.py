# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL BM25 search over LightRAG document chunks."""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any

from dlightrag.core.retrieval.bm25_language import (
    BM25_FALLBACK_LANGUAGE,
    BM25_LANGUAGE_COLUMN,
    BM25LanguageClassifier,
    normalize_language_code,
)
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.storage.sql_identifiers import pg_identifier, pg_qualified_identifier

BM25_INDEX_PREFIX = pg_identifier("idx_lightrag_doc_chunks_bm25")
BM25_LANGUAGE_INDEX = pg_identifier("idx_lightrag_doc_chunks_dlightrag_bm25_language")
BM25_TABLE = pg_identifier("LIGHTRAG_DOC_CHUNKS")
logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _format_bm25_top(chunks: list[dict[str, Any]], *, limit: int = 3) -> str:
    parts: list[str] = []
    for chunk in chunks[:limit]:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "?")
        profile = str(chunk.get("bm25_profile") or "?")
        score = chunk.get("score")
        if score is None:
            score_text = "?"
        else:
            try:
                score_text = f"{float(score):.3f}"
            except (TypeError, ValueError):
                score_text = "?"
        parts.append(f"{chunk_id}:{profile}:{score_text}")
    if len(chunks) > limit:
        parts.append(f"+{len(chunks) - limit}")
    return ",".join(parts) if parts else "none"


def _sql_language_literal(language: str) -> str:
    """Return a validated BM25 language code as a SQL string literal."""
    return pg_identifier(normalize_language_code(language))


def validate_profile_text_config(text_config: str) -> str:
    """Validate a pg_textsearch text_config name before embedding it in DDL."""
    value = str(text_config).strip()
    try:
        return pg_qualified_identifier(value)
    except ValueError as exc:
        raise ValueError(f"unsafe BM25 text_config: {text_config!r}") from exc


@dataclass(frozen=True)
class BM25Profile:
    """A chunk-language-aware pg_textsearch BM25 index profile."""

    name: str
    text_config: str
    languages: tuple[str, ...] = ()
    fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", pg_identifier(str(self.name).strip()))
        object.__setattr__(self, "text_config", validate_profile_text_config(self.text_config))
        normalized_languages = tuple(
            code for language in self.languages if (code := normalize_language_code(language))
        )
        if self.fallback and normalized_languages:
            raise ValueError("BM25 fallback profile must not declare languages")
        if not self.fallback and len(normalized_languages) != 1:
            raise ValueError("BM25 language profiles must declare exactly one language")
        object.__setattr__(
            self,
            "languages",
            normalized_languages,
        )

    @property
    def index_name(self) -> str:
        return pg_identifier(f"{BM25_INDEX_PREFIX}_{self.name}")

    @property
    def language_bucket(self) -> str | None:
        if self.fallback or not self.languages:
            return None
        return self.languages[0]


BM25_PROFILE_FALLBACK = BM25Profile(name="simple", text_config="simple", fallback=True)


@dataclass(frozen=True)
class BM25IndexOptions:
    """pg_textsearch BM25 index options managed by DlightRAG."""

    profile: BM25Profile = BM25_PROFILE_FALLBACK
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")

    def create_index_sql(self) -> str:
        sql = (
            f"CREATE INDEX {self.profile.index_name} ON {BM25_TABLE} USING bm25(content) "
            f"WITH (text_config='{self.profile.text_config}', "
            f"k1={_format_float(self.k1)}, b={_format_float(self.b)})"
        )
        if self.profile.language_bucket is not None:
            sql += (
                f" WHERE {BM25_LANGUAGE_COLUMN} = "
                f"'{_sql_language_literal(self.profile.language_bucket)}'"
            )
        return sql

    def matches_indexdef(self, indexdef: str | None) -> bool:
        if not indexdef:
            return False
        normalized = re.sub(r"\s+", "", indexdef.lower().replace('"', "").replace("'", ""))
        text_config = self.profile.text_config.lower()
        matches = (
            "usingbm25(content)" in normalized
            and self.profile.index_name.lower() in normalized
            and (
                f"text_config={text_config}" in normalized
                or f"text_config={text_config}::regconfig" in normalized
            )
            and f"k1={_format_float(self.k1)}" in normalized
            and f"b={_format_float(self.b)}" in normalized
        )
        if not matches:
            return False
        if self.profile.language_bucket is None:
            return "where" not in normalized
        return (
            "where" in normalized
            and BM25_LANGUAGE_COLUMN.lower() in normalized
            and self.profile.language_bucket.lower() in normalized
        )


def build_bm25_sql(
    *,
    index_name: str,
    candidate_ids: set[str] | None,
    limit: int,
    language: str | None = None,
) -> str:
    """Build a pg_textsearch BM25 query with optional hard candidate filter."""
    safe_index = pg_identifier(index_name)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    candidate_clause = "AND id = ANY($3::text[])" if candidate_ids is not None else ""
    language_clause = (
        f"AND {BM25_LANGUAGE_COLUMN} = '{_sql_language_literal(language)}'" if language else ""
    )
    limit_placeholder = "$4" if candidate_ids is not None else "$3"
    return f"""
        SELECT id, content, file_path,
               -(content <@> to_bm25query($1, '{safe_index}')) AS score
        FROM {BM25_TABLE}
        WHERE workspace = $2
        {candidate_clause}
        {language_clause}
        ORDER BY content <@> to_bm25query($1, '{safe_index}')
        LIMIT {limit_placeholder}
    """


class PostgresBM25:
    """BM25 retriever backed by pg_textsearch."""

    def __init__(
        self,
        *,
        pool: Any,
        workspace: str,
        top_k: int = 40,
        profiles: list[BM25Profile] | tuple[BM25Profile, ...] | None = None,
    ) -> None:
        self._pool = pool
        self._workspace = workspace
        self._top_k = top_k
        self._profiles = tuple(profiles or (BM25_PROFILE_FALLBACK,))
        self._language_classifier = BM25LanguageClassifier(self._profile_languages)
        if not any(profile.fallback for profile in self._profiles):
            raise ValueError("At least one BM25 profile must be marked fallback")

    async def _run(self, operation):
        run = getattr(self._pool, "run", None)
        if callable(run) and inspect.iscoroutinefunction(run):
            return await run(operation)
        async with self._pool.acquire() as conn:
            return await operation(conn)

    async def ensure_indexes(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        options_by_profile = [
            BM25IndexOptions(profile=profile, k1=k1, b=b) for profile in self._profiles
        ]

        async def _operation(conn: Any) -> None:
            await self._ensure_schema(conn)
            for options in options_by_profile:
                await self._verify_text_config(conn, options.profile.text_config)
                indexdef = await self._fetch_indexdef(conn, options.profile.index_name)
                if options.matches_indexdef(indexdef):
                    continue
                if indexdef:
                    await conn.execute(f"DROP INDEX IF EXISTS {options.profile.index_name}")
                await conn.execute(options.create_index_sql())
            await self._drop_stale_indexes(
                conn, {option.profile.index_name for option in options_by_profile}
            )

        await self._run(_operation)

    async def verify_indexes(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Verify BM25 indexes exist and match runtime config without DDL."""
        options_by_profile = [
            BM25IndexOptions(profile=profile, k1=k1, b=b) for profile in self._profiles
        ]

        async def _operation(conn: Any) -> None:
            await self._verify_schema(conn)
            for options in options_by_profile:
                indexdef = await self._fetch_indexdef(conn, options.profile.index_name)
                if not options.matches_indexdef(indexdef):
                    raise RuntimeError(
                        f"{options.profile.index_name} is missing or does not match configured "
                        "BM25 options; create it on the primary first"
                    )

        await self._run(_operation)

    @staticmethod
    async def _ensure_schema(conn: Any) -> None:
        await conn.execute(
            f"ALTER TABLE {BM25_TABLE} ADD COLUMN IF NOT EXISTS "
            f"{BM25_LANGUAGE_COLUMN} TEXT NOT NULL DEFAULT '{BM25_FALLBACK_LANGUAGE}'"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {BM25_LANGUAGE_INDEX} "
            f"ON {BM25_TABLE}(workspace, {BM25_LANGUAGE_COLUMN})"
        )

    @staticmethod
    async def _verify_schema(conn: Any) -> None:
        exists = await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_chunks'
              AND column_name = $1
            LIMIT 1
            """,
            BM25_LANGUAGE_COLUMN,
        )
        if not exists:
            raise RuntimeError(
                f"{BM25_TABLE}.{BM25_LANGUAGE_COLUMN} is missing; initialize it on the primary first"
            )

    @staticmethod
    async def _verify_text_config(conn: Any, text_config: str) -> None:
        if "." in text_config:
            schema, name = text_config.split(".", maxsplit=1)
            exists = await conn.fetchval(
                """
                SELECT 1
                FROM pg_ts_config c
                JOIN pg_namespace n ON n.oid = c.cfgnamespace
                WHERE n.nspname = $1 AND c.cfgname = $2
                LIMIT 1
                """,
                schema,
                name,
            )
        else:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_ts_config WHERE cfgname = $1 LIMIT 1",
                text_config,
            )
        if not exists:
            raise RuntimeError(f"PostgreSQL text search config {text_config!r} is missing")

    @staticmethod
    async def _fetch_indexdef(conn: Any, index_name: str) -> str | None:
        return await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
            index_name,
        )

    @staticmethod
    async def _drop_stale_indexes(conn: Any, desired_indexes: set[str]) -> None:
        rows = await conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'lightrag_doc_chunks'
              AND indexname LIKE $1
              AND indexname <> $2
            """,
            f"{BM25_INDEX_PREFIX}%",
            BM25_LANGUAGE_INDEX,
        )
        for row in rows:
            index_name = str(row["indexname"])
            if index_name not in desired_indexes:
                await conn.execute(f"DROP INDEX IF EXISTS {pg_identifier(index_name)}")

    async def search(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if candidate_ids is not None and len(candidate_ids) == 0:
            logger.info(
                "[BM25] search: workspace=%s query=%r profiles=none candidate_scope=0 "
                "top_k=%s returned=0 top=none",
                self._workspace,
                query,
                top_k or self._top_k,
            )
            return []
        limit = self._top_k if top_k is None else top_k
        profiles = self._profiles_for_query(query)

        async def _operation(conn: Any) -> list[list[dict[str, Any]]]:
            rankings: list[list[dict[str, Any]]] = []
            for profile in profiles:
                sql = build_bm25_sql(
                    index_name=profile.index_name,
                    candidate_ids=candidate_ids,
                    limit=limit,
                    language=profile.language_bucket,
                )
                if candidate_ids is None:
                    rows = await conn.fetch(sql, query, self._workspace, int(limit))
                else:
                    rows = await conn.fetch(
                        sql,
                        query,
                        self._workspace,
                        list(candidate_ids),
                        int(limit),
                    )
                rankings.append([self._row_to_chunk(row, profile=profile) for row in rows])
            return rankings

        rankings = await self._run(_operation)
        if len(rankings) == 1:
            result = rankings[0]
        else:
            result = rrf_fuse(rankings)[: int(limit)]
        logger.info(
            "[BM25] search: workspace=%s query=%r profiles=%s candidate_scope=%s "
            "top_k=%d returned=%d top=%s",
            self._workspace,
            query,
            ",".join(profile.name for profile in profiles) or "none",
            len(candidate_ids) if candidate_ids is not None else "all",
            int(limit),
            len(result),
            _format_bm25_top(result),
        )
        return result

    def _profiles_for_query(self, query: str) -> tuple[BM25Profile, ...]:
        language_profiles = tuple(
            profile for profile in self._profiles if not profile.fallback and profile.languages
        )
        selected: list[BM25Profile] = []
        if language_profiles:
            language = self._language_classifier.detect(query)
            for profile in language_profiles:
                if language in profile.languages:
                    selected.append(profile)
        if not selected:
            selected.extend(profile for profile in self._profiles if profile.fallback)

        deduped: list[BM25Profile] = []
        seen: set[str] = set()
        for profile in selected:
            if profile.name in seen:
                continue
            seen.add(profile.name)
            deduped.append(profile)
        return tuple(deduped)

    @property
    def _profile_languages(self) -> tuple[str, ...]:
        return tuple(
            language
            for profile in self._profiles
            if not profile.fallback
            for language in profile.languages
        )

    @staticmethod
    def _row_to_chunk(row: Any, *, profile: BM25Profile) -> dict[str, Any]:
        return {
            "chunk_id": row["id"],
            "content": row["content"],
            "file_path": row["file_path"],
            "bm25_profile": profile.name,
            "score": float(row["score"]),
        }
