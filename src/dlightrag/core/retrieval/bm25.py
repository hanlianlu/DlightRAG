# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL BM25 search over LightRAG document chunks."""

import asyncio
import inspect
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dlightrag_rag.retrieval import ContextRow, MetadataScope, format_bm25_top, rrf_fuse

from dlightrag.core.retrieval.bm25_language import (
    BM25_FALLBACK_LANGUAGE,
    BM25_LANGUAGE_COLUMN,
    BM25LanguageClassifier,
    normalize_language_code,
)
from dlightrag.storage.sql_identifiers import pg_identifier, pg_qualified_identifier

BM25_INDEX_PREFIX = pg_identifier("idx_lightrag_doc_chunks_bm25")
BM25_LANGUAGE_INDEX = pg_identifier("idx_lightrag_doc_chunks_dlightrag_bm25_language")
BM25_DOC_INDEX = pg_identifier("idx_lightrag_doc_chunks_dlightrag_full_doc_id")
BM25_TABLE = pg_identifier("LIGHTRAG_DOC_CHUNKS")
logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    return f"{float(value):g}"


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


def profiles_from_config(config_profiles: Iterable[Any]) -> tuple[BM25Profile, ...]:
    """Convert typed settings profiles into immutable runtime profiles."""
    return tuple(
        BM25Profile(
            name=profile.name,
            text_config=profile.text_config,
            languages=tuple(profile.languages),
            fallback=profile.fallback,
        )
        for profile in config_profiles
    )


def required_postgres_extensions(profiles: Iterable[BM25Profile]) -> tuple[str, ...]:
    """Return PostgreSQL extensions required by the configured BM25 profiles."""
    extensions = ["pg_textsearch"]
    if any(profile.text_config == "public.jiebacfg" for profile in profiles):
        extensions.append("pg_jieba")
    return tuple(extensions)


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
    scoped: bool,
    limit: int,
    language: str | None = None,
) -> str:
    """Build a pg_textsearch BM25 query with an optional hard document filter."""
    safe_index = pg_identifier(index_name)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    candidate_clause = "AND full_doc_id = ANY($3::text[])" if scoped else ""
    language_clause = (
        f"AND {BM25_LANGUAGE_COLUMN} = '{_sql_language_literal(language)}'" if language else ""
    )
    limit_placeholder = "$4" if scoped else "$3"
    return (
        f"SELECT id, content, file_path, full_doc_id, "  # noqa: S608
        f"-(content <@> to_bm25query($1, '{safe_index}')) AS score "
        f"FROM {BM25_TABLE} "
        "WHERE workspace = $2 "
        f"{candidate_clause} "
        f"{language_clause} "
        f"ORDER BY content <@> to_bm25query($1, '{safe_index}') "
        f"LIMIT {limit_placeholder}"
    )


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
        """Verify BM25 schema and indexes exist without emitting DDL (reader)."""
        options_by_profile = [
            BM25IndexOptions(profile=profile, k1=k1, b=b) for profile in self._profiles
        ]

        async def _operation(conn: Any) -> None:
            await self._verify_schema(conn)
            for options in options_by_profile:
                await self._verify_text_config(conn, options.profile.text_config)
                indexdef = await self._fetch_indexdef(conn, options.profile.index_name)
                if not options.matches_indexdef(indexdef):
                    raise RuntimeError(
                        f"BM25 index {options.profile.index_name} is missing or does not "
                        "match configured options; initialize it on the writer first"
                    )

        await self._run(_operation)

    async def relabel_chunk_languages(self, *, batch_size: int = 500) -> dict[str, int]:
        """Refresh BM25 language labels for every chunk in this workspace."""
        batch_limit = int(batch_size)
        if batch_limit < 1:
            raise ValueError("BM25 batch_size must be positive")

        async def _operation(conn: Any) -> dict[str, int]:
            stats = {"processed_chunks": 0, "updated_chunks": 0}
            cursor = ""
            while True:
                rows = await conn.fetch(
                    f"""
                    SELECT id, content
                    FROM {BM25_TABLE}
                    WHERE workspace = $1 AND id > $2
                    ORDER BY id
                    LIMIT $3
                    """,  # noqa: S608 - internal table constant.
                    self._workspace,
                    cursor,
                    batch_limit,
                )
                if not rows:
                    break

                chunk_ids = [str(row["id"]) for row in rows]
                contents = [str(row["content"] or "") for row in rows]
                # Detection is CPU-bound; keep it off the loop while this holds
                # a pooled connection for the whole workspace relabel.
                languages = await asyncio.to_thread(
                    lambda texts=contents: [
                        self._language_classifier.detect(text) for text in texts
                    ]
                )
                await conn.execute(
                    (
                        f"UPDATE {BM25_TABLE} AS chunks "  # noqa: S608 - internal constants.
                        f"SET {BM25_LANGUAGE_COLUMN}=labels.language, "
                        "update_time=CURRENT_TIMESTAMP "
                        "FROM UNNEST($2::text[], $3::text[]) AS labels(id, language) "
                        "WHERE chunks.workspace=$1 AND chunks.id=labels.id"
                    ),
                    self._workspace,
                    chunk_ids,
                    languages,
                )
                stats["processed_chunks"] += len(rows)
                stats["updated_chunks"] += len(chunk_ids)
                cursor = chunk_ids[-1]
                if len(rows) < batch_limit:
                    break
            return stats

        return await self._run(_operation)

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
                f"{BM25_TABLE}.{BM25_LANGUAGE_COLUMN} is missing; initialize it on the writer first"
            )

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
        # Metadata filters scope retrieval by document, so full_doc_id is the
        # lookup key for both the chunk count and the BM25 candidate filter.
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {BM25_DOC_INDEX} ON {BM25_TABLE}(workspace, full_doc_id)"
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
        scope: MetadataScope | None,
        top_k: int | None = None,
    ) -> list[ContextRow]:
        if scope is not None and not scope:
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
        doc_ids = scope.as_list() if scope is not None else None

        async def _operation(conn: Any) -> list[list[ContextRow]]:
            rankings: list[list[ContextRow]] = []
            for profile in profiles:
                sql = build_bm25_sql(
                    index_name=profile.index_name,
                    scoped=doc_ids is not None,
                    limit=limit,
                    language=profile.language_bucket,
                )
                if doc_ids is None:
                    rows = await conn.fetch(sql, query, self._workspace, int(limit))
                else:
                    rows = await conn.fetch(
                        sql,
                        query,
                        self._workspace,
                        doc_ids,
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
            f"{len(doc_ids)}doc" if doc_ids is not None else "all",
            int(limit),
            len(result),
            format_bm25_top(result),
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
    def _row_to_chunk(row: Any, *, profile: BM25Profile) -> ContextRow:
        chunk = {
            "chunk_id": row["id"],
            "content": row["content"],
            "file_path": row["file_path"],
            "bm25_profile": profile.name,
            "score": float(row["score"]),
        }
        if row.get("full_doc_id"):
            chunk["full_doc_id"] = row["full_doc_id"]
        return chunk


async def create_postgres_bm25(
    config: Any,
    *,
    pool: Any,
    profiles: tuple[BM25Profile, ...] | None = None,
) -> PostgresBM25 | None:
    """Construct and provision workspace BM25 for the configured service role."""
    if not config.bm25_enabled:
        return None

    runtime_profiles = profiles or profiles_from_config(config.bm25_profiles)
    retriever = PostgresBM25(
        pool=pool,
        workspace=config.workspace,
        profiles=runtime_profiles,
    )
    if config.is_reader:
        await retriever.verify_indexes(k1=config.bm25_k1, b=config.bm25_b)
    else:
        await retriever.ensure_indexes(k1=config.bm25_k1, b=config.bm25_b)
    return retriever


async def rebuild_postgres_bm25(
    config: Any,
    *,
    pool: Any,
    batch_size: int = 500,
) -> dict[str, int]:
    """Provision workspace BM25 and refresh every stored chunk language."""
    if config.bm25_enabled and config.is_reader:
        raise RuntimeError("BM25 rebuild requires the writer service role")
    retriever = await create_postgres_bm25(config, pool=pool)
    if retriever is None:
        return {"processed_chunks": 0, "updated_chunks": 0}
    return await retriever.relabel_chunk_languages(batch_size=batch_size)
