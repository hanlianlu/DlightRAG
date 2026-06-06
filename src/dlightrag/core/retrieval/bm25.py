# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL BM25 search over LightRAG document chunks."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.storage.sql_identifiers import pg_identifier, pg_qualified_identifier

BM25_INDEX_PREFIX = pg_identifier("idx_lightrag_doc_chunks_bm25")
BM25_TABLE = pg_identifier("LIGHTRAG_DOC_CHUNKS")
_DEFAULT_DETECTION_LANGUAGES = ("zh", "en")
_LINGUA_MIN_RELATIVE_DISTANCE = 0.08
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def validate_profile_text_config(text_config: str) -> str:
    """Validate a pg_textsearch text_config name before embedding it in DDL."""
    value = str(text_config).strip()
    try:
        return pg_qualified_identifier(value)
    except ValueError as exc:
        raise ValueError(f"unsafe BM25 text_config: {text_config!r}") from exc


def _normalize_language_code(language: str) -> str:
    value = str(language).strip().lower().replace("_", "-")
    if not value:
        return ""
    if value.startswith("zh") or value in {"cn", "chinese"}:
        return "zh"
    return value.split("-", maxsplit=1)[0]


def _language_codes_for_detection(supported_languages: tuple[str, ...] | None) -> tuple[str, ...]:
    raw_languages = supported_languages or _DEFAULT_DETECTION_LANGUAGES
    language_codes: list[str] = []
    seen: set[str] = set()
    for language in raw_languages:
        code = _normalize_language_code(language)
        if not code or code in seen:
            continue
        seen.add(code)
        language_codes.append(code)
    return tuple(language_codes)


def _lingua_language_for_code(code: str) -> Any | None:
    try:
        from lingua import IsoCode639_1, Language
    except Exception:
        return None
    try:
        iso_code = IsoCode639_1.from_str(code.upper())
        return Language.from_iso_code_639_1(iso_code)
    except Exception:
        try:
            return Language.from_str(code)
        except Exception:
            return None


@lru_cache(maxsize=32)
def _lingua_detector(language_codes: tuple[str, ...]) -> tuple[Any | None, dict[str, str]]:
    try:
        from lingua import LanguageDetectorBuilder
    except Exception:
        return None, {}

    lingua_languages: list[Any] = []
    code_by_lingua_name: dict[str, str] = {}
    for code in language_codes:
        lingua_language = _lingua_language_for_code(code)
        if lingua_language is None:
            continue
        lingua_name = lingua_language.name
        if lingua_name in code_by_lingua_name:
            continue
        code_by_lingua_name[lingua_name] = code
        lingua_languages.append(lingua_language)

    if not lingua_languages:
        return None, {}
    detector = (
        LanguageDetectorBuilder.from_languages(*lingua_languages)
        .with_minimum_relative_distance(_LINGUA_MIN_RELATIVE_DISTANCE)
        .build()
    )
    return detector, code_by_lingua_name


def detect_query_language(
    query: str,
    *,
    supported_languages: tuple[str, ...] | None = None,
) -> str:
    """Detect the query language for BM25 profile routing."""
    language_codes = _language_codes_for_detection(supported_languages)
    language_code_set = set(language_codes)
    detector, code_by_lingua_name = _lingua_detector(language_codes)
    if detector is not None:
        try:
            detected = detector.detect_language_of(query)
        except Exception:
            detected = None
        if detected:
            code = code_by_lingua_name.get(detected.name)
            if code:
                return code

    if (
        _CJK_RE.search(query)
        and "zh" in language_code_set
        and language_code_set.isdisjoint({"ja", "ko"})
    ):
        return "zh"
    return "unknown"


@dataclass(frozen=True)
class BM25Profile:
    """A query-language-routed pg_textsearch BM25 index profile."""

    name: str
    text_config: str
    languages: tuple[str, ...] = ()
    fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", pg_identifier(str(self.name).strip()))
        object.__setattr__(self, "text_config", validate_profile_text_config(self.text_config))
        object.__setattr__(
            self,
            "languages",
            tuple(
                code for language in self.languages if (code := _normalize_language_code(language))
            ),
        )

    @property
    def index_name(self) -> str:
        return pg_identifier(f"{BM25_INDEX_PREFIX}_{self.name}")


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
        return (
            f"CREATE INDEX {self.profile.index_name} ON {BM25_TABLE} USING bm25(content) "
            f"WITH (text_config='{self.profile.text_config}', "
            f"k1={_format_float(self.k1)}, b={_format_float(self.b)})"
        )

    def matches_indexdef(self, indexdef: str | None) -> bool:
        if not indexdef:
            return False
        normalized = re.sub(r"\s+", "", indexdef.lower().replace('"', "").replace("'", ""))
        text_config = self.profile.text_config.lower()
        return (
            "usingbm25(content)" in normalized
            and self.profile.index_name.lower() in normalized
            and (
                f"text_config={text_config}" in normalized
                or f"text_config={text_config}::regconfig" in normalized
            )
            and f"k1={_format_float(self.k1)}" in normalized
            and f"b={_format_float(self.b)}" in normalized
        )


def build_bm25_sql(*, index_name: str, candidate_ids: set[str] | None, limit: int) -> str:
    """Build a pg_textsearch BM25 query with optional hard candidate filter."""
    safe_index = pg_identifier(index_name)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    candidate_clause = "AND id = ANY($3::text[])" if candidate_ids is not None else ""
    limit_placeholder = "$4" if candidate_ids is not None else "$3"
    return f"""
        SELECT id, content, file_path,
               -(content <@> to_bm25query($1, '{safe_index}')) AS score
        FROM {BM25_TABLE}
        WHERE workspace = $2
        {candidate_clause}
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
            for options in options_by_profile:
                await self._verify_text_config(conn, options.profile.text_config)
                indexdef = await self._fetch_indexdef(conn, options.profile.index_name)
                if options.matches_indexdef(indexdef):
                    continue
                if indexdef:
                    await conn.execute(f"DROP INDEX IF EXISTS {options.profile.index_name}")
                await conn.execute(options.create_index_sql())

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
            for options in options_by_profile:
                indexdef = await self._fetch_indexdef(conn, options.profile.index_name)
                if not options.matches_indexdef(indexdef):
                    raise RuntimeError(
                        f"{options.profile.index_name} is missing or does not match configured "
                        "BM25 options; create it on the primary first"
                    )

        await self._run(_operation)

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

    async def search(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if candidate_ids is not None and len(candidate_ids) == 0:
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
            return rankings[0]
        return rrf_fuse(rankings)[: int(limit)]

    def _profiles_for_query(self, query: str) -> tuple[BM25Profile, ...]:
        language_profiles = tuple(
            profile for profile in self._profiles if not profile.fallback and profile.languages
        )
        selected: list[BM25Profile] = []
        if language_profiles:
            language = detect_query_language(query, supported_languages=self._profile_languages)
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
