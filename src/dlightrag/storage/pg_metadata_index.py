# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed document metadata index for structured queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from dlightrag.core.retrieval.metadata_fields import (
    METADATA_FIELDS,
    field_by_id,
    system_field_ids,
)
from dlightrag.core.retrieval.models import MetadataFilter

logger = logging.getLogger(__name__)


def _build_create_table() -> str:
    cols = [
        "workspace       VARCHAR(255) NOT NULL",
        "doc_id          VARCHAR(255) NOT NULL",
    ]
    for f in METADATA_FIELDS:
        cols.append(f"    {f.field_id}    {f.pg_type}")
    cols.append("    PRIMARY KEY (workspace, doc_id)")
    return "CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata (\n" + ",\n".join(cols) + "\n)"


_CREATE_TABLE = _build_create_table()


def _build_create_indexes() -> list[str]:
    indexes = []
    for f in METADATA_FIELDS:
        idx_clause = _index_clause(f.field_id, f.pg_type, f.index_type)
        if idx_clause is not None:
            indexes.append(
                f"CREATE INDEX IF NOT EXISTS idx_dm_{f.field_id} "
                f"ON dlightrag_doc_metadata{idx_clause}"
            )
    return indexes


def _index_clause(field_id: str, pg_type: str, index_type: str | None) -> str | None:
    if index_type == "gin":
        return f" USING gin ({field_id})"
    if index_type != "btree":
        return None

    if _is_string_pg_type(pg_type):
        return f" (LOWER({field_id}))"
    return f" ({field_id})"


def _is_string_pg_type(pg_type: str) -> bool:
    normalized = pg_type.upper()
    return normalized.startswith(("TEXT", "VARCHAR", "CHAR", "CHARACTER"))


def _json_param(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


_CREATE_INDEXES = _build_create_indexes()

_UPSERT = """\
INSERT INTO dlightrag_doc_metadata
    (workspace, doc_id, filename, filename_stem, file_path, file_extension,
     doc_title, doc_author, creation_date, original_format,
     page_count, ingest_strategy, parse_engine, process_options, artifact_status,
     custom_metadata, metadata_json)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
ON CONFLICT (workspace, doc_id) DO UPDATE SET
    filename = COALESCE(EXCLUDED.filename, dlightrag_doc_metadata.filename),
    filename_stem = COALESCE(EXCLUDED.filename_stem, dlightrag_doc_metadata.filename_stem),
    file_path = COALESCE(EXCLUDED.file_path, dlightrag_doc_metadata.file_path),
    file_extension = COALESCE(EXCLUDED.file_extension, dlightrag_doc_metadata.file_extension),
    doc_title = COALESCE(EXCLUDED.doc_title, dlightrag_doc_metadata.doc_title),
    doc_author = COALESCE(EXCLUDED.doc_author, dlightrag_doc_metadata.doc_author),
    creation_date = COALESCE(EXCLUDED.creation_date, dlightrag_doc_metadata.creation_date),
    original_format = COALESCE(EXCLUDED.original_format, dlightrag_doc_metadata.original_format),
    page_count = COALESCE(EXCLUDED.page_count, dlightrag_doc_metadata.page_count),
    ingest_strategy = COALESCE(EXCLUDED.ingest_strategy, dlightrag_doc_metadata.ingest_strategy),
    parse_engine = COALESCE(EXCLUDED.parse_engine, dlightrag_doc_metadata.parse_engine),
    process_options = COALESCE(EXCLUDED.process_options, dlightrag_doc_metadata.process_options),
    artifact_status = COALESCE(EXCLUDED.artifact_status, dlightrag_doc_metadata.artifact_status),
    custom_metadata = dlightrag_doc_metadata.custom_metadata || EXCLUDED.custom_metadata,
    metadata_json = dlightrag_doc_metadata.metadata_json || EXCLUDED.metadata_json"""


class PGMetadataIndex:
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    Supports exact match, explicit pattern, range, and JSONB queries.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool: Any = None

    async def initialize(self, *, read_only: bool = False) -> None:
        """Create table and indexes. Call once during service startup."""
        from dlightrag.storage.pool import pg_pool

        pool = await pg_pool.get()
        self._pool = pool
        if read_only:
            await self._verify_table()
            return
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                await conn.execute(idx_sql)

    async def _verify_table(self) -> None:
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'dlightrag_doc_metadata'"
                ")"
            )
            if not exists:
                raise RuntimeError(
                    "dlightrag_doc_metadata is missing; initialize it on the primary first"
                )

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        sys_fields = system_field_ids()
        system = {k: metadata.get(k) for k in sys_fields}
        system["parse_engine"] = metadata.get("parse_engine") or metadata.get("lightrag.parse_engine")
        system["process_options"] = metadata.get("process_options") or metadata.get(
            "lightrag.process_options"
        )
        system["artifact_status"] = metadata.get("artifact_status")

        filterable = metadata.get("metadata_filterable")
        if isinstance(filterable, dict):
            custom = filterable
        else:
            custom = {
                k: v
                for k, v in metadata.items()
                if k not in sys_fields
                and k not in {"ingested_at", "metadata_filterable", "user_metadata"}
                and not k.startswith("lightrag.")
            }

        raw_json = metadata.get("metadata_json")
        metadata_json = raw_json if isinstance(raw_json, dict) else {}
        process_options = system.get("process_options")

        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT,
                self._workspace,
                doc_id,
                system.get("filename"),
                system.get("filename_stem"),
                system.get("file_path"),
                system.get("file_extension"),
                system.get("doc_title"),
                system.get("doc_author"),
                system.get("creation_date"),
                system.get("original_format"),
                system.get("page_count"),
                system.get("ingest_strategy"),
                system.get("parse_engine"),
                _json_param(process_options),
                system.get("artifact_status"),
                json.dumps(custom),
                json.dumps(metadata_json),
            )

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters.

        Match strategy per field type:
        - string fields: exact match (case-insensitive)
        - integer fields: exact match
        - date fields: range queries (from/to)
        - JSONB: containment (@>)
        - filename_pattern: explicit ILIKE with user/LLM-provided wildcards
        """
        conditions: list[str] = ["workspace = $1"]
        params: list[Any] = [self._workspace]
        idx = 2

        # Searchable fields — dispatch by field type
        for attr in ("filename", "filename_stem", "file_extension", "doc_title", "doc_author"):
            value = getattr(filters, attr, None)
            if value is None:
                continue
            fdef = field_by_id(attr)
            if fdef is None:
                continue
            if fdef.pg_type.upper() in {"INTEGER", "BIGINT", "SMALLINT", "INT"}:
                conditions.append(f"{attr} = ${idx}")
                params.append(value)
            else:
                # Deterministic metadata matching: exact case-insensitive.
                conditions.append(f"LOWER({attr}) = LOWER(${idx})")
                params.append(value)
            idx += 1

        # Filename pattern (ILIKE with user wildcards)
        if filters.filename_pattern:
            conditions.append(f"filename ILIKE ${idx}")
            params.append(filters.filename_pattern)
            idx += 1

        # Date range
        if filters.date_from:
            conditions.append(f"creation_date >= ${idx}")
            params.append(filters.date_from)
            idx += 1
        if filters.date_to:
            conditions.append(f"creation_date <= ${idx}")
            params.append(filters.date_to)
            idx += 1

        # JSONB containment
        if filters.custom:
            conditions.append(f"custom_metadata @> ${idx}::jsonb")
            params.append(json.dumps(filters.custom))
            idx += 1

        where = " AND ".join(conditions)
        sql = f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [r["doc_id"] for r in rows]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a single document."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )
        if not row:
            return None
        return dict(row)

    async def delete(self, doc_id: str) -> None:
        """Delete metadata for a document."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

    async def clear(self) -> None:
        """Delete all metadata for this workspace."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1",
                self._workspace,
            )
            logger.info("PGMetadataIndex cleared for workspace %s: %s", self._workspace, result)

    # Internal columns excluded from schema hints — not user-filterable
    _INTERNAL_COLS = frozenset({"workspace", "doc_id", "ingested_at"})

    async def get_field_schema(self) -> dict[str, Any]:
        """Read table structure dynamically.

        - Column names/types from information_schema (table structure)
        - JSONB keys from custom_metadata (JSONB-internal structure)

        Used by QueryAnalyzer to build a context-aware LLM prompt.
        """
        async with self._pool.acquire() as conn:
            col_rows = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'dlightrag_doc_metadata' ORDER BY ordinal_position"
            )
            key_rows = await conn.fetch(
                "SELECT DISTINCT jsonb_object_keys(custom_metadata) AS k "
                "FROM dlightrag_doc_metadata "
                "WHERE workspace=$1 AND custom_metadata != '{}'",
                self._workspace,
            )

        columns = [
            {"name": r["column_name"], "type": r["data_type"]}
            for r in col_rows
            if r["column_name"] not in self._INTERNAL_COLS
        ]
        custom_keys = [r["k"] for r in key_rows]
        return {"columns": columns, "custom_keys": custom_keys}

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by case-insensitive filename match."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata WHERE workspace=$1 AND LOWER(filename)=LOWER($2)",
                self._workspace,
                name,
            )
        return [r["doc_id"] for r in rows]
