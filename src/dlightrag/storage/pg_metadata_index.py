# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed document metadata index for structured queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage.metadata_fields import (
    METADATA_FIELDS,
    field_by_id,
    system_field_ids,
)

logger = logging.getLogger(__name__)

_TRIGRAM_THRESHOLD = 0.3


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
    idx_mapping = {
        "gin_trgm": " USING gin ({field} gin_trgm_ops)",
        "btree": " ({field})",
        "gin": " USING gin ({field})",
    }
    for f in METADATA_FIELDS:
        if f.index_type and f.index_type in idx_mapping:
            indexes.append(
                f"CREATE INDEX IF NOT EXISTS idx_dm_{f.field_id} "
                f"ON dlightrag_doc_metadata{idx_mapping[f.index_type].format(field=f.field_id)}"
            )
    return indexes


_CREATE_INDEXES = _build_create_indexes()

_UPSERT = """\
INSERT INTO dlightrag_doc_metadata
    (workspace, doc_id, filename, filename_stem, file_path, file_extension,
     doc_title, doc_author, creation_date, original_format,
     page_count, rag_mode, custom_metadata)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
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
    rag_mode = COALESCE(EXCLUDED.rag_mode, dlightrag_doc_metadata.rag_mode),
    custom_metadata = dlightrag_doc_metadata.custom_metadata || EXCLUDED.custom_metadata"""


class PGMetadataIndex:
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    Supports exact match, fuzzy match (pg_trgm), range, and JSONB queries.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool: Any = None

    async def initialize(self) -> None:
        """Create table and indexes. Call once during service startup."""
        from dlightrag.storage.pool import pg_pool

        pool = await pg_pool.get()
        self._pool = pool
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                try:
                    await conn.execute(idx_sql)
                except Exception:
                    logger.warning(
                        "Index creation skipped (pg_trgm may not be available): %s",
                        idx_sql[:60],
                    )

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        sys_fields = system_field_ids()
        system = {k: metadata.get(k) for k in sys_fields}
        custom = {k: v for k, v in metadata.items() if k not in sys_fields and k != "ingested_at"}

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
                system.get("rag_mode"),
                json.dumps(custom),
            )

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters.

        Match strategy per field type:
        - trigram fields: similarity() > threshold (fuzzy, handles typos/variants)
        - btree string fields: exact match (case-insensitive)
        - integer fields: exact match
        - date fields: range queries (from/to)
        - JSONB: containment (@>)
        - filename_pattern: ILIKE with user-provided wildcards
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
            if fdef.trigram:
                conditions.append(f"similarity({attr}, ${idx}) > {_TRIGRAM_THRESHOLD}")
                params.append(value)
            elif fdef.pg_type.upper() in {"INTEGER", "BIGINT", "SMALLINT", "INT"}:
                conditions.append(f"{attr} = ${idx}")
                params.append(value)
            else:
                # btree / non-trigram string: exact case-insensitive
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

        # Exact enum match
        if filters.rag_mode:
            conditions.append(f"rag_mode = ${idx}")
            params.append(filters.rag_mode)
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

        doc_ids = [r["doc_id"] for r in rows]

        # Fallback: if strict trigram/exact returned 0, retry with ILIKE
        if not doc_ids and len(conditions) > 1:
            doc_ids = await self._fallback_ilike_query(filters)

        return doc_ids

    async def _fallback_ilike_query(self, filters: MetadataFilter) -> list[str]:
        """Relaxed fallback: retry text fields with ILIKE instead of trigram/exact."""
        conditions: list[str] = ["workspace = $1"]
        params: list[Any] = [self._workspace]
        idx = 2

        for attr in ("filename", "doc_title", "doc_author"):
            value = getattr(filters, attr, None)
            if value is None:
                continue
            conditions.append(f"{attr} ILIKE ${idx}")
            params.append(f"%{value}%")
            idx += 1

        if len(conditions) <= 1:
            return []

        where = " AND ".join(conditions)
        sql = f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        if rows:
            logger.info("Metadata fallback (ILIKE): %d doc(s) found", len(rows))
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
