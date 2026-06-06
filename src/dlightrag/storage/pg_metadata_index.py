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
from dlightrag.storage.migrations import Migration, apply_migrations, verify_migrations
from dlightrag.storage.sql_identifiers import pg_identifier

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


def _build_schema_migrations() -> tuple[Migration, ...]:
    migrations = [
        Migration(
            "0001_base",
            "Create document metadata table",
            (_CREATE_TABLE,),
        )
    ]
    for f in METADATA_FIELDS:
        migrations.append(
            Migration(
                f"column_{f.field_id}",
                f"Ensure document metadata column {f.field_id}",
                (
                    "ALTER TABLE dlightrag_doc_metadata "
                    f"ADD COLUMN IF NOT EXISTS {pg_identifier(f.field_id)} {f.pg_type}",
                ),
            )
        )
    for f in METADATA_FIELDS:
        idx_clause = _index_clause(f.field_id, f.pg_type, f.index_type)
        if idx_clause is None:
            continue
        migrations.append(
            Migration(
                f"index_{f.field_id}",
                f"Ensure document metadata index {f.field_id}",
                (
                    f"CREATE INDEX IF NOT EXISTS idx_dm_{f.field_id} "
                    f"ON dlightrag_doc_metadata{idx_clause}",
                ),
            )
        )
    return tuple(migrations)


_SCHEMA_MIGRATIONS = _build_schema_migrations()

_JSONB_MERGE_FIELDS = frozenset({"custom_metadata", "metadata_json"})
_UPSERT_FIELDS = tuple(f for f in METADATA_FIELDS if f.field_id != "ingested_at")
_UPSERT_FIELD_IDS = tuple(f.field_id for f in _UPSERT_FIELDS)


def _build_upsert() -> str:
    columns = ("workspace", "doc_id", *_UPSERT_FIELD_IDS)
    insert_columns = ", ".join(columns)
    placeholders = ",".join(f"${idx}" for idx in range(1, len(columns) + 1))
    updates = []
    for field_id in _UPSERT_FIELD_IDS:
        if field_id in _JSONB_MERGE_FIELDS:
            updates.append(
                f"    {field_id} = "
                f"dlightrag_doc_metadata.{field_id} || EXCLUDED.{field_id}"
            )
        else:
            updates.append(
                f"    {field_id} = "
                f"COALESCE(EXCLUDED.{field_id}, dlightrag_doc_metadata.{field_id})"
            )
    return (
        "INSERT INTO dlightrag_doc_metadata\n"
        f"    ({insert_columns})\n"
        f"VALUES ({placeholders})\n"
        "ON CONFLICT (workspace, doc_id) DO UPDATE SET\n"
        + ",\n".join(updates)
    )


def _build_upsert_params(
    *,
    workspace: str,
    doc_id: str,
    system: dict[str, Any],
    custom: dict[str, Any],
    metadata_json: dict[str, Any],
) -> list[Any]:
    values: list[Any] = [workspace, doc_id]
    for field_id in _UPSERT_FIELD_IDS:
        if field_id == "custom_metadata":
            values.append(json.dumps(custom))
        elif field_id == "metadata_json":
            values.append(json.dumps(metadata_json))
        elif field_id == "process_options":
            values.append(_json_param(system.get(field_id)))
        else:
            values.append(system.get(field_id))
    return values


_UPSERT = _build_upsert()


class PGMetadataIndex:
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    Supports exact match, explicit pattern, range, and JSONB queries.
    """

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace

    async def _run(self, operation):
        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation)

    async def initialize(self, *, read_only: bool = False) -> None:
        """Create table and indexes. Call once during service startup."""
        if read_only:
            await self._verify_schema()
            return

        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=_SCHEMA_MIGRATIONS,
            )

        await self._run(_operation)

    async def _verify_schema(self) -> None:
        async def _operation(conn: Any) -> None:
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
            await verify_migrations(
                conn,
                scope="doc_metadata",
                migrations=_SCHEMA_MIGRATIONS,
            )

        await self._run(_operation)

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        sys_fields = system_field_ids()
        system = {k: metadata.get(k) for k in sys_fields}
        system["parse_engine"] = metadata.get("parse_engine") or metadata.get(
            "lightrag.parse_engine"
        )
        system["process_options"] = metadata.get("process_options") or metadata.get(
            "lightrag.process_options"
        )

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

        async def _operation(conn: Any) -> None:
            await conn.execute(
                _UPSERT,
                *_build_upsert_params(
                    workspace=self._workspace,
                    doc_id=doc_id,
                    system=system,
                    custom=custom,
                    metadata_json=metadata_json,
                ),
            )

        await self._run(_operation)

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

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(sql, *params)

        rows = await self._run(_operation)

        return [r["doc_id"] for r in rows]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a single document."""
        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

        row = await self._run(_operation)
        if not row:
            return None
        return dict(row)

    async def delete(self, doc_id: str) -> None:
        """Delete metadata for a document."""
        async def _operation(conn: Any) -> None:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

        await self._run(_operation)

    async def clear(self) -> None:
        """Delete all metadata for this workspace."""
        async def _operation(conn: Any) -> str:
            return await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1",
                self._workspace,
            )

        result = await self._run(_operation)
        logger.info("PGMetadataIndex cleared for workspace %s: %s", self._workspace, result)

    # Internal columns excluded from schema hints — not user-filterable
    _INTERNAL_COLS = frozenset({"workspace", "doc_id", "ingested_at"})

    async def get_field_schema(self) -> dict[str, Any]:
        """Read table structure dynamically.

        - Column names/types from information_schema (table structure)
        - JSONB keys from custom_metadata (JSONB-internal structure)

        Used by QueryAnalyzer to build a context-aware LLM prompt.
        """
        async def _operation(conn: Any) -> tuple[list[Any], list[Any]]:
            col_rows = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = 'dlightrag_doc_metadata' "
                "ORDER BY ordinal_position"
            )
            key_rows = await conn.fetch(
                "SELECT DISTINCT jsonb_object_keys(custom_metadata) AS k "
                "FROM dlightrag_doc_metadata "
                "WHERE workspace=$1 AND custom_metadata != '{}'",
                self._workspace,
            )
            return col_rows, key_rows

        col_rows, key_rows = await self._run(_operation)

        columns = [
            {"name": r["column_name"], "type": r["data_type"]}
            for r in col_rows
            if r["column_name"] not in self._INTERNAL_COLS
        ]
        custom_keys = [r["k"] for r in key_rows]
        return {"columns": columns, "custom_keys": custom_keys}

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by case-insensitive filename match."""
        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata WHERE workspace=$1 AND LOWER(filename)=LOWER($2)",
                self._workspace,
                name,
            )

        rows = await self._run(_operation)
        return [r["doc_id"] for r in rows]
