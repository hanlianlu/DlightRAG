# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed document metadata index for structured queries."""

import json
import logging
from typing import Any

from dlightrag.core.retrieval.metadata_fields import (
    FILTER_FIELD_COLUMNS,
    METADATA_FIELDS,
    system_field_ids,
)
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage.migrations import Migration, apply_migrations
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


# One canonical comparison for every text match: neither case nor padding is a
# meaningful difference in a filter, and both sides must fold identically.
def _canonical(expr: str) -> str:
    return f"LOWER(TRIM({expr}))"


def _index_clause(field_id: str, pg_type: str, index_type: str | None) -> str | None:
    if index_type == "gin":
        return f" USING gin ({field_id})"
    if index_type != "btree":
        return None

    if _is_string_pg_type(pg_type):
        return f" ({_canonical(field_id)})"
    return f" ({field_id})"


def _is_string_pg_type(pg_type: str) -> bool:
    normalized = pg_type.upper()
    return normalized.startswith(("TEXT", "VARCHAR", "CHAR", "CHARACTER"))


def _json_param(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def _build_schema_migrations() -> tuple[Migration, ...]:
    """Converge the table on what METADATA_FIELDS declares.

    Every statement is derived from the registry rather than recorded as history,
    so a fresh database and an existing one reach the same shape and the list
    does not grow with each schema change.
    """
    migrations = [
        Migration(
            "0001_base",
            "Create document metadata table",
            (_CREATE_TABLE,),
        ),
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
    migrations.append(
        Migration(
            "index_workspace_download_locator",
            "Index exact source download ownership lookups",
            (
                "CREATE INDEX IF NOT EXISTS idx_dm_workspace_download_locator "
                "ON dlightrag_doc_metadata (workspace, download_locator)",
            ),
        )
    )
    for f in METADATA_FIELDS:
        idx_clause = _index_clause(f.field_id, f.pg_type, f.index_type)
        if idx_clause is None:
            continue
        # Versioned by the expression: an index only serves a match whose
        # canonical form it was built with, so changing one must rebuild it.
        migrations.append(
            Migration(
                f"index_{f.field_id}_canonical",
                f"Ensure document metadata index {f.field_id}",
                (
                    f"DROP INDEX IF EXISTS idx_dm_{f.field_id}",
                    f"CREATE INDEX IF NOT EXISTS idx_dm_{f.field_id} "
                    f"ON dlightrag_doc_metadata{idx_clause}",
                ),
            )
        )
    return tuple(migrations)


_SCHEMA_MIGRATIONS = _build_schema_migrations()

_JSONB_MERGE_FIELDS = frozenset({"custom_metadata"})
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
                f"    {field_id} = dlightrag_doc_metadata.{field_id} || EXCLUDED.{field_id}"
            )
        else:
            updates.append(
                f"    {field_id} = COALESCE(EXCLUDED.{field_id}, dlightrag_doc_metadata.{field_id})"
            )
    return (
        "INSERT INTO dlightrag_doc_metadata\n"
        f"    ({insert_columns})\n"
        f"VALUES ({placeholders})\n"
        "ON CONFLICT (workspace, doc_id) DO UPDATE SET\n" + ",\n".join(updates)
    )


def _build_upsert_params(
    *,
    workspace: str,
    doc_id: str,
    system: dict[str, Any],
    custom: dict[str, Any],
) -> list[Any]:
    values: list[Any] = [workspace, doc_id]
    for field_id in _UPSERT_FIELD_IDS:
        if field_id == "custom_metadata":
            values.append(json.dumps(custom))
        else:
            values.append(system.get(field_id))
    return values


_UPSERT = _build_upsert()

_FILTERABLE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(column for columns in FILTER_FIELD_COLUMNS.values() for column in columns)
)


def _build_field_schema() -> str:
    """Report which filter columns hold data, plus the custom keys in use.

    Deliberately reports populated columns rather than the table definition: the
    planner cannot see the corpus, so naming a column it can never match is what
    sends it down an empty filter. One row out regardless of document count.
    """
    populated = ",\n    ".join(
        f"bool_or({column} IS NOT NULL) AS {column}" for column in _FILTERABLE_COLUMNS
    )
    return f"""
SELECT
    {populated},
    (
        SELECT array_agg(DISTINCT key)
        FROM (
            SELECT jsonb_object_keys(custom_metadata) AS key
            FROM dlightrag_doc_metadata
            WHERE workspace = ANY($1::text[])
              AND custom_metadata != '{{}}'
        ) AS keys
    ) AS custom_keys
FROM dlightrag_doc_metadata
WHERE workspace = ANY($1::text[])
"""  # noqa: S608 - column names come from the field registry, never from input


_FIELD_SCHEMA = _build_field_schema()


# A named file is matched against both the full name and the stem, so a caller
# who omits the extension still hits the functional lower() indexes on both.
_FILENAME_EXACT_CONDITION = (
    "(LOWER(TRIM(filename)) = LOWER(TRIM(${idx})) "
    "OR LOWER(TRIM(filename_stem)) = LOWER(TRIM(${idx})))"
)
_FILENAME_CONTAINS_CONDITION = "filename ILIKE ${idx}"


def _as_ilike_pattern(value: str) -> str:
    """Wrap a bare name in wildcards, leaving a caller's own pattern intact."""
    return value if "%" in value or "_" in value else f"%{value}%"


def _decoded_row(row: Any) -> dict[str, Any]:
    """asyncpg hands JSONB back as text, which callers and comparisons must not see."""
    decoded = dict(row)
    raw = decoded.get("custom_metadata")
    decoded["custom_metadata"] = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return decoded


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
        """Create table and indexes, or verify them (read-only reader)."""
        if read_only:
            await self._verify_schema()
            return

        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=_SCHEMA_MIGRATIONS,
                require_applied_prefix=False,
            )

        await self._run(_operation)

    async def _verify_schema(self) -> None:
        """Confirm the metadata table exists without emitting DDL."""

        async def _operation(conn: Any) -> None:
            exists = await conn.fetchval("SELECT to_regclass('dlightrag_doc_metadata') IS NOT NULL")
            if not exists:
                raise RuntimeError(
                    "dlightrag_doc_metadata is missing; initialize it on the writer first"
                )

        await self._run(_operation)

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        system = {k: metadata.get(k) for k in system_field_ids()}
        custom = metadata.get("custom_metadata")

        async def _operation(conn: Any) -> None:
            await conn.execute(
                _UPSERT,
                *_build_upsert_params(
                    workspace=self._workspace,
                    doc_id=doc_id,
                    system=system,
                    custom=custom if isinstance(custom, dict) else {},
                ),
            )

        await self._run(_operation)

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters.

        Match strategy per field:
        - string fields: exact match (case-insensitive)
        - date fields: range queries (from/to)
        - custom metadata: case-insensitive match on the stored JSONB value
        - filename: exact on name or stem, falling back to a contains match
        """
        conditions: list[str] = ["workspace = $1"]
        params: list[Any] = [self._workspace]
        idx = 2

        for attr in ("file_extension", "title", "author"):
            value = getattr(filters, attr, None)
            if value is None:
                continue
            conditions.append(f"{_canonical(attr)} = {_canonical(f'${idx}')}")
            params.append(value)
            idx += 1

        filename_slot: tuple[int, int] | None = None
        if filters.filename:
            filename_slot = (len(conditions), len(params))
            conditions.append(_FILENAME_EXACT_CONDITION.format(idx=idx))
            params.append(filters.filename)
            idx += 1

        # Date range
        if filters.creation_date_from:
            conditions.append(f"creation_date >= ${idx}")
            params.append(filters.creation_date_from)
            idx += 1
        if filters.creation_date_to:
            conditions.append(f"creation_date <= ${idx}")
            params.append(filters.creation_date_to)
            idx += 1

        # JSONB values are stored verbatim, so the canonical fold happens here
        # rather than being baked into what was written.
        for key, value in (filters.custom or {}).items():
            conditions.append(
                f"{_canonical(f'custom_metadata ->> ${idx}')} = {_canonical(f'${idx + 1}')}"
            )
            params.append(key)
            params.append(value if isinstance(value, str) else json.dumps(value))
            idx += 2

        doc_ids = await self._select_doc_ids(conditions, params)
        if doc_ids or filename_slot is None:
            return doc_ids

        # The caller named a file the corpus does not carry verbatim. A planner
        # cannot know whether a name is complete, and a human rarely types one,
        # so widen that single clause rather than returning nothing.
        condition_slot, param_slot = filename_slot
        conditions[condition_slot] = _FILENAME_CONTAINS_CONDITION.format(idx=param_slot + 1)
        params[param_slot] = _as_ilike_pattern(str(filters.filename))
        return await self._select_doc_ids(conditions, params)

    async def _select_doc_ids(self, conditions: list[str], params: list[Any]) -> list[str]:
        where = " AND ".join(conditions)
        sql = f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where}"  # noqa: S608

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
        return _decoded_row(row)

    async def get_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get metadata for multiple documents in one query."""
        unique_doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not unique_doc_ids:
            return {}

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id = ANY($2::text[])",
                self._workspace,
                unique_doc_ids,
            )

        rows = await self._run(_operation)
        return {str(row["doc_id"]): _decoded_row(row) for row in rows}

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

    async def get_field_schema(
        self,
        *,
        workspaces: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Report the filters this workspace can actually satisfy.

        Returns the planner-facing filter field names backed by at least one
        populated column, plus the custom metadata keys in use. Formats and
        meanings are static and live in the planner prompt; only availability
        is workspace-dependent.
        """
        workspace_scope = list(dict.fromkeys(workspaces or (self._workspace,)))

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_FIELD_SCHEMA, workspace_scope)

        row = await self._run(_operation)
        if row is None:
            return {"filters": [], "custom_keys": []}

        filters = [
            field
            for field, columns in FILTER_FIELD_COLUMNS.items()
            if any(row[column] for column in columns)
        ]
        return {"filters": filters, "custom_keys": list(row["custom_keys"] or ())}

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

    async def find_by_file_path(self, file_path: str) -> list[str]:
        """Find doc_ids by exact stored file_path match."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata WHERE workspace=$1 AND file_path=$2",
                self._workspace,
                file_path,
            )

        rows = await self._run(_operation)
        return [r["doc_id"] for r in rows]

    async def find_by_download_locator(self, download_locator: str) -> list[str]:
        """Find doc_ids owning one exact download locator in this workspace."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata "
                "WHERE workspace=$1 AND download_locator=$2",
                self._workspace,
                download_locator,
            )

        rows = await self._run(_operation)
        return [r["doc_id"] for r in rows]
