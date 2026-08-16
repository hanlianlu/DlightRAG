# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed document metadata index for structured queries."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from dlightrag_rag.ports import CorpusSchemaError
from dlightrag_rag.retrieval import MetadataFilter
from dlightrag_rag.retrieval.metadata_fields import (
    FILTER_FIELD_COLUMNS,
    METADATA_FIELD_IDS,
    canonical_metadata_key,
)

from dlightrag.adapters.postgres._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres._operations import PostgresOperationRunner
from dlightrag.adapters.postgres.identifiers import pg_identifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PGMetadataColumn:
    field_id: str
    pg_type: str
    indexed: bool = False


_PG_FIELD_TYPES = {
    "filename": "VARCHAR(512)",
    "filename_stem": "VARCHAR(512)",
    "source_uri": "TEXT",
    "download_locator": "TEXT",
    "file_extension": "VARCHAR(32)",
    "title": "TEXT",
    "author": "VARCHAR(255)",
    # RAG normalizes values to naive UTC before binding.
    "creation_date": "TIMESTAMP",
    "ingested_at": "TIMESTAMPTZ DEFAULT NOW()",
    "custom_metadata": "JSONB DEFAULT '{}'",
}
_PG_INDEXED_FIELDS = frozenset(
    {"filename", "filename_stem", "file_extension", "title", "author", "creation_date"}
)
if set(_PG_FIELD_TYPES) != set(METADATA_FIELD_IDS):
    raise RuntimeError("PostgreSQL metadata columns do not match the RAG metadata field registry")
_PG_METADATA_COLUMNS = tuple(
    _PGMetadataColumn(
        field_id,
        _PG_FIELD_TYPES[field_id],
        indexed=field_id in _PG_INDEXED_FIELDS,
    )
    for field_id in METADATA_FIELD_IDS
)


def _build_create_table() -> str:
    cols = [
        "workspace       VARCHAR(255) NOT NULL",
        "doc_id          VARCHAR(255) NOT NULL",
    ]
    for f in _PG_METADATA_COLUMNS:
        cols.append(f"    {f.field_id}    {f.pg_type}")
    cols.append("    PRIMARY KEY (workspace, doc_id)")
    return "CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata (\n" + ",\n".join(cols) + "\n)"


_CREATE_TABLE = _build_create_table()


# One canonical comparison for every text match: neither case nor padding is a
# meaningful difference in a filter, and both sides must fold identically.
def _canonical(expr: str) -> str:
    return f"LOWER(TRIM({expr}))"


def _index_clause(field_id: str, pg_type: str, indexed: bool) -> str | None:
    if not indexed:
        return None
    if _is_string_pg_type(pg_type):
        return f" ({_canonical(field_id)})"
    return f" ({field_id})"


def _is_string_pg_type(pg_type: str) -> bool:
    normalized = pg_type.upper()
    return normalized.startswith(("TEXT", "VARCHAR", "CHAR", "CHARACTER"))


def _build_schema_migrations() -> tuple[Migration, ...]:
    """Add declared metadata columns; never remove what is no longer declared.

    Every statement is derived from the registry rather than recorded as history,
    so a fresh database and an existing one reach the same shape and the list
    does not grow with each schema change. Indexes are derived, so they are
    rebuilt in place. Columns hold data, so an undeclared one is left standing
    for an operator to drop — a rollback must not be able to erase a column.
    """
    migrations = [
        Migration(
            "0001_base",
            "Create document metadata table",
            (_CREATE_TABLE,),
        ),
    ]
    for f in _PG_METADATA_COLUMNS:
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
    for f in _PG_METADATA_COLUMNS:
        idx_clause = _index_clause(f.field_id, f.pg_type, f.indexed)
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

_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_doc_metadata",
        columns=("workspace", "doc_id", *METADATA_FIELD_IDS),
        primary_key=("workspace", "doc_id"),
        indexes=(
            "idx_dm_workspace_download_locator",
            *(f"idx_dm_{f.field_id}" for f in _PG_METADATA_COLUMNS if f.indexed),
        ),
    ),
)

_CUSTOM = "custom_metadata"
_UPSERT_FIELD_IDS = tuple(field_id for field_id in METADATA_FIELD_IDS if field_id != "ingested_at")


def _field_assignment(field_id: str, placeholder: str, table_qualified: str) -> str:
    if field_id == _CUSTOM:
        # `||` yields NULL if either side is: a merge must never erase the column.
        return (
            f"{field_id} = COALESCE({table_qualified}, '{{}}'::jsonb) "
            f"|| COALESCE({placeholder}::jsonb, '{{}}'::jsonb)"
        )
    return f"{field_id} = COALESCE({placeholder}, {table_qualified})"


def _build_upsert() -> str:
    columns = ("workspace", "doc_id", *_UPSERT_FIELD_IDS)
    insert_columns = ", ".join(columns)
    placeholders = ",".join(f"${idx}" for idx in range(1, len(columns) + 1))
    updates = [
        "    "
        + _field_assignment(field_id, f"EXCLUDED.{field_id}", f"dlightrag_doc_metadata.{field_id}")
        for field_id in _UPSERT_FIELD_IDS
    ]
    return (
        "INSERT INTO dlightrag_doc_metadata\n"
        f"    ({insert_columns})\n"
        f"VALUES ({placeholders})\n"
        "ON CONFLICT (workspace, doc_id) DO UPDATE SET\n" + ",\n".join(updates)
    )


def _build_update() -> str:
    """Same assignments as the upsert, but a missing document stays missing."""
    assignments = [
        "    " + _field_assignment(field_id, f"${idx}", field_id)
        for idx, field_id in enumerate(_UPSERT_FIELD_IDS, start=3)
    ]
    return (
        "UPDATE dlightrag_doc_metadata SET\n"
        + ",\n".join(assignments)
        + "\nWHERE workspace = $1 AND doc_id = $2"
    )


def _build_params(workspace: str, doc_id: str, metadata: dict[str, Any]) -> list[Any]:
    custom = metadata.get(_CUSTOM)
    return [
        workspace,
        doc_id,
        *(
            json.dumps(custom if isinstance(custom, dict) else {})
            if field_id == _CUSTOM
            else metadata.get(field_id)
            for field_id in _UPSERT_FIELD_IDS
        ),
    ]


_UPSERT = _build_upsert()
_UPDATE = _build_update()

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
# A caller types a name, not a pattern, so the widened match is literal substring
# search. ILIKE would mean escaping %, _ and \ back out of the pattern language.
_FILENAME_CONTAINS_CONDITION = "STRPOS(LOWER(TRIM(filename)), LOWER(TRIM(${idx}))) > 0"

# Deletion resolves a name, so it matches the full name only, never the stem.
_FIND_BY_FILENAME = (
    "SELECT doc_id FROM dlightrag_doc_metadata "  # noqa: S608 - fixed text; only $-params
    f"WHERE workspace=$1 AND {_canonical('filename')} = {_canonical('$2')}"
)


def _decoded_row(row: Any) -> dict[str, Any]:
    """asyncpg hands JSONB back as text, which callers and comparisons must not see."""
    decoded = dict(row)
    raw = decoded.get("custom_metadata")
    decoded["custom_metadata"] = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return decoded


class PGMetadataIndex(PostgresOperationRunner):
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    """

    def __init__(self, workspace: str = "default") -> None:
        super().__init__()
        self._workspace = workspace

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create table and indexes, or validate them (reader)."""

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope="doc_metadata",
                    migrations=_SCHEMA_MIGRATIONS,
                    tables=_SCHEMA_TABLES,
                    schema_error=CorpusSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=CorpusSchemaError,
                require_applied_prefix=False,
            )

        await self._run(_operation)

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        params = _build_params(self._workspace, doc_id, metadata)

        async def _operation(conn: Any) -> None:
            await conn.execute(_UPSERT, *params)

        await self._run(_operation)

    async def merge_custom_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        """Update an existing document, reporting whether one was there to update."""
        params = _build_params(self._workspace, doc_id, metadata)

        async def _operation(conn: Any) -> str:
            return await conn.execute(_UPDATE, *params)

        return (await self._run(_operation)) != "UPDATE 0"

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters."""
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
        # rather than being baked into what was written. Keys were folded on the
        # way in, so the same fold applies to the key the caller filters on.
        for key, value in (filters.custom or {}).items():
            conditions.append(
                f"{_canonical(f'custom_metadata ->> ${idx}')} = {_canonical(f'${idx + 1}')}"
            )
            params.append(canonical_metadata_key(key))
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
        params[param_slot] = str(filters.filename)
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
        custom_keys = list(row["custom_keys"] or ())
        if custom_keys:
            # The planner is told every other field must stay null, so a key it
            # may filter on is useless unless `custom` is named as available.
            filters.append("custom")
        return {"filters": filters, "custom_keys": custom_keys}

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by case-insensitive filename match."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                _FIND_BY_FILENAME,
                self._workspace,
                name,
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
