# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Physical workspace partitioning foundation for retrieval-critical tables.

DlightRAG hides PostgreSQL LIST partitioning behind this seam:

* application SQL keeps querying parent table names and always carries the
  authenticated ``workspace = $n`` predicate; nothing in the retrieval path
  ever names a child table;
* partition names are internal deterministic hashes, never raw workspace
  identifiers;
* every retrieval-critical parent carries a shared DEFAULT partition so new
  workspaces work without per-workspace DDL, while Commit 3 promotion attaches
  dedicated children for hot workspaces.

The seam converts LightRAG's fresh empty upstream tables in place (never a
populated table) and fails loudly with an actionable one-time development reset
message when it meets the old unpartitioned corpus shape. Readers only
validate; they never issue DDL.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from dlightrag.adapters.postgres.core._locks import advisory_lock_key
from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.adapters.postgres.core.identifiers import pg_identifier
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

logger = logging.getLogger(__name__)

PARTITION_COLUMN = "workspace"

_RESET_HINT = (
    "The previous unpartitioned development corpus is not migrated automatically "
    "and is never rebuilt destructively. Run the one-time development reset "
    "(uv run scripts/reset_development.py --mode docker, or --mode native for a "
    "dedicated local database), then re-ingest. A per-workspace data reset cannot "
    "change this database-wide physical schema."
)

# Relation kinds reported by pg_class.
_RELKIND_PLAIN = "r"
_RELKIND_PARTITIONED = "p"


@dataclass(frozen=True, slots=True)
class PartitionedTableSpec:
    """What one partitioned retrieval-critical table must look like.

    ``missing_ok`` marks DlightRAG-owned tables whose partitioned DDL is issued
    by their own migration scope; a missing LightRAG-owned table is a hard
    contract failure either way. ``convert_empty_plain`` is reserved for the
    fresh empty tables LightRAG creates upstream. DlightRAG-owned legacy shapes
    set it false and fail loudly even when empty, rather than pretending their
    old migration/index definitions are compatible.
    """

    name: str
    required_columns: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    required_indexes: tuple[str, ...] = ()
    required_index_markers: tuple[str, ...] = ()
    missing_ok: bool = False
    convert_empty_plain: bool = True


def default_child_name(table_name: str) -> str:
    """Return the deterministic internal name of one parent's DEFAULT child."""
    parent_digest = hashlib.sha256(pg_identifier(table_name).lower().encode("utf-8")).hexdigest()[
        :10
    ]
    return f"p_{parent_digest}_w_default"


def child_partition_name(table_name: str, workspace: str) -> str:
    """Return the deterministic internal name of one workspace's child.

    Never derived from the raw workspace text: the name is a parent digest plus
    a workspace digest, so adversarial workspace identifiers cannot reach SQL
    identifiers and hot partitions stay reproducible across processes.
    """
    parent_digest = hashlib.sha256(pg_identifier(table_name).lower().encode("utf-8")).hexdigest()[
        :10
    ]
    workspace_digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:16]
    return f"p_{parent_digest}_w_{workspace_digest}"


def _staging_name(table_name: str) -> str:
    parent_digest = hashlib.sha256(pg_identifier(table_name).lower().encode("utf-8")).hexdigest()[
        :10
    ]
    return f"t_{parent_digest}_prepartition"


def _loud_incompatible(reason: str) -> CorpusSchemaError:
    return CorpusSchemaError(f"{reason} {_RESET_HINT}")


async def _table_relkind(conn: Any, table_name: str) -> str | None:
    return await conn.fetchval(
        "SELECT c.relkind::text FROM pg_catalog.pg_class c WHERE c.oid = to_regclass($1)",
        table_name,
    )


async def _table_is_empty(conn: Any, table_name: str) -> bool:
    return await conn.fetchval(
        f"SELECT NOT EXISTS (SELECT 1 FROM {pg_identifier(table_name)} LIMIT 1)"  # noqa: S608
    )


async def _primary_key(conn: Any, table_name: str) -> tuple[str, tuple[str, ...]] | None:
    row = await conn.fetchrow(
        """
        SELECT con.conname AS name,
               array_agg(a.attname ORDER BY k.ord) AS columns
        FROM pg_catalog.pg_constraint con
        JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
        JOIN pg_catalog.pg_attribute a
          ON a.attrelid = con.conrelid AND a.attnum = k.attnum
        WHERE con.conrelid = to_regclass($1) AND con.contype = 'p'
        GROUP BY con.conname
        """,
        table_name,
    )
    if row is None:
        return None
    return str(row["name"]), tuple(str(column) for column in row["columns"])


async def _table_indexdefs(conn: Any, table_name: str) -> list[tuple[str, str]]:
    rows = await conn.fetch(
        """
        SELECT index_rel.relname AS indexname,
               pg_get_indexdef(i.indexrelid) AS indexdef
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class index_rel ON index_rel.oid = i.indexrelid
        WHERE i.indrelid = to_regclass($1) AND i.indisvalid AND i.indisready
        ORDER BY index_rel.relname
        """,
        table_name,
    )
    return [(str(row["indexname"]), str(row["indexdef"])) for row in rows]


async def _partition_children(conn: Any, table_name: str) -> set[str]:
    rows = await conn.fetch(
        """
        SELECT c.relname AS name
        FROM pg_catalog.pg_inherits i
        JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid
        WHERE i.inhparent = to_regclass($1)
        """,
        table_name,
    )
    return {str(row["name"]) for row in rows}


async def _parent_indexes(conn: Any, table_name: str) -> list[tuple[str, str]]:
    rows = await conn.fetch(
        """
        SELECT c.relname AS name, pg_get_indexdef(i.indexrelid) AS definition
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        WHERE i.indrelid = to_regclass($1) AND i.indisvalid
        ORDER BY c.relname
        """,
        table_name,
    )
    return [(str(row["name"]), str(row["definition"])) for row in rows]


async def _parent_index_has_child(conn: Any, index_name: str, table_name: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_inherits i
                JOIN pg_catalog.pg_class parent_idx ON parent_idx.oid = i.inhparent
                JOIN pg_catalog.pg_index pi ON pi.indexrelid = parent_idx.oid
                JOIN pg_catalog.pg_class table_rel ON table_rel.oid = pi.indrelid
                WHERE parent_idx.relname = $1 AND table_rel.relname = $2
            )
            """,
            index_name,
            table_name.lower(),
        )
    )


async def _validate_partitioned_state(
    conn: Any,
    spec: PartitionedTableSpec,
    *,
    detail_error: type[RuntimeError] = CorpusSchemaError,
) -> None:
    """Confirm one partitioned parent matches its spec (read-only)."""
    absent: list[str] = []
    present = await _names(
        conn,
        "SELECT a.attname AS name FROM pg_catalog.pg_attribute a "
        "WHERE a.attrelid = to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped",
        spec.name,
    )
    for column in spec.required_columns:
        if column not in present:
            absent.append(f"column {spec.name}.{column}")
    if spec.primary_key:
        key = await _primary_key(conn, spec.name)
        if key is None or key[1] != spec.primary_key:
            absent.append(f"primary key {spec.name} {spec.primary_key}")
    index_rows = await _parent_indexes(conn, spec.name)
    index_names = {name for name, _definition in index_rows}
    for index_name in spec.required_indexes:
        if index_name not in index_names:
            absent.append(f"index {index_name}")
    for marker in spec.required_index_markers:
        lowered = marker.lower()
        if not any(lowered in definition.lower() for _name, definition in index_rows):
            absent.append(f"index matching {marker!r}")
    for name, _definition in index_rows:
        if not await _parent_index_has_child(conn, name, spec.name):
            absent.append(f"child index for {name}")
    children = await _partition_children(conn, spec.name)
    default_child = default_child_name(spec.name)
    if default_child not in children:
        absent.append(f"partition child {default_child}")
    if absent:
        raise detail_error(
            f"Partitioned table {spec.name} is missing: {'; '.join(absent)}. {_RESET_HINT}"
        )


async def _names(conn: Any, query: str, table_name: str) -> set[str]:
    rows = await conn.fetch(query, table_name)
    return {str(row["name"]) for row in rows}


async def _convert_empty_plain_table(conn: Any, spec: PartitionedTableSpec) -> None:
    """Convert one verified-empty unpartitioned table into a partitioned parent.

    Only empty tables are converted, so no data can be lost: the staging table
    holds nothing but column definitions while the parent is rebuilt. A crash
    between the rename and the rebuild is repaired by ``_recover_staging`` on
    the next startup.
    """
    table_name = spec.name
    staging = _staging_name(table_name)
    indexdefs = await _table_indexdefs(conn, table_name)
    primary_key = await _primary_key(conn, table_name)
    logger.info("Converting empty table %s to a workspace-partitioned parent", table_name)
    await conn.execute(f"ALTER TABLE {pg_identifier(table_name)} RENAME TO {staging}")
    await conn.execute(
        f"CREATE TABLE {pg_identifier(table_name)} "  # noqa: S608 - validated identifiers
        f"(LIKE {staging} INCLUDING DEFAULTS INCLUDING STORAGE INCLUDING COMPRESSION) "
        f"PARTITION BY LIST ({PARTITION_COLUMN})"
    )
    await conn.execute(f"DROP TABLE {staging}")
    if primary_key is not None:
        constraint_name, columns = primary_key
        await conn.execute(
            f"ALTER TABLE {pg_identifier(table_name)} "  # noqa: S608
            f"ADD CONSTRAINT {pg_identifier(constraint_name)} "
            f"PRIMARY KEY ({', '.join(pg_identifier(column) for column in columns)})"
        )
    await conn.execute(
        f"CREATE TABLE {default_child_name(table_name)} "  # noqa: S608
        f"PARTITION OF {pg_identifier(table_name)} DEFAULT"
    )
    for indexname, indexdef in indexdefs:
        if primary_key is not None and indexname == primary_key[0]:
            continue  # the PK constraint re-created the backing index above
        await conn.execute(indexdef)
    await _validate_partitioned_state(conn, spec)
    logger.info(
        "Converted %s: parent + DEFAULT child + %d replayed index(es)",
        table_name,
        len(indexdefs) - (1 if primary_key is not None else 0),
    )


async def _recover_staging(conn: Any, spec: PartitionedTableSpec) -> None:
    """Restore a half-converted table so ensure() can re-run conversion."""
    staging = _staging_name(spec.name)
    relkind = await _table_relkind(conn, staging)
    if relkind is None:
        return
    parent_relkind = await _table_relkind(conn, spec.name)
    if parent_relkind is None:
        await conn.execute(
            f"ALTER TABLE {staging} RENAME TO {pg_identifier(spec.name)}"  # noqa: S608
        )
        return
    if parent_relkind == _RELKIND_PARTITIONED:
        # Transactional DDL should never leave this pair behind. Treat a
        # non-empty scratch relation as user data and fail instead of guessing
        # that it is disposable.
        await conn.execute(f"LOCK TABLE {staging} IN ACCESS EXCLUSIVE MODE")
        if not await _table_is_empty(conn, staging):
            raise _loud_incompatible(
                f"Partitioned table {spec.name} has a non-empty stale staging "
                f"relation {staging}; it was not removed automatically."
            )
        await conn.execute(f"DROP TABLE {staging}")
        return
    raise _loud_incompatible(
        f"Table {spec.name} and stale staging relation {staging} coexist in an "
        "unexpected schema state."
    )


async def ensure_partitioned_tables(
    conn: Any,
    *,
    specs: tuple[PartitionedTableSpec, ...],
) -> None:
    """Writer path: create/convert parents and DEFAULT children, or fail loudly."""
    for spec in specs:
        pg_identifier(spec.name)  # fail before touching the server
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1)",
                advisory_lock_key("dlightrag_partition_foundation", spec.name),
            )
            await _recover_staging(conn, spec)
            relkind = await _table_relkind(conn, spec.name)
            if relkind is None:
                if spec.missing_ok:
                    continue
                raise _loud_incompatible(
                    f"Required LightRAG table {spec.name} is missing after storage "
                    "initialization; LightRAG did not create it."
                )
            if relkind == _RELKIND_PARTITIONED:
                await _validate_partitioned_state(conn, spec)
                continue
            if relkind != _RELKIND_PLAIN:
                raise _loud_incompatible(
                    f"Table {spec.name} has unexpected relation kind {relkind!r}."
                )
            if not spec.convert_empty_plain:
                raise _loud_incompatible(
                    f"Table {spec.name} uses the old unpartitioned schema, which this "
                    "fresh-schema release does not migrate in place."
                )
            # Serialize the emptiness decision with every concurrent user of
            # the legacy table. Without this lock, an old process could commit
            # a row after NOT EXISTS and before RENAME, and the staging drop
            # would then destroy that row.
            await conn.execute(
                f"LOCK TABLE {spec.name} IN ACCESS EXCLUSIVE MODE"  # noqa: S608
            )
            if not await _table_is_empty(conn, spec.name):
                raise _loud_incompatible(
                    f"Table {spec.name} is an unpartitioned table that already holds "
                    "rows, so the new workspace-partitioned schema cannot be applied "
                    "in place."
                )
            await _convert_empty_plain_table(conn, spec)


async def verify_partitioned_tables(
    conn: Any,
    *,
    specs: tuple[PartitionedTableSpec, ...],
) -> None:
    """Reader path: validate partitioned parents without issuing any DDL."""
    for spec in specs:
        pg_identifier(spec.name)
        relkind = await _table_relkind(conn, spec.name)
        if relkind != _RELKIND_PARTITIONED:
            if relkind is None:
                raise _loud_incompatible(
                    f"Required table {spec.name} is missing; initialize the corpus "
                    "on a writer instance first."
                )
            raise _loud_incompatible(
                f"Table {spec.name} is not partitioned by workspace (relation kind "
                f"{relkind!r}); the running corpus predates the partitioned schema."
            )
        await _validate_partitioned_state(conn, spec)


async def attach_workspace_partition(
    conn: Any,
    *,
    table_name: str,
    workspace: str,
) -> str:
    """Create one deterministic attached child for compact planner tests.

    This helper creates a new ``PARTITION OF`` directly. Commit 3's production
    cutover must instead use ``ATTACH PARTITION`` for its detached, pre-indexed
    staging table. The raw workspace value is bound as a quoted literal, never
    as an identifier.
    """
    parent = pg_identifier(table_name)
    child = child_partition_name(parent, workspace)
    literal = await conn.fetchval("SELECT quote_literal($1)", str(workspace))
    await conn.execute(
        f"CREATE TABLE IF NOT EXISTS {child} "  # noqa: S608 - generated identifier
        f"PARTITION OF {parent} FOR VALUES IN ({literal})"
    )
    return child


class PGPartitionFoundation(PostgresOperationRunner):
    """Runner facade for the partition seam over the domain pool."""

    async def ensure_tables(self, *, specs: tuple[PartitionedTableSpec, ...]) -> None:
        """Create/convert partitioned parents on the writer path."""

        async def operation(conn: Any) -> None:
            await ensure_partitioned_tables(conn, specs=specs)

        await self._run(operation)

    async def verify_tables(self, *, specs: tuple[PartitionedTableSpec, ...]) -> None:
        """Validate partitioned parents on the reader path."""

        async def operation(conn: Any) -> None:
            await verify_partitioned_tables(conn, specs=specs)

        await self._run(operation)


__all__ = [
    "PARTITION_COLUMN",
    "PGPartitionFoundation",
    "PartitionedTableSpec",
    "attach_workspace_partition",
    "child_partition_name",
    "default_child_name",
    "ensure_partitioned_tables",
    "verify_partitioned_tables",
]
