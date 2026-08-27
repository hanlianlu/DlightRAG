# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight migrations for DlightRAG-owned PostgreSQL schemas."""

from dataclasses import dataclass
from typing import Any

from dlightrag.adapters.postgres.core._locks import advisory_lock_key

_CREATE_LEDGER = """CREATE TABLE IF NOT EXISTS dlightrag_schema_migrations (
    scope       TEXT NOT NULL,
    version     TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope, version)
)
"""

_INSERT_APPLIED = """INSERT INTO dlightrag_schema_migrations (scope, version, description)
VALUES ($1, $2, $3)
ON CONFLICT (scope, version) DO NOTHING
"""

_SELECT_APPLIED = """SELECT version
FROM dlightrag_schema_migrations
WHERE scope = $1
ORDER BY version
"""

_LEDGER_EXISTS = "SELECT to_regclass('dlightrag_schema_migrations') IS NOT NULL"

# Structured catalog reads: every required object is resolved through the same
# search_path the migration DDL used, then looked up by relation OID.
_TABLE_OID = """SELECT c.oid
FROM pg_catalog.pg_class c
WHERE c.oid = to_regclass($1) AND c.relkind IN ('r', 'p')
"""

_TABLE_COLUMNS = """SELECT a.attname AS name
FROM pg_catalog.pg_attribute a
WHERE a.attrelid = $1 AND a.attnum > 0 AND NOT a.attisdropped
"""

_TABLE_INDEXES = """SELECT c.relname AS name
FROM pg_catalog.pg_index i
JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
WHERE i.indrelid = $1 AND i.indisvalid AND i.indisready
"""

_TABLE_UNIQUE_INDEXES = """SELECT c.relname AS name
FROM pg_catalog.pg_index i
JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
WHERE i.indrelid = $1 AND i.indisvalid AND i.indisready AND i.indisunique
"""

_TABLE_CHECKS = """SELECT con.conname AS name
FROM pg_catalog.pg_constraint con
WHERE con.conrelid = $1 AND con.contype = 'c' AND con.convalidated
"""

_TABLE_KEYS = """SELECT con.contype::text AS contype,
       array_agg(a.attname ORDER BY k.ord) AS columns
FROM pg_catalog.pg_constraint con
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
JOIN pg_catalog.pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
WHERE con.conrelid = $1 AND con.contype IN ('p', 'u')
GROUP BY con.oid, con.contype
"""

_TABLE_FOREIGN_KEYS = """SELECT cf.relname AS referenced,
       array_agg(a.attname ORDER BY k.ord) AS columns
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class cf ON cf.oid = con.confrelid
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
JOIN pg_catalog.pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
WHERE con.conrelid = $1 AND con.contype = 'f'
GROUP BY con.oid, cf.relname
"""


@dataclass(frozen=True)
class Migration:
    """One idempotent DlightRAG-owned PostgreSQL schema migration."""

    version: str
    description: str
    statements: tuple[str, ...]


@dataclass(frozen=True)
class ForeignKeyRequirement:
    """Local columns that must still reference ``references``."""

    columns: tuple[str, ...]
    references: str


@dataclass(frozen=True)
class TableRequirement:
    """Schema objects one revision requires on one table.

    ``unique_indexes`` names the partial unique indexes that enforce an invariant
    no constraint can express; the catalog must report them as unique, because a
    same-named index rebuilt without uniqueness would silently retire that invariant.
    """

    name: str
    columns: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    unique: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKeyRequirement, ...] = ()
    checks: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    unique_indexes: tuple[str, ...] = ()


async def apply_migrations(
    conn: Any,
    *,
    scope: str,
    migrations: tuple[Migration, ...],
    schema_error: type[RuntimeError],
    require_applied_prefix: bool = True,
) -> None:
    """Ensure idempotent migrations and record their versions in the ledger.

    A per-scope session advisory lock serializes concurrent callers (e.g. app
    replicas first-touching a lazily-initialized store), so they cannot race on
    the same ``IF NOT EXISTS`` DDL. WorkspaceRag startup already holds a broader
    init lock; this keeps ``apply_migrations`` safe on its own path too.

    ``require_applied_prefix`` keeps static migration scopes fail-fast by
    default. Dynamic scopes can opt out and replay missing declared versions in
    the current order. Both modes reject undeclared ledger versions and require
    a development-data reset.
    """
    _validate_unique_versions(migrations)
    lock_key = advisory_lock_key("dlightrag_schema_migration", scope)
    await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
    try:
        await conn.execute(_CREATE_LEDGER)
        applied_versions = await _applied_versions_for_scope(conn, scope)
        _validate_applied_state(
            scope,
            migrations,
            applied_versions,
            schema_error=schema_error,
            require_applied_prefix=require_applied_prefix,
        )
        for migration in migrations:
            if migration.version in applied_versions:
                continue
            async with conn.transaction():
                for statement in migration.statements:
                    await conn.execute(statement)
                await conn.execute(_INSERT_APPLIED, scope, migration.version, migration.description)
            applied_versions.add(migration.version)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)


async def _applied_versions_for_scope(conn: Any, scope: str) -> set[str]:
    rows = await conn.fetch(_SELECT_APPLIED, scope)
    return {_version_from_row(row) for row in rows}


async def verify_migrations(
    conn: Any,
    *,
    scope: str,
    migrations: tuple[Migration, ...],
    tables: tuple[TableRequirement, ...],
    schema_error: type[RuntimeError],
) -> None:
    """Confirm this revision's schema is already present, issuing no DDL.

    Reader processes do not own schema: they must attach to a schema a writer
    already migrated. The ledger alone cannot prove that, because a recorded
    version survives a table, column, index, or constraint that was later
    dropped, so each scope also names the objects it requires and they are read
    back from the PostgreSQL catalog.
    """
    _validate_unique_versions(migrations)
    if not await conn.fetchval(_LEDGER_EXISTS):
        raise schema_error(
            "dlightrag_schema_migrations is missing; apply DlightRAG migrations "
            "on a writer instance before starting a reader"
        )
    applied_versions = await _applied_versions_for_scope(conn, scope)
    _validate_applied_state(
        scope,
        migrations,
        applied_versions,
        schema_error=schema_error,
        require_applied_prefix=False,
    )
    missing = [
        migration.version for migration in migrations if migration.version not in applied_versions
    ]
    if missing:
        raise schema_error(
            f"Schema migration scope '{scope}' is missing versions: {', '.join(missing)}; "
            "apply DlightRAG migrations on a writer instance before starting a reader"
        )

    absent: list[str] = []
    for table in tables:
        absent.extend(await _absent_table_objects(conn, table))
    if absent:
        raise schema_error(
            f"Schema migration scope '{scope}' records every version but is missing: "
            f"{'; '.join(absent)}; apply DlightRAG migrations on a writer instance "
            "before starting a reader"
        )


async def _absent_table_objects(conn: Any, table: TableRequirement) -> list[str]:
    """Name every declared object of ``table`` the catalog does not report."""
    oid = await conn.fetchval(_TABLE_OID, table.name)
    if oid is None:
        return [f"table {table.name}"]

    absent: list[str] = []
    if table.columns:
        present = await _names(conn, _TABLE_COLUMNS, oid)
        absent += [f"column {table.name}.{name}" for name in table.columns if name not in present]
    if table.indexes:
        present = await _names(conn, _TABLE_INDEXES, oid)
        absent += [f"index {name}" for name in table.indexes if name not in present]
    if table.unique_indexes:
        present = await _names(conn, _TABLE_UNIQUE_INDEXES, oid)
        absent += [f"unique index {name}" for name in table.unique_indexes if name not in present]
    if table.checks:
        present = await _names(conn, _TABLE_CHECKS, oid)
        absent += [f"constraint {name}" for name in table.checks if name not in present]
    if table.primary_key or table.unique:
        keys = [
            (str(row["contype"]), tuple(row["columns"]))
            for row in await conn.fetch(_TABLE_KEYS, oid)
        ]
        if table.primary_key and ("p", table.primary_key) not in keys:
            absent.append(f"primary key {_columns(table.name, table.primary_key)}")
        unique_keys = {columns for _contype, columns in keys}
        absent += [
            f"unique key {_columns(table.name, columns)}"
            for columns in table.unique
            if columns not in unique_keys
        ]
    if table.foreign_keys:
        present_keys = {
            (tuple(row["columns"]), str(row["referenced"]))
            for row in await conn.fetch(_TABLE_FOREIGN_KEYS, oid)
        }
        absent += [
            f"foreign key {_columns(table.name, key.columns)} -> {key.references}"
            for key in table.foreign_keys
            if (key.columns, key.references) not in present_keys
        ]
    return absent


async def _names(conn: Any, query: str, oid: Any) -> set[str]:
    return {str(row["name"]) for row in await conn.fetch(query, oid)}


def _columns(table: str, columns: tuple[str, ...]) -> str:
    return f"{table} ({', '.join(columns)})"


def _version_from_row(row: Any) -> str:
    try:
        return str(row["version"])
    except (KeyError, TypeError) as exc:  # pragma: no cover - guarded by unit fakes
        raise TypeError("schema migration rows must expose a 'version' field") from exc


def _validate_unique_versions(migrations: tuple[Migration, ...]) -> None:
    seen: set[str] = set()
    for migration in migrations:
        if migration.version in seen:
            raise ValueError(f"Duplicate schema migration version: {migration.version}")
        seen.add(migration.version)


def _validate_applied_state(
    scope: str,
    migrations: tuple[Migration, ...],
    applied_versions: set[str],
    *,
    schema_error: type[RuntimeError],
    require_applied_prefix: bool,
) -> None:
    declared_versions = [migration.version for migration in migrations]
    undeclared_versions = sorted(applied_versions - set(declared_versions))
    if undeclared_versions:
        raise schema_error(
            f"Schema migration scope '{scope}' contains undeclared versions: "
            f"{', '.join(undeclared_versions)}; reset the development database "
            "before starting this revision"
        )
    if not require_applied_prefix:
        return

    applied_indices = [
        index for index, version in enumerate(declared_versions) if version in applied_versions
    ]
    if not applied_indices:
        return

    last_applied_index = max(applied_indices)
    missing_versions = [
        version
        for index, version in enumerate(declared_versions[:last_applied_index])
        if version not in applied_versions
    ]
    if not missing_versions:
        return

    first_missing_index = next(
        index for index, version in enumerate(declared_versions) if version not in applied_versions
    )
    out_of_order_versions = [
        version
        for version in declared_versions[first_missing_index:]
        if version in applied_versions
    ]
    raise schema_error(
        "Schema migration ledger for "
        f"scope '{scope}' is non-prefix across current migrations; "
        f"missing current versions: {', '.join(missing_versions)}; "
        "out-of-order recorded current versions: "
        f"{', '.join(out_of_order_versions)}"
    )
