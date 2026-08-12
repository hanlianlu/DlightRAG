# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight migrations for DlightRAG-owned PostgreSQL schemas."""

import hashlib
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class Migration:
    """One idempotent DlightRAG-owned PostgreSQL schema migration."""

    version: str
    description: str
    statements: tuple[str, ...]


class SchemaValidationError(RuntimeError):
    """A required schema is absent or incompatible with this software revision.

    Raised only by validation paths. It is terminal for startup: no process can
    repair it by staying up, so it must never degrade into partial readiness.
    """


_LOCK_NAMESPACE = "dlightrag_schema_migration"


def _advisory_lock_key(scope: str) -> int:
    """Stable signed 64-bit advisory-lock key for a migration scope."""
    digest = hashlib.blake2b(f"{_LOCK_NAMESPACE}:{scope}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def apply_migrations(
    conn: Any,
    *,
    scope: str,
    migrations: tuple[Migration, ...],
    require_applied_prefix: bool = True,
) -> None:
    """Ensure idempotent migrations and record their versions in the ledger.

    A per-scope session advisory lock serializes concurrent callers (e.g. app
    replicas first-touching a lazily-initialized store), so they cannot race on
    the same ``IF NOT EXISTS`` DDL. RAGService startup already holds a broader
    init lock; this keeps ``apply_migrations`` safe on its own path too.

    ``require_applied_prefix`` keeps static migration scopes fail-fast by
    default. Dynamic scopes that may insert idempotent declared versions later
    can opt out and replay any missing versions in the current declared order.
    """
    _validate_unique_versions(migrations)
    lock_key = _advisory_lock_key(scope)
    await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
    try:
        await conn.execute(_CREATE_LEDGER)
        applied_versions = await _applied_versions_for_scope(conn, scope)
        _validate_applied_state(
            scope,
            migrations,
            applied_versions,
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
) -> None:
    """Confirm this revision's declared versions are already applied, issuing no DDL.

    Reader processes do not own schema: they must attach to a schema a writer
    already migrated. The ledger is the authority, so one read per scope proves
    every table, column, and index that revision declares is present, and a
    reader whose deployment ran ahead of its writer fails startup by name
    instead of serving traffic against an older schema.
    """
    _validate_unique_versions(migrations)
    if not await conn.fetchval(_LEDGER_EXISTS):
        raise SchemaValidationError(
            "dlightrag_schema_migrations is missing; apply DlightRAG migrations "
            "on a writer instance before starting a reader"
        )
    applied_versions = await _applied_versions_for_scope(conn, scope)
    missing = [
        migration.version for migration in migrations if migration.version not in applied_versions
    ]
    if missing:
        raise SchemaValidationError(
            f"Schema migration scope '{scope}' is missing versions: {', '.join(missing)}; "
            "apply DlightRAG migrations on a writer instance before starting a reader"
        )


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
    require_applied_prefix: bool,
) -> None:
    if not require_applied_prefix:
        return

    declared_versions = [migration.version for migration in migrations]
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
    raise RuntimeError(
        "Schema migration ledger for "
        f"scope '{scope}' is non-prefix across current migrations; "
        f"missing current versions: {', '.join(missing_versions)}; "
        "out-of-order recorded current versions: "
        f"{', '.join(out_of_order_versions)}"
    )
