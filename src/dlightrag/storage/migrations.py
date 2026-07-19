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


@dataclass(frozen=True)
class Migration:
    """One idempotent DlightRAG-owned PostgreSQL schema migration."""

    version: str
    description: str
    statements: tuple[str, ...]


_LOCK_NAMESPACE = "dlightrag_schema_migration"


def _advisory_lock_key(scope: str) -> int:
    """Stable signed 64-bit advisory-lock key for a migration scope."""
    digest = hashlib.blake2b(f"{_LOCK_NAMESPACE}:{scope}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


async def apply_migrations(conn: Any, *, scope: str, migrations: tuple[Migration, ...]) -> None:
    """Ensure idempotent migrations and record their versions in the ledger.

    A per-scope session advisory lock serializes concurrent callers (e.g. app
    replicas first-touching a lazily-initialized store), so they cannot race on
    the same ``IF NOT EXISTS`` DDL. RAGService startup already holds a broader
    init lock; this keeps ``apply_migrations`` safe on its own path too.
    """
    _validate_unique_versions(migrations)
    lock_key = _advisory_lock_key(scope)
    await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
    try:
        await conn.execute(_CREATE_LEDGER)
        for migration in migrations:
            async with conn.transaction():
                for statement in migration.statements:
                    await conn.execute(statement)
                await conn.execute(_INSERT_APPLIED, scope, migration.version, migration.description)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)


def _validate_unique_versions(migrations: tuple[Migration, ...]) -> None:
    seen: set[str] = set()
    for migration in migrations:
        if migration.version in seen:
            raise ValueError(f"Duplicate schema migration version: {migration.version}")
        seen.add(migration.version)
