# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one PostgreSQL test-support surface: defaults, availability, scratch-database teardown,
and reading a migrated catalog back in the vocabulary a scope declares its schema in.

Every integration suite talks to the same server and owns at most one scratch database, so those
concerns live here instead of being re-implemented in each file.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any, Protocol

import asyncpg
import pytest

from dlightrag.adapters.postgres.core._migrations import ForeignKeyRequirement, TableRequirement

# `localhost` is ambiguous on a machine that also runs a host PostgreSQL on [::1]: the resolver
# prefers that IPv6 instance over the container's IPv4 mapping, so the suite silently tested against
# a server whose `dlightrag` role is not the superuser CI provides - which surfaced as unrelated
# "permission denied to terminate process" and "must be superuser to create extension" failures.
# Pin the container mapping and keep PGHOST for CI and other deployments.
PG_CONN_KWARGS: dict[str, Any] = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "dlightrag"),
    password=os.environ.get("PGPASSWORD", "dlightrag"),
    database=os.environ.get("PGDATABASE", "dlightrag"),
)

_DROP_ATTEMPTS = 5
_DROP_RETRY_SECONDS = 0.2


async def postgres_available() -> bool:
    """Whether the configured server accepts a connection."""
    try:
        connection = await asyncpg.connect(**PG_CONN_KWARGS)
    except Exception:
        return False
    try:
        await connection.fetchval("SELECT 1")
    finally:
        await connection.close()
    return True


async def skip_without_postgres() -> None:
    """Skip the calling suite when no PostgreSQL is reachable."""
    if not await postgres_available():
        pytest.skip("PostgreSQL not available")


async def require_postgres() -> None:
    """Fail a validation command before skip-capable suites if PostgreSQL is unavailable."""
    if not await postgres_available():
        raise RuntimeError("PostgreSQL is not available")


class DropAdmin(Protocol):
    """The slice of a connection a drop needs, so a test can stand in for it."""

    async def execute(self, query: str) -> str: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...


async def drop_scratch_database(admin: DropAdmin, database: str) -> None:
    """Drop one scratch database over an open administrative connection.

    `DROP DATABASE ... WITH (FORCE)` terminates whatever is still attached, but it needs
    `pg_signal_backend` for another role's backend and refuses outright for a superuser-owned one,
    such as the autovacuum worker that may start on a database a test has just created. That
    backend always detaches on its own, so retry briefly and, if it never does, report what was
    still attached instead of surfacing a bare `InsufficientPrivilegeError`.
    """
    for attempt in range(_DROP_ATTEMPTS):
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
            return
        except asyncpg.exceptions.InsufficientPrivilegeError:
            if attempt + 1 == _DROP_ATTEMPTS:
                break
            await asyncio.sleep(_DROP_RETRY_SECONDS)
    attached = await admin.fetch(
        "select pid, state, coalesce(left(query, 60), '-') from pg_stat_activity"
        " where datname = $1 and pid <> pg_backend_pid()",
        database,
    )
    raise RuntimeError(f"cannot drop {database}: backends still attached: {attached}")


class RunDeleter(Protocol):
    """The run store's caller-owned-transaction deletion seam."""

    async def delete_runs_in(self, conn: Any, *, owner_id: str, run_ids: Sequence[str]) -> Any: ...


async def delete_runs(
    pool: Any, store: RunDeleter, *, owner_id: str, run_ids: Sequence[str]
) -> Any:
    """Delete runs in a transaction of their own.

    Production deletes runs only inside the transaction of the owner that links them (a Web
    conversation), so the store exposes that seam alone; a suite exercising run deletion by
    itself supplies the transaction here.
    """
    async with pool.acquire() as conn, conn.transaction():
        return await store.delete_runs_in(conn, owner_id=owner_id, run_ids=run_ids)


async def drop_database(database: str) -> None:
    """Drop one scratch database, opening and closing the administrative connection it needs.

    The connection goes to the configured maintenance database, never to `database` itself: no
    server can drop the database a connection is currently using.
    """
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await drop_scratch_database(admin, database)
    finally:
        await admin.close()


_CATALOG_TABLES = """SELECT c.relname AS name
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND c.relname LIKE 'dlightrag%'
"""

_CATALOG_COLUMNS = """SELECT a.attname AS name
FROM pg_catalog.pg_attribute a
WHERE a.attrelid = $1::regclass AND a.attnum > 0 AND NOT a.attisdropped
"""

_CATALOG_KEYS = """SELECT con.contype::text AS contype, array_agg(a.attname ORDER BY k.ord) AS columns
FROM pg_catalog.pg_constraint con
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
JOIN pg_catalog.pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
WHERE con.conrelid = $1::regclass AND con.contype IN ('p', 'u')
GROUP BY con.oid, con.contype
"""

_CATALOG_FOREIGN_KEYS = """SELECT cf.relname AS referenced, array_agg(a.attname ORDER BY k.ord) AS columns
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class cf ON cf.oid = con.confrelid
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
JOIN pg_catalog.pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
WHERE con.conrelid = $1::regclass AND con.contype = 'f'
GROUP BY con.oid, cf.relname
"""

_CATALOG_CHECKS = """SELECT con.conname AS name
FROM pg_catalog.pg_constraint con
WHERE con.conrelid = $1::regclass AND con.contype = 'c'
"""

# An index that backs a primary key or unique constraint is declared through that
# constraint; every other index is declared by name, plain or unique.
_CATALOG_INDEXES = """SELECT c.relname AS name, i.indisunique AS is_unique
FROM pg_catalog.pg_index i
JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
WHERE i.indrelid = $1::regclass
  AND NOT EXISTS (
      SELECT 1 FROM pg_catalog.pg_constraint con
      WHERE con.conrelid = i.indrelid AND con.conindid = i.indexrelid
        AND con.contype IN ('p', 'u', 'x')
  )
"""

_CATALOG_TRIGGERS = """SELECT t.tgname AS name
FROM pg_catalog.pg_trigger t
WHERE t.tgrelid = $1::regclass AND NOT t.tgisinternal
"""


# Every object's full definition, keyed by table-qualified name.
_CATALOG_DEFINITIONS = {
    "columns": """SELECT c.relname || '.' || a.attname AS key,
            format_type(a.atttypid, a.atttypmod) || ' not null=' || a.attnotnull
            || ' default=' || coalesce(pg_get_expr(d.adbin, d.adrelid), '') AS definition
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND c.relname LIKE 'dlightrag%'
          AND a.attnum > 0 AND NOT a.attisdropped""",
    "constraints": """SELECT c.relname || '.' || con.conname AS key,
            pg_get_constraintdef(con.oid) AS definition
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname LIKE 'dlightrag%'""",
    "indexes": """SELECT i.relname AS key, pg_get_indexdef(i.oid) AS definition
        FROM pg_catalog.pg_index x
        JOIN pg_catalog.pg_class i ON i.oid = x.indexrelid
        JOIN pg_catalog.pg_class t ON t.oid = x.indrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public' AND t.relname LIKE 'dlightrag%'""",
    "triggers": """SELECT t.tgname AS key, pg_get_triggerdef(t.oid) AS definition
        FROM pg_catalog.pg_trigger t
        JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND NOT t.tgisinternal AND c.relname LIKE 'dlightrag%'""",
    "functions": """SELECT p.proname AS key, pg_get_functiondef(p.oid) AS definition
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname LIKE 'dlightrag%'""",
}


async def catalog_definitions(conn: Any) -> dict[str, dict[str, str]]:
    """Every DlightRAG column, constraint, index, trigger, and function, by definition."""
    return {
        kind: {str(row["key"]): str(row["definition"]) for row in await conn.fetch(query)}
        for kind, query in _CATALOG_DEFINITIONS.items()
    }


async def catalog_tables(conn: Any) -> set[str]:
    """Name every DlightRAG table the connected database holds, ledger included."""
    return {str(row["name"]) for row in await conn.fetch(_CATALOG_TABLES)}


async def catalog_table(conn: Any, name: str) -> TableRequirement:
    """Read one table back as the requirement that would declare exactly what it has."""
    keys = [
        (str(row["contype"]), tuple(row["columns"]))
        for row in await conn.fetch(_CATALOG_KEYS, name)
    ]
    indexes = await conn.fetch(_CATALOG_INDEXES, name)
    return TableRequirement(
        name=name,
        columns=tuple(sorted(str(row["name"]) for row in await conn.fetch(_CATALOG_COLUMNS, name))),
        primary_key=next((columns for kind, columns in keys if kind == "p"), ()),
        unique=tuple(sorted(columns for kind, columns in keys if kind == "u")),
        foreign_keys=tuple(
            sorted(
                (
                    ForeignKeyRequirement(
                        columns=tuple(row["columns"]), references=str(row["referenced"])
                    )
                    for row in await conn.fetch(_CATALOG_FOREIGN_KEYS, name)
                ),
                key=lambda key: (key.columns, key.references),
            )
        ),
        checks=tuple(sorted(str(row["name"]) for row in await conn.fetch(_CATALOG_CHECKS, name))),
        indexes=tuple(sorted(str(row["name"]) for row in indexes if not row["is_unique"])),
        unique_indexes=tuple(sorted(str(row["name"]) for row in indexes if row["is_unique"])),
        triggers=tuple(
            sorted(str(row["name"]) for row in await conn.fetch(_CATALOG_TRIGGERS, name))
        ),
    )


def declared_shape(table: TableRequirement) -> TableRequirement:
    """The same requirement with every declaration in catalog order, for comparison."""
    return TableRequirement(
        name=table.name,
        columns=tuple(sorted(table.columns)),
        primary_key=table.primary_key,
        unique=tuple(sorted(table.unique)),
        foreign_keys=tuple(
            sorted(table.foreign_keys, key=lambda key: (key.columns, key.references))
        ),
        checks=tuple(sorted(table.checks)),
        indexes=tuple(sorted(table.indexes)),
        unique_indexes=tuple(sorted(table.unique_indexes)),
        triggers=tuple(sorted(table.triggers)),
    )
