# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one PostgreSQL test-support surface: defaults, availability, and scratch-database teardown.

Every integration suite talks to the same server and owns at most one scratch database, so those
three concerns live here instead of being re-implemented in each file.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol

import asyncpg
import pytest

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
