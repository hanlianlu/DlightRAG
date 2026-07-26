# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG-owned PostgreSQL schema migrations."""

from collections.abc import Sequence
from typing import Any

import pytest

from dlightrag.storage.migrations import Migration, apply_migrations


class _Tx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> None:
        self._conn.transaction_events.append("begin")
        return None

    async def __aexit__(self, exc_type: object, *args: object) -> None:
        self._conn.transaction_events.append("rollback" if exc_type else "commit")
        return None


class _Record:
    def __init__(self, version: str) -> None:
        self._version = version

    def __getitem__(self, key: str) -> str:
        if key != "version":
            raise KeyError(key)
        return self._version


class _Conn:
    def __init__(self, *, row_shape: str = "dict") -> None:
        self.applied: set[tuple[str, str]] = set()
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.transaction_events: list[str] = []
        self.failures: dict[str, int] = {}
        self.row_shape = row_shape

    def transaction(self) -> _Tx:
        return _Tx(self)

    async def fetch(self, query: str, *args: Any) -> Sequence[dict[str, str] | _Record]:
        self.fetches.append((query, args))
        scope = str(args[0])
        versions = sorted(
            version for applied_scope, version in self.applied if applied_scope == scope
        )
        if self.row_shape == "record":
            return [_Record(version) for version in versions]
        return [{"version": version} for version in versions]

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        remaining_failures = self.failures.get(query, 0)
        if remaining_failures > 0:
            self.failures[query] = remaining_failures - 1
            raise RuntimeError(f"boom: {query}")
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))
            return "INSERT 0 1"
        if query.startswith("SELECT pg_advisory_lock"):
            return "SELECT 1"
        if query.startswith("SELECT pg_advisory_unlock"):
            return "SELECT 1"
        return "OK"


def _example_migrations() -> tuple[Migration, ...]:
    return (
        Migration("0001", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("0002", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )


async def test_apply_migrations_skips_versions_already_recorded_for_scope() -> None:
    conn = _Conn()
    migrations = _example_migrations()

    await apply_migrations(conn, scope="example", migrations=migrations)
    await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 1
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 1
    assert conn.applied == {("example", "0001"), ("example", "0002")}


async def test_apply_migrations_runs_only_newly_appended_versions() -> None:
    conn = _Conn(row_shape="record")
    initial = (Migration("0001", "first", ("CREATE TABLE example (id TEXT)",)),)
    appended = initial + (
        Migration("0002", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )

    await apply_migrations(conn, scope="example", migrations=initial)
    await apply_migrations(conn, scope="example", migrations=appended)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 1
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 1
    assert conn.applied == {("example", "0001"), ("example", "0002")}


async def test_apply_migrations_does_not_record_failed_versions() -> None:
    conn = _Conn()
    migrations = (
        Migration("0001", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("0002", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )
    conn.failures["ALTER TABLE example ADD COLUMN name TEXT"] = 1

    with pytest.raises(RuntimeError, match="ALTER TABLE example ADD COLUMN name TEXT"):
        await apply_migrations(conn, scope="example", migrations=migrations)

    assert conn.applied == {("example", "0001")}
    assert ("example", "0002") not in conn.applied
    assert conn.transaction_events == ["begin", "commit", "begin", "rollback"]
    assert not any(
        query.startswith("INSERT INTO dlightrag_schema_migrations")
        and args[:2] == ("example", "0002")
        for query, args in conn.executed
    )


async def test_apply_migrations_isolates_applied_versions_per_scope() -> None:
    conn = _Conn()
    migrations = _example_migrations()

    await apply_migrations(conn, scope="alpha", migrations=migrations)
    await apply_migrations(conn, scope="beta", migrations=migrations)

    assert conn.applied == {
        ("alpha", "0001"),
        ("alpha", "0002"),
        ("beta", "0001"),
        ("beta", "0002"),
    }
    applied_reads = [
        args[0] for query, args in conn.fetches if "dlightrag_schema_migrations" in query
    ]
    assert applied_reads == ["alpha", "beta"]


async def test_apply_migrations_rejects_gapped_current_ledger_before_running_migrations() -> None:
    conn = _Conn()
    conn.applied.add(("example", "0002"))
    migrations = _example_migrations()

    with pytest.raises(
        RuntimeError,
        match=(
            r"scope 'example'.*missing current versions: 0001.*"
            r"out-of-order recorded current versions: 0002"
        ),
    ):
        await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert "CREATE TABLE example (id TEXT)" not in executed_sql
    assert "ALTER TABLE example ADD COLUMN name TEXT" not in executed_sql
    assert not any(
        query.startswith("INSERT INTO dlightrag_schema_migrations") for query in executed_sql
    )
    assert conn.applied == {("example", "0002")}
    assert conn.transaction_events == []


async def test_apply_migrations_tolerates_historical_unknown_ledger_versions() -> None:
    conn = _Conn()
    conn.applied.update({("example", "0001"), ("example", "0999")})
    migrations = _example_migrations()

    await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 0
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 1
    assert conn.applied == {("example", "0001"), ("example", "0002"), ("example", "0999")}


async def test_apply_migrations_releases_lock_when_gap_validation_fails() -> None:
    conn = _Conn()
    conn.applied.add(("example", "0002"))

    with pytest.raises(RuntimeError, match=r"scope 'example'"):
        await apply_migrations(conn, scope="example", migrations=_example_migrations())

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("SELECT pg_advisory_lock($1)") == 1
    assert executed_sql.count("SELECT pg_advisory_unlock($1)") == 1
    assert executed_sql[-1] == "SELECT pg_advisory_unlock($1)"


async def test_apply_migrations_rejects_duplicate_versions_before_mutating_db() -> None:
    conn = _Conn()
    duplicate = (
        Migration("0001", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("0001", "first again", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )

    with pytest.raises(ValueError, match="Duplicate schema migration version: 0001"):
        await apply_migrations(conn, scope="example", migrations=duplicate)

    assert conn.executed == []
