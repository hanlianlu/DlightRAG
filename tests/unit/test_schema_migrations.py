# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG-owned PostgreSQL schema migrations."""

from typing import Any

from dlightrag.storage.migrations import Migration, apply_migrations


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Conn:
    def __init__(self) -> None:
        self.applied: set[tuple[str, str]] = set()
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _Tx:
        return _Tx()

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))


async def test_apply_migrations_records_versions_and_replays_idempotent_statements() -> None:
    conn = _Conn()
    migrations = (
        Migration("0001", "first", ("CREATE TABLE example (id TEXT)",)),
        Migration("0002", "second", ("ALTER TABLE example ADD COLUMN name TEXT",)),
    )

    await apply_migrations(conn, scope="example", migrations=migrations)
    await apply_migrations(conn, scope="example", migrations=migrations)

    executed_sql = [query for query, _ in conn.executed]
    assert executed_sql.count("CREATE TABLE example (id TEXT)") == 2
    assert executed_sql.count("ALTER TABLE example ADD COLUMN name TEXT") == 2
    assert conn.applied == {("example", "0001"), ("example", "0002")}
