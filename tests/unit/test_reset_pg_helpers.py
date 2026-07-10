# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for reset PostgreSQL helper boundaries."""

from unittest.mock import MagicMock

import pytest

from dlightrag.core.reset import (
    _clean_orphan_tables,
    _clean_workspace_meta,
    _list_all_workspaces,
)


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetchval(self, query: str) -> bool:
        assert "dlightrag_workspace_meta" in query
        return True

    async def execute(self, query: str, *args: object) -> None:
        self.executed.append((query, args))

    async def fetch(self, query: str) -> list[dict[str, str]]:
        assert "dlightrag_workspace_meta" in query
        return [{"workspace": "default"}, {"workspace": "research"}]

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def config() -> MagicMock:
    cfg = MagicMock()
    cfg.pg_connection_kwargs.return_value = {
        "host": "localhost",
        "port": 5432,
        "user": "dlightrag",
        "password": "test",
        "database": "dlightrag",
    }
    return cfg


async def test_clean_workspace_meta_uses_configured_pg_endpoint(monkeypatch, config) -> None:
    conn = _Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)

    await _clean_workspace_meta("research", config=config)

    config.pg_connection_kwargs.assert_called_once_with()
    assert conn.executed == [
        ("DELETE FROM dlightrag_workspace_meta WHERE workspace = $1", ("research",))
    ]
    assert conn.closed is True


async def test_list_all_workspaces_uses_configured_pg_endpoint(monkeypatch, config) -> None:
    conn = _Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)

    workspaces = await _list_all_workspaces(config=config)

    config.pg_connection_kwargs.assert_called_once_with()
    assert workspaces == ["default", "research"]
    assert conn.closed is True


async def test_clean_orphan_tables_quotes_public_table_identifiers(monkeypatch, config) -> None:
    class Conn:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...]]] = []
            self.closed = False

        async def fetch(self, query: str) -> list[dict[str, str]]:
            assert "pg_tables" in query
            return [{"tablename": 'dlightrag_bad"name'}]

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            if "information_schema.columns" in query:
                assert "table_schema = 'public'" in query
                assert args == ('dlightrag_bad"name',)
                return {"?column?": 1}
            if "COUNT(*)" in query:
                assert query == (
                    'SELECT COUNT(*) as count FROM public."dlightrag_bad""name" '
                    "WHERE workspace = $1"
                )
                assert args == ("research",)
                return {"count": 1}
            raise AssertionError(query)

        async def fetchval(self, query: str, *args: object) -> str:
            assert query == "SELECT quote_ident($1)"
            assert args == ('dlightrag_bad"name',)
            return '"dlightrag_bad""name"'

        async def execute(self, query: str, *args: object) -> None:
            self.executed.append((query, args))

        async def close(self) -> None:
            self.closed = True

    conn = Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    monkeypatch.setattr("dlightrag.config.get_config", lambda: config)

    cleaned = await _clean_orphan_tables("research", dry_run=False)

    assert cleaned == 1
    assert conn.executed == [
        (
            'DELETE FROM public."dlightrag_bad""name" WHERE workspace = $1',
            ("research",),
        ),
    ]
    assert conn.closed is True


async def test_clean_orphan_tables_never_drops_migration_managed_tables(
    monkeypatch, config
) -> None:
    """Reset DELETEs rows but MUST NOT DROP dlightrag_ migration-managed tables.

    Regression: dlightrag_checkpoints/doc_metadata/ingest_jobs are global tables
    with a workspace column, owned by dlightrag_schema_migrations. Resetting the
    last workspace empties them; dropping an emptied one orphaned the ledger and
    left the running app raising UndefinedTableError until the next restart.
    """

    class Conn:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...]]] = []
            self.closed = False

        async def fetch(self, query: str) -> list[dict[str, str]]:
            assert "pg_tables" in query
            return [{"tablename": "dlightrag_checkpoints"}]

        async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
            if "information_schema.columns" in query:
                return {"?column?": 1}
            if "COUNT(*)" in query:
                return {"count": 1}
            # A "SELECT EXISTS ... has_rows" probe would mean the DROP path is back.
            raise AssertionError(f"unexpected has_rows/DROP probe: {query}")

        async def fetchval(self, query: str, *args: object) -> str:
            assert query == "SELECT quote_ident($1)"
            return "dlightrag_checkpoints"

        async def execute(self, query: str, *args: object) -> None:
            self.executed.append((query, args))

        async def close(self) -> None:
            self.closed = True

    conn = Conn()

    async def fake_connect(**kwargs):
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    monkeypatch.setattr("dlightrag.config.get_config", lambda: config)

    cleaned = await _clean_orphan_tables("default", dry_run=False)

    assert cleaned == 1
    assert conn.executed == [
        ("DELETE FROM public.dlightrag_checkpoints WHERE workspace = $1", ("default",)),
    ]
    assert not any("DROP TABLE" in query for query, _ in conn.executed)
    assert conn.closed is True
