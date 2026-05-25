# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for reset PostgreSQL helper boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlightrag.core.reset import _clean_workspace_meta, _list_all_workspaces


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


async def test_clean_workspace_meta_uses_primary_config(monkeypatch, config) -> None:
    conn = _Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)

    await _clean_workspace_meta("research", config=config)

    config.pg_connection_kwargs.assert_called_once_with("primary")
    assert conn.executed == [
        ("DELETE FROM dlightrag_workspace_meta WHERE workspace = $1", ("research",))
    ]
    assert conn.closed is True


async def test_list_all_workspaces_uses_primary_config(monkeypatch, config) -> None:
    conn = _Conn()

    async def fake_connect(**kwargs):
        assert kwargs == config.pg_connection_kwargs.return_value
        return conn

    import asyncpg

    monkeypatch.setattr(asyncpg, "connect", fake_connect)

    workspaces = await _list_all_workspaces(config=config)

    config.pg_connection_kwargs.assert_called_once_with("primary")
    assert workspaces == ["default", "research"]
    assert conn.closed is True
