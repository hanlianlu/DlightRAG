# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Profile Memory brings an installed pg_textsearch up to the release its server carries."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import pytest
from dlightrag_memory._storage.pg_bm25 import install_text_search


class _Conn:
    """A server whose pg_textsearch is installed at ``installed`` and carries ``available``."""

    def __init__(self, installed: str | None, available: str | None, refuse: bool = False) -> None:
        self.executed: list[str] = []
        self._versions = {"installed_version": installed, "default_version": available}
        self._refuse = refuse

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield

    async def execute(self, statement: str) -> None:
        if self._refuse and statement.startswith("ALTER EXTENSION"):
            raise asyncpg.InsufficientPrivilegeError("must be owner of extension pg_textsearch")
        self.executed.append(statement)

    async def fetchrow(self, query: str) -> dict[str, str | None]:
        assert "pg_available_extensions" in query
        return self._versions


_CREATED = [
    "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
    "CREATE EXTENSION IF NOT EXISTS pg_jieba",
]


async def test_an_installed_pg_textsearch_behind_the_server_is_updated_before_pg_jieba() -> None:
    conn = _Conn("1.4.0", "1.5.1")

    await install_text_search(conn)

    assert conn.executed == [
        "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
        "ALTER EXTENSION pg_textsearch UPDATE",
        "CREATE EXTENSION IF NOT EXISTS pg_jieba",
    ]


@pytest.mark.parametrize(
    ("installed", "available"),
    [("1.5.1", "1.5.1"), ("1.6.0", "1.5.1"), (None, "1.5.1"), ("1.5.1-dev", "1.5.1")],
)
async def test_a_current_newer_or_unnumbered_pg_textsearch_is_left_alone(
    installed: str | None, available: str
) -> None:
    conn = _Conn(installed, available)

    await install_text_search(conn)

    assert conn.executed == _CREATED


async def test_a_refused_update_is_logged_and_pg_jieba_still_installs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn = _Conn("1.4.0", "1.5.1", refuse=True)

    with caplog.at_level(logging.WARNING):
        await install_text_search(conn)

    assert conn.executed == _CREATED
    (record,) = caplog.records
    assert "left pg_textsearch at 1.4.0 while the server carries 1.5.1" in record.getMessage()
