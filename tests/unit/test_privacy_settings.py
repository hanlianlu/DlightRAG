# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Conversation Contribution when the connection to PostgreSQL is lost.

A connection can drop after a statement committed and before the client heard of it. The pool
answers that by trying a read again and a write never, and the setting depends on the difference:
a turn-on tried again seconds later, after the owner has turned it off, would give back a
permission they had withdrawn.
"""

from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.privacy import PrivacySettings

_BEGAN = datetime.datetime(2026, 10, 9, 20, 15, 3, tzinfo=datetime.UTC)
_LOST = asyncpg.exceptions.ConnectionDoesNotExistError


class _LosesItsConnectionOnce:
    """A connection whose first statement is cut off, as a dropped connection cuts it."""

    def __init__(self) -> None:
        self.statements = 0

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        self.statements += 1
        if self.statements == 1:
            raise _LOST("the commit's outcome is unknown")
        return {"contribute": True, "contribute_since": _BEGAN}


@pytest.fixture
def connection(monkeypatch: pytest.MonkeyPatch) -> _LosesItsConnectionOnce:
    connection = _LosesItsConnectionOnce()

    async def run(operation: Callable[[Any], Awaitable[Any]]) -> Any:
        """What the pool does for a read: a lost connection is answered by trying again."""
        try:
            return await operation(connection)
        except _LOST:
            return await operation(connection)

    async def run_once(operation: Callable[[Any], Awaitable[Any]]) -> Any:
        """What the pool does for a write whose outcome matters: one attempt."""
        return await operation(connection)

    monkeypatch.setattr(pg_pool, "run", run)
    monkeypatch.setattr(pg_pool, "run_once", run_once)
    return connection


async def test_a_write_that_loses_its_connection_is_not_made_again(
    connection: _LosesItsConnectionOnce,
) -> None:
    with pytest.raises(_LOST):
        await PGRunStore().write_privacy_settings(owner_id="alice", contribute=True)

    assert connection.statements == 1


async def test_a_read_that_loses_its_connection_is_tried_again(
    connection: _LosesItsConnectionOnce,
) -> None:
    settings = await PGRunStore().read_privacy_settings(owner_id="alice")

    assert settings == PrivacySettings(contribute=True, since=_BEGAN)
    assert connection.statements == 2
