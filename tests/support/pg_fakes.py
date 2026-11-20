# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the asyncpg pool a PostgreSQL store takes in its constructor."""

from __future__ import annotations

from typing import Any


class FakeAcquire:
    """``async with pool.acquire() as conn`` yielding the one connection a test supplies."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


class FakePool:
    """A pool whose every ``acquire()`` hands out the same connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._conn)


class RecordingConnection:
    """Records every read, so a refusal can show it reached no storage."""

    def __init__(self) -> None:
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        return []
