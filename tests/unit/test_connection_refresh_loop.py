# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Connections refresh loop with nothing to claim sleeps until the next refresh falls due.

The store's wakes (a NOTIFY, a resynchronization) end any sleep early, so polling the claim on
a fixed cadence would only add load. The loop is observed through ``start`` and
``stop_refresh``, with a store whose schedule the test scripts one idle scan at a time.
"""

import asyncio
from typing import Any, cast

import pytest

from dlightrag.application.connections import ConnectionPolicy, Connections
from dlightrag.application.connections.credentials import CredentialCipher


class _Store:
    """Scripted scans: an exception fails that scan's claim; anything else is its due time.

    The loop is parked once it has slept as often as there are scans, so a loop that sleeps
    on some other rule fails the assertion instead of spinning.
    """

    def __init__(self, *scans: float | Exception | None) -> None:
        self.scans = list(scans)
        self.budget = len(scans)
        self.sleeps: list[float] = []
        self.parked = asyncio.Event()

    async def initialize(self, *, validate_only: bool) -> None:
        pass

    async def start_notifications(self) -> None:
        pass

    async def stop_notifications(self) -> None:
        pass

    async def claim(self, **_: Any) -> None:
        scan = self.scans[0]
        if isinstance(scan, Exception):
            self.scans.pop(0)
            raise scan
        return None

    async def seconds_until_refresh(self) -> float | None:
        due = self.scans.pop(0)
        assert not isinstance(due, Exception)
        return due

    async def wait_refresh(self, timeout: float) -> None:
        self.sleeps.append(timeout)
        if len(self.sleeps) == self.budget:
            self.parked.set()
            await asyncio.Event().wait()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_an_idle_loop_sleeps_until_the_next_refresh_is_due_and_at_most_thirty_seconds():
    store = _Store(12.5, None, 3600.0, 0.0, RuntimeError("store unavailable"))
    connections = Connections(
        store=cast(Any, store),
        mcp=cast(Any, None),
        policy=ConnectionPolicy(discovery_concurrency=1),
        cipher=CredentialCipher(None),
    )
    await connections.start(validate_only=True)
    try:
        async with asyncio.timeout(5):
            await store.parked.wait()
    finally:
        await connections.stop_refresh()
    # Due in 12.5 s; nothing scheduled; due in an hour; due already; a failing store.
    assert store.sleeps == [12.5, 30.0, 30.0, 0.0, 1.0]
