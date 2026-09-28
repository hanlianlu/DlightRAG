# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The shared PostgreSQL LISTEN hub: fan-out, resynchronization, reconnect, release."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from dlightrag.adapters.postgres.core import _notifications
from dlightrag.adapters.postgres.core._notifications import PGNotificationHub


class _Connection:
    """The slice of an asyncpg connection the hub drives."""

    def __init__(self) -> None:
        self.listeners: dict[str, Callable[..., None]] = {}
        self.statements: list[str] = []
        self.keepalive_error: Exception | None = None
        self._termination: list[Callable[[Any], None]] = []
        self._closed = False

    async def add_listener(self, channel: str, callback: Callable[..., None]) -> None:
        self.statements.append(f"LISTEN {channel}")
        self.listeners[channel] = callback

    async def remove_listener(self, channel: str, _callback: Callable[..., None]) -> None:
        self.statements.append(f"UNLISTEN {channel}")
        self.listeners.pop(channel, None)

    def add_termination_listener(self, callback: Callable[[Any], None]) -> None:
        self._termination.append(callback)

    def remove_termination_listener(self, callback: Callable[[Any], None]) -> None:
        if callback in self._termination:
            self._termination.remove(callback)

    async def fetchval(self, query: str, *, timeout: float | None = None) -> int:
        self.statements.append(query)
        if self.keepalive_error is not None:
            raise self.keepalive_error
        return 1

    def is_closed(self) -> bool:
        return self._closed

    def terminate(self) -> None:
        if self._closed:
            return
        self._closed = True
        for callback in tuple(self._termination):
            asyncio.get_running_loop().call_soon(callback, self)

    def notify(self, channel: str, payload: str) -> None:
        self.listeners[channel](self, 4242, channel, payload)


class _Endpoint:
    """Hands out fake connections and records which ones were given back."""

    def __init__(self, *, failures: int = 0) -> None:
        self.opened: list[_Connection] = []
        self.released: list[_Connection] = []
        self._failures = failures

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_Connection]:
        if self._failures:
            self._failures -= 1
            raise ConnectionRefusedError("database starting up")
        connection = _Connection()
        self.opened.append(connection)
        try:
            yield connection
        finally:
            self.released.append(connection)


async def _until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


@pytest.fixture(autouse=True)
def _fast_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.0)


async def test_a_subscriber_receives_only_its_channel_and_a_resync_on_connect() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    async with hub.listen("runs", received.append):
        await _until(lambda: received == [None])
        connection = endpoint.opened[0]
        connection.notify("runs", "wake-1")
        assert "other" not in connection.listeners

    assert received == [None, "wake-1"]
    await hub.aclose()


async def test_subscribers_share_one_connection_and_the_last_one_releases_it() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    first: list[str | None] = []
    second: list[str | None] = []

    await hub.subscribe("runs", first.append)
    await _until(lambda: first == [None])
    await hub.subscribe("runs", second.append)
    await hub.subscribe("catalogue", second.append)
    connection = endpoint.opened[0]
    connection.notify("runs", "shared")

    assert len(endpoint.opened) == 1
    assert connection.statements == ["LISTEN runs", "LISTEN catalogue"]
    assert first == [None, "shared"]
    assert second == ["shared"]

    await hub.unsubscribe("runs", first.append)
    await hub.unsubscribe("runs", second.append)
    assert endpoint.released == []
    assert connection.statements[-1] == "UNLISTEN runs"
    await hub.unsubscribe("catalogue", second.append)
    await _until(lambda: endpoint.released == [connection])
    assert connection.listeners == {}
    await hub.aclose()


async def test_a_lost_connection_is_replaced_and_every_subscriber_resynchronizes() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    async with hub.listen("runs", received.append):
        await _until(lambda: received == [None])
        endpoint.opened[0].terminate()
        await _until(lambda: received == [None, None])
        replacement = endpoint.opened[1]
        assert replacement.statements == ["LISTEN runs"]
        replacement.notify("runs", "after reconnect")

    assert received == [None, None, "after reconnect"]
    assert endpoint.released[0] is endpoint.opened[0]
    await hub.aclose()


async def test_a_failed_keepalive_drops_the_connection_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    async with hub.listen("runs", received.append):
        await _until(lambda: received == [None])
        half_open = endpoint.opened[0]
        half_open.keepalive_error = TimeoutError("no answer")
        await _until(lambda: received == [None, None])

    assert half_open.is_closed()
    assert "SELECT 1" in half_open.statements
    await hub.aclose()


async def test_a_refused_connection_is_retried_until_the_database_answers() -> None:
    endpoint = _Endpoint(failures=2)
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    async with hub.listen("runs", received.append):
        await _until(lambda: received == [None])

    assert len(endpoint.opened) == 1
    await hub.aclose()


async def test_a_failing_subscriber_does_not_starve_the_others() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    def broken(_payload: str | None) -> None:
        raise RuntimeError("subscriber bug")

    await hub.subscribe("runs", broken)
    await hub.subscribe("runs", received.append)
    await _until(lambda: received == [None])
    endpoint.opened[0].notify("runs", "still delivered")

    assert received == [None, "still delivered"]
    await hub.aclose()


async def test_closing_the_hub_releases_its_connection_and_refuses_new_subscribers() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []
    await hub.subscribe("runs", received.append)
    await _until(lambda: received == [None])
    connection = endpoint.opened[0]

    await hub.aclose()

    assert endpoint.released == [connection]
    assert connection.listeners == {}
    with pytest.raises(RuntimeError, match="closed"):
        await hub.subscribe("runs", received.append)
