# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The shared PostgreSQL LISTEN hub: fan-out, resynchronization, reconnect, release."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from itertools import pairwise
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.core import _notifications
from dlightrag.adapters.postgres.core._notifications import PGNotificationHub


class _Connection:
    """The slice of an asyncpg connection the hub drives."""

    def __init__(self) -> None:
        self.listeners: dict[str, Callable[..., None]] = {}
        self.statements: list[str] = []
        self.keepalive_error: Exception | None = None
        self.hang: set[str] = set()  # statements that never complete
        self.gates: dict[str, asyncio.Event] = {}  # statements that complete once opened
        self.entered: set[str] = set()  # hanging or gated statements that have started
        self.dies_after_listen = False  # the server drops it right after a LISTEN
        self.delay = 0.0  # how long every statement takes
        self._termination: list[Callable[[Any], None]] = []
        self._closed = False
        self._detached = False

    def _check(self) -> None:
        if self._detached:
            raise asyncpg.InterfaceError("connection has been released back to the pool")

    async def _statement(self, statement: str) -> None:
        self._check()
        self.statements.append(statement)
        if self.delay:
            await asyncio.sleep(self.delay)
        if statement in self.hang or statement in self.gates:
            self.entered.add(statement)
            await self.gates.get(statement, asyncio.Event()).wait()

    async def add_listener(self, channel: str, callback: Callable[..., None]) -> None:
        await self._statement(f"LISTEN {channel}")
        self.listeners[channel] = callback
        if self.dies_after_listen:
            asyncio.get_running_loop().call_soon(self._end)

    async def remove_listener(self, channel: str, _callback: Callable[..., None]) -> None:
        await self._statement(f"UNLISTEN {channel}")
        self.listeners.pop(channel, None)

    def add_termination_listener(self, callback: Callable[[Any], None]) -> None:
        self._check()
        self._termination.append(callback)

    def remove_termination_listener(self, callback: Callable[[Any], None]) -> None:
        self._check()
        if callback in self._termination:
            self._termination.remove(callback)

    async def fetchval(self, query: str) -> int:
        await self._statement(query)
        if self.keepalive_error is not None:
            raise self.keepalive_error
        return 1

    def is_closed(self) -> bool:
        self._check()
        return self._closed

    async def close(self) -> None:
        await self._statement("close")
        self._end()

    def terminate(self) -> None:
        self._check()
        self._end()

    def detach(self) -> None:
        """What asyncpg does to a pool connection the server closed: clean up, detach."""
        self._end()
        self._detached = True

    def _end(self) -> None:
        if self._closed:
            return
        self._closed = True
        for callback in tuple(self._termination):
            asyncio.get_running_loop().call_soon(callback, self)
        self._termination.clear()

    def notify(self, channel: str, payload: str) -> None:
        self.listeners[channel](self, 4242, channel, payload)


class _Endpoint:
    """Hands out fake connections and records which ones were given back."""

    def __init__(
        self, *, failures: int = 0, doomed: int = 0, delays: tuple[float, ...] = ()
    ) -> None:
        self.opened: list[_Connection] = []
        self.opened_at: list[float] = []
        self.released: list[_Connection] = []
        self._failures = failures
        self._doomed = doomed  # how many connections, first to last, die after a LISTEN
        self._delays = delays  # per connection, first to last: how long statements take

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_Connection]:
        if self._failures:
            self._failures -= 1
            raise ConnectionRefusedError("database starting up")
        connection = _Connection()
        index = len(self.opened)
        connection.dies_after_listen = index < self._doomed
        connection.delay = self._delays[index] if index < len(self._delays) else 0.0
        self.opened.append(connection)
        self.opened_at.append(asyncio.get_running_loop().time())
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


async def test_reconnects_back_off_until_a_connection_passes_a_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection the server drops right after its LISTEN must not spin the hub.

    Each replacement waits twice as long as the one before, until a connection
    passes a keepalive; losing that one costs only the base delay again.
    """
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.05)
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.02)
    endpoint = _Endpoint(doomed=3)
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    await hub.subscribe("runs", received.append)
    await _until(lambda: len(endpoint.opened) == 4 and "SELECT 1" in endpoint.opened[3].statements)
    lost_at = asyncio.get_running_loop().time()
    endpoint.opened[3].terminate()
    await _until(lambda: len(endpoint.opened) == 5 and len(received) == 5)

    waits = [later - earlier for earlier, later in pairwise(endpoint.opened_at[:4])]
    slack = 0.001  # the event loop may run a timer up to its clock resolution early
    expected = (0.05, 0.1, 0.2)
    assert all(w >= e - slack for w, e in zip(waits, expected, strict=True)), waits
    # The passed keepalive reset the delay, which would otherwise have reached 0.4 s.
    assert 0.05 - slack <= endpoint.opened_at[4] - lost_at < 0.3
    assert received == [None] * 5
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


async def test_a_subscribe_cancelled_during_its_listen_leaves_nothing_registered() -> None:
    """Only a channel's first subscriber LISTENs, so a leftover would mute every later one."""
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    kept: list[str | None] = []
    await hub.subscribe("keep", kept.append)
    await _until(lambda: kept == [None])
    first = endpoint.opened[0]
    first.hang.add("LISTEN runs")
    abandoned: list[str | None] = []

    async def wait_for_runs() -> None:
        async with hub.listen("runs", abandoned.append):
            await asyncio.Event().wait()

    waiter = asyncio.create_task(wait_for_runs())
    await _until(lambda: "LISTEN runs" in first.entered)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    assert "runs" not in hub._subscribers  # noqa: SLF001
    # The abandoned statement cost its connection; the replacement serves the rest.
    await _until(lambda: len(endpoint.opened) == 2 and kept == [None, None])
    replacement = endpoint.opened[1]
    assert replacement.statements == ["LISTEN keep"]
    later: list[str | None] = []
    await hub.subscribe("runs", later.append)
    replacement.notify("runs", "reply")
    assert later == ["reply"]
    assert abandoned == []
    await hub.aclose()


async def test_a_listener_cancelled_again_while_its_exit_waits_leaves_nothing_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enclosing timeout can cancel listen()'s exit while a hung keepalive holds the lock.

    A registration left behind would keep the hub, and its pool connection, forever
    and call a dead callback on every NOTIFY.
    """
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.5)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    kept: list[str | None] = []
    await hub.subscribe("keep", kept.append)
    await _until(lambda: kept == [None])
    first = endpoint.opened[0]
    abandoned: list[str | None] = []
    inside = asyncio.Event()

    async def wait_for_runs() -> None:
        async with hub.listen("runs", abandoned.append):
            inside.set()
            await asyncio.Event().wait()

    waiter = asyncio.create_task(wait_for_runs())
    await inside.wait()
    first.hang.add("SELECT 1")
    await _until(lambda: "SELECT 1" in first.entered)
    waiter.cancel()
    await asyncio.sleep(0.01)  # listen()'s exit now waits for the lock
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    assert hub._subscribers == {"keep": [kept.append]}  # noqa: SLF001
    # The hung keepalive costs its connection; the replacement LISTENs what is left.
    await _until(lambda: len(endpoint.opened) == 2 and kept == [None, None])
    assert endpoint.opened[1].statements == ["LISTEN keep"]
    await hub.unsubscribe("keep", kept.append)
    await _until(lambda: endpoint.released == endpoint.opened)
    assert abandoned == []  # not even the replacement's resynchronization
    await hub.aclose()


async def test_a_channel_subscribed_again_before_its_unlisten_runs_stays_listened() -> None:
    """The newcomer's LISTEN found the channel still LISTENed, so the UNLISTEN must yield."""
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    leaving: list[str | None] = []
    await hub.subscribe("runs", leaving.append)
    await _until(lambda: leaving == [None])
    connection = endpoint.opened[0]
    gate = connection.gates.setdefault("LISTEN other", asyncio.Event())
    other: list[str | None] = []
    holder = asyncio.create_task(hub.subscribe("other", other.append))
    await _until(lambda: "LISTEN other" in connection.entered)
    arriving: list[str | None] = []
    arrival = asyncio.create_task(hub.subscribe("runs", arriving.append))
    await asyncio.sleep(0)  # the newcomer waits for the lock first
    departure = asyncio.create_task(hub.unsubscribe("runs", leaving.append))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(holder, arrival, departure)

    assert "UNLISTEN runs" not in connection.statements
    connection.notify("runs", "still heard")
    assert arriving == ["still heard"]
    assert leaving == [None]
    await hub.aclose()


async def test_a_failed_listen_keeps_the_subscriber_for_the_replacement_connection() -> None:
    """A server-closed pool connection is a detached proxy: every call raises, terminate too."""
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    kept: list[str | None] = []
    await hub.subscribe("keep", kept.append)
    await _until(lambda: kept == [None])
    endpoint.opened[0].detach()
    fresh: list[str | None] = []

    await hub.subscribe("fresh", fresh.append)

    await _until(lambda: len(endpoint.opened) == 2 and fresh == [None])
    replacement = endpoint.opened[1]
    assert sorted(replacement.statements) == ["LISTEN fresh", "LISTEN keep"]
    assert kept == [None, None]
    replacement.notify("fresh", "wake")
    assert fresh == [None, "wake"]
    await hub.aclose()


async def test_a_hung_unlisten_is_bounded_and_costs_only_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.05)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    kept: list[str | None] = []
    leaving: list[str | None] = []
    await hub.subscribe("keep", kept.append)
    await hub.subscribe("leaving", leaving.append)
    await _until(lambda: kept == [None])
    half_open = endpoint.opened[0]
    half_open.hang.add("UNLISTEN leaving")

    await asyncio.wait_for(hub.unsubscribe("leaving", leaving.append), timeout=1)

    assert half_open.is_closed()
    await _until(lambda: len(endpoint.opened) == 2 and kept == [None, None])
    assert endpoint.opened[1].statements == ["LISTEN keep"]
    await hub.aclose()


async def test_a_hung_keepalive_is_bounded_and_the_connection_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.05)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    async with hub.listen("runs", received.append):
        await _until(lambda: received == [None])
        half_open = endpoint.opened[0]
        half_open.hang.add("SELECT 1")
        await _until(lambda: received == [None, None])

    assert half_open.is_closed()
    await hub.aclose()


async def test_the_listen_batch_of_a_new_connection_is_bounded_as_a_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch holds the lock throughout, so each LISTEN within the bound is not enough."""
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.15)
    channels = ("a", "b", "c", "d", "e")
    endpoint = _Endpoint(delays=(0.0, 0.05))  # the first replacement answers slowly
    hub = PGNotificationHub(connect=endpoint.connect)
    received: dict[str, list[str | None]] = {channel: [] for channel in channels}
    for channel in channels:
        await hub.subscribe(channel, received[channel].append)
    await _until(lambda: all(sink == [None] for sink in received.values()))

    endpoint.opened[0].terminate()

    await _until(
        lambda: len(endpoint.opened) == 3 and all(s == [None, None] for s in received.values())
    )
    slow = endpoint.opened[1]
    assert slow.is_closed()
    assert len(slow.statements) < len(channels)
    assert endpoint.opened[2].statements == [f"LISTEN {channel}" for channel in channels]
    await hub.aclose()


async def test_a_dedicated_connection_that_never_finishes_closing_is_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.05)
    half_open = _Connection()
    half_open.hang.add("close")

    async def connect(**_kwargs: Any) -> _Connection:
        return half_open

    monkeypatch.setattr(asyncpg, "connect", connect)

    async with asyncio.timeout(1):
        async with _notifications.dedicated_connection({"host": "unused"}) as connection:
            assert connection is half_open

    assert half_open.entered == {"close"}
    assert half_open.is_closed()
