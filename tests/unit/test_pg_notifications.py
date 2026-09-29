# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The PostgreSQL notification hub: fixed LISTENs, fan-out, resynchronization, reconnect,
and the channel watcher that runs one coroutine after every wake."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from itertools import pairwise
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.core import _notifications
from dlightrag.adapters.postgres.core._channels import (
    CHANNELS,
    MODEL_CATALOGUE_CHANNEL,
    RUN_ACTIVITY_CHANNEL,
    RUN_CANCEL_CHANNEL,
)
from dlightrag.adapters.postgres.core._notifications import ChannelWatcher, PGNotificationHub

RUNS = RUN_ACTIVITY_CHANNEL
CATALOGUE = MODEL_CATALOGUE_CHANNEL
CANCEL = RUN_CANCEL_CHANNEL
LISTEN_ALL = [f"LISTEN {channel}" for channel in CHANNELS]


class _Connection:
    """The slice of an asyncpg connection the hub drives."""

    def __init__(self, *, delay: float = 0.0, dies_after_listen: bool = False) -> None:
        self.listeners: dict[str, Callable[..., None]] = {}
        self.statements: list[str] = []
        self.keepalive_error: Exception | None = None
        self.hang: set[str] = set()  # statements that never complete
        self._delay = delay  # how long every statement takes
        self._dies_after_listen = dies_after_listen  # the server drops it once it LISTENs
        self._termination: list[Callable[[Any], None]] = []
        self._closed = False

    async def _statement(self, statement: str) -> None:
        if self._closed:
            raise asyncpg.ConnectionDoesNotExistError("connection was closed")
        self.statements.append(statement)
        if self._delay:
            await asyncio.sleep(self._delay)
        if statement in self.hang:
            await asyncio.Event().wait()

    async def add_listener(self, channel: str, callback: Callable[..., None]) -> None:
        await self._statement(f"LISTEN {channel}")
        self.listeners[channel] = callback
        if self._dies_after_listen and len(self.listeners) == len(CHANNELS):
            asyncio.get_running_loop().call_soon(self._end)

    def add_termination_listener(self, callback: Callable[[Any], None]) -> None:
        self._termination.append(callback)

    async def fetchval(self, query: str) -> int:
        await self._statement(query)
        if self.keepalive_error is not None:
            raise self.keepalive_error
        return 1

    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        await self._statement("close")
        self._end()

    def terminate(self) -> None:
        self._end()

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
        self._doomed = doomed  # how many connections, first to last, die once they LISTEN
        self._delays = delays  # per connection, first to last: how long statements take

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_Connection]:
        if self._failures:
            self._failures -= 1
            raise ConnectionRefusedError("database starting up")
        index = len(self.opened)
        connection = _Connection(
            delay=self._delays[index] if index < len(self._delays) else 0.0,
            dies_after_listen=index < self._doomed,
        )
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


async def test_one_connection_listens_every_declared_channel_once_and_never_unlistens() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    first: list[str | None] = []
    second: list[str | None] = []

    hub.subscribe(RUNS, first.append)
    await _until(lambda: first == [None])
    hub.subscribe(RUNS, second.append)
    hub.subscribe(CATALOGUE, second.append)
    hub.unsubscribe(RUNS, first.append)
    hub.unsubscribe(RUNS, second.append)
    hub.unsubscribe(CATALOGUE, second.append)
    await asyncio.sleep(0.01)

    (connection,) = endpoint.opened
    assert connection.statements == LISTEN_ALL
    assert not connection.is_closed()
    await hub.aclose()


async def test_a_subscriber_receives_its_channel_and_none_once_the_hub_is_live() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    runs: list[str | None] = []
    catalogue: list[str | None] = []

    with hub.listen(RUNS, runs.append), hub.listen(CATALOGUE, catalogue.append):
        await _until(lambda: runs == [None] and catalogue == [None])
        connection = endpoint.opened[0]
        connection.notify(RUNS, "wake-1")
        connection.notify(CATALOGUE, "revision-2")

    assert runs == [None, "wake-1"]
    assert catalogue == [None, "revision-2"]
    await hub.aclose()


async def test_a_subscriber_joining_a_live_hub_is_resynchronized_on_its_own() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    first: list[str | None] = []
    joining: list[str | None] = []
    hub.subscribe(RUNS, first.append)
    await _until(lambda: first == [None])

    hub.subscribe(RUNS, joining.append)
    await _until(lambda: joining == [None])

    assert first == [None]
    assert endpoint.opened[0].statements == LISTEN_ALL
    await hub.aclose()


async def test_a_subscriber_that_leaves_before_its_resynchronization_gets_nothing() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    first: list[str | None] = []
    leaving: list[str | None] = []
    later: list[str | None] = []
    hub.subscribe(RUNS, first.append)
    await _until(lambda: first == [None])

    hub.subscribe(RUNS, leaving.append)
    hub.unsubscribe(RUNS, leaving.append)
    hub.subscribe(RUNS, later.append)
    await _until(lambda: later == [None])  # resynchronizations are delivered in join order

    assert leaving == []
    await hub.aclose()


async def test_nothing_reaches_a_subscriber_once_it_unsubscribed() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    staying: list[str | None] = []
    leaving: list[str | None] = []
    hub.subscribe(RUNS, staying.append)
    hub.subscribe(RUNS, leaving.append)
    await _until(lambda: staying == [None] and leaving == [None])

    hub.unsubscribe(RUNS, leaving.append)
    endpoint.opened[0].notify(RUNS, "after leaving")
    endpoint.opened[0].terminate()
    await _until(lambda: staying == [None, "after leaving", None])

    assert leaving == [None]
    await hub.aclose()


async def test_an_undeclared_channel_is_refused() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)

    with pytest.raises(ValueError, match="undeclared"):
        hub.subscribe("dlightrag_typo", lambda _payload: None)
    with pytest.raises(ValueError, match="undeclared"):
        with hub.listen("dlightrag_typo", lambda _payload: None):
            pass
    await asyncio.sleep(0.01)

    assert endpoint.opened == []
    await hub.aclose()


async def test_each_passing_keepalive_resynchronizes_every_subscriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    runs: list[str | None] = []
    catalogue: list[str | None] = []
    hub.subscribe(RUNS, runs.append)
    hub.subscribe(CATALOGUE, catalogue.append)

    await _until(lambda: len(runs) >= 3 and len(catalogue) >= 3)

    (connection,) = endpoint.opened
    assert set(runs) == set(catalogue) == {None}
    assert connection.statements[: len(CHANNELS)] == LISTEN_ALL
    assert set(connection.statements[len(CHANNELS) :]) == {"SELECT 1"}
    await hub.aclose()


async def test_a_lost_connection_is_replaced_listened_again_and_resynchronized() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    with hub.listen(RUNS, received.append):
        await _until(lambda: received == [None])
        endpoint.opened[0].terminate()
        await _until(lambda: received == [None, None])
        replacement = endpoint.opened[1]
        assert replacement.statements == LISTEN_ALL
        replacement.notify(RUNS, "after reconnect")

    assert received == [None, None, "after reconnect"]
    assert endpoint.released[0] is endpoint.opened[0]
    await hub.aclose()


async def test_a_failed_keepalive_terminates_the_connection_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []
    hub.subscribe(RUNS, received.append)
    await _until(lambda: received == [None])
    broken = endpoint.opened[0]
    broken.keepalive_error = OSError("connection reset")

    await _until(lambda: len(endpoint.opened) == 2 and endpoint.opened[1].statements == LISTEN_ALL)

    assert broken.is_closed()
    assert broken.statements[-1] == "SELECT 1"
    await hub.aclose()


async def test_a_hung_keepalive_is_bounded_and_the_connection_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.05)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []
    hub.subscribe(RUNS, received.append)
    await _until(lambda: received == [None])
    half_open = endpoint.opened[0]
    half_open.hang.add("SELECT 1")

    await _until(lambda: len(endpoint.opened) == 2 and endpoint.opened[1].statements == LISTEN_ALL)

    assert half_open.is_closed()
    await hub.aclose()


async def test_the_listens_of_a_new_connection_are_bounded_as_a_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every LISTEN within the bound is not enough when the batch as a whole outlasts it."""
    monkeypatch.setattr(_notifications, "_STATEMENT_TIMEOUT_SECONDS", 0.15)
    endpoint = _Endpoint(delays=(0.0, 0.05))  # the first replacement answers slowly
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []
    hub.subscribe(RUNS, received.append)
    await _until(lambda: received == [None])

    endpoint.opened[0].terminate()

    await _until(lambda: len(endpoint.opened) == 3 and received == [None, None])
    slow = endpoint.opened[1]
    assert slow.is_closed()
    assert len(slow.statements) < len(CHANNELS)
    assert endpoint.opened[2].statements == LISTEN_ALL
    await hub.aclose()


async def test_a_refused_connection_is_retried_until_the_database_answers() -> None:
    endpoint = _Endpoint(failures=2)
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    with hub.listen(RUNS, received.append):
        await _until(lambda: received == [None])

    assert len(endpoint.opened) == 1
    await hub.aclose()


async def test_reconnects_back_off_until_a_connection_passes_a_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection the server drops right after its LISTENs must not spin the hub.

    Each replacement waits twice as long as the one before, until a connection
    passes a keepalive; losing that one costs only the base delay again.
    """
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.05)
    monkeypatch.setattr(_notifications, "_KEEPALIVE_SECONDS", 0.02)
    endpoint = _Endpoint(doomed=3)
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    hub.subscribe(RUNS, received.append)
    await _until(lambda: len(endpoint.opened) == 4 and "SELECT 1" in endpoint.opened[3].statements)
    lost_at = asyncio.get_running_loop().time()
    endpoint.opened[3].terminate()
    await _until(lambda: len(endpoint.opened) == 5 and endpoint.opened[4].statements == LISTEN_ALL)

    waits = [later - earlier for earlier, later in pairwise(endpoint.opened_at[:4])]
    slack = 0.001  # the event loop may run a timer up to its clock resolution early
    expected = (0.05, 0.1, 0.2)
    assert all(w >= e - slack for w, e in zip(waits, expected, strict=True)), waits
    # The passed keepalive reset the delay, which would otherwise have reached 0.4 s.
    assert 0.05 - slack <= endpoint.opened_at[4] - lost_at < 0.3
    assert set(received) == {None}
    await hub.aclose()


async def test_a_failing_subscriber_does_not_starve_the_others() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []

    def broken(_payload: str | None) -> None:
        raise RuntimeError("subscriber bug")

    hub.subscribe(RUNS, broken)
    hub.subscribe(RUNS, received.append)
    await _until(lambda: received == [None])
    endpoint.opened[0].notify(RUNS, "still delivered")

    assert received == [None, "still delivered"]
    await hub.aclose()


async def test_closing_the_hub_releases_its_connection_and_refuses_new_subscribers() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    received: list[str | None] = []
    hub.subscribe(RUNS, received.append)
    await _until(lambda: received == [None])
    connection = endpoint.opened[0]

    await hub.aclose()

    assert endpoint.released == [connection]
    hub.unsubscribe(RUNS, received.append)  # a late leaver is harmless
    with pytest.raises(RuntimeError, match="closed"):
        hub.subscribe(RUNS, received.append)


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

    assert half_open.statements == ["close"]
    assert half_open.is_closed()


async def test_a_watcher_runs_once_the_channel_is_live_and_again_after_each_wake() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    runs = 0

    async def on_wake() -> None:
        nonlocal runs
        runs += 1

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await asyncio.wait_for(watcher.ready.wait(), timeout=2)
    assert runs == 1

    endpoint.opened[0].notify(CANCEL, "wake")
    await _until(lambda: runs == 2)
    endpoint.opened[0].notify(RUNS, "another channel")
    await asyncio.sleep(0.01)

    assert runs == 2
    await watcher.aclose()
    await hub.aclose()


async def test_wakes_during_a_run_coalesce_into_one_more_run() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    release = asyncio.Event()
    runs = 0

    async def on_wake() -> None:
        nonlocal runs
        runs += 1
        if runs == 2:
            await release.wait()

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await asyncio.wait_for(watcher.ready.wait(), timeout=2)
    connection = endpoint.opened[0]
    connection.notify(CANCEL, "first")
    await _until(lambda: runs == 2)
    for payload in ("second", "third", "fourth"):
        connection.notify(CANCEL, payload)
    release.set()
    await _until(lambda: runs == 3)
    await asyncio.sleep(0.01)

    assert runs == 3
    await watcher.aclose()
    await hub.aclose()


async def test_a_watcher_is_ready_only_after_a_run_that_began_once_the_channel_was_live() -> None:
    endpoint = _Endpoint(delays=(0.02,))  # the LISTENs take a while
    hub = PGNotificationHub(connect=endpoint.connect)
    runs = 0

    async def on_wake() -> None:
        nonlocal runs
        runs += 1

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await _until(lambda: bool(endpoint.opened) and CANCEL in endpoint.opened[0].listeners)
    endpoint.opened[0].notify(CANCEL, "before the hub is live")
    await _until(lambda: runs == 1)

    assert not watcher.ready.is_set()
    await asyncio.wait_for(watcher.ready.wait(), timeout=2)
    assert runs == 2
    await watcher.aclose()
    await hub.aclose()


async def test_until_ready_a_failed_run_is_retried_with_a_doubling_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.05)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    started: list[float] = []

    async def on_wake() -> None:
        started.append(asyncio.get_running_loop().time())
        if len(started) < 3:
            raise RuntimeError("authoritative read unavailable")

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await asyncio.wait_for(watcher.ready.wait(), timeout=2)

    waits = [later - earlier for earlier, later in pairwise(started)]
    slack = 0.001  # the event loop may run a timer up to its clock resolution early
    assert all(w >= e - slack for w, e in zip(waits, (0.05, 0.1), strict=True)), waits
    await watcher.aclose()
    await hub.aclose()


async def test_once_ready_a_failed_run_waits_for_the_next_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.01)
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    runs = 0

    async def on_wake() -> None:
        nonlocal runs
        runs += 1
        if runs == 2:
            raise RuntimeError("authoritative read unavailable")

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await asyncio.wait_for(watcher.ready.wait(), timeout=2)
    endpoint.opened[0].notify(CANCEL, "fails")
    await _until(lambda: runs == 2)
    await asyncio.sleep(0.1)  # ten retry delays: none is taken

    assert runs == 2
    endpoint.opened[0].notify(CANCEL, "next wake")
    await _until(lambda: runs == 3)
    await watcher.aclose()
    await hub.aclose()


async def test_closing_a_watcher_unsubscribes_it_and_abandons_its_run() -> None:
    endpoint = _Endpoint()
    hub = PGNotificationHub(connect=endpoint.connect)
    entered = asyncio.Event()
    abandoned = asyncio.Event()
    runs = 0

    async def on_wake() -> None:
        nonlocal runs
        runs += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            abandoned.set()

    watcher = ChannelWatcher(lambda: hub, CANCEL, on_wake, name="Test watcher")
    await watcher.start()
    await asyncio.wait_for(entered.wait(), timeout=2)

    await watcher.aclose()
    endpoint.opened[0].notify(CANCEL, "after closing")
    await asyncio.sleep(0.01)

    assert abandoned.is_set()
    assert runs == 1
    await hub.aclose()
