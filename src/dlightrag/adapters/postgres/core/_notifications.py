# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One PostgreSQL LISTEN connection per process, fanned out to subscribers.

A NOTIFY is only a wake hint: authoritative state is always re-read from tables.
The hub holds one connection of its own, outside any pool, LISTENs every declared
channel on it once, and never UNLISTENs; subscribing and unsubscribing only change
who is called.

A subscriber's callback receives each NOTIFY payload on its channel, and ``None``,
meaning "your channel is live; re-read your authoritative state". ``None`` reaches
every subscriber after each (re)connect's LISTENs succeed and after each keepalive
passes, which resynchronizes everyone periodically, and reaches a subscriber that
joins a live hub once on its own, unless it leaves first. A notification can only
be missed while no connection listens, so the ``None`` that follows closes the gap.
Callbacks run on the event loop and must not block; one that raises is logged and
does not affect the others.

A lost connection, or one whose LISTENs or keepalive fail or hang, is terminated
and replaced after a delay that doubles from one second up to thirty, and starts
over once a replacement passes a keepalive.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, suppress
from typing import Any

import asyncpg

from dlightrag.adapters.postgres.core._channels import CHANNELS

logger = logging.getLogger(__name__)

type NotificationCallback = Callable[[str | None], None]

_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0
# A LISTEN connection is otherwise idle, so a half-open socket would go unnoticed;
# the keepalive turns that into a reconnect, and each one that passes resynchronizes
# every subscriber.
_KEEPALIVE_SECONDS = 30.0
# Bounds a new connection's LISTENs as a whole, each keepalive, and a dedicated
# connection's graceful close, so a half-open socket cannot hold any of them up.
_STATEMENT_TIMEOUT_SECONDS = 5.0


@asynccontextmanager
async def dedicated_connection(connect_kwargs: Mapping[str, Any]) -> AsyncIterator[Any]:
    """Open one connection outside any pool, closing it on exit."""
    connection = await asyncpg.connect(**connect_kwargs)
    try:
        yield connection
    finally:
        # A graceful close waits for the server to hang up, which a half-open socket
        # never does; past the bound the connection is terminated instead.
        try:
            await asyncio.wait_for(connection.close(), timeout=_STATEMENT_TIMEOUT_SECONDS)
        except TimeoutError:
            connection.terminate()


class PGNotificationHub:
    """Deliver every declared channel to its subscribers over one LISTEN connection."""

    def __init__(self, *, connect: Callable[[], AbstractAsyncContextManager[Any]]) -> None:
        self._connect = connect
        self._subscribers: dict[str, list[NotificationCallback]] = {
            channel: [] for channel in CHANNELS
        }
        self._live = False
        self._closed = False
        self._task: asyncio.Task[None] | None = None
        self._reconnect_delay = _RECONNECT_BASE_SECONDS

    @contextmanager
    def listen(self, channel: str, callback: NotificationCallback) -> Iterator[None]:
        """Deliver ``channel`` to ``callback`` for the duration of the block."""
        self.subscribe(channel, callback)
        try:
            yield
        finally:
            self.unsubscribe(channel, callback)

    def subscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Start delivering ``channel`` to ``callback``; the first subscriber starts the hub."""
        callbacks = self._subscribers.get(channel)
        if callbacks is None:
            raise ValueError(f"undeclared notification channel: {channel!r}")
        if self._closed:
            raise RuntimeError("notification hub is closed")
        callbacks.append(callback)
        if self._live:
            asyncio.get_running_loop().call_soon(self._welcome, channel, callback)
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="dlightrag-pg-notifications")

    def unsubscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Stop delivering ``channel`` to ``callback``; nothing reaches it afterwards."""
        callbacks = self._subscribers.get(channel, [])
        if callback in callbacks:
            callbacks.remove(callback)

    async def aclose(self) -> None:
        """Stop listening and close the connection; the hub cannot be reused."""
        self._closed = True
        for callbacks in self._subscribers.values():
            callbacks.clear()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        while True:
            try:
                async with self._connect() as connection:
                    await self._serve(connection)
                logger.warning(
                    "PostgreSQL notification connection was lost; reconnecting in %.1fs",
                    self._reconnect_delay,
                )
            except Exception:
                logger.warning(
                    "PostgreSQL notification connection failed; reconnecting in %.1fs",
                    self._reconnect_delay,
                    exc_info=True,
                )
            # Every replacement waits, and the wait doubles until a connection passes a
            # keepalive, so one that dies right after its LISTENs cannot spin the hub
            # through reconnects and resynchronizations.
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX_SECONDS)

    async def _serve(self, connection: Any) -> None:
        """Serve every subscriber on one connection until it is lost; raise if it fails."""
        lost = asyncio.Event()
        connection.add_termination_listener(lambda _connection: lost.set())
        try:
            await asyncio.wait_for(self._listen_all(connection), _STATEMENT_TIMEOUT_SECONDS)
            self._live = True
            self._resynchronize()
            while not await _set_within(lost, _KEEPALIVE_SECONDS):
                await asyncio.wait_for(connection.fetchval("SELECT 1"), _STATEMENT_TIMEOUT_SECONDS)
                self._reconnect_delay = _RECONNECT_BASE_SECONDS
                self._resynchronize()
        except Exception:
            # A failed or hung statement leaves the connection's state unknown.
            connection.terminate()
            raise
        finally:
            self._live = False

    async def _listen_all(self, connection: Any) -> None:
        for channel in self._subscribers:
            await connection.add_listener(channel, self._dispatch)

    def _dispatch(self, _connection: object, _pid: object, channel: str, payload: str) -> None:
        for callback in tuple(self._subscribers[channel]):
            _deliver(callback, payload)

    def _resynchronize(self) -> None:
        for callbacks in self._subscribers.values():
            for callback in tuple(callbacks):
                _deliver(callback, None)

    def _welcome(self, channel: str, callback: NotificationCallback) -> None:
        if self._live and callback in self._subscribers[channel]:
            _deliver(callback, None)


async def _set_within(event: asyncio.Event, timeout: float) -> bool:
    """Wait up to ``timeout`` for ``event``; return whether it is set."""
    with suppress(TimeoutError):
        await asyncio.wait_for(event.wait(), timeout)
    return event.is_set()


def _deliver(callback: NotificationCallback, payload: str | None) -> None:
    try:
        callback(payload)
    except Exception:
        logger.warning("Notification subscriber failed", exc_info=True)


__all__ = ["NotificationCallback", "PGNotificationHub", "dedicated_connection"]
