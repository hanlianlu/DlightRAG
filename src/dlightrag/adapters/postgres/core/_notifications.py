# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One shared PostgreSQL LISTEN connection per pool, fanned out to subscribers.

A subscriber names a channel and a callback. The callback receives every NOTIFY
payload on that channel, and ``None`` each time the hub (re)connects: a
notification can only be missed while no connection was listening, so ``None``
asks the subscriber to re-read its authoritative state. Notifications are wake
hints, never authority. Callbacks run on the event loop and must not block.

The hub holds its connection only while something subscribes, replaces a lost
connection (termination, or a failed keepalive) with bounded exponential backoff,
and releases the connection when the last subscriber leaves.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

type NotificationCallback = Callable[[str | None], None]

_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0
# A LISTEN connection is otherwise idle, so a half-open socket would go unnoticed;
# the keepalive turns that into a reconnect and a resynchronization.
_KEEPALIVE_SECONDS = 30.0
_KEEPALIVE_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def dedicated_connection(connect_kwargs: Mapping[str, Any]) -> AsyncIterator[Any]:
    """Open one connection outside any pool, closing it on exit."""
    connection = await asyncpg.connect(**connect_kwargs)
    try:
        yield connection
    finally:
        await connection.close()


class PGNotificationHub:
    """Share one LISTEN connection between every subscriber on one endpoint."""

    def __init__(self, *, connect: Callable[[], AbstractAsyncContextManager[Any]]) -> None:
        self._connect = connect
        self._subscribers: dict[str, list[NotificationCallback]] = {}
        self._connection: Any = None
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closing = False

    @asynccontextmanager
    async def listen(self, channel: str, callback: NotificationCallback) -> AsyncIterator[None]:
        """Deliver ``channel`` to ``callback`` for the duration of the block."""
        await self.subscribe(channel, callback)
        try:
            yield
        finally:
            await self.unsubscribe(channel, callback)

    async def subscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Start delivering ``channel`` to ``callback``.

        When the hub is connected, the channel is LISTENed before this returns, so a
        subscriber that reads its state afterwards misses nothing. Otherwise the
        connection it is waiting for delivers ``None`` once it listens.
        """
        if self._closing:
            raise RuntimeError("notification hub is closed")
        async with self._lock:
            callbacks = self._subscribers.setdefault(channel, [])
            callbacks.append(callback)
            if len(callbacks) == 1 and self._connection is not None:
                try:
                    await self._connection.add_listener(channel, self._dispatch)
                except Exception:
                    logger.warning(
                        "LISTEN failed on the notification connection; reconnecting",
                        exc_info=True,
                    )
                    # The replacement connection listens on every channel and
                    # resynchronizes every subscriber, this one included.
                    self._connection.terminate()
                    self._connection = None
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="dlightrag-pg-notifications")

    async def unsubscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Stop delivering ``channel`` to ``callback``; the last one out releases the hub."""
        async with self._lock:
            callbacks = self._subscribers.get(channel)
            if not callbacks or callback not in callbacks:
                return
            callbacks.remove(callback)
            if callbacks:
                return
            del self._subscribers[channel]
            if self._connection is not None:
                with suppress(Exception):
                    await self._connection.remove_listener(channel, self._dispatch)
        if not self._subscribers:
            self._changed.set()

    async def aclose(self) -> None:
        """Stop listening and release the connection; the hub cannot be reused."""
        self._closing = True
        async with self._lock:
            channels = tuple(self._subscribers)
            self._subscribers.clear()
            if self._connection is not None:
                for channel in channels:
                    with suppress(Exception):
                        await self._connection.remove_listener(channel, self._dispatch)
        self._changed.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        backoff = _RECONNECT_BASE_SECONDS
        while self._subscribers and not self._closing:
            try:
                await self._serve()
                backoff = _RECONNECT_BASE_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._subscribers or self._closing:
                    return
                logger.warning(
                    "PostgreSQL notification connection failed; reconnecting in %.1fs",
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def _serve(self) -> None:
        async with self._connect() as connection:
            lost = asyncio.Event()

            def _terminated(_connection: object) -> None:
                lost.set()
                self._changed.set()

            connection.add_termination_listener(_terminated)
            failed = False
            try:
                await self._listen_on(connection, lost)
            except Exception:
                failed = True
                raise
            finally:
                async with self._lock:
                    if self._connection is connection:
                        self._connection = None
                # A pool connection that asyncpg already cleaned up is detached from
                # its proxy, and every call on it raises InterfaceError.
                with suppress(asyncpg.InterfaceError):
                    connection.remove_termination_listener(_terminated)
                    if failed or connection.is_closed():
                        # One that failed an operation may be half-open, and asyncpg can
                        # report a server-closed one closed before cleaning it up, which
                        # is what frees its pool slot. terminate() does both, now.
                        connection.terminate()

    async def _listen_on(self, connection: Any, lost: asyncio.Event) -> None:
        async with self._lock:
            for channel in self._subscribers:
                await connection.add_listener(channel, self._dispatch)
            self._connection = connection
        self._deliver_to_all(None)
        while self._subscribers and not lost.is_set() and not self._closing:
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=_KEEPALIVE_SECONDS)
            except TimeoutError:
                async with self._lock:
                    await connection.fetchval("SELECT 1", timeout=_KEEPALIVE_TIMEOUT_SECONDS)

    def _dispatch(self, _connection: object, _pid: object, channel: str, payload: str) -> None:
        for callback in tuple(self._subscribers.get(channel, ())):
            _deliver(callback, payload)

    def _deliver_to_all(self, payload: str | None) -> None:
        for callbacks in tuple(self._subscribers.values()):
            for callback in tuple(callbacks):
                _deliver(callback, payload)


def _deliver(callback: NotificationCallback, payload: str | None) -> None:
    try:
        callback(payload)
    except Exception:
        logger.warning("Notification subscriber failed", exc_info=True)


__all__ = ["NotificationCallback", "PGNotificationHub", "dedicated_connection"]
