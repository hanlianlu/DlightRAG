# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One shared PostgreSQL LISTEN connection per pool, fanned out to subscribers.

A subscriber names a channel and a callback. The callback receives every NOTIFY
payload on that channel, and ``None`` each time the hub (re)connects: a
notification can only be missed while no connection was listening, so ``None``
asks the subscriber to re-read its authoritative state. Notifications are wake
hints, never authority. Callbacks run on the event loop and must not block.

The hub holds its connection only while something subscribes, replaces a lost
connection (termination, or a failed or hung statement) after a delay that
doubles until a connection passes a keepalive, and releases the connection when
the last subscriber leaves. A registered subscriber is always LISTENed on the
current connection, or will be by the next one.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from functools import partial
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

type NotificationCallback = Callable[[str | None], None]

_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0
# A LISTEN connection is otherwise idle, so a half-open socket would go unnoticed;
# the keepalive turns that into a reconnect and a resynchronization. A connection
# that passes one has proven itself, which resets the reconnect delay.
_KEEPALIVE_SECONDS = 30.0
# Bounds every statement the hub runs: LISTEN, UNLISTEN, and the keepalive. They
# run under the hub lock, because asyncpg runs one statement per connection at a
# time, so this is also the longest a half-open socket can hold up subscribe and
# unsubscribe before the connection is dropped and replaced.
_STATEMENT_TIMEOUT_SECONDS = 5.0


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
        self._reconnect_delay = _RECONNECT_BASE_SECONDS

    @asynccontextmanager
    async def listen(self, channel: str, callback: NotificationCallback) -> AsyncIterator[None]:
        """Deliver ``channel`` to ``callback`` for the duration of the block."""
        try:
            await self.subscribe(channel, callback)
            yield
        finally:
            await self.unsubscribe(channel, callback)

    async def subscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Start delivering ``channel`` to ``callback``.

        When the hub is connected, the channel is LISTENed before this returns, so a
        subscriber that reads its state afterwards misses nothing. When it is not,
        or that LISTEN fails, the subscriber stays registered and the next
        connection LISTENs it and delivers ``None``. A subscribe cancelled during its
        LISTEN leaves nothing registered.
        """
        if self._closing:
            raise RuntimeError("notification hub is closed")
        async with self._lock:
            callbacks = self._subscribers.setdefault(channel, [])
            callbacks.append(callback)
            connection = self._connection
            if len(callbacks) == 1 and connection is not None:
                try:
                    await self._statement(
                        connection, partial(connection.add_listener, channel, self._dispatch)
                    )
                except Exception:
                    # The failed connection was dropped; its replacement LISTENs every
                    # registered channel and resynchronizes every subscriber.
                    logger.warning(
                        "LISTEN failed on the notification connection; reconnecting",
                        exc_info=True,
                    )
                except BaseException:
                    # A registration without a LISTEN would swallow this channel for
                    # every later subscriber, since only the first one LISTENs.
                    self._unregister(channel, callback)
                    raise
        self._ensure_running()

    async def unsubscribe(self, channel: str, callback: NotificationCallback) -> None:
        """Stop delivering ``channel`` to ``callback``; the last one out releases the hub.

        The callback is unregistered before anything is awaited, so an unsubscribe
        cancelled while it waits for the lock, which a hung statement holds for up to
        the statement bound, still leaves nothing behind. The channel's UNLISTEN
        follows under the lock unless the channel was subscribed again meanwhile: that
        subscriber's LISTEN found the channel still LISTENed and issued none, so an
        UNLISTEN now would leave it deaf.
        """
        if not self._unregister(channel, callback):
            return
        try:
            async with self._lock:
                connection = self._connection
                if connection is None or channel in self._subscribers:
                    return
                # A failed UNLISTEN only costs the connection: the replacement LISTENs
                # the channels still registered, which no longer include this one.
                with suppress(Exception):
                    await self._statement(
                        connection, partial(connection.remove_listener, channel, self._dispatch)
                    )
        finally:
            if not self._subscribers:
                self._changed.set()  # the last one out releases the hub

    async def aclose(self) -> None:
        """Stop listening and release the connection; the hub cannot be reused."""
        self._closing = True
        async with self._lock:
            channels = tuple(self._subscribers)
            self._subscribers.clear()
            connection = self._connection
            for channel in channels:
                if connection is None or self._connection is not connection:
                    break
                with suppress(Exception):
                    await self._statement(
                        connection, partial(connection.remove_listener, channel, self._dispatch)
                    )
        self._changed.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _unregister(self, channel: str, callback: NotificationCallback) -> bool:
        """Drop one registration without awaiting; return whether it emptied ``channel``."""
        callbacks = self._subscribers.get(channel)
        if not callbacks or callback not in callbacks:
            return False
        callbacks.remove(callback)
        if callbacks:
            return False
        del self._subscribers[channel]
        return True

    def _ensure_running(self) -> None:
        if not self._closing and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run(), name="dlightrag-pg-notifications")

    async def _statement(self, connection: Any, statement: Callable[[], Awaitable[object]]) -> None:
        """Run one statement on the connection; one that fails is never used again.

        Callers hold the lock. A statement that fails, hangs past the timeout, or is
        abandoned by a cancelled caller leaves the connection's state unknown, so the
        connection is dropped and terminated, which ends its serve loop.
        """
        try:
            await asyncio.wait_for(statement(), timeout=_STATEMENT_TIMEOUT_SECONDS)
        except BaseException:
            if self._connection is connection:
                self._connection = None
            # A pool connection asyncpg already cleaned up is a detached proxy, on which
            # even terminate() raises; its termination was already signalled.
            with suppress(asyncpg.InterfaceError):
                connection.terminate()
            raise

    async def _run(self) -> None:
        self._reconnect_delay = _RECONNECT_BASE_SECONDS
        while self._subscribers and not self._closing:
            try:
                lost = await self._serve()
            except Exception:
                if not self._subscribers or self._closing:
                    return
                logger.warning(
                    "PostgreSQL notification connection failed; reconnecting in %.1fs",
                    self._reconnect_delay,
                    exc_info=True,
                )
            else:
                if not lost or not self._subscribers or self._closing:
                    continue  # released while idle; serve again only if someone came back
                logger.warning(
                    "PostgreSQL notification connection was lost; reconnecting in %.1fs",
                    self._reconnect_delay,
                )
            # Every replacement waits, and the wait doubles until a connection passes a
            # keepalive, so one that dies right after its LISTEN cannot spin the hub
            # through reconnects and resynchronizations.
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX_SECONDS)

    async def _serve(self) -> bool:
        """Serve the subscribers on one connection; return whether it was lost, not released."""
        async with self._connect() as connection:
            lost = asyncio.Event()

            def _terminated(_connection: object) -> None:
                lost.set()
                self._changed.set()

            connection.add_termination_listener(_terminated)
            try:
                return await self._listen_on(connection, lost)
            finally:
                async with self._lock:
                    if self._connection is connection:
                        self._connection = None
                # A pool connection that asyncpg already cleaned up is detached from
                # its proxy, and every call on it raises InterfaceError.
                with suppress(asyncpg.InterfaceError):
                    connection.remove_termination_listener(_terminated)
                    if connection.is_closed():
                        # asyncpg can report a server-closed connection closed before
                        # cleaning it up, which is what frees its pool slot.
                        connection.terminate()

    async def _listen_on(self, connection: Any, lost: asyncio.Event) -> bool:
        async with self._lock:
            self._connection = connection
            for channel in tuple(self._subscribers):
                await self._statement(
                    connection, partial(connection.add_listener, channel, self._dispatch)
                )
        self._deliver_to_all(None)
        while self._subscribers and not self._closing:
            if lost.is_set():
                return True
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=_KEEPALIVE_SECONDS)
            except TimeoutError:
                async with self._lock:
                    if self._connection is not connection:
                        return True  # dropped by a failed statement; the next one replaces it
                    await self._statement(connection, partial(connection.fetchval, "SELECT 1"))
                self._reconnect_delay = _RECONNECT_BASE_SECONDS
        return False

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
