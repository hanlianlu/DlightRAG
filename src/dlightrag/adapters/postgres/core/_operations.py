# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared execution plumbing for PostgreSQL adapters."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

from dlightrag.adapters.postgres.core._errors import guard_payload
from dlightrag.adapters.postgres.core._notifications import PGNotificationHub
from dlightrag.adapters.postgres.core._pool import pg_pool

T = TypeVar("T")


class ConnectionPool(Protocol):
    """Raw connection pool accepted by focused adapters and integration tests."""

    def acquire(self) -> Any: ...


class PostgresOperationRunner:
    """Run adapter operations through an injected raw pool or the process pool."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        notifications: PGNotificationHub | None = None,
    ) -> None:
        self._operation_pool = pool
        self._notifications = notifications

    def _notification_hub(self) -> PGNotificationHub:
        """The hub this runner's subscribers listen through.

        Every adapter on the process pool shares the process hub. An injected pool
        comes with the hub for its database, since no hub can be derived from it.
        """
        if self._notifications is not None:
            return self._notifications
        if self._operation_pool is not None:
            raise RuntimeError("an injected pool needs an injected notification hub")
        return pg_pool.notifications

    async def _run(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        if self._operation_pool is None:
            return await guard_payload(pg_pool.run(operation), surface="Durable record")
        async with self._operation_pool.acquire() as connection:
            return await guard_payload(operation(connection), surface="Durable record")

    async def _run_once(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        """Run an outcome-sensitive mutation without replaying it."""
        if self._operation_pool is None:
            return await guard_payload(pg_pool.run_once(operation), surface="Durable record")
        async with self._operation_pool.acquire() as connection:
            return await guard_payload(operation(connection), surface="Durable record")

    async def _stream(self, operation: Callable[[Any], AsyncIterator[T]]) -> AsyncIterator[T]:
        """Stream a read through one connection; the caller drains the iterator."""
        if self._operation_pool is None:
            async for piece in pg_pool.stream(operation):
                yield piece
            return
        async with self._operation_pool.acquire() as connection:
            async with connection.transaction():
                async for piece in operation(connection):
                    yield piece


__all__ = ["ConnectionPool", "PostgresOperationRunner"]
