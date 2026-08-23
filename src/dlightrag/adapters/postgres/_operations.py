# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared execution plumbing for PostgreSQL adapters."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

from dlightrag.adapters.postgres._pool import pg_pool

T = TypeVar("T")


class ConnectionPool(Protocol):
    """Raw connection pool accepted by focused adapters and integration tests."""

    def acquire(self) -> Any: ...


class PostgresOperationRunner:
    """Run adapter operations through an injected raw pool or the process pool."""

    def __init__(self, *, pool: ConnectionPool | None = None) -> None:
        self._operation_pool = pool

    async def _run(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        if self._operation_pool is None:
            return await pg_pool.run(operation)
        async with self._operation_pool.acquire() as connection:
            return await operation(connection)

    async def _run_once(self, operation: Callable[[Any], Awaitable[T]]) -> T:
        """Run an outcome-sensitive mutation without replaying it."""
        if self._operation_pool is None:
            return await pg_pool.run_once(operation)
        async with self._operation_pool.acquire() as connection:
            return await operation(connection)

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
