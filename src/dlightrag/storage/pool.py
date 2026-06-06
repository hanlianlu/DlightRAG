# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process-wide asyncpg connection pool for DlightRAG domain stores.

Independent from LightRAG's internal ClientManager pool to avoid contention
under concurrent ingestion/retrieval workloads.

Usage::

    from dlightrag.storage.pool import pg_pool

    pool = await pg_pool.get()           # lazily creates the pool
    async with pool.acquire() as conn:
        await conn.execute(...)

    await pg_pool.close()                # call once at shutdown
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import asyncpg

from dlightrag.config import PostgresTarget

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10
_DEFAULT_RETRY_ATTEMPTS = 10
_DEFAULT_RETRY_BACKOFF = 3.0
_DEFAULT_RETRY_BACKOFF_MAX = 30.0
_DEFAULT_POOL_CLOSE_TIMEOUT = 5.0

T = TypeVar("T")


class PGPool:
    """Lazy-created asyncpg pools keyed by primary/replica target."""

    def __init__(self) -> None:
        self._pools: dict[PostgresTarget, asyncpg.Pool] = {}
        self._lock: asyncio.Lock | None = None
        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, target: PostgresTarget | None = None) -> asyncpg.Pool:
        """Return a shared pool, creating it on first call."""
        from dlightrag.config import get_config

        config = get_config()
        resolved_target = target or config.pg_target_for_runtime()
        if resolved_target in self._pools:
            return self._pools[resolved_target]
        async with self._get_lock():
            if resolved_target in self._pools:
                return self._pools[resolved_target]

            min_size = getattr(config, "postgres_pool_min_size", _DEFAULT_MIN_SIZE)
            max_size = getattr(config, "postgres_pool_max_size", _DEFAULT_MAX_SIZE)
            endpoint = config.pg_connection_kwargs(resolved_target)
            pool_kwargs: dict[str, Any] = dict(endpoint)
            pool_kwargs["min_size"] = min_size
            pool_kwargs["max_size"] = max_size
            statement_cache_size = getattr(config, "postgres_statement_cache_size", None)
            if statement_cache_size is not None:
                pool_kwargs["statement_cache_size"] = int(statement_cache_size)
            server_settings = config.postgres_server_settings_dict()
            if server_settings:
                pool_kwargs["server_settings"] = server_settings
            pool = await asyncpg.create_pool(**pool_kwargs)
            self._pools[resolved_target] = pool
            logger.info(
                "DlightRAG %s domain store pool created (min=%d, max=%d)",
                resolved_target,
                min_size,
                max_size,
            )
            return pool

    async def run(
        self,
        operation: Callable[[Any], Awaitable[T]],
        *,
        target: PostgresTarget | None = None,
    ) -> T:
        """Execute an operation through the shared pool with transient retry."""
        from dlightrag.config import get_config

        config = get_config()
        resolved_target = target or config.pg_target_for_runtime()
        attempts = max(
            1,
            int(getattr(config, "postgres_connection_retries", _DEFAULT_RETRY_ATTEMPTS)),
        )
        backoff = max(
            0.0,
            float(getattr(config, "postgres_connection_retry_backoff", _DEFAULT_RETRY_BACKOFF)),
        )
        backoff_max = max(
            backoff,
            float(
                getattr(
                    config,
                    "postgres_connection_retry_backoff_max",
                    _DEFAULT_RETRY_BACKOFF_MAX,
                )
            ),
        )
        pool_close_timeout = max(
            0.0,
            float(getattr(config, "postgres_pool_close_timeout", _DEFAULT_POOL_CLOSE_TIMEOUT)),
        )

        for attempt in range(1, attempts + 1):
            try:
                pool = await self.get(resolved_target)
                async with pool.acquire() as conn:
                    return await operation(conn)
            except self._transient_exceptions as exc:
                await self.close(resolved_target, timeout=pool_close_timeout)
                if attempt >= attempts:
                    raise
                sleep_for = min(backoff * (2 ** (attempt - 1)), backoff_max)
                logger.warning(
                    "DlightRAG %s domain store transient PostgreSQL error on attempt %d/%d: %r",
                    resolved_target,
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        raise RuntimeError("unreachable PostgreSQL retry state")

    async def close(
        self,
        target: PostgresTarget | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        """Close one pool or all pools. Safe to call multiple times."""
        targets: list[PostgresTarget] = [target] if target is not None else ["primary", "replica"]
        for resolved_target in targets:
            pool = self._pools.get(resolved_target)
            if pool is None:
                continue
            try:
                if timeout is None:
                    await pool.close()
                else:
                    await asyncio.wait_for(pool.close(), timeout=timeout)
                logger.info("DlightRAG %s domain store pool closed", resolved_target)
            except TimeoutError:
                logger.error(
                    "DlightRAG %s domain store pool close timed out after %.2fs",
                    resolved_target,
                    timeout,
                )
            except Exception:
                logger.warning("DlightRAG domain store pool close failed", exc_info=True)
            finally:
                self._pools.pop(resolved_target, None)


pg_pool = PGPool()
