# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process-wide asyncpg connection pool for DlightRAG domain stores.

Independent from LightRAG's internal ClientManager pool to avoid contention
under concurrent ingestion/retrieval workloads.

Usage::

    from dlightrag.adapters.postgres.core._pool import pg_pool

    pool = await pg_pool.get()           # lazily creates the pool
    async with pool.acquire() as conn:
        await conn.execute(...)

    await pg_pool.close()                # call once at shutdown
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import asyncpg

from dlightrag.adapters.postgres.core._errors import is_postgres_unavailable

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10
_DEFAULT_RETRY_ATTEMPTS = 10
_DEFAULT_RETRY_BACKOFF = 3.0
_DEFAULT_RETRY_BACKOFF_MAX = 30.0

T = TypeVar("T")


class PGPool:
    """Lazy-created asyncpg pool for DlightRAG domain stores."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._lock: asyncio.Lock | None = None
        self._config: Any = None

    @staticmethod
    def _binding_signature(config: Any) -> tuple[Any, ...]:
        """Connection-identity fields that must not change under one process."""
        return (
            config.storage.postgres.host,
            config.storage.postgres.port,
            config.storage.postgres.user,
            config.storage.postgres.database,
            config.deployment.service_role,
            config.storage.postgres.ssl_mode,
            tuple(sorted(config.domain_pool_server_settings().items())),
        )

    def bind(self, config: Any) -> None:
        """Bind the pool to an explicit config during Application startup.

        Prevents the domain pool from silently diverging from the process's
        service config -- notably the reader read-only invariant -- when a
        caller constructs a config without ``set_config()``. A second,
        connection-incompatible config in the same process fails fast.
        """
        if self._config is not None and self._binding_signature(
            self._config
        ) != self._binding_signature(config):
            raise RuntimeError(
                "DlightRAG domain pool is already bound to a different PostgreSQL "
                "endpoint/role; one process supports a single service config."
            )
        self._config = config

    def _active_config(self) -> Any:
        if self._config is not None:
            return self._config
        from dlightrag.application.config import get_config

        return get_config()

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self) -> asyncpg.Pool:
        """Return a shared pool, creating it on first call."""
        config = self._active_config()
        if self._pool is not None:
            return self._pool
        async with self._get_lock():
            if self._pool is not None:
                return self._pool

            min_size = getattr(config.storage.postgres, "pool_min_size", _DEFAULT_MIN_SIZE)
            max_size = getattr(config.storage.postgres, "pool_max_size", _DEFAULT_MAX_SIZE)
            endpoint = config.pg_connection_kwargs()
            pool_kwargs: dict[str, Any] = dict(endpoint)
            pool_kwargs["min_size"] = min_size
            pool_kwargs["max_size"] = max_size
            statement_cache_size = getattr(config.storage.postgres, "statement_cache_size", None)
            if statement_cache_size is not None:
                pool_kwargs["statement_cache_size"] = int(statement_cache_size)
            if config.storage.postgres.command_timeout is not None:
                pool_kwargs["command_timeout"] = config.storage.postgres.command_timeout
            server_settings = config.domain_pool_server_settings()
            if server_settings:
                pool_kwargs["server_settings"] = server_settings
            pool = await asyncpg.create_pool(**pool_kwargs)
            self._pool = pool
            logger.info(
                "DlightRAG domain store pool created (min=%d, max=%d)",
                min_size,
                max_size,
            )
            return pool

    async def run(
        self,
        operation: Callable[[Any], Awaitable[T]],
    ) -> T:
        """Execute an operation through the shared pool with transient retry.

        Retries the *operation* on transient errors — never destroys the pool.
        asyncpg's built-in connection health checks handle bad connections;
        destroying the pool amplifies one transient failure into a cascading
        disruption for all concurrent callers.
        """
        config = self._active_config()
        attempts = max(
            1,
            int(
                getattr(
                    config.storage.postgres,
                    "connection_retries",
                    _DEFAULT_RETRY_ATTEMPTS,
                )
            ),
        )
        backoff = max(
            0.0,
            float(
                getattr(
                    config.storage.postgres,
                    "connection_retry_backoff",
                    _DEFAULT_RETRY_BACKOFF,
                )
            ),
        )
        backoff_max = max(
            backoff,
            float(
                getattr(
                    config.storage.postgres,
                    "connection_retry_backoff_max",
                    _DEFAULT_RETRY_BACKOFF_MAX,
                )
            ),
        )

        for attempt in range(1, attempts + 1):
            try:
                pool = await self.get()
                async with pool.acquire(timeout=config.storage.postgres.acquire_timeout) as conn:
                    return await operation(conn)
            except Exception as exc:
                if not is_postgres_unavailable(exc):
                    raise
                if attempt >= attempts:
                    raise
                sleep_for = min(backoff * (2 ** (attempt - 1)), backoff_max)
                logger.warning(
                    "DlightRAG domain store transient PostgreSQL error on attempt %d/%d: %r",
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        raise RuntimeError("unreachable PostgreSQL retry state")

    async def run_once(
        self,
        operation: Callable[[Any], Awaitable[T]],
    ) -> T:
        """Execute an operation once without replaying ambiguous outcomes.

        Use this for outcome-sensitive mutations whose transaction may have
        committed before a connection error reaches the client.
        """
        config = self._active_config()
        pool = await self.get()
        async with pool.acquire(timeout=config.storage.postgres.acquire_timeout) as conn:
            return await operation(conn)

    async def stream(
        self,
        operation: Callable[[Any], AsyncIterator[T]],
    ) -> AsyncIterator[T]:
        """Stream a read through one pooled connection without retry.

        The transaction wrapper exists because asyncpg server-side cursors
        require one; ordinary reads are unaffected.
        """
        config = self._active_config()
        pool = await self.get()
        async with pool.acquire(timeout=config.storage.postgres.acquire_timeout) as conn:
            async with conn.transaction():
                async for piece in operation(conn):
                    yield piece

    async def close(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        """Close the shared pool. Safe to call multiple times."""
        pool = self._pool
        if pool is None:
            self._config = None
            return
        try:
            if timeout is None:
                await pool.close()
            else:
                await asyncio.wait_for(pool.close(), timeout=timeout)
            logger.info("DlightRAG domain store pool closed")
        except TimeoutError:
            logger.error(
                "DlightRAG domain store pool close timed out after %.2fs; terminating", timeout
            )
            pool.terminate()
        except Exception:
            logger.warning("DlightRAG domain store pool close failed", exc_info=True)
        finally:
            self._pool = None
            self._config = None


pg_pool = PGPool()
