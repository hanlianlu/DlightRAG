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

import asyncpg

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10


class PGPool:
    """Lazy-created asyncpg pool singleton with double-check locking."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self) -> asyncpg.Pool:
        """Return the shared pool, creating it on first call."""
        if self._pool is not None:
            return self._pool
        async with self._get_lock():
            if self._pool is not None:
                return self._pool
            from dlightrag.config import get_config

            config = get_config()
            self._pool = await asyncpg.create_pool(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                database=config.postgres_database,
                min_size=_DEFAULT_MIN_SIZE,
                max_size=_DEFAULT_MAX_SIZE,
            )
            logger.info(
                "DlightRAG domain store pool created (min=%d, max=%d)",
                _DEFAULT_MIN_SIZE,
                _DEFAULT_MAX_SIZE,
            )
            return self._pool

    async def close(self) -> None:
        """Close the pool. Safe to call multiple times."""
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("DlightRAG domain store pool closed")
            except Exception:
                logger.warning("DlightRAG domain store pool close failed", exc_info=True)
            finally:
                self._pool = None


pg_pool = PGPool()
