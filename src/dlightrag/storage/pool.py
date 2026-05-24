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

from dlightrag.config import PostgresTarget

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SIZE = 2
_DEFAULT_MAX_SIZE = 10


class PGPool:
    """Lazy-created asyncpg pools keyed by primary/replica target."""

    def __init__(self) -> None:
        self._pools: dict[PostgresTarget, asyncpg.Pool] = {}
        self._lock: asyncio.Lock | None = None

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
            pool = await asyncpg.create_pool(
                host=endpoint["host"],
                port=endpoint["port"],
                user=endpoint["user"],
                password=endpoint["password"],
                database=endpoint["database"],
                min_size=min_size,
                max_size=max_size,
            )
            self._pools[resolved_target] = pool
            logger.info(
                "DlightRAG %s domain store pool created (min=%d, max=%d)",
                resolved_target,
                min_size,
                max_size,
            )
            return pool

    async def close(self, target: PostgresTarget | None = None) -> None:
        """Close one pool or all pools. Safe to call multiple times."""
        targets: list[PostgresTarget] = [target] if target is not None else ["primary", "replica"]
        for resolved_target in targets:
            pool = self._pools.get(resolved_target)
            if pool is None:
                continue
            try:
                await pool.close()
                logger.info("DlightRAG %s domain store pool closed", resolved_target)
            except Exception:
                logger.warning("DlightRAG domain store pool close failed", exc_info=True)
            finally:
                self._pools.pop(resolved_target, None)


pg_pool = PGPool()
