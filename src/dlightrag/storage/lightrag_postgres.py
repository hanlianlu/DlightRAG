# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG PostgreSQL integration helpers owned by DlightRAG."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qsl

import asyncpg
from lightrag.kg.postgres_impl import PostgreSQLDB
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


def parse_postgres_server_settings(raw: Any) -> dict[str, str]:
    """Parse LightRAG's POSTGRES_SERVER_SETTINGS query-string format."""
    if raw is None:
        return {}
    return {key: value for key, value in parse_qsl(str(raw), keep_blank_values=False)}


class ReadOnlyPostgreSQLDB(PostgreSQLDB):
    """PostgreSQLDB variant that attaches to an existing schema without DDL.

    LightRAG's normal ``initdb()`` bootstraps extensions, tables, indexes, and
    AGE graph labels. Query workers running against replicas must never enter
    that path, including after a transient pool reset. This subclass preserves
    LightRAG's query-time retry machinery while overriding pool creation to be
    strictly read-only.
    """

    def _pool_kwargs(self) -> dict[str, Any]:
        pool_kwargs: dict[str, Any] = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "min_size": 1,
            "max_size": self.max,
        }
        if self.statement_cache_size is not None:
            pool_kwargs["statement_cache_size"] = int(self.statement_cache_size)

        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            pool_kwargs["ssl"] = ssl_context
        elif self.ssl_mode:
            ssl_mode = str(self.ssl_mode).lower()
            if ssl_mode in {"require", "prefer"}:
                pool_kwargs["ssl"] = True
            elif ssl_mode == "disable":
                pool_kwargs["ssl"] = False

        server_settings = parse_postgres_server_settings(self.server_settings)
        if server_settings:
            pool_kwargs["server_settings"] = server_settings
        return pool_kwargs

    async def _init_read_only_connection(self, connection: asyncpg.Connection) -> None:
        if self.enable_vector:
            await register_vector(connection)
        if self.enable_vector and self.vector_index_type == "VCHORDRQ":
            await self.configure_vchordrq(connection)

    async def _reset_read_only_connection(self, connection: asyncpg.Connection) -> None:
        try:
            reset_query = connection.get_reset_query()
            if reset_query:
                await connection.execute(reset_query)
        except Exception as exc:
            logger.error(
                "[%s] PostgreSQL read-only pool reset cleanup failed: %r",
                self.workspace,
                exc,
            )
            raise

        if self.enable_vector and self.vector_index_type == "VCHORDRQ":
            await self.configure_vchordrq(connection)

    async def initdb(self) -> None:
        """Create a read-only asyncpg pool without LightRAG bootstrap DDL."""
        attempts = max(1, int(self.connection_retry_attempts))
        backoff = max(0.0, float(self.connection_retry_backoff))
        backoff_max = max(backoff, float(self.connection_retry_backoff_max))

        for attempt in range(1, attempts + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    **self._pool_kwargs(),
                    init=self._init_read_only_connection,
                    reset=self._reset_read_only_connection,
                )
                return
            except self._transient_exceptions as exc:
                self.pool = None
                if attempt >= attempts:
                    raise
                sleep_for = min(backoff * (2 ** (attempt - 1)), backoff_max)
                logger.warning(
                    "PostgreSQL read-only pool transient connection issue on attempt %d/%d: %r",
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
