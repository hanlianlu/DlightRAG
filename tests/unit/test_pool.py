# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dlightrag.storage.pool.PGPool singleton."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest


class TestPGPoolGet:
    """Tests for PGPool.get()."""

    @pytest.mark.asyncio
    async def test_get_creates_pool(self) -> None:
        """First call to get() creates the asyncpg pool with config values."""
        from dlightrag.storage.pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_host = "testhost"
        mock_config.postgres_port = 5432
        mock_config.postgres_user = "testuser"
        mock_config.postgres_password = "testpass"
        mock_config.postgres_database = "testdb"
        mock_config.postgres_pool_min_size = 2
        mock_config.postgres_pool_max_size = 10
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_command_timeout = None
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {
            "host": "testhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpass",
            "database": "testdb",
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result = await pool.get()

        assert result is mock_pool
        mock_create.assert_called_once_with(
            host="testhost",
            port=5432,
            user="testuser",
            password="testpass",
            database="testdb",
            min_size=2,
            max_size=10,
        )

    @pytest.mark.asyncio
    async def test_double_get_reuses_pool(self) -> None:
        """Calling get() twice returns the same pool and creates it only once."""
        from dlightrag.storage.pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost"
        mock_config.postgres_port = 5432
        mock_config.postgres_user = "u"
        mock_config.postgres_password = "p"
        mock_config.postgres_database = "db"
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {
            "host": "localhost",
            "port": 5432,
            "user": "u",
            "password": "p",
            "database": "db",
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result1 = await pool.get()
            result2 = await pool.get()

        assert result1 is mock_pool
        assert result2 is mock_pool
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        """close() calls pool.close() and resets internal state."""
        from dlightrag.storage.pool import PGPool

        mock_pool = AsyncMock()
        pool = PGPool()
        pool._pool = mock_pool  # inject directly, skip creation

        await pool.close()

        mock_pool.close.assert_called_once()
        assert pool._pool is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """close() on an already-closed pool does not raise."""
        from dlightrag.storage.pool import PGPool

        pool = PGPool()
        # pool._pool is None — calling close should be a no-op
        await pool.close()  # must not raise

    @pytest.mark.asyncio
    async def test_get_after_close_recreates_pool(self) -> None:
        """After close(), get() creates a fresh pool."""
        from dlightrag.storage.pool import PGPool

        mock_pool1 = AsyncMock()
        mock_pool2 = AsyncMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_host = "localhost"
        mock_config.postgres_port = 5432
        mock_config.postgres_user = "u"
        mock_config.postgres_password = "p"
        mock_config.postgres_database = "db"
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {
            "host": "localhost",
            "port": 5432,
            "user": "u",
            "password": "p",
            "database": "db",
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool",
                new=AsyncMock(side_effect=[mock_pool1, mock_pool2]),
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            r1 = await pool.get()
            await pool.close()
            r2 = await pool.get()

        assert r1 is mock_pool1
        assert r2 is mock_pool2
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_get_applies_session_settings_and_statement_cache(self) -> None:
        """Domain store pools should use the same PG session tuning as LightRAG."""
        from dlightrag.storage.pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_pool_min_size = 2
        mock_config.postgres_pool_max_size = 10
        mock_config.postgres_statement_cache_size = 128
        mock_config.postgres_command_timeout = None
        mock_config.postgres_server_settings_dict.return_value = {
            "hnsw.ef_search": "384",
            "application_name": "dlightrag",
        }
        mock_config.pg_connection_kwargs.return_value = {
            "host": "primary",
            "port": 5432,
            "user": "writer",
            "password": "secret",
            "database": "dlightrag",
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result = await pool.get()

        assert result is mock_pool
        mock_create.assert_called_once_with(
            host="primary",
            port=5432,
            user="writer",
            password="secret",
            database="dlightrag",
            min_size=2,
            max_size=10,
            statement_cache_size=128,
            server_settings={
                "hnsw.ef_search": "384",
                "application_name": "dlightrag",
            },
        )

    @pytest.mark.asyncio
    async def test_get_forwards_ssl_connection_kwargs(self) -> None:
        """Managed PostgreSQL SSL settings must reach DlightRAG domain pools."""
        from dlightrag.storage.pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_pool_min_size = 2
        mock_config.postgres_pool_max_size = 10
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_command_timeout = None
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {
            "host": "primary",
            "port": 5432,
            "user": "writer",
            "password": "secret",
            "database": "dlightrag",
            "ssl": True,
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result = await pool.get()

        assert result is mock_pool
        mock_create.assert_called_once_with(
            host="primary",
            port=5432,
            user="writer",
            password="secret",
            database="dlightrag",
            ssl=True,
            min_size=2,
            max_size=10,
        )

    @pytest.mark.asyncio
    async def test_run_retries_operation_on_transient_error_without_destroying_pool(self) -> None:
        """Transient errors retry the operation on the same pool — the pool survives."""
        from dlightrag.storage.pool import PGPool

        the_pool = MagicMock()
        the_pool.close = AsyncMock()
        pool = PGPool()

        stale_conn = object()
        fresh_conn = object()
        # First acquire yields the stale connection; second gives a healthy one.
        the_pool.acquire.return_value.__aenter__.side_effect = [stale_conn, fresh_conn]

        mock_config = MagicMock()
        mock_config.postgres_pool_min_size = 2
        mock_config.postgres_pool_max_size = 10
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.postgres_connection_retries = 2
        mock_config.postgres_connection_retry_backoff = 0
        mock_config.postgres_connection_retry_backoff_max = 0
        mock_config.pg_connection_kwargs.return_value = {
            "host": "primary",
            "port": 5432,
            "user": "writer",
            "password": "secret",
            "database": "dlightrag",
        }

        calls = 0

        async def operation(conn):  # noqa: ANN001, ANN202
            nonlocal calls
            calls += 1
            if calls == 1:
                assert conn is stale_conn
                raise asyncpg.exceptions.ConnectionDoesNotExistError("stale connection")
            assert conn is fresh_conn
            return "ok"

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool",
                new=AsyncMock(return_value=the_pool),
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result = await pool.run(operation)

        assert result == "ok"
        assert calls == 2
        assert mock_create.call_count == 1
        # The pool must NOT be closed — pool.acquire() handles stale connections internally
        the_pool.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_applies_command_timeout(self) -> None:
        """A configured command_timeout is forwarded to asyncpg.create_pool."""
        from dlightrag.storage.pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_pool_min_size = 2
        mock_config.postgres_pool_max_size = 10
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_command_timeout = 60.0
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {
            "host": "primary",
            "port": 5432,
            "user": "writer",
            "password": "secret",
            "database": "dlightrag",
        }

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)
            ) as mock_create,
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            await pool.get()

        assert mock_create.call_args.kwargs["command_timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_run_acquires_with_configured_timeout(self) -> None:
        """run() passes the configured acquire timeout to pool.acquire()."""
        from dlightrag.storage.pool import PGPool

        the_pool = MagicMock()
        the_pool.acquire.return_value.__aenter__.return_value = object()
        pool = PGPool()

        mock_config = MagicMock()
        mock_config.postgres_statement_cache_size = None
        mock_config.postgres_command_timeout = None
        mock_config.postgres_acquire_timeout = 12.5
        mock_config.postgres_connection_retries = 1
        mock_config.postgres_server_settings_dict.return_value = {}
        mock_config.pg_connection_kwargs.return_value = {"host": "h", "port": 5432}

        async def operation(conn):  # noqa: ANN001, ANN202
            return "ok"

        with (
            patch(
                "dlightrag.storage.pool.asyncpg.create_pool",
                new=AsyncMock(return_value=the_pool),
            ),
            patch("dlightrag.config.get_config", return_value=mock_config),
        ):
            result = await pool.run(operation)

        assert result == "ok"
        the_pool.acquire.assert_called_once_with(timeout=12.5)

    @pytest.mark.asyncio
    async def test_close_terminates_on_timeout(self) -> None:
        """If graceful close exceeds the timeout, the pool is force-terminated."""
        from dlightrag.storage.pool import PGPool

        async def _hang() -> None:
            await asyncio.sleep(10)

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock(side_effect=_hang)
        mock_pool.terminate = MagicMock()
        pool = PGPool()
        pool._pool = mock_pool

        await pool.close(timeout=0.01)

        mock_pool.terminate.assert_called_once()
        assert pool._pool is None
