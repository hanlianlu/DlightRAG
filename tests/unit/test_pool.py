# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dlightrag.adapters.postgres.core._pool.PGPool singleton."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from tests.support.config_helpers import mutate_config


def _session_settings(
    mock_config: MagicMock, *, ef_search: int = 256, extra: dict[str, str] | None = None
) -> None:
    """Give the mock config the plain settings the pool renders its session GUCs from."""
    mutate_config(mock_config, "storage.lightrag.hnsw_ef_search", ef_search)
    mutate_config(mock_config, "storage.postgres.session_settings", dict(extra or {}))


def _endpoint(mock_config: MagicMock, *, ssl_mode: str | None = None, **endpoint: object) -> None:
    """Give the mock config the PostgreSQL endpoint and TLS mode the pool connects with."""
    fields = {"host": "h", "port": 5432, "user": "u", "password": "p", "database": "db"}
    for name, value in {**fields, **endpoint, "ssl_mode": ssl_mode}.items():
        mutate_config(mock_config, f"storage.postgres.{name}", value)


class TestPGPoolGet:
    """Tests for PGPool.get()."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ({}, {}),
            # Domain store pools use the same session tuning as LightRAG.
            (
                {
                    "statement_cache_size": 128,
                    "ef_search": 384,
                    "session_settings": {"application_name": "dlightrag"},
                },
                {
                    "statement_cache_size": 128,
                    "server_settings": {
                        "hnsw.ef_search": "384",
                        "application_name": "dlightrag",
                    },
                },
            ),
            # Managed PostgreSQL SSL settings must reach the domain pools.
            ({"ssl_mode": "require"}, {"ssl": True}),
            ({"command_timeout": 60.0}, {"command_timeout": 60.0}),
        ],
        ids=[
            "endpoint-and-sizes",
            "session-settings-and-statement-cache",
            "ssl",
            "command-timeout",
        ],
    )
    async def test_get_forwards_the_configured_connection_settings_to_asyncpg(
        self, configured: dict[str, object], expected: dict[str, object]
    ) -> None:
        from dlightrag.adapters.postgres.core._pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.pool_min_size", 2)
        mutate_config(mock_config, "storage.postgres.pool_max_size", 10)
        mutate_config(
            mock_config,
            "storage.postgres.statement_cache_size",
            configured.get("statement_cache_size"),
        )
        mutate_config(
            mock_config, "storage.postgres.command_timeout", configured.get("command_timeout")
        )
        _session_settings(
            mock_config,
            ef_search=configured.get("ef_search", 256),  # type: ignore[arg-type]
            extra=configured.get("session_settings"),  # type: ignore[arg-type]
        )
        _endpoint(
            mock_config,
            host="primary",
            user="writer",
            password="secret",
            database="dlightrag",
            ssl_mode=configured.get("ssl_mode"),  # type: ignore[arg-type]
        )

        with (
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(return_value=mock_pool),
            ) as mock_create,
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
        ):
            result = await pool.get()

        assert result is mock_pool
        mock_create.assert_called_once_with(
            **{
                "host": "primary",
                "port": 5432,
                "user": "writer",
                "password": "secret",
                "database": "dlightrag",
                "min_size": 2,
                "max_size": 10,
                "server_settings": {"hnsw.ef_search": "256"},
                **expected,
            }
        )

    @pytest.mark.asyncio
    async def test_double_get_reuses_pool(self) -> None:
        """Calling get() twice returns the same pool and creates it only once."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        mock_pool = MagicMock()
        pool = PGPool()

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.statement_cache_size", None)
        _session_settings(mock_config)
        _endpoint(mock_config, host="localhost")

        with (
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(return_value=mock_pool),
            ) as mock_create,
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
        ):
            result1 = await pool.get()
            result2 = await pool.get()

        assert result1 is mock_pool
        assert result2 is mock_pool
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        """close() calls pool.close() and resets internal state."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        mock_pool = AsyncMock()
        pool = PGPool()
        pool._pool = mock_pool  # inject directly, skip creation

        await pool.close()

        mock_pool.close.assert_called_once()
        assert pool._pool is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """close() on an already-closed pool does not raise."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        pool = PGPool()
        # pool._pool is None — calling close should be a no-op
        await pool.close()  # must not raise

    @pytest.mark.asyncio
    async def test_get_after_close_recreates_pool(self) -> None:
        """After close(), get() creates a fresh pool."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        mock_pool1 = AsyncMock()
        mock_pool2 = AsyncMock()
        pool = PGPool()

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.statement_cache_size", None)
        _session_settings(mock_config)
        _endpoint(mock_config, host="localhost")

        with (
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(side_effect=[mock_pool1, mock_pool2]),
            ) as mock_create,
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
        ):
            r1 = await pool.get()
            await pool.close()
            r2 = await pool.get()

        assert r1 is mock_pool1
        assert r2 is mock_pool2
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_run_retries_operation_on_transient_error_without_destroying_pool(self) -> None:
        """Transient errors retry the operation on the same pool — the pool survives."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        the_pool = MagicMock()
        the_pool.close = AsyncMock()
        pool = PGPool()

        stale_conn = object()
        fresh_conn = object()
        # First acquire yields the stale connection; second gives a healthy one.
        the_pool.acquire.return_value.__aenter__.side_effect = [stale_conn, fresh_conn]

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.pool_min_size", 2)
        mutate_config(mock_config, "storage.postgres.pool_max_size", 10)
        mutate_config(mock_config, "storage.postgres.statement_cache_size", None)
        _session_settings(mock_config)
        mutate_config(mock_config, "storage.postgres.connection_retries", 2)
        mutate_config(mock_config, "storage.postgres.connection_retry_backoff", 0)
        mutate_config(mock_config, "storage.postgres.connection_retry_backoff_max", 0)
        _endpoint(
            mock_config, host="primary", user="writer", password="secret", database="dlightrag"
        )

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
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(return_value=the_pool),
            ) as mock_create,
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
        ):
            result = await pool.run(operation)

        assert result == "ok"
        assert calls == 2
        assert mock_create.call_count == 1
        # The pool must NOT be closed — pool.acquire() handles stale connections internally
        the_pool.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_once_does_not_replay_operation_after_transient_error(self) -> None:
        """Outcome-sensitive writes get one attempt even for transient errors."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        the_pool = MagicMock()
        connection = object()
        the_pool.acquire.return_value.__aenter__.return_value = connection
        pool = PGPool()

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.statement_cache_size", None)
        mutate_config(mock_config, "storage.postgres.command_timeout", None)
        mutate_config(mock_config, "storage.postgres.acquire_timeout", 12.5)
        _session_settings(mock_config)
        _endpoint(mock_config)

        calls = 0

        async def operation(conn):  # noqa: ANN001, ANN202
            nonlocal calls
            calls += 1
            assert conn is connection
            raise asyncpg.exceptions.ConnectionDoesNotExistError("commit outcome unknown")

        with (
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(return_value=the_pool),
            ),
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
            pytest.raises(
                asyncpg.exceptions.ConnectionDoesNotExistError,
                match="commit outcome unknown",
            ),
        ):
            await pool.run_once(operation)

        assert calls == 1
        the_pool.acquire.assert_called_once_with(timeout=12.5)

    @pytest.mark.asyncio
    async def test_run_acquires_with_configured_timeout(self) -> None:
        """run() passes the configured acquire timeout to pool.acquire()."""
        from dlightrag.adapters.postgres.core._pool import PGPool

        the_pool = MagicMock()
        the_pool.acquire.return_value.__aenter__.return_value = object()
        pool = PGPool()

        mock_config = MagicMock()
        mutate_config(mock_config, "storage.postgres.statement_cache_size", None)
        mutate_config(mock_config, "storage.postgres.command_timeout", None)
        mutate_config(mock_config, "storage.postgres.acquire_timeout", 12.5)
        mutate_config(mock_config, "storage.postgres.pool_min_size", 2)
        mutate_config(mock_config, "storage.postgres.pool_max_size", 10)
        mutate_config(mock_config, "storage.postgres.connection_retries", 1)
        mutate_config(mock_config, "storage.postgres.connection_retry_backoff", 0)
        mutate_config(mock_config, "storage.postgres.connection_retry_backoff_max", 0)
        _session_settings(mock_config)
        _endpoint(mock_config)

        async def operation(conn):  # noqa: ANN001, ANN202
            return "ok"

        with (
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool",
                new=AsyncMock(return_value=the_pool),
            ),
            patch("dlightrag.application.config.loading.get_config", return_value=mock_config),
        ):
            result = await pool.run(operation)

        assert result == "ok"
        the_pool.acquire.assert_called_once_with(timeout=12.5)

    @pytest.mark.asyncio
    async def test_close_terminates_on_timeout(self) -> None:
        """If graceful close exceeds the timeout, the pool is force-terminated."""
        from dlightrag.adapters.postgres.core._pool import PGPool

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

    @pytest.mark.asyncio
    async def test_notifications_listen_on_a_connection_of_their_own(self) -> None:
        """The process hub connects to the bound endpoint directly and holds no pool slot.

        It carries the pool's session settings, such as a relaxed idle-session timeout that
        would otherwise end the idle LISTEN connection between keepalives.
        """
        from dlightrag.adapters.postgres.core._channels import RUN_CANCEL_CHANNEL
        from dlightrag.adapters.postgres.core._pool import PGPool

        connection = MagicMock()
        connection.add_listener = AsyncMock()
        connection.close = AsyncMock()
        pool = PGPool()
        mock_config = MagicMock()
        _endpoint(mock_config)
        _session_settings(mock_config, extra={"idle_session_timeout": "0"})
        pool.bind(mock_config)
        received: list[str | None] = []

        with (
            patch("asyncpg.connect", new=AsyncMock(return_value=connection)) as connect,
            patch(
                "dlightrag.adapters.postgres.core._pool.asyncpg.create_pool", new=AsyncMock()
            ) as create_pool,
        ):
            pool.notifications.subscribe(RUN_CANCEL_CHANNEL, received.append)
            async with asyncio.timeout(2):
                while received != [None]:
                    await asyncio.sleep(0.005)
            await pool.close()

        connect.assert_awaited_once_with(
            host="h",
            port=5432,
            user="u",
            password="p",
            database="db",
            server_settings={"hnsw.ef_search": "256", "idle_session_timeout": "0"},
        )
        create_pool.assert_not_called()
        connection.close.assert_awaited_once()
