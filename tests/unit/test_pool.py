# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dlightrag.storage.pool.PGPool singleton."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
