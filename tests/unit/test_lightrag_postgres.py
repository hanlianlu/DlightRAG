# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's LightRAG PostgreSQL integration helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from lightrag.kg.postgres_impl import PostgreSQLDB


def _pg_config() -> dict[str, object]:
    return {
        "host": "replica",
        "port": 5432,
        "user": "reader",
        "password": "secret",
        "database": "dlightrag",
        "workspace": "default",
        "max_connections": 4,
        "ssl_mode": None,
        "ssl_cert": None,
        "ssl_key": None,
        "ssl_root_cert": None,
        "ssl_crl": None,
        "enable_vector": False,
        "vector_index_type": "HNSW",
        "hnsw_m": 32,
        "hnsw_ef": 256,
        "ivfflat_lists": 100,
        "vchordrq_build_options": "",
        "vchordrq_probes": "",
        "vchordrq_epsilon": 1.9,
        "server_settings": "application_name=dlightrag",
        "statement_cache_size": 128,
        "connection_retry_attempts": 2,
        "connection_retry_backoff": 0.0,
        "connection_retry_backoff_max": 0.0,
        "pool_close_timeout": 5.0,
    }


@pytest.mark.asyncio
async def test_read_only_postgres_reconnect_does_not_call_upstream_initdb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.storage import lightrag_postgres
    from dlightrag.storage.lightrag_postgres import ReadOnlyPostgreSQLDB

    pools = [object(), object()]
    create_pool = AsyncMock(side_effect=pools)
    monkeypatch.setattr(lightrag_postgres.asyncpg, "create_pool", create_pool)

    async def forbidden_upstream_initdb(self):  # noqa: ANN001, ANN202
        raise AssertionError("read-only reconnect must not run upstream initdb")

    monkeypatch.setattr(PostgreSQLDB, "initdb", forbidden_upstream_initdb)

    db = ReadOnlyPostgreSQLDB(_pg_config())
    await db.initdb()
    assert db.pool is pools[0]

    db.pool = None
    await db._ensure_pool()

    assert db.pool is pools[1]
    assert create_pool.call_count == 2
    kwargs = create_pool.await_args_list[-1].kwargs
    assert kwargs["host"] == "replica"
    assert kwargs["user"] == "reader"
    assert kwargs["min_size"] == 1
    assert kwargs["max_size"] == 4
    assert kwargs["statement_cache_size"] == 128
    assert kwargs["server_settings"] == {"application_name": "dlightrag"}
    assert callable(kwargs["init"])
    assert callable(kwargs["reset"])


@pytest.mark.asyncio
async def test_read_only_postgres_initdb_retries_transient_pool_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.storage import lightrag_postgres
    from dlightrag.storage.lightrag_postgres import ReadOnlyPostgreSQLDB

    pool = object()
    create_pool = AsyncMock(side_effect=[OSError("transient"), pool])
    monkeypatch.setattr(lightrag_postgres.asyncpg, "create_pool", create_pool)

    db = ReadOnlyPostgreSQLDB(_pg_config())
    await db.initdb()

    assert db.pool is pool
    assert create_pool.call_count == 2
