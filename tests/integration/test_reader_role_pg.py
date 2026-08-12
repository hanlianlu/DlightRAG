# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for reader-role PostgreSQL session and schema semantics.

A reader is corpus-read-only, not process-read-only: its DlightRAG domain pool
must accept operational writes while its LightRAG corpus pool must be read-only
in PostgreSQL itself. Reader startup must validate the migrated domain schema
without issuing any DDL and must fail before serving when that schema is absent.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import asyncpg
import pytest

from dlightrag.config import DlightragConfig, EmbeddingConfig
from dlightrag.storage.answer_runs import (
    ANSWER_RUN_MIGRATION_SCOPE,
    ANSWER_RUN_MIGRATIONS,
    PGAnswerRunStore,
)
from dlightrag.storage.migrations import apply_migrations, verify_migrations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _restore_process_env() -> Iterator[None]:
    """Config construction bridges PostgreSQL settings into LightRAG's env API."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
async def database() -> AsyncIterator[str]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_reader_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    try:
        yield db_name
    finally:
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _config(database: str, *, service_role: str) -> DlightragConfig:
    return cast(Any, DlightragConfig)(
        _env_file=None,
        service_role=service_role,
        postgres_host=_PG_CONN_KWARGS["host"],
        postgres_port=_PG_CONN_KWARGS["port"],
        postgres_user=_PG_CONN_KWARGS["user"],
        postgres_password=_PG_CONN_KWARGS["password"],
        postgres_database=database,
        embedding=EmbeddingConfig(
            provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
        ),
    )


async def _pool(config: DlightragConfig, settings: dict[str, str]) -> Any:
    return await asyncpg.create_pool(
        **config.pg_connection_kwargs(),
        min_size=1,
        max_size=2,
        server_settings=settings,
    )


# ---------------------------------------------------------------------------
# Session modes
# ---------------------------------------------------------------------------


async def test_reader_domain_pool_accepts_operational_writes(database: str) -> None:
    config = _config(database, service_role="reader")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "off"
            transaction = conn.transaction()
            await transaction.start()
            try:
                await conn.execute("CREATE TABLE reader_probe (id INT)")
                await conn.execute("INSERT INTO reader_probe VALUES (1)")
                assert await conn.fetchval("SELECT COUNT(*) FROM reader_probe") == 1
            finally:
                await transaction.rollback()
    finally:
        await pool.close()


async def test_reader_corpus_pool_rejects_writes_in_postgres(database: str) -> None:
    config = _config(database, service_role="reader")
    writer_pool = await _pool(
        _config(database, service_role="writer"),
        _config(database, service_role="writer").domain_pool_server_settings(),
    )
    try:
        async with writer_pool.acquire() as conn:
            await conn.execute("CREATE TABLE corpus_probe (id INT)")
    finally:
        await writer_pool.close()

    pool = await _pool(config, config.lightrag_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "on"
            assert await conn.fetchval("SELECT COUNT(*) FROM corpus_probe") == 0
            with pytest.raises(asyncpg.exceptions.ReadOnlySQLTransactionError):
                await conn.execute("INSERT INTO corpus_probe VALUES (1)")
    finally:
        await pool.close()


async def test_writer_corpus_pool_stays_writable(database: str) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.lightrag_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            assert str(await conn.fetchval("SHOW transaction_read_only")).lower() == "off"
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Validation-only startup
# ---------------------------------------------------------------------------


async def test_reader_startup_fails_before_the_writer_migrates(database: str) -> None:
    config = _config(database, service_role="reader")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        store = PGAnswerRunStore(pool=pool)
        with pytest.raises(RuntimeError, match="dlightrag_schema_migrations"):
            await store.initialize(validate_only=True)

        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT to_regclass('dlightrag_answer_runs')") is None
    finally:
        await pool.close()


async def test_reader_startup_validates_a_migrated_schema_without_ddl(database: str) -> None:
    writer_config = _config(database, service_role="writer")
    writer_pool = await _pool(writer_config, writer_config.domain_pool_server_settings())
    try:
        await PGAnswerRunStore(pool=writer_pool).initialize()
    finally:
        await writer_pool.close()

    reader_config = _config(database, service_role="reader")
    reader_pool = await _pool(reader_config, reader_config.lightrag_pool_server_settings())
    try:
        # A read-only session proves validation issues no DDL and no ledger write.
        await PGAnswerRunStore(pool=reader_pool).initialize(validate_only=True)
    finally:
        await reader_pool.close()


async def test_reader_startup_fails_when_a_declared_version_is_missing(database: str) -> None:
    config = _config(database, service_role="writer")
    pool = await _pool(config, config.domain_pool_server_settings())
    try:
        async with pool.acquire() as conn:
            await apply_migrations(
                conn,
                scope=ANSWER_RUN_MIGRATION_SCOPE,
                migrations=ANSWER_RUN_MIGRATIONS[:1],
            )
            await conn.execute(
                "DELETE FROM dlightrag_schema_migrations WHERE scope = $1 AND version = $2",
                ANSWER_RUN_MIGRATION_SCOPE,
                ANSWER_RUN_MIGRATIONS[0].version,
            )
            with pytest.raises(RuntimeError, match=ANSWER_RUN_MIGRATION_SCOPE):
                await verify_migrations(
                    conn,
                    scope=ANSWER_RUN_MIGRATION_SCOPE,
                    migrations=ANSWER_RUN_MIGRATIONS,
                )
    finally:
        await pool.close()
