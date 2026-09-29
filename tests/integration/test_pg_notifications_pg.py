# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The notification hub against a real PostgreSQL: delivery, and a lost backend replaced."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.core import _notifications
from dlightrag.adapters.postgres.core._channels import MODEL_CATALOGUE_CHANNEL, RUN_CANCEL_CHANNEL
from tests.support.pg import PG_CONN_KWARGS, drop_database, notification_hub, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# The hub's backend, other than one already terminated: the last thing it ran was a LISTEN.
_HUB_BACKENDS = """SELECT pid FROM pg_stat_activity
WHERE datname = current_database() AND pid NOT IN (pg_backend_pid(), $1)
  AND query LIKE 'LISTEN %'"""


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_notifications_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=2
    )
    try:
        yield created
    finally:
        await created.close()
        await drop_database(database)


async def _until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_a_notify_reaches_the_subscribers_of_its_channel_only(pool: Any) -> None:
    cancels: list[str | None] = []
    catalogue: list[str | None] = []
    async with notification_hub(pool) as hub:
        hub.subscribe(RUN_CANCEL_CHANNEL, cancels.append)
        hub.subscribe(MODEL_CATALOGUE_CHANNEL, catalogue.append)
        await _until(lambda: cancels == [None] and catalogue == [None])

        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_notify($1, $2)", RUN_CANCEL_CHANNEL, "wake")
        await _until(lambda: cancels == [None, "wake"])

    assert catalogue == [None]


async def test_a_terminated_backend_is_replaced_listened_again_and_resynchronized(
    pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_notifications, "_RECONNECT_BASE_SECONDS", 0.05)
    received: list[str | None] = []
    async with notification_hub(pool) as hub, pool.acquire() as connection:
        hub.subscribe(RUN_CANCEL_CHANNEL, received.append)
        await _until(lambda: received == [None])
        (backend,) = [row["pid"] for row in await connection.fetch(_HUB_BACKENDS, 0)]

        assert await connection.fetchval("SELECT pg_terminate_backend($1)", backend)
        await _until(lambda: received == [None, None])
        assert len(await connection.fetch(_HUB_BACKENDS, backend)) == 1
        await connection.execute("SELECT pg_notify($1, $2)", RUN_CANCEL_CHANNEL, "after")
        await _until(lambda: received == [None, None, "after"])
