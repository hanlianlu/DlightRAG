# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL CAS persistence and NOTIFY synchronization for model catalogue overlays."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.model_catalogue import PGModelCatalogueStore
from dlightrag.engine.ai.catalog import catalogue_overlay_revision
from tests.support.pg import PG_CONN_KWARGS, drop_database, notification_hub, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_catalogue_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": database}, min_size=1, max_size=4
    )
    try:
        yield created
    finally:
        await created.close()
        await drop_database(database)


async def test_writer_initializes_singleton_and_publish_is_compare_and_set(pool: Any) -> None:
    store = PGModelCatalogueStore(initial_revision=catalogue_overlay_revision(()), pool=pool)

    await store.initialize(validate_only=False)
    initial = await store.load()
    changed_revision = "sha256:" + "1" * 64

    assert initial.revision == catalogue_overlay_revision(())
    assert initial.overlay == []
    assert await store.publish(
        expected_revision=initial.revision,
        revision=changed_revision,
        overlay=[],
        actor="integration",
    )
    assert not await store.publish(
        expected_revision=initial.revision,
        revision="sha256:" + "2" * 64,
        overlay=[],
        actor="stale",
    )
    assert (await store.load()).revision == changed_revision


async def test_notify_listener_synchronizes_after_publish(pool: Any) -> None:
    writer = PGModelCatalogueStore(initial_revision=catalogue_overlay_revision(()), pool=pool)
    await writer.initialize(validate_only=False)
    synchronized = asyncio.Event()
    calls = 0

    async def on_change() -> None:
        nonlocal calls
        calls += 1
        synchronized.set()

    async with notification_hub(pool) as hub:
        listener = PGModelCatalogueStore(
            initial_revision=catalogue_overlay_revision(()), pool=pool, notifications=hub
        )
        await listener.start_listener(on_change)
        await asyncio.wait_for(synchronized.wait(), timeout=5)
        synchronized.clear()
        initial = await writer.load()
        assert await writer.publish(
            expected_revision=initial.revision,
            revision="sha256:" + "3" * 64,
            overlay=[],
            actor="integration",
        )

        await asyncio.wait_for(synchronized.wait(), timeout=5)
        assert calls >= 2
        await listener.aclose()
