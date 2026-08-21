# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Memory Record store: owner isolation, supersede, forget, purge."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.memory import PGAnswerMemoryStore
from dlightrag.answer.memory import MemoryProvenance, MemoryRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_PG: dict[str, Any] = dict(
    host="localhost", port=5432, user="dlightrag", password="dlightrag", database="dlightrag"
)


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def store() -> AsyncIterator[PGAnswerMemoryStore]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=4)
    created = PGAnswerMemoryStore(pool=pool)
    await created.initialize()
    try:
        yield created
    finally:
        await pool.close()
        admin = await asyncpg.connect(**_PG)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _record(*, owner: str = "alpha", body: str = "No email.") -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=str(uuid.uuid4()),
        kind="preference",
        body=body,
        confidence=0.9,
        provenance=MemoryProvenance(run_id=str(uuid.uuid4()), session_id=str(uuid.uuid4())),
        created_at=now,
        updated_at=now,
    )


async def test_pg_owners_are_isolated(store: PGAnswerMemoryStore) -> None:
    await store.insert(_record(owner="alpha", body="Alpha only."))
    await store.insert(_record(owner="beta", body="Beta only."))
    alpha = await store.list_active(owner_id="alpha")
    beta = await store.list_active(owner_id="beta")
    assert [row.body for row in alpha] == ["Alpha only."]
    assert [row.body for row in beta] == ["Beta only."]


async def test_pg_supersede_and_forget(store: PGAnswerMemoryStore) -> None:
    old = _record(body="Old preference.")
    await store.insert(old)
    new = _record(body="New preference.")
    await store.supersede(owner_id="alpha", old_id=old.memory_id, new=new)
    active = await store.list_active(owner_id="alpha")
    assert [row.body for row in active] == ["New preference."]
    superseded = await store.get(owner_id="alpha", memory_id=old.memory_id)
    assert superseded is not None
    assert superseded.status == "superseded"
    assert await store.forget(owner_id="alpha", memory_id=new.memory_id) is True
    assert await store.get(owner_id="alpha", memory_id=new.memory_id) is None
    assert await store.forget(owner_id="beta", memory_id=old.memory_id) is False


async def test_pg_purge_superseded(store: PGAnswerMemoryStore) -> None:
    old = _record(body="Stale.")
    await store.insert(old)
    await store.supersede(owner_id="alpha", old_id=old.memory_id, new=_record(body="Fresh."))
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_answer_memory_records "
            "SET updated_at = NOW() - INTERVAL '400 days' "
            "WHERE memory_id = $1",
            uuid.UUID(old.memory_id),
        )
    removed = await store.purge_superseded(older_than=datetime.now(UTC) - timedelta(days=365))
    assert removed == 1
    assert await store.get(owner_id="alpha", memory_id=old.memory_id) is None


async def test_pg_prune_write_log(store: PGAnswerMemoryStore) -> None:
    await store.insert(_record(body="Logged."))
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_answer_memory_write_log SET written_at = NOW() - INTERVAL '3 hours'"
        )
    removed = await store.prune_write_log(older_than=datetime.now(UTC) - timedelta(hours=2))
    assert removed >= 1
