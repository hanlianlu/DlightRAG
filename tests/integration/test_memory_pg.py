# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Profile Memory records, journal, atomic receipts, undo, and recall."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import Memory, MemoryProvenance, MemoryRecord
from dlightrag_memory._storage.pg_bm25 import index_name
from dlightrag_memory.postgres import PostgresMemoryStore

from dlightrag.adapters.postgres.memory_settings import (
    MEMORY_SETTINGS_DDL,
    PGMemorySettingsStore,
)
from dlightrag.answer.errors import MemoryDisabledError, MemoryWriteRejectedError
from dlightrag.services.memory import MemoryService

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
async def store() -> AsyncIterator[PostgresMemoryStore]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")
    db_name = f"dlightrag_mem_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(**{**_PG, "database": db_name}, min_size=1, max_size=4)
    created = PostgresMemoryStore(pool=pool)
    await created.initialize()
    try:
        yield created
    finally:
        await created.aclose()
        await pool.close()
        admin = await asyncpg.connect(**_PG)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _provenance(run: str = "run-1") -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run", origin_id=run, run_id=run, session_id="session-1"
    )


def _record(*, owner: str = "alpha", body: str = "No email.") -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=str(uuid.uuid4()),
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


async def test_pg_owners_are_isolated(store: PostgresMemoryStore) -> None:
    await store.insert(_record(owner="alpha", body="Alpha only."))
    await store.insert(_record(owner="beta", body="Beta only."))
    assert [row.body for row in await store.list_active(owner_id="alpha")] == ["Alpha only."]
    assert [row.body for row in await store.list_active(owner_id="beta")] == ["Beta only."]


async def test_pg_initialization_rejects_legacy_confidence_schema(
    store: PostgresMemoryStore,
) -> None:
    pool = store._operation_pool
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE dlightrag_memory_records "
            "ADD COLUMN confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0"
        )
    with pytest.raises(RuntimeError, match="removed confidence"):
        await store.initialize()


async def test_pg_operation_replay_duplicate_cap_and_schema(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    first = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    replay = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="call-1",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    duplicate = await memory.remember(
        owner_id="alpha",
        kind="preference",
        body="  use chinese. ",
        provenance=_provenance(),
        idempotency_key="call-2",
        mutation_scope="run-1",
        mutation_limit=1,
    )
    assert replay == first
    assert duplicate.outcome == "unchanged"
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Lives in Gothenburg.",
            provenance=_provenance(),
            idempotency_key="call-3",
            mutation_scope="run-1",
            mutation_limit=1,
        )
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM dlightrag_memory_operations WHERE owner_id = 'alpha'"
            )
            == 2
        )
        assert not await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'dlightrag_memory_records' AND column_name = 'confidence'"
        )


async def test_pg_owner_lock_rechecks_deactivation_before_mutation_commit(
    store: PostgresMemoryStore,
) -> None:
    pool = store._operation_pool
    assert pool is not None
    async with pool.acquire() as conn:
        for statement in MEMORY_SETTINGS_DDL:
            await conn.execute(statement)
    service = MemoryService(store, settings_store=PGMemorySettingsStore(pool=pool))

    async with pool.acquire() as conn, conn.transaction():
        await conn.fetchval("SELECT pg_advisory_xact_lock(hashtext($1))", "alpha")
        await conn.execute(
            "INSERT INTO dlightrag_answer_memory_settings "
            "(owner_id, enabled, epoch) VALUES ($1, FALSE, 1)",
            "alpha",
        )
        pending = asyncio.create_task(
            service.remember(
                owner_id="alpha",
                auth_mode="jwt",
                kind="fact",
                body="Stable.",
                provenance=_provenance(),
                idempotency_key="call-after-disable",
            )
        )
        await asyncio.sleep(0.05)
        assert not pending.done()

    with pytest.raises(MemoryDisabledError):
        await pending
    assert await store.count_active(owner_id="alpha") == 0


async def test_pg_supersede_forget_and_compensating_undo(store: PostgresMemoryStore) -> None:
    memory = Memory(store)
    old = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Beijing.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    replacement = await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Lives in Gothenburg.",
        provenance=_provenance(),
        idempotency_key="call-2",
        supersedes_id=old.memory_id,
    )
    undone = await memory.undo(
        owner_id="alpha",
        change_id=replacement.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    assert undone.outcome == "changed"
    assert [row.body for row in await memory.list_active(owner_id="alpha")] == ["Lives in Beijing."]

    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=undone.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    restored = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-2"),
        idempotency_key="undo-2",
    )
    assert restored.outcome == "changed"
    assert [row.body for row in await memory.list_active(owner_id="alpha")] == ["Lives in Beijing."]


async def test_pg_clear_physically_erases_records_and_operations(
    store: PostgresMemoryStore,
) -> None:
    memory = Memory(store)
    await memory.remember(
        owner_id="alpha",
        kind="fact",
        body="Stable.",
        provenance=_provenance(),
        idempotency_key="call-1",
    )
    assert await memory.clear(owner_id="alpha") == 1
    assert await memory.list_active(owner_id="alpha") == ()


async def test_pg_purge_expired_non_active_rows(store: PostgresMemoryStore) -> None:
    old = _record(body="Stale.")
    await store.insert(old)
    await store.supersede(owner_id="alpha", old_id=old.memory_id, new=_record(body="Fresh."))
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        await conn.execute(
            "UPDATE dlightrag_memory_records "
            "SET updated_at = NOW() - INTERVAL '400 days' WHERE memory_id = $1",
            uuid.UUID(old.memory_id),
        )
    removed = await store.purge_superseded(older_than=datetime.now(UTC) - timedelta(days=365))
    assert removed == 1
    assert await store.get(owner_id="alpha", memory_id=old.memory_id) is None


async def test_pg_recall_legs_find_only_the_owner(store: PostgresMemoryStore) -> None:
    await store.insert(_record(owner="alpha", body="No email."))
    await store.insert(_record(owner="beta", body="No email."))
    await store.insert(_record(owner="alpha", body="Deploy at midnight."))

    candidates = await store.search_candidates(owner_id="alpha", query="No email", limit=10)
    assert "No email." in {candidate.record.body for candidate in candidates}
    assert all(candidate.record.owner_id == "alpha" for candidate in candidates)


async def test_pg_bm25_indexes_are_provisioned(store: PostgresMemoryStore) -> None:
    async with store._operation_pool.acquire() as conn:  # type: ignore[union-attr]
        rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'dlightrag_memory_records' "
            "AND indexname LIKE $1",
            f"{index_name('simple')}%",
        )
        assert index_name("simple") in {str(row["indexname"]) for row in rows}
