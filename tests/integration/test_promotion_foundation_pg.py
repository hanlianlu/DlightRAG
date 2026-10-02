# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL integration coverage for the promotion control-plane schema.

Compact real-PostgreSQL fixtures prove the durable registry control-plane
fields and the promotion-job table: leased claims, retry backoff, fenced
transitions, legal-state constraints, and the bounded claim indexes. Jobs are
queued the way a corpus mutation window queues them; only the adapter
interfaces drive them from there. The worker has its own suite.
"""

import datetime
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres
from tests.support.promotion import queue_promotion

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_WORKSPACE = "pf_promotion_ws"
_WORKSPACE_B = "pf_promotion_ws_b"
_WORKSPACE_C = "pf_promotion_ws_c"


@pytest.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    await skip_without_postgres()
    database = f"dlightrag_promotion_foundation_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=2
    )
    try:
        yield pool
    finally:
        await pool.close()
        await drop_database(database)


async def test_registry_control_plane_fields_are_durable_and_constrained(pool: Any) -> None:
    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    # Simulate the pre-foundation registry upgrade path: columns may exist
    # while the new checks do not. Replaying the final migration must make
    # the writer-created schema pass the same catalog contract readers use.
    from dlightrag.adapters.postgres.corpus import workspaces

    constraint_names = [name for name, _expression in workspaces._WORKSPACE_CHECK_EXPRESSIONS]
    async with pool.acquire() as conn:
        for name in constraint_names:
            await conn.execute(f"ALTER TABLE dlightrag_workspace_meta DROP CONSTRAINT {name}")
        await conn.execute(
            "DELETE FROM dlightrag_schema_migrations "
            "WHERE scope = 'workspace_registry' "
            "AND version = 'workspace_meta_promotion_constraints'"
        )
    await registry.initialize()
    async with pool.acquire() as conn:
        installed = {
            str(row["conname"])
            for row in await conn.fetch(
                "SELECT conname FROM pg_catalog.pg_constraint "
                "WHERE conrelid = 'dlightrag_workspace_meta'::regclass"
            )
        }
    assert set(constraint_names) <= installed

    await registry.upsert(
        workspace=_WORKSPACE,
        display_name="Promotion Workspace",
        embedding_model="pf-it-fake",
    )
    row = await registry.get_row(_WORKSPACE)
    assert row is not None
    assert row["ingested_docs_total"] == 0
    assert row["ingested_chunks_total"] == 0
    assert row["storage_tier"] == "shared"
    assert row["promotion_state"] == "none"

    # Creating a workspace never renames one that exists.
    assert not await registry.insert(
        workspace=_WORKSPACE,
        display_name="Renamed",
        embedding_model="pf-it-fake",
        created_by="owner-b",
    )
    row = await registry.get_row(_WORKSPACE)
    assert row is not None and row["display_name"] == "Promotion Workspace"
    created = "pf_registry_created"
    assert await registry.insert(
        workspace=created,
        display_name="Created",
        embedding_model="pf-it-fake",
        created_by="owner-a",
    )
    assert await registry.exists(created)
    # Only a created workspace names its creator; the registered one has none.
    assert await registry.workspace_creators([created, _WORKSPACE]) == {created: "owner-a"}
    assert await registry.delete(created)
    assert not await registry.exists(created)
    assert not await registry.delete(created)

    # Promotion observability transitions, with retry bookkeeping.
    assert await registry.set_promotion_state(workspace=_WORKSPACE, state="pending")
    retry_at = datetime.datetime.now(datetime.UTC)
    assert await registry.set_promotion_state(
        workspace=_WORKSPACE,
        state="failed",
        error="cutover invariant mismatch",
        next_retry_at=retry_at,
    )
    row = await registry.get_row(_WORKSPACE)
    assert row is not None
    assert row["promotion_state"] == "failed"
    assert row["promotion_last_error"] == "cutover invariant mismatch"
    assert row["promotion_retry_count"] == 1
    assert row["promotion_next_retry_at"] is not None

    # Write fence: expired requested leases are rejected; valid leases can
    # be acquired, extended, and released only by their owner token.
    assert not await registry.acquire_write_fence(
        workspace=_WORKSPACE,
        owner="stale-worker",
        until=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1),
    )
    until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)
    assert await registry.acquire_write_fence(workspace=_WORKSPACE, owner="worker-1", until=until)
    assert not await registry.acquire_write_fence(
        workspace=_WORKSPACE, owner="worker-2", until=until
    )
    assert not await registry.release_write_fence(workspace=_WORKSPACE, owner="worker-2")
    row = await registry.get_row(_WORKSPACE)
    assert row is not None
    assert (row["write_fence_owner"], row["write_fence_until"]) == ("worker-1", until)
    assert await registry.release_write_fence(workspace=_WORKSPACE, owner="worker-1")
    row = await registry.get_row(_WORKSPACE)
    assert row is not None
    assert (row["write_fence_owner"], row["write_fence_until"]) == (None, None)

    # Legal-state constraints reject impossible registry states outright.
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                UPDATE dlightrag_workspace_meta SET storage_tier = 'promoting'
                WHERE workspace = $1
                """,
                _WORKSPACE,
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                UPDATE dlightrag_workspace_meta
                SET promotion_next_retry_at = NOW(), promotion_state = 'none'
                WHERE workspace = $1
                """,
                _WORKSPACE,
            )


async def test_promotion_jobs_are_leased_fenced_and_retried_after_backoff(pool: Any) -> None:
    from dlightrag.adapters.postgres.corpus.promotion_jobs import (
        PGPromotionJobStore,
        mark_done_in,
        mark_failed_in,
    )
    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

    def lease() -> datetime.datetime:
        # Each claim call asks for an expiry of its own: the expiry is that call's
        # replay token.
        return datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)

    async def mark_done(**identity: Any) -> bool:
        async with pool.acquire() as conn, conn.transaction():
            return await mark_done_in(conn, **identity)

    async def mark_failed(**identity: Any) -> bool:
        async with pool.acquire() as conn, conn.transaction():
            return await mark_failed_in(
                conn, error="promotion staging verification failed", **identity
            )

    async def job(job_id: int) -> dict[str, Any]:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state, lease_owner, lease_generation, attempt_count, last_error"
                " FROM dlightrag_promotion_jobs WHERE job_id = $1",
                job_id,
            )
        assert row is not None
        return dict(row)

    store = PGPromotionJobStore(pool=pool)
    await store.initialize()
    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    for workspace in (_WORKSPACE_B, _WORKSPACE_C):
        await registry.upsert(
            workspace=workspace, display_name=workspace, embedding_model="pf-it-fake"
        )
    await queue_promotion(_WORKSPACE_B, pool=pool)

    claimed = await store.claim_next(owner="worker-1", lease_until=lease())
    assert claimed is not None
    assert claimed["workspace"] == _WORKSPACE_B
    assert claimed["attempt_count"] == 1
    job_id = int(claimed["job_id"])
    generation = int(claimed["lease_generation"])

    # An unexpired lease is never taken over: another worker finds nothing to claim.
    assert await store.claim_next(owner="worker-2", lease_until=lease()) is None
    assert await job(job_id) == {
        "state": "promoting",
        "lease_owner": "worker-1",
        "lease_generation": generation,
        "attempt_count": 1,
        "last_error": None,
    }

    # Only the current owner and generation extend the lease, and only forward.
    later = lease() + datetime.timedelta(minutes=5)
    past = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
    assert await store.renew_lease(
        job_id=job_id, owner="worker-1", lease_generation=generation, lease_until=later
    )
    assert not await store.renew_lease(
        job_id=job_id, owner="worker-2", lease_generation=generation, lease_until=later
    )
    assert not await store.renew_lease(
        job_id=job_id, owner="worker-1", lease_generation=generation + 1, lease_until=later
    )
    assert not await store.renew_lease(
        job_id=job_id, owner="worker-1", lease_generation=generation, lease_until=past
    )

    # Owner + monotonically increasing generation + lease time form the
    # fencing identity. A different owner cannot finish this attempt.
    assert not await mark_done(job_id=job_id, owner="worker-2", lease_generation=generation)
    assert await mark_done(job_id=job_id, owner="worker-1", lease_generation=generation)

    # A failed job is not claimed again before its retry time.
    await queue_promotion(_WORKSPACE_C, pool=pool)
    claimed = await store.claim_next(owner="worker-1", lease_until=lease())
    assert claimed is not None
    assert claimed["workspace"] == _WORKSPACE_C
    retry_job_id = int(claimed["job_id"])
    retry_generation = int(claimed["lease_generation"])
    assert await mark_failed(
        job_id=retry_job_id,
        owner="worker-1",
        lease_generation=retry_generation,
        next_retry_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
    )
    assert await store.claim_next(owner="worker-1", lease_until=lease()) is None
    assert await job(retry_job_id) == {
        "state": "failed",
        "lease_owner": None,
        "lease_generation": retry_generation,
        "attempt_count": 1,
        "last_error": "promotion staging verification failed",
    }

    # Once the backoff has passed, the same job is retried under a new generation.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_promotion_jobs SET next_retry_at = NOW() - INTERVAL '1 second'"
            " WHERE job_id = $1",
            retry_job_id,
        )
    reclaimed = await store.claim_next(owner="worker-1", lease_until=lease())
    assert reclaimed is not None
    assert int(reclaimed["job_id"]) == retry_job_id
    assert int(reclaimed["attempt_count"]) == 2
    assert int(reclaimed["lease_generation"]) == retry_generation + 1

    # Even the same process owner cannot let its stale attempt fail or finish
    # the row after it has been reclaimed with a newer generation.
    assert not await mark_failed(
        job_id=retry_job_id,
        owner="worker-1",
        lease_generation=retry_generation,
        next_retry_at=datetime.datetime.now(datetime.UTC),
    )
    assert not await mark_done(
        job_id=retry_job_id, owner="worker-1", lease_generation=retry_generation
    )
    assert await job(retry_job_id) == {
        "state": "promoting",
        "lease_owner": "worker-1",
        "lease_generation": retry_generation + 1,
        "attempt_count": 2,
        "last_error": None,
    }
    assert await mark_done(
        job_id=retry_job_id, owner="worker-1", lease_generation=retry_generation + 1
    )

    # Legal-state constraints reject impossible rows outright.
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_promotion_jobs (workspace, state, lease_owner) "
                "VALUES ('pf_bad_lease', 'promoting', NULL)"
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_promotion_jobs "
                "(workspace, state, last_error, next_retry_at) "
                "VALUES ('pf_bad_error', 'failed', NULL, NOW())"
            )
