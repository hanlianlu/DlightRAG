# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A lapsed lease stays reclaimable, and a live owner may re-assert it first.

The lease is the window after which a *dead* worker's Run may be taken over, not a cap on how long a
Run may last. Renewal is fenced by ownership -- the same ``lease_owner`` at the same
``fencing_epoch`` on a running row -- so a worker that stalled past the window and re-asserts its
claim before another worker's sweep reaches the row keeps the Run it was working on, and a Run that
was actually reclaimed (the epoch moved) refuses it. A stall longer than the window, with no
re-assertion, is still a Run handed over -- which is what a lease is for, and the reason the window
is sized for work that takes minutes rather than seconds. Both halves are pinned against Postgres
rather than against the statement text, because the behaviour is the contract and the SQL is only how
it is written.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from dlightrag.adapters.postgres.answer.session_repository import _INSERT_MEMORY_OPERATION_EVENT
from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.adapters.postgres.runtime._child import _parent_origin_holds_run
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.integration.test_connection_dispatch_pg import dispatch_fixture
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


async def _claimed_run(store: Any, *, key: str) -> Any:
    accepted = await store.accept_run(
        envelope=run_envelope("retrieval", key=key), run_id=str(uuid.uuid7())
    )
    claimed = await store.claim_next(
        worker_id="worker-1", run_kinds=("retrieval",), lanes=("query",)
    )
    assert claimed is not None
    assert claimed.execution.run_id == accepted.run.run_id
    return claimed


async def test_a_live_owner_re_asserts_a_lapsed_lease() -> None:
    """A stall past the window is not ownership loss: the worker still holds the row."""
    async with isolated_run_runtime("lease_lapse") as (store, pool):
        claimed = await _claimed_run(store, key="lapsed")
        # The stall: the window lapses while the worker is inside a tool that does not checkpoint.
        await pool.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second' "
            "WHERE run_id = $1",
            uuid.UUID(claimed.execution.run_id),
        )

        renewal = await store.heartbeat(
            owner_id=claimed.execution.owner_id,
            run_id=claimed.execution.run_id,
            worker_id=claimed.execution.worker_id,
            fencing_epoch=claimed.execution.fencing_epoch,
        )

        assert renewal.renewed is True
        assert (
            await pool.fetchval(
                "SELECT lease_expires_at > NOW() FROM dlightrag_runs WHERE run_id = $1",
                uuid.UUID(claimed.execution.run_id),
            )
            is True
        )


async def test_a_reclaimed_run_refuses_its_previous_owner() -> None:
    """The epoch, not the clock, decides ownership: a reclaim locks the old owner out."""
    async with isolated_run_runtime("lease_reclaim") as (store, pool):
        claimed = await _claimed_run(store, key="reclaimed")
        # A reclaim: another worker owns the row at a higher epoch, as claiming writes it.
        await pool.execute(
            "UPDATE dlightrag_runs SET lease_owner = 'worker-2', "
            "fencing_epoch = fencing_epoch + 1, lease_expires_at = NOW() + INTERVAL '300 seconds' "
            "WHERE run_id = $1",
            uuid.UUID(claimed.execution.run_id),
        )

        renewal = await store.heartbeat(
            owner_id=claimed.execution.owner_id,
            run_id=claimed.execution.run_id,
            worker_id=claimed.execution.worker_id,
            fencing_epoch=claimed.execution.fencing_epoch,
        )

        assert renewal.renewed is False


async def test_a_write_after_a_lapse_is_the_owner_s_not_the_clock_s_business() -> None:
    """The write path carries the same fence as the heartbeat: a live owner may still write.

    This is the case the runtime actually hits: a tool runs long, the window lapses, the worker comes
    back and its first durable action is a checkpoint -- not a heartbeat. Fencing that write on the
    clock handed the Run to a second worker for being busy, which is the failure the lease change
    exists to remove, so the fence is ownership alone and the write extends the lease itself.
    """
    async with isolated_run_runtime("lease_write") as (store, pool):
        claimed = await _claimed_run(store, key="write-after-lapse")
        execution = claimed.execution
        await pool.execute(
            "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second' "
            "WHERE run_id = $1",
            uuid.UUID(execution.run_id),
        )

        wrote = await store.write_checkpoint(
            owner_id=execution.owner_id,
            run_id=execution.run_id,
            worker_id=execution.worker_id,
            fencing_epoch=execution.fencing_epoch,
            checkpoint={"step": 1},
        )
        appended = await store.append_event(
            owner_id=execution.owner_id,
            run_id=execution.run_id,
            worker_id=execution.worker_id,
            fencing_epoch=execution.fencing_epoch,
            phase=None,
            event_type="progress",
            payload={"phase": "researching"},
        )

        assert wrote is True
        assert appended is not None
        assert (
            await pool.fetchval(
                "SELECT lease_expires_at > NOW() FROM dlightrag_runs WHERE run_id = $1",
                uuid.UUID(execution.run_id),
            )
            is True
        )


async def test_a_reclaimed_run_refuses_its_previous_owner_s_writes() -> None:
    """Ownership, not the clock: the epoch locks a stale worker out of every write."""
    async with isolated_run_runtime("lease_write_reclaim") as (store, pool):
        claimed = await _claimed_run(store, key="write-after-reclaim")
        execution = claimed.execution
        await pool.execute(
            "UPDATE dlightrag_runs SET lease_owner = 'worker-2', "
            "fencing_epoch = fencing_epoch + 1, lease_expires_at = NOW() + INTERVAL '300 seconds' "
            "WHERE run_id = $1",
            uuid.UUID(execution.run_id),
        )

        wrote = await store.write_checkpoint(
            owner_id=execution.owner_id,
            run_id=execution.run_id,
            worker_id=execution.worker_id,
            fencing_epoch=execution.fencing_epoch,
            checkpoint={"step": 2},
        )
        appended = await store.append_event(
            owner_id=execution.owner_id,
            run_id=execution.run_id,
            worker_id=execution.worker_id,
            fencing_epoch=execution.fencing_epoch,
            phase=None,
            event_type="progress",
            payload={"phase": "researching"},
        )

        assert wrote is False
        assert appended is None


async def test_the_fencing_epoch_alone_refuses_a_renewal() -> None:
    """Ownership is owner *and* epoch: a reclaim that kept the owner still locks the old worker out.

    Pinned separately because a test that changes the owner as well passes on the owner predicate
    alone, and would not notice an epoch comparison relaxed from ``=`` to ``>=``.
    """
    async with isolated_run_runtime("lease_epoch") as (store, pool):
        claimed = await _claimed_run(store, key="epoch-only")
        execution = claimed.execution
        await pool.execute(
            "UPDATE dlightrag_runs SET fencing_epoch = fencing_epoch + 1 WHERE run_id = $1",
            uuid.UUID(execution.run_id),
        )

        renewal = await store.heartbeat(
            owner_id=execution.owner_id,
            run_id=execution.run_id,
            worker_id=execution.worker_id,
            fencing_epoch=execution.fencing_epoch,
        )

        assert renewal.renewed is False


_LAPSE = "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second'"
_RECLAIM = "UPDATE dlightrag_runs SET fencing_epoch = fencing_epoch + 1"


async def test_a_nonterminal_event_needs_an_owner_not_an_unexpired_lease() -> None:
    """The events trigger guards integrity; who owns the Run is the writing statement's predicate.

    The memory operation event is a write that does not extend the lease, so after a lapse it is
    the case that reaches the trigger with a stale clock. A live owner's event is accepted and the
    lease stays lapsed; once a reclaim moved the epoch the same statement writes nothing.
    """
    async with isolated_run_runtime("lease_event") as (store, pool):
        claimed = await _claimed_run(store, key="event-after-lapse")
        execution = claimed.execution
        run_id = uuid.UUID(execution.run_id)
        await pool.execute(_LAPSE)

        async def settle(epoch: int) -> str:
            return await pool.execute(
                _INSERT_MEMORY_OPERATION_EVENT,
                execution.owner_id,
                run_id,
                execution.worker_id,
                epoch,
                json.dumps({"operation": "remember"}),
            )

        assert await settle(execution.fencing_epoch) == "INSERT 0 1"
        assert (
            await pool.fetchval(
                "SELECT lease_expires_at < NOW() FROM dlightrag_runs WHERE run_id = $1", run_id
            )
            is True
        )

        await pool.execute(_RECLAIM)

        assert await settle(execution.fencing_epoch) == "INSERT 0 0"


async def test_a_parent_origin_holds_the_run_by_ownership_not_by_the_clock() -> None:
    """A parent's control of its Child is honoured while the same owner and epoch hold the Run."""
    async with isolated_run_runtime("lease_origin") as (store, pool):
        claimed = await _claimed_run(store, key="origin-after-lapse")
        execution = claimed.execution
        run_id = uuid.UUID(execution.run_id)

        async def holds() -> bool:
            row = await pool.fetchrow(
                "SELECT status, lease_owner, lease_expires_at, fencing_epoch"
                " FROM dlightrag_runs WHERE run_id = $1",
                run_id,
            )
            return _parent_origin_holds_run(
                row, worker_id=execution.worker_id, fencing_epoch=execution.fencing_epoch
            )

        await pool.execute(_LAPSE)
        assert await holds() is True

        await pool.execute(_RECLAIM)
        assert await holds() is False


async def test_a_connection_effect_keeps_its_execution_by_ownership_not_by_the_clock() -> None:
    """Whether an external effect may run again is the pending effect's business, not the clock's."""
    async with isolated_run_runtime("lease_connection") as (runs, pool):
        *_, claim, _tool, runtime = await dispatch_fixture(runs, pool)

        async def live() -> bool:
            async with pool.acquire() as conn:
                return await PGConnectionsStore._live_execution(
                    conn, claim=claim, runtime=runtime, lock=False
                )

        await pool.execute(_LAPSE)
        assert await live() is True

        await pool.execute(_RECLAIM)
        assert await live() is False
