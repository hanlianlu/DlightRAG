# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for durable Answer run state on PostgreSQL 18.

Exercises the real contract against a live database: the declared schema and its
constraints, owner-scoped creation and idempotency, queued cancellation, slot-safe
claiming across workers, lease fencing, gap-free event sequences, checkpoint
compare-and-set, terminal transitions, graceful requeue, the crash-recovery bound,
event trimming, retention pruning, and ownership-safe artifact cleanup.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.storage.answer_runs import (
    MAX_CONSECUTIVE_RECOVERIES,
    RUN_ABANDONED_ERROR_KIND,
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    PGAnswerRunStore,
)

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

_OWNER = "owner-alpha"
_OTHER_OWNER = "owner-beta"
_WORKER = "worker-1"


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    """Provision an isolated throwaway database and yield a pool bound to it."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_runs_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def store(pool: Any) -> PGAnswerRunStore:
    created = PGAnswerRunStore(pool=pool)
    await created.initialize()
    # Retention exempts conversation-linked runs, so the whole operational schema
    # is established here exactly as a real process establishes it at startup.
    from dlightrag.storage.web_conversations import PGWebConversationStore

    await PGWebConversationStore(pool=pool, run_store=created).initialize()
    return created


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {"query": query, "workspaces": ["alpha"], **extra}


async def _expire_lease(pool: Any, run_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs "
            "SET lease_expires_at = NOW() - INTERVAL '1 second' WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _backdate_finish(pool: Any, run_id: str, *, days: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs "
            "SET finished_at = NOW() - ($2 * INTERVAL '1 day') WHERE run_id = $1",
            uuid.UUID(run_id),
            days,
        )


async def _event_types(pool: Any, run_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT event_type FROM dlightrag_answer_run_events "
            "WHERE run_id = $1 ORDER BY event_sequence",
            uuid.UUID(run_id),
        )
    return [str(row["event_type"]) for row in rows]


async def _checkpoint(pool: Any, run_id: str) -> str | None:
    """Read the raw retained checkpoint; the record type deliberately hides it."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT checkpoint_json FROM dlightrag_answer_runs WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _claimed(store: PGAnswerRunStore, *, worker_id: str = _WORKER) -> Any:
    claim = await store.claim_next(worker_id=worker_id)
    assert claim is not None
    return claim


def _delete_action(value: Any) -> str:
    """``pg_constraint.confdeltype`` is a one-byte ``"char"`` column."""
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    async def test_creates_exactly_the_four_contract_tables(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'dlightrag_answer%'"
            )
        assert {str(row["table_name"]) for row in rows} == {
            "dlightrag_answer_runs",
            "dlightrag_answer_run_events",
            "dlightrag_answer_artifacts",
            "dlightrag_answer_run_artifacts",
        }

    async def test_run_columns_match_the_contract(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                "dlightrag_answer_runs",
            )
        assert {str(row["column_name"]) for row in rows} == {
            "owner_id",
            "run_id",
            "idempotency_key",
            "request_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "completed_turns",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "recovery_count",
            "next_event_sequence",
            "events_trimmed_at",
            "checkpoint_json",
            "result_json",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        }

    async def test_foreign_keys_cascade_runs_and_restrict_blobs(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.conrelid::regclass::text AS child,
                       c.confrelid::regclass::text AS parent,
                       c.confdeltype AS on_delete
                FROM pg_constraint AS c
                WHERE c.contype = 'f'
                  AND c.conrelid::regclass::text LIKE 'dlightrag_answer%'
                """
            )
        actions = {
            (str(row["child"]), str(row["parent"])): _delete_action(row["on_delete"])
            for row in rows
        }
        assert actions[("dlightrag_answer_run_events", "dlightrag_answer_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_answer_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_answer_artifacts")] == "r"

    async def test_rejects_unknown_status_phase_and_event_type(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "UPDATE dlightrag_answer_runs SET status = 'paused' WHERE run_id = $1", run_id
                )
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "UPDATE dlightrag_answer_runs SET phase = 'polishing' WHERE run_id = $1", run_id
                )
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_answer_run_events "
                    "(owner_id, run_id, event_sequence, event_type) VALUES ($1, $2, 1, 'thinking')",
                    _OWNER,
                    run_id,
                )

    async def test_allows_only_one_terminal_event_per_run(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_answer_run_events "
                "(owner_id, run_id, event_sequence, event_type) VALUES ($1, $2, 1, 'done')",
                _OWNER,
                run_id,
            )
            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_answer_run_events "
                    "(owner_id, run_id, event_sequence, event_type) VALUES ($1, $2, 2, 'error')",
                    _OWNER,
                    run_id,
                )

    async def test_preserves_ingest_job_migration_scope(self, store, pool) -> None:
        from dlightrag.storage.ingest_jobs import PGIngestJobStore

        await PGIngestJobStore(pool=pool).initialize()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT scope, version FROM dlightrag_schema_migrations")
        recorded = {(str(row["scope"]), str(row["version"])) for row in rows}
        assert ("answer_runs", "0001_answer_runs") in recorded
        assert ("ingest_jobs", "0001_ingest_jobs") in recorded


# ---------------------------------------------------------------------------
# Creation, idempotency, owner scoping
# ---------------------------------------------------------------------------


class TestCreation:
    async def test_creates_queued_run_with_uuid7_identity(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        assert creation.replayed is False
        assert uuid.UUID(creation.run.run_id).version == 7
        assert creation.run.status == "queued"
        assert creation.run.next_event_sequence == 1
        assert creation.run.request == _request()

    async def test_run_ids_are_unique_and_time_ordered(self, store) -> None:
        run_ids = [
            uuid.UUID((await store.create_run(owner_id=_OWNER, request=_request())).run.run_id)
            for _ in range(16)
        ]
        assert run_ids == sorted(run_ids)
        assert len(set(run_ids)) == len(run_ids)

    async def test_replays_same_key_and_input(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        second = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        assert second.replayed is True
        assert second.run.run_id == first.run.run_id

    async def test_replay_returns_current_status_not_queued(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        await _claimed(store)
        replay = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        assert replay.run.run_id == first.run.run_id
        assert replay.run.status == "running"

    async def test_rejects_same_key_with_different_input(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        with pytest.raises(IdempotencyKeyConflict):
            await store.create_run(owner_id=_OWNER, request=_request("other"), idempotency_key="k1")

    async def test_replay_normalizes_key_order_but_not_list_order(self, store) -> None:
        first = await store.create_run(
            owner_id=_OWNER,
            request={"query": "why", "workspaces": ["alpha", "beta"]},
            idempotency_key="k1",
        )
        reordered = await store.create_run(
            owner_id=_OWNER,
            request={"workspaces": ["alpha", "beta"], "query": "why"},
            idempotency_key="k1",
        )
        assert reordered.replayed is True
        assert reordered.run.run_id == first.run.run_id
        with pytest.raises(IdempotencyKeyConflict):
            await store.create_run(
                owner_id=_OWNER,
                request={"query": "why", "workspaces": ["beta", "alpha"]},
                idempotency_key="k1",
            )

    async def test_scopes_idempotency_keys_per_owner(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        second = await store.create_run(
            owner_id=_OTHER_OWNER, request=_request(), idempotency_key="k1"
        )
        assert second.replayed is False
        assert second.run.run_id != first.run.run_id

    async def test_creation_without_key_always_creates_a_new_run(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request())
        second = await store.create_run(owner_id=_OWNER, request=_request())
        assert first.run.run_id != second.run.run_id

    async def test_foreign_owner_and_unknown_ids_read_identically(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        assert await store.get_run(owner_id=_OTHER_OWNER, run_id=creation.run.run_id) is None
        assert await store.get_run(owner_id=_OWNER, run_id=str(uuid.uuid7())) is None
        assert await store.get_run(owner_id=_OWNER, run_id="not-a-uuid") is None
        assert await store.read_event_page(owner_id=_OTHER_OWNER, run_id=creation.run.run_id) == ()

    async def test_rejects_empty_owner(self, store) -> None:
        with pytest.raises(ValueError):
            await store.create_run(owner_id="  ", request=_request())


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_queued_run_cancels_in_one_transaction(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert outcome.outcome == "cancelled"
        assert outcome.run is not None
        assert outcome.run.status == "cancelled"
        assert outcome.run.finished_at is not None
        assert await _event_types(pool, creation.run.run_id) == ["done"]
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].sequence == 1
        assert events[0].payload == {"status": "cancelled"}

    async def test_running_run_records_pending_request(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert outcome.outcome == "pending"
        assert outcome.run is not None
        assert outcome.run.status == "running"
        assert outcome.run.cancel_requested is True

    async def test_cancelling_a_terminal_run_is_a_no_op(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        repeat = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert repeat.outcome == "already_terminal"
        assert repeat.run is not None
        assert repeat.run.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_unknown_and_foreign_runs_cancel_identically(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        foreign = await store.request_cancellation(
            owner_id=_OTHER_OWNER, run_id=creation.run.run_id
        )
        unknown = await store.request_cancellation(owner_id=_OWNER, run_id=str(uuid.uuid7()))
        assert foreign == unknown
        assert foreign.outcome == "unknown"

    async def test_success_yields_to_a_cancellation_that_won_the_row(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "hello"},
        )
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert await _event_types(pool, creation.run.run_id) == ["done"]


# ---------------------------------------------------------------------------
# Claim, lease, fencing
# ---------------------------------------------------------------------------


class TestClaiming:
    async def test_claims_the_oldest_queued_run_first(self, store) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request("a"))
        second = await store.create_run(owner_id=_OWNER, request=_request("b"))
        claim = await _claimed(store)
        assert claim.run.run_id == first.run.run_id
        assert claim.run.status == "running"
        assert claim.run.lease_owner == _WORKER
        assert claim.run.fencing_epoch == 1
        assert claim.run.started_at is not None
        assert (await _claimed(store, worker_id="worker-2")).run.run_id == second.run.run_id

    async def test_concurrent_workers_never_share_a_row(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request("a"))
        await store.create_run(owner_id=_OWNER, request=_request("b"))
        claims = await asyncio.gather(
            *(store.claim_next(worker_id=f"worker-{index}") for index in range(6))
        )
        claimed = [claim for claim in claims if claim is not None]
        assert len(claimed) == 2
        assert len({claim.run.run_id for claim in claimed}) == 2

    async def test_returns_none_when_no_row_is_eligible(self, store) -> None:
        assert await store.claim_next(worker_id=_WORKER) is None

    async def test_does_not_claim_a_live_lease(self, store) -> None:
        await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        assert await store.claim_next(worker_id="worker-2") is None

    async def test_reclaims_an_expired_lease_with_a_new_epoch(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        reclaim = await _claimed(store, worker_id="worker-2")
        assert reclaim.run.run_id == creation.run.run_id
        assert reclaim.run.fencing_epoch == 2
        assert reclaim.run.recovery_count == 1
        assert reclaim.run.lease_owner == "worker-2"

    async def test_skips_cancel_pending_rows(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert await store.claim_next(worker_id="worker-2") is None

    async def test_racing_hosts_reclaim_one_expired_lease_once(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        claims = await asyncio.gather(
            *(store.claim_next(worker_id=f"host-{index}") for index in range(5))
        )
        winners = [claim for claim in claims if claim is not None]
        assert len(winners) == 1
        assert winners[0].run.fencing_epoch == 2
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.lease_owner == winners[0].run.lease_owner
        assert record.recovery_count == 1

    async def test_restores_the_latest_checkpoint_on_reclaim(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.commit_checkpoint(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            version=3,
            state={"evidence": ["a"]},
        )
        await _expire_lease(pool, creation.run.run_id)
        reclaim = await _claimed(store, worker_id="worker-2")
        assert reclaim.checkpoint is not None
        assert reclaim.checkpoint.version == 3
        assert reclaim.checkpoint.completed_turns == 1
        assert reclaim.checkpoint.state == {"evidence": ["a"]}
        assert reclaim.run.completed_turns == 1


class TestLeaseFencing:
    async def test_heartbeat_renews_and_reports_cancellation(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        renewal = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert renewal.renewed is True
        assert renewal.cancel_requested is False
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert (
            await store.heartbeat(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
        ).cancel_requested is True

    async def test_expired_lease_is_never_revived(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        renewal = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert renewal.renewed is False

    async def test_stale_worker_cannot_write_after_reclaim(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        stale = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        fresh = await _claimed(store, worker_id="worker-2")
        stale_args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=stale.run.fencing_epoch,
        )

        assert (await store.heartbeat(**stale_args)).renewed is False
        assert await store.record_phase(**stale_args, phase="searching") is None
        assert await store.append_token_batch(**stale_args, text="stale") is None
        assert await store.append_reset(**stale_args) is None
        assert (
            await store.finish_success(**stale_args, result={"answer": "stale"})
        ).committed is False
        assert (
            await store.finish_failure(
                **stale_args, error_kind="provider_error", error_message="stale"
            )
        ).committed is False
        assert await store.release_for_shutdown(**stale_args) == "lease_lost"
        commit = await store.commit_checkpoint(
            **stale_args, expected_completed_turns=0, version=1, state={}
        )
        assert commit.outcome == "lease_lost"

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"
        assert record.lease_owner == "worker-2"
        assert record.fencing_epoch == fresh.run.fencing_epoch
        assert await _event_types(pool, creation.run.run_id) == []

    async def test_same_worker_with_a_stale_epoch_is_fenced_out(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        await _claimed(store)
        assert (
            await store.record_phase(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                phase="planning",
            )
            is None
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_phase_and_progress_event_advance_together(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        sequence = await store.record_phase(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase="researching",
        )
        assert sequence == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.phase == "researching"
        assert record.next_event_sequence == 2
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.sequence, event.event_type, event.payload) for event in events] == [
            (1, "progress", {"phase": "researching"})
        ]

    async def test_sequences_are_gap_free_under_concurrent_appends(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        sequences = await asyncio.gather(
            *(store.append_token_batch(**args, text=f"chunk-{index}") for index in range(24))
        )
        assert sorted(sequence for sequence in sequences if sequence is not None) == list(
            range(1, 25)
        )
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [event.sequence for event in events] == list(range(1, 25))

    async def test_replay_resumes_after_a_cursor(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.record_phase(**args, phase="planning")
        await store.append_token_batch(**args, text="one")
        await store.append_reset(**args)
        await store.append_token_batch(**args, text="two")
        events = await store.read_event_page(
            owner_id=_OWNER, run_id=creation.run.run_id, after_sequence=2
        )
        assert [(event.sequence, event.event_type) for event in events] == [
            (3, "reset"),
            (4, "token"),
        ]

    async def test_appending_an_event_renews_the_lease(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_runs "
                "SET lease_expires_at = NOW() + INTERVAL '1 second' WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        await store.append_token_batch(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            text="hi",
        )
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.lease_expires_at is not None
        assert await store.claim_next(worker_id="worker-2") is None


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestCheckpoints:
    async def test_commits_one_turn_and_resets_the_recovery_counter(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        claim = await _claimed(store, worker_id="worker-2")
        assert claim.run.recovery_count == 1
        commit = await store.commit_checkpoint(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id="worker-2",
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            version=1,
            state={"episode": []},
        )
        assert commit.outcome == "committed"
        assert commit.completed_turns == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.completed_turns == 1
        assert record.recovery_count == 0

    async def test_a_turn_far_behind_the_row_is_corrupt(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"turn": 1}
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=1, version=1, state={"turn": 2}
        )
        conflict = await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"turn": 1}
        )
        assert conflict.outcome == "corrupt"
        assert conflict.completed_turns == 2

    async def test_a_checkpoint_turn_that_disagrees_with_the_row_is_corrupt(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"turn": 1}
        )
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_runs "
                "SET checkpoint_json = jsonb_set(checkpoint_json, '{completed_turns}', '9') "
                "WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        conflict = await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"turn": 1}
        )
        assert conflict.outcome == "corrupt"

    async def test_indeterminate_commit_resolves_as_committed(self, store, pool) -> None:
        """A committed turn plus a replayed compare-and-set must not fail the run."""
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=2, state={"turn": 1}
        )
        # The worker never saw the commit result and retries the same transaction.
        replay = await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=2, state={"turn": 1}
        )
        assert replay.outcome == "committed"
        assert replay.completed_turns == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.completed_turns == 1

    async def test_missing_run_resolves_as_lease_lost(self, store) -> None:
        commit = await store.commit_checkpoint(
            owner_id=_OWNER,
            run_id=str(uuid.uuid7()),
            worker_id=_WORKER,
            fencing_epoch=1,
            expected_completed_turns=0,
            version=1,
            state={},
        )
        assert commit.outcome == "lease_lost"


# ---------------------------------------------------------------------------
# Terminal transitions, requeue, recovery bound
# ---------------------------------------------------------------------------


class TestTerminalTransitions:
    async def test_success_stores_the_canonical_result_and_one_done_event(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        result = {"answer": "yes", "references": [{"id": "r1"}]}
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result=result,
            stop_reason="converged",
        )
        assert outcome.committed is True
        assert outcome.status == "succeeded"
        assert outcome.event_sequence == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "succeeded"
        assert record.result == result
        assert record.stop_reason == "converged"
        assert record.lease_owner is None
        assert record.finished_at is not None
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].event_type == "done"
        assert events[0].payload == {"status": "succeeded", "result": result}
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_terminal_transitions_are_not_repeatable(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert (
            await store.finish_failure(**args, error_kind="provider_error", error_message="boom")
        ).committed is True
        assert (
            await store.finish_failure(**args, error_kind="provider_error", error_message="boom")
        ).committed is False
        assert (await store.finish_success(**args, result={"answer": "no"})).committed is False
        assert await _event_types(pool, creation.run.run_id) == ["error"]
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == "provider_error"
        assert record.error_message == "boom"

    async def test_worker_observed_cancellation_commits_cancelled(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await store.finish_cancelled(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_a_fenced_terminal_transition_drops_the_checkpoint(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"episode": ["kept"]}
        )
        assert await _checkpoint(pool, creation.run.run_id) is not None

        await store.finish_success(**args, result={"answer": "final"})

        assert await _checkpoint(pool, creation.run.run_id) is None
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.completed_turns == 1
        assert record.result == {"answer": "final"}

    async def test_an_unleased_finalization_drops_the_checkpoint(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.commit_checkpoint(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            version=1,
            state={"episode": ["kept"]},
        )
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        await _expire_lease(pool, creation.run.run_id)

        assert (await store.sweep_once()).cancelled == 1

        assert await _checkpoint(pool, creation.run.run_id) is None


class TestShutdownAndRecovery:
    async def test_graceful_requeue_preserves_progress_without_counting_recovery(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        await store.commit_checkpoint(
            **args, expected_completed_turns=0, version=1, state={"turn": 1}
        )
        release = await store.release_for_shutdown(**args)
        assert release == "requeued"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "queued"
        assert record.lease_owner is None
        assert record.completed_turns == 1
        assert record.recovery_count == 0
        reclaim = await _claimed(store, worker_id="worker-2")
        assert reclaim.run.run_id == creation.run.run_id
        assert reclaim.run.recovery_count == 0
        assert reclaim.checkpoint is not None
        assert reclaim.checkpoint.state == {"turn": 1}
        assert await _event_types(pool, creation.run.run_id) == []

    async def test_shutdown_finalizes_a_cancel_pending_run(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        release = await store.release_for_shutdown(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert release == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_four_reclaims_are_allowed_and_the_next_abandons(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        for expected in range(1, MAX_CONSECUTIVE_RECOVERIES + 1):
            await _expire_lease(pool, creation.run.run_id)
            reclaim = await _claimed(store, worker_id=f"worker-{expected}")
            assert reclaim.run.recovery_count == expected

        await _expire_lease(pool, creation.run.run_id)
        assert await store.claim_next(worker_id="worker-final") is None

        sweep = await store.sweep_once()
        assert sweep.cancelled == 0
        assert sweep.abandoned == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == RUN_ABANDONED_ERROR_KIND
        assert record.lease_owner is None
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [event.event_type for event in events] == ["error"]
        assert events[0].payload["kind"] == RUN_ABANDONED_ERROR_KIND

    async def test_a_checkpoint_lets_a_long_run_survive_more_restarts(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        turns = 0
        for _ in range(MAX_CONSECUTIVE_RECOVERIES + 2):
            claim = await _claimed(store)
            commit = await store.commit_checkpoint(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                expected_completed_turns=turns,
                version=1,
                state={"turn": turns + 1},
            )
            assert commit.outcome == "committed"
            turns += 1
            await _expire_lease(pool, creation.run.run_id)
        assert (await store.sweep_once()).abandoned == 0
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"
        assert record.completed_turns == turns

    async def test_sweeper_finalizes_an_unleased_cancellation(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        await _expire_lease(pool, creation.run.run_id)
        sweep = await store.sweep_once()
        assert sweep.cancelled == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert await _event_types(pool, creation.run.run_id) == ["done"]

    async def test_sweeper_leaves_live_leases_alone(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        sweep = await store.sweep_once()
        assert sweep.cancelled == 0
        assert sweep.abandoned == 0
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _reference(
    digest: str, *, resource_id: str = "res-1", ordinal: int = 0
) -> PendingArtifactReference:
    return PendingArtifactReference(
        resource_id=resource_id,
        reference_kind="current_attachment",
        ordinal=ordinal,
        digest=digest,
        filename="report.pdf",
        mime_type="application/pdf",
        transform_locator={"page": 2},
    )


class TestArtifacts:
    async def test_creation_stores_blobs_and_ordered_references(self, store) -> None:
        artifact = PendingArtifact(content=b"%PDF-1.7 payload")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(references) == 1
        assert references[0].digest == artifact.digest
        assert references[0].reference_kind == "current_attachment"
        assert references[0].transform_locator == {"page": 2}
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) == (
            artifact.content
        )

    async def test_blobs_deduplicate_within_one_owner_only(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"shared bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await store.create_run(
            owner_id=_OWNER,
            request=_request("b"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await store.create_run(
            owner_id=_OTHER_OWNER,
            request=_request("c"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        async with pool.acquire() as conn:
            owners = await conn.fetch(
                "SELECT owner_id FROM dlightrag_answer_artifacts WHERE digest = $1",
                artifact.digest,
            )
        assert sorted(str(row["owner_id"]) for row in owners) == sorted([_OWNER, _OTHER_OWNER])
        assert await store.load_artifact(owner_id="ghost", digest=artifact.digest) is None
        assert first.run.run_id is not None

    async def test_run_scoped_fetch_attaches_bytes_and_reference_together(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        fetched = PendingArtifact(content=b"<html>page</html>")
        outcome = await store.attach_artifacts(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            artifacts=[fetched],
            references=[
                PendingArtifactReference(
                    resource_id="res-web",
                    reference_kind="fetched_resource",
                    ordinal=0,
                    digest=fetched.digest,
                    filename="page.html",
                    mime_type="text/html",
                )
            ],
        )
        assert outcome == "attached"
        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [reference.reference_kind for reference in references] == ["fetched_resource"]
        assert await store.load_artifact(owner_id=_OWNER, digest=fetched.digest) == (
            fetched.content
        )

    async def test_deleting_one_run_keeps_bytes_another_run_still_links(self, store) -> None:
        artifact = PendingArtifact(content=b"shared bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        second = await store.create_run(
            owner_id=_OWNER,
            request=_request("b"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        deletion = await store.delete_runs(owner_id=_OWNER, run_ids=[first.run.run_id])
        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) is not None

        final = await store.delete_runs(owner_id=_OWNER, run_ids=[second.run.run_id])
        assert final.runs == 1
        assert final.artifacts == 1
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) is None

    async def test_deletion_is_owner_scoped(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        deletion = await store.delete_runs(owner_id=_OTHER_OWNER, run_ids=[creation.run.run_id])
        assert deletion.runs == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is not None

    async def test_cleanup_yields_to_a_run_still_adopting_the_blob(self, store, pool) -> None:
        """An uncommitted adoption holds the blob's key-share lock, so cleanup skips it."""
        artifact = PendingArtifact(content=b"contended bytes")
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("a"),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        second = await store.create_run(owner_id=_OWNER, request=_request("b"))

        adopting = await pool.acquire()
        transaction = adopting.transaction()
        await transaction.start()
        try:
            await adopting.execute(
                "INSERT INTO dlightrag_answer_run_artifacts "
                "(owner_id, run_id, resource_id, reference_kind, ordinal, digest, "
                "filename, mime_type) "
                "VALUES ($1, $2, 'res-late', 'fetched_resource', 0, $3, 'late.pdf', "
                "'application/pdf')",
                _OWNER,
                uuid.UUID(second.run.run_id),
                artifact.digest,
            )
            deletion = await asyncio.wait_for(
                store.delete_runs(owner_id=_OWNER, run_ids=[first.run.run_id]), timeout=10
            )
        finally:
            await transaction.commit()
            await pool.release(adopting)

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) == (
            artifact.content
        )
        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=second.run.run_id)
        assert [reference.resource_id for reference in references] == ["res-late"]

    async def test_a_referenced_blob_cannot_be_deleted_directly(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"still linked")
        await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.RestrictViolationError):
                await conn.execute(
                    "DELETE FROM dlightrag_answer_artifacts WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    async def test_trim_removes_expired_event_logs_and_keeps_the_result(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "kept"},
        )
        assert await store.trim_expired_event_logs() == 0

        await _backdate_finish(pool, creation.run.run_id, days=31)
        assert await store.trim_expired_event_logs() == 1
        assert await store.trim_expired_event_logs() == 0

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.events_trimmed_at is not None
        assert record.result == {"answer": "kept"}
        assert await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id) == ()

    async def test_prune_deletes_expired_runs_with_events_and_blobs(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"expiring bytes")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "old"},
        )
        fresh = await store.prune_expired_runs()
        assert fresh.runs == 0
        assert fresh.artifacts == 0

        await _backdate_finish(pool, creation.run.run_id, days=31)
        outcome = await store.prune_expired_runs()
        assert outcome.runs == 1
        assert outcome.artifacts == 1
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) is None
        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "SELECT count(*) FROM dlightrag_answer_run_events WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        assert int(remaining) == 0

    async def test_prune_leaves_unfinished_runs_alone(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_runs "
                "SET created_at = NOW() - INTERVAL '90 days' WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        assert (await store.prune_expired_runs()).runs == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is not None


# ---------------------------------------------------------------------------
# Blob-cleanup failure isolation
# ---------------------------------------------------------------------------


async def _force_blob_delete_restrict(pool: Any) -> None:
    """Make every blob delete raise SQLSTATE 23001, exactly as an adopted blob does.

    A real adopter usually loses to ``FOR UPDATE SKIP LOCKED``, so the surviving
    window is too narrow to schedule. The trigger reproduces the identical
    ``RestrictViolationError`` inside a real transaction, which is what decides
    whether the caller's run deletion survives.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE FUNCTION dlightrag_test_restrict() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "RAISE EXCEPTION 'blob adopted concurrently' USING ERRCODE = '23001'; "
            "END; $$"
        )
        await conn.execute(
            "CREATE TRIGGER dlightrag_test_restrict_trigger "
            "BEFORE DELETE ON dlightrag_answer_artifacts "
            "FOR EACH ROW EXECUTE FUNCTION dlightrag_test_restrict()"
        )


async def _succeed_and_expire(store: PGAnswerRunStore, pool: Any, run_id: str) -> None:
    claim = await _claimed(store)
    await store.finish_success(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
        result={"answer": run_id},
    )
    await _backdate_finish(pool, run_id, days=31)


class TestBlobCleanupFailureIsolation:
    async def test_deletion_commits_when_blob_cleanup_hits_restrict(self, store, pool) -> None:
        artifact = PendingArtifact(content=b"contended bytes")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[artifact],
            references=[_reference(artifact.digest)],
        )
        await _force_blob_delete_restrict(pool)

        deletion = await store.delete_runs(owner_id=_OWNER, run_ids=[creation.run.run_id])

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await store.load_artifact(owner_id=_OWNER, digest=artifact.digest) is not None

    async def test_retention_advances_past_a_contended_head_batch(self, store, pool) -> None:
        run_ids: list[str] = []
        digests: list[str] = []
        for index in range(3):
            artifact = PendingArtifact(content=f"batch bytes {index}".encode())
            creation = await store.create_run(
                owner_id=_OWNER,
                request=_request(f"q{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )
            await _succeed_and_expire(store, pool, creation.run.run_id)
            run_ids.append(creation.run.run_id)
            digests.append(artifact.digest)
        await _force_blob_delete_restrict(pool)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 3
        assert outcome.artifacts == 0
        for run_id in run_ids:
            assert await store.get_run(owner_id=_OWNER, run_id=run_id) is None
        assert (await store.prune_expired_runs()).runs == 0
        for digest in digests:
            assert await store.load_artifact(owner_id=_OWNER, digest=digest) is not None


# ---------------------------------------------------------------------------
# Bounded event replay
# ---------------------------------------------------------------------------


class TestEventPaging:
    async def test_paged_replay_crosses_the_boundary_without_gaps(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        appended = 501
        for index in range(appended):
            await store.append_token_batch(**args, text=f"chunk-{index}")
        assert (await store.finish_success(**args, result={"answer": "end"})).committed is True

        pages: list[tuple[int, ...]] = []
        cursor = 0
        while True:
            page = await store.read_event_page(
                owner_id=_OWNER, run_id=creation.run.run_id, after_sequence=cursor
            )
            if not page:
                break
            pages.append(tuple(event.sequence for event in page))
            cursor = page[-1].sequence

        total = appended + 1
        replayed = [sequence for page in pages for sequence in page]
        assert len(pages) >= 2
        assert len(pages[0]) < total
        assert replayed == list(range(1, total + 1))


# ---------------------------------------------------------------------------
# Fetched-resource replay and fenced attachment
# ---------------------------------------------------------------------------


def _fetched(
    digest: str, *, resource_id: str = "res-web", ordinal: int = 0
) -> PendingArtifactReference:
    return PendingArtifactReference(
        resource_id=resource_id,
        reference_kind="fetched_resource",
        ordinal=ordinal,
        digest=digest,
        filename=f"{resource_id}.html",
        mime_type="text/html",
    )


class TestFetchedResourceReplay:
    async def test_exact_replay_stores_one_reference_and_one_blob(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        page = PendingArtifact(content=b"<html>one</html>")
        for _ in range(2):
            assert (
                await store.attach_artifacts(
                    owner_id=_OWNER,
                    run_id=creation.run.run_id,
                    worker_id=_WORKER,
                    fencing_epoch=claim.run.fencing_epoch,
                    expected_completed_turns=0,
                    artifacts=[page],
                    references=[_fetched(page.digest)],
                )
                == "attached"
            )

        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(r.resource_id, r.digest) for r in references] == [("res-web", page.digest)]
        assert await store.load_artifact(owner_id=_OWNER, digest=page.digest) == page.content

    async def test_replay_rebinds_the_slot_to_changed_bytes(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        first = PendingArtifact(content=b"<html>monday</html>")
        second = PendingArtifact(content=b"<html>tuesday, longer</html>")
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
        )
        await store.attach_artifacts(
            **args, artifacts=[first], references=[_fetched(first.digest, resource_id="res-a")]
        )
        assert (
            await store.attach_artifacts(
                **args,
                artifacts=[second],
                references=[_fetched(second.digest, resource_id="res-b")],
            )
            == "attached"
        )

        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(r.resource_id, r.digest) for r in references] == [("res-b", second.digest)]
        assert await store.load_artifact(owner_id=_OWNER, digest=second.digest) == second.content
        assert await store.load_artifact(owner_id=_OWNER, digest=first.digest) is None

    async def test_rebind_keeps_bytes_another_run_still_references(self, store) -> None:
        shared = PendingArtifact(content=b"<html>shared</html>")
        keeper = await store.create_run(
            owner_id=_OWNER,
            request=_request("keeper"),
            artifacts=[shared],
            references=[_reference(shared.digest)],
        )
        creation = await store.create_run(owner_id=_OWNER, request=_request("replayer"))
        claim = await _claimed(store, worker_id="worker-2")
        assert claim.run.run_id == keeper.run.run_id
        claim = await _claimed(store)
        args = dict(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
        )
        await store.attach_artifacts(
            **args, artifacts=[shared], references=[_fetched(shared.digest, resource_id="res-a")]
        )
        replacement = PendingArtifact(content=b"<html>changed</html>")
        await store.attach_artifacts(
            **args,
            artifacts=[replacement],
            references=[_fetched(replacement.digest, resource_id="res-b")],
        )

        assert await store.load_artifact(owner_id=_OWNER, digest=shared.digest) == shared.content
        keeper_references = await store.list_run_artifacts(
            owner_id=_OWNER, run_id=keeper.run.run_id
        )
        assert [r.digest for r in keeper_references] == [shared.digest]

    async def test_a_fetched_replay_never_rewrites_an_accepted_input(self, store) -> None:
        accepted = PendingArtifact(content=b"%PDF accepted")
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(),
            artifacts=[accepted],
            references=[_reference(accepted.digest, resource_id="res-1")],
        )
        claim = await _claimed(store)
        intruder = PendingArtifact(content=b"<html>intruder</html>")

        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await store.attach_artifacts(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                expected_completed_turns=0,
                artifacts=[intruder],
                references=[_fetched(intruder.digest, resource_id="res-1")],
            )

        references = await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(r.resource_id, r.reference_kind, r.digest) for r in references] == [
            ("res-1", "current_attachment", accepted.digest)
        ]
        assert await store.load_artifact(owner_id=_OWNER, digest=intruder.digest) is None


class TestFencedAttachment:
    async def test_a_stale_epoch_writes_nothing(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        page = PendingArtifact(content=b"<html>stale</html>")

        outcome = await store.attach_artifacts(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch - 1,
            expected_completed_turns=0,
            artifacts=[page],
            references=[_fetched(page.digest)],
        )

        assert outcome == "lease_lost"
        assert await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id) == ()
        assert await store.load_artifact(owner_id=_OWNER, digest=page.digest) is None

    async def test_an_expired_lease_writes_nothing(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        page = PendingArtifact(content=b"<html>expired</html>")

        outcome = await store.attach_artifacts(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            artifacts=[page],
            references=[_fetched(page.digest)],
        )

        assert outcome == "lease_lost"
        assert await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id) == ()
        assert await store.load_artifact(owner_id=_OWNER, digest=page.digest) is None

    async def test_a_wrong_completed_turn_writes_nothing(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        page = PendingArtifact(content=b"<html>wrong turn</html>")

        outcome = await store.attach_artifacts(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=3,
            artifacts=[page],
            references=[_fetched(page.digest)],
        )

        assert outcome == "turn_mismatch"
        assert await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id) == ()
        assert await store.load_artifact(owner_id=_OWNER, digest=page.digest) is None

    async def test_a_foreign_owner_writes_nothing(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        page = PendingArtifact(content=b"<html>foreign</html>")

        outcome = await store.attach_artifacts(
            owner_id=_OTHER_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            expected_completed_turns=0,
            artifacts=[page],
            references=[_fetched(page.digest)],
        )

        assert outcome == "lease_lost"
        assert await store.list_run_artifacts(owner_id=_OWNER, run_id=creation.run.run_id) == ()
        assert await store.load_artifact(owner_id=_OTHER_OWNER, digest=page.digest) is None

    async def test_an_unknown_run_writes_nothing(self, store) -> None:
        page = PendingArtifact(content=b"<html>ghost</html>")

        outcome = await store.attach_artifacts(
            owner_id=_OWNER,
            run_id="not-a-uuid",
            worker_id=_WORKER,
            fencing_epoch=1,
            expected_completed_turns=0,
            artifacts=[page],
            references=[_fetched(page.digest)],
        )

        assert outcome == "lease_lost"
        assert await store.load_artifact(owner_id=_OWNER, digest=page.digest) is None
