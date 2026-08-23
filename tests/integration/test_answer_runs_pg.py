# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for durable Answer run state on PostgreSQL 18.

Exercises the real contract against a live database: the declared schema and its
constraints, owner-scoped creation and idempotency, queued cancellation, slot-safe
claiming across workers, lease fencing, gap-free event sequences, journal
compare-and-set, terminal transitions, graceful requeue, the crash-recovery bound,
event trimming, retention pruning, and ownership-safe artifact cleanup.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import asyncio
import datetime
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.runtime import (
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    answer_run_request_fingerprint,
)
from tests.conftest import FingerprintingAnswerRunStore

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
    created = FingerprintingAnswerRunStore(pool=pool)
    await created.initialize()
    # Retention exempts conversation-linked runs, so the whole operational schema
    # is established here exactly as a real process establishes it at startup.
    from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore

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


async def _prepared_input(pool: Any, run_id: str) -> str | None:
    """Read the raw prepared input column."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT prepared_input_json FROM dlightrag_answer_runs WHERE run_id = $1",
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
    async def test_creates_exactly_the_answer_schema_tables(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'dlightrag_%'"
            )
        assert {str(row["table_name"]) for row in rows} - {"dlightrag_schema_migrations"} == {
            "dlightrag_answer_runs",
            "dlightrag_answer_run_events",
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_agent_context_projections",
            "dlightrag_agent_effects",
            "dlightrag_answer_run_stages",
            "dlightrag_answer_evidence",
            "dlightrag_answer_resources",
            "dlightrag_blobs",
            "dlightrag_blob_chunks",
            "dlightrag_answer_run_artifacts",
            "dlightrag_answer_workspace_inventory",
            "dlightrag_answer_committed_spills",
            "dlightrag_answer_run_routing",
            "dlightrag_answer_child_sessions",
            "dlightrag_agent_controls",
            "dlightrag_answer_memory_settings",
        }

    async def test_memory_settings_default_and_roundtrip(self, store, pool) -> None:
        """Enablement defaults on for absent rows and persists across updates."""
        from dlightrag.adapters.postgres.memory_settings import PGMemorySettingsStore

        settings = PGMemorySettingsStore(pool=pool)

        assert await settings.enabled(owner_id="alpha") is True
        await settings.set_enabled(owner_id="alpha", enabled=False)
        assert await settings.enabled(owner_id="alpha") is False
        assert await settings.enabled(owner_id="beta") is True
        await settings.set_enabled(owner_id="alpha", enabled=True)
        assert await settings.enabled(owner_id="alpha") is True

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
            "prepared_input_json",
            "accepted_input_json",
            "request_fingerprint",
            "status",
            "phase",
            "stop_reason",
            "cancel_requested_at",
            "lease_owner",
            "lease_expires_at",
            "fencing_epoch",
            "durable_progress_version",
            "last_reclaim_progress_version",
            "reclaims_without_progress",
            "next_event_sequence",
            "events_trimmed_at",
            "result_json",
            "error_kind",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "workspace_epoch",
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
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_blobs")] == "r"

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
        from dlightrag.adapters.postgres.ingest_jobs import PGIngestJobStore

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
        assert creation.run.prepared_input["query"] == "why"

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
        lookup = await store.replay_run(
            owner_id=_OWNER,
            idempotency_key="k1",
            idempotency_fingerprint=answer_run_request_fingerprint(_request()),
        )
        assert lookup is not None
        assert lookup.replayed is True
        assert lookup.run.run_id == first.run.run_id

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
        assert events[0].payload.get("status", "cancelled") == "cancelled"

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
        assert reclaim.run.reclaims_without_progress == 1
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
        assert record.reclaims_without_progress == 1


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
        from dlightrag.agent.session.entries import UserMessageEntry
        from dlightrag.agent.session.ids import EntryId, SessionId

        assert creation.run.prepared_input is not None
        stale_session = SessionId(str(creation.run.prepared_input["session_id"]))
        from dlightrag.adapters.postgres.session_journal import PGJournalStore

        stale_journal = PGJournalStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=int(stale_args["fencing_epoch"]),
        )
        stale_append = await stale_journal.append(
            session_id=stale_session,
            expected_version=0,
            entries=[
                UserMessageEntry(
                    entry_id=EntryId.new(),
                    session_id=stale_session,
                    timestamp=datetime.datetime.now(datetime.UTC),
                    content="stale",
                )
            ],
        )
        assert stale_append.__class__.__name__ == "LeaseLost"

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

    async def test_a_fenced_terminal_transition_clears_the_prepared_input(
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
        assert await _prepared_input(pool, creation.run.run_id) is not None

        await store.finish_success(**args, result={"answer": "final"})

        assert await _prepared_input(pool, creation.run.run_id) is None
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.durable_progress_version == 0
        assert record.result == {"answer": "final"}

    async def test_an_unleased_finalization_clears_the_prepared_input(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.record_phase(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase="searching",
        )
        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        await _expire_lease(pool, creation.run.run_id)

        assert (await store.sweep_once()).cancelled == 1

        assert await _prepared_input(pool, creation.run.run_id) is None


class TestShutdownAndRecovery:
    async def test_active_requirements_include_only_work_that_may_execute(
        self,
        store,
        pool,
    ) -> None:
        cancelled = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="cancelled", pinned_models=[]),
        )
        await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=cancelled.run.run_id)
        await _expire_lease(pool, cancelled.run.run_id)

        abandoned = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="abandoned", pinned_models=[]),
        )
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(abandoned.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        recoverable = await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="recoverable", pinned_models=[]),
        )
        await _claimed(store)
        await _expire_lease(pool, recoverable.run.run_id)
        await store.create_run(
            owner_id=_OWNER,
            request=_request(context_policy_revision="queued", pinned_models=[]),
        )

        requirements = await store.list_active_run_requirements()

        assert {row["context_policy_revision"] for row in requirements} == {
            "queued",
            "recoverable",
        }

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
                "SELECT owner_id FROM dlightrag_blobs WHERE digest = $1",
                artifact.digest,
            )
        assert sorted(str(row["owner_id"]) for row in owners) == sorted([_OWNER, _OTHER_OWNER])
        assert await store.load_artifact(owner_id="ghost", digest=artifact.digest) is None
        assert first.run.run_id is not None

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
                    "DELETE FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
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

        await _backdate_finish(pool, creation.run.run_id, days=370)
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

        await _backdate_finish(pool, creation.run.run_id, days=370)
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


async def _force_blob_delete_restrict(pool: Any, *, only_digest: str | None = None) -> None:
    """Make blob deletes raise SQLSTATE 23001, exactly as an adopted blob does.

    A real adopter usually loses to ``FOR UPDATE SKIP LOCKED``, so the surviving
    window is too narrow to schedule. The trigger reproduces the identical
    ``RestrictViolationError`` inside a real transaction, which is what decides
    whether the caller's run deletion survives. ``only_digest`` contends exactly
    one blob so a mixed batch can be observed.
    """
    guard = "IF OLD.digest = $q$" + only_digest + "$q$ THEN" if only_digest else "IF true THEN"
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE FUNCTION dlightrag_test_restrict() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            f"{guard} "
            "RAISE EXCEPTION 'blob adopted concurrently' USING ERRCODE = '23001'; "
            "END IF; RETURN OLD; "
            "END; $$"
        )
        await conn.execute(
            "CREATE TRIGGER dlightrag_test_restrict_trigger "
            "BEFORE DELETE ON dlightrag_blobs "
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
    await _backdate_finish(pool, run_id, days=370)


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

    async def test_one_contended_digest_does_not_shield_unrelated_orphans(
        self, store, pool
    ) -> None:
        """A mixed batch must not let one adopted blob keep every other orphan alive."""
        digests: list[str] = []
        run_ids: list[str] = []
        for index in range(3):
            artifact = PendingArtifact(content=f"mixed bytes {index}".encode())
            creation = await store.create_run(
                owner_id=_OWNER,
                request=_request(f"mixed-{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )
            await _succeed_and_expire(store, pool, creation.run.run_id)
            digests.append(artifact.digest)
            run_ids.append(creation.run.run_id)
        contended = digests[1]
        await _force_blob_delete_restrict(pool, only_digest=contended)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 3
        assert outcome.artifacts == 2
        for run_id in run_ids:
            assert await store.get_run(owner_id=_OWNER, run_id=run_id) is None
        assert await store.load_artifact(owner_id=_OWNER, digest=contended) is not None
        for digest in (digests[0], digests[2]):
            assert await store.load_artifact(owner_id=_OWNER, digest=digest) is None


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


class TestAgentControlsAndChildren:
    async def test_controls_are_ordered_and_consumed_under_the_run_lease(self, store) -> None:
        creation = await store.create_run(
            owner_id=_OWNER,
            request=_request(mode="research"),
        )
        first = await store.enqueue_agent_control(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            kind="steer",
            content="first",
        )
        second = await store.enqueue_agent_control(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            kind="steer",
            content="second",
        )
        claim = await _claimed(store)

        controls = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        acknowledged = await store.acknowledge_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            control_sequences=tuple(int(item["control_sequence"]) for item in controls or ()),
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        replay = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        assert acknowledged
        assert first is not None and first["control_sequence"] == 1
        assert second is not None and second["control_sequence"] == 2
        assert [item["content"] for item in controls or ()] == ["first", "second"]
        assert replay == ()

    async def test_one_spawn_call_persists_multiple_child_lineages(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        children = (str(uuid.uuid7()), str(uuid.uuid7()))
        parent_intent_id = str(uuid.uuid7())

        for position, child_id in enumerate(children):
            held = await store.upsert_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                parent_session_id=parent_id,
                parent_call_id="one-call",
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                parent_intent_id=parent_intent_id,
                objective=f"child {position}",
                context_mode="parent",
                model_role="extract",
                tools=("search_knowledge_base",),
            )
            assert held
            # Tool execution repeats the roster upsert after the intent-bound
            # precreate. It must not lose the parent intent or lease.
            assert await store.upsert_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                parent_session_id=parent_id,
                parent_call_id="one-call",
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
            assert await store.finish_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                status="succeeded",
                summary="done",
                usage={"input_tokens": 10 + position},
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )

        roster = await store.list_child_sessions(owner_id=_OWNER, run_id=creation.run.run_id)
        assert {item["child_session_id"] for item in roster} == set(children)
        assert {item["parent_session_id"] for item in roster} == {parent_id}
        assert {item["parent_intent_id"] for item in roster} == {parent_intent_id}
        assert [item["objective"] for item in roster] == ["child 0", "child 1"]
        assert [item["usage"]["input_tokens"] for item in roster] == [10, 11]
