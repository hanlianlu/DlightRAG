# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for durable Answer run state on PostgreSQL 18.

Exercises the real contract against a live database: the declared schema and its
constraints, owner-scoped creation and idempotency, queued cancellation, slot-safe
claiming across workers, lease fencing, gap-free event sequences, Session
compare-and-set, terminal transitions, graceful requeue, the crash-recovery bound,
event trimming, retention pruning, and ownership-safe artifact cleanup.

Every test runs inside a throwaway database created and dropped per test, so the
configured administrative database is never mutated. Connection settings come
from the shared integration-test PostgreSQL environment; skipped if unavailable.
"""

import asyncio
import datetime
import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer.agentspace import (
    PGAgentspaceStore,
    write_committed_spill,
    write_inventory,
)
from dlightrag.adapters.postgres.core._migrations import Migration, apply_migrations
from dlightrag.adapters.postgres.runtime.run_blob_store import (
    BlobSizeConflict,
    PGRunBlobStore,
    write_blob_content,
    write_complete_blob,
)
from dlightrag.adapters.postgres.runtime.run_store import (
    RUN_SCHEMA_TABLES,
    PGRunStore,
)
from dlightrag.engine.agent.session.ids import SessionId, StageIntentId
from dlightrag.engine.answer.research.runtime import _check_child_write
from dlightrag.engine.answer.tools.subagents import SubagentHost, subagent_tools
from dlightrag.engine.runtime.agentspace import (
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
)
from dlightrag.engine.runtime.blob_chunks import BLOB_CHUNK_BYTES
from dlightrag.engine.runtime.errors import RunSchemaError
from dlightrag.engine.runtime.policy import (
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
)
from dlightrag.engine.runtime.progress import StageTerminalCommit
from dlightrag.engine.runtime.records import (
    ActiveRunLimitExceededError,
    DeletedRun,
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    PendingPublication,
    RunAdmissionLimitExceededError,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from tests.conftest import FingerprintingRunStore
from tests.support.pg import (
    PG_CONN_KWARGS,
    catalog_table,
    catalog_tables,
    declared_shape,
    delete_runs,
    drop_database,
    notification_hub,
    skip_without_postgres,
)
from tests.support.tool_helpers import tool_runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_OWNER = "owner-alpha"
_OTHER_OWNER = "owner-beta"
_WORKER = "worker-1"
_ABANDONED_ERROR_MESSAGE = "Run exceeded its reclaim-without-progress bound."


def _blob_store(store: PGRunStore) -> PGRunBlobStore:
    pool = store._operation_pool  # noqa: SLF001 - bind the separate blob seam in tests
    assert pool is not None
    return PGRunBlobStore(pool=pool)


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    """Provision an isolated throwaway database and yield a pool bound to it."""
    await skip_without_postgres()

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
        await drop_database(db_name)


@pytest.fixture
async def store(pool: Any) -> AsyncIterator[PGRunStore]:
    async with notification_hub(pool) as hub:
        created = FingerprintingRunStore(pool=pool, notifications=hub)
        await created.initialize()
        # Establish the complete operational schema exactly as a real process does.
        from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore

        await PGWebConversationStore(pool=pool, run_store=created).initialize()
        yield created


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {
        "query": query,
        "workspaces": ["alpha"],
        "agent_session_id": "00000000-0000-7000-8000-000000000001",
        "agent_lane_id": "main",
        **extra,
    }


async def _expire_lease(pool: Any, run_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET lease_expires_at = NOW() - INTERVAL '1 second' WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _backdate_finish(pool: Any, run_id: str, *, days: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs "
            "SET finished_at = NOW() - ($2 * INTERVAL '1 day'), "
            "purge_after = NOW() - ($2 * INTERVAL '1 day') "
            "+ make_interval(secs => retention_seconds::double precision) WHERE run_id = $1",
            uuid.UUID(run_id),
            days,
        )


async def _event_types(pool: Any, run_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT event_type FROM dlightrag_run_events WHERE run_id = $1 ORDER BY event_sequence",
            uuid.UUID(run_id),
        )
    return [str(row["event_type"]) for row in rows]


async def _prepared_input(pool: Any, run_id: str) -> str | None:
    """Read the raw prepared input column."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT prepared_input_json FROM dlightrag_runs WHERE run_id = $1",
            uuid.UUID(run_id),
        )


async def _write_fetched_blob(pool: Any, run_id: str, content: bytes) -> str:
    """Write complete fetched bytes and their run-owned resource reference."""
    digest = hashlib.sha256(content).hexdigest()
    locator = f"https://example.test/{digest[:12]}".encode()
    async with pool.acquire() as conn, conn.transaction():
        await write_blob_content(
            conn,
            owner_id=_OWNER,
            digest=digest,
            content=content,
        )
        await conn.execute(
            "INSERT INTO dlightrag_answer_resources "
            "(owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities, "
            "ordinal, blob_digest, locator_digest, source_locator) "
            "VALUES ($1, $2, $3, 'fetched_blob', 'page.html', 'text/html', $4::jsonb, "
            "0, $5, $6, $7)",
            _OWNER,
            uuid.UUID(run_id),
            f"fetched-{digest[:16]}",
            '{"resource_kind":"web"}',
            digest,
            hashlib.sha256(locator).hexdigest(),
            locator,
        )
    return digest


async def _claimed(store: PGRunStore, *, worker_id: str = _WORKER) -> Any:
    claim = await store.claim_next(worker_id=worker_id)
    assert claim is not None
    return claim


def _delete_action(value: Any) -> str:
    """``pg_constraint.confdeltype`` is a one-byte ``"char"`` column."""
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


async def _assert_run_event_parent_guard(store: FingerprintingRunStore, pool: Any) -> None:
    """Exercise the direct-DML event invariants on either schema path."""
    async with pool.acquire() as conn:
        trigger_rows = await conn.fetch(
            "SELECT trigger_name, event_manipulation, action_timing "
            "FROM information_schema.triggers "
            "WHERE trigger_schema = 'public' "
            "AND event_object_table = 'dlightrag_run_events'"
        )
    assert {
        (
            str(row["trigger_name"]),
            str(row["event_manipulation"]),
            str(row["action_timing"]),
        )
        for row in trigger_rows
    } == {
        ("trg_dlightrag_run_events_enforce", "INSERT", "BEFORE"),
        ("trg_dlightrag_run_events_enforce", "UPDATE", "BEFORE"),
    }

    async def remove(run_id: str) -> None:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM dlightrag_runs WHERE run_id = $1", uuid.UUID(run_id))

    # Even a live lease cannot publish a sequence the parent has not allocated.
    creation = await store.create_run(owner_id=_OWNER, request=_request("future sequence"))
    await _claimed(store)
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '[]'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # Nonterminal events require a running parent with a lease owner; a lapsed window is not
    # a missing owner (test_run_lease_renewal_pg pins that half).
    creation = await store.create_run(owner_id=_OWNER, request=_request("queued event"))
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    creation = await store.create_run(owner_id=_OWNER, request=_request("unleased event"))
    await _claimed(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET lease_owner = NULL, "
            "lease_expires_at = NULL, next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'progress', '{}'::jsonb)",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # Terminal events are admitted only after the matching parent transition.
    creation = await store.create_run(owner_id=_OWNER, request=_request("early terminal"))
    await _claimed(store)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET next_event_sequence = 2 WHERE run_id = $1",
            uuid.UUID(creation.run.run_id),
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'done', "
                '\'{"status":"succeeded","result":{}}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    creation = await store.create_run(owner_id=_OWNER, request=_request("terminal mismatch"))
    claim = await _claimed(store)
    assert (
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "complete"},
        )
    ).committed
    async with pool.acquire() as conn:
        # Cleanup and retention DELETEs remain legal; the guard covers only writes.
        assert (
            await conn.execute(
                "DELETE FROM dlightrag_run_events WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        ) == "DELETE 1"
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'done', "
                '\'{"status":"succeeded","result":{"answer":"wrong"}}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_run_events "
                "(owner_id, run_id, event_sequence, event_type, payload) "
                "VALUES ($1, $2, 1, 'error', "
                '\'{"kind":"provider_error","message":"boom"}\'::jsonb)',
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
    await remove(creation.run.run_id)

    # UPDATEs pass through the same parent-sequence guard.
    creation = await store.create_run(owner_id=_OWNER, request=_request("event update"))
    claim = await _claimed(store)
    assert (
        await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="progress",
            payload={"phase": "working"},
        )
    ) == 1
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "UPDATE dlightrag_run_events SET event_sequence = 2 "
                "WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        assert (
            await conn.execute(
                "DELETE FROM dlightrag_run_events WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
        ) == "DELETE 1"
    await remove(creation.run.run_id)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    async def test_a_refused_ledger_releases_the_scope_lock(self, pool) -> None:
        """A runner that refuses its ledger must not leave other callers waiting on it."""
        created = Migration("lock_probe", "first", ("CREATE TABLE lock_probe (id TEXT)",))
        renamed = Migration("lock_probe_renamed", "first", ("SELECT 1",))
        async with pool.acquire() as first, pool.acquire() as second:
            probe = {"scope": "lock_probe", "tables": (), "schema_error": RunSchemaError}
            await apply_migrations(first, migrations=(created,), **probe)
            # The ledger records a version this declaration no longer names.
            with pytest.raises(RunSchemaError):
                await apply_migrations(first, migrations=(renamed,), **probe)
            await asyncio.wait_for(
                apply_migrations(second, migrations=(created,), **probe), timeout=5
            )

    async def test_fresh_catalog_is_exactly_its_declaration(self, pool) -> None:
        """Readers verify the declaration, so every object the migrations create is in it."""
        await FingerprintingRunStore(pool=pool).initialize()

        async with pool.acquire() as conn:
            assert await catalog_tables(conn) - {"dlightrag_schema_migrations"} == {
                declared.name for declared in RUN_SCHEMA_TABLES
            }
            for declared in RUN_SCHEMA_TABLES:
                assert await catalog_table(conn, declared.name) == declared_shape(declared)

    @pytest.mark.parametrize("mode", ["DISABLE", "ENABLE REPLICA"])
    async def test_readers_refuse_an_event_guard_that_does_not_fire(
        self, store, pool, mode: str
    ) -> None:
        """A present but disabled or replica-only guard enforces nothing, so it is missing."""
        async with pool.acquire() as conn:
            await conn.execute(
                f"ALTER TABLE dlightrag_run_events {mode} TRIGGER trg_dlightrag_run_events_enforce"
            )

        with pytest.raises(RunSchemaError, match="trigger trg_dlightrag_run_events_enforce"):
            await PGRunStore(pool=pool).initialize(validate_only=True)

    async def test_fresh_schema_enforces_run_event_parent_contract(self, store, pool) -> None:
        await _assert_run_event_parent_guard(store, pool)

    async def test_creates_exactly_the_answer_schema_tables(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'dlightrag_%'"
            )
        assert {str(row["table_name"]) for row in rows} - {"dlightrag_schema_migrations"} == {
            "dlightrag_runs",
            "dlightrag_run_events",
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_agent_session_registers",
            "dlightrag_answer_run_stages",
            "dlightrag_answer_goals",
            "dlightrag_answer_evidence",
            "dlightrag_answer_resources",
            "dlightrag_blobs",
            "dlightrag_blob_chunks",
            "dlightrag_answer_run_artifacts",
            "dlightrag_answer_artifact_attachments",
            "dlightrag_answer_session_notes",
            "dlightrag_session_workspace_heads",
            "dlightrag_session_work_heads",
            "dlightrag_session_workspace_snapshots",
            "dlightrag_session_memory_progress",
            "dlightrag_answer_feedback",
            "dlightrag_answer_workspace_inventory",
            "dlightrag_answer_committed_spills",
            "dlightrag_answer_run_routing",
            "dlightrag_answer_child_sessions",
            "dlightrag_answer_child_operations",
            "dlightrag_answer_child_guidance",
            "dlightrag_agent_controls",
            "dlightrag_answer_memory_settings",
            "dlightrag_agent_browser_leases",
            "dlightrag_agent_accounts",
            "dlightrag_agent_account_settings",
            "dlightrag_corpus_mutation_windows",
            "dlightrag_runtime_policies",
            "dlightrag_privacy_settings",
            "dlightrag_schedules",
            "dlightrag_schedule_fires",
            "dlightrag_owner_principals",
        }

    async def test_artifact_reference_constraint_uses_one_output_kind(self, store, pool) -> None:
        async with pool.acquire() as conn:
            definition = await conn.fetchval(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'dlightrag_answer_run_artifacts_kind_check'"
            )

        assert definition == (
            "CHECK ((reference_kind = ANY (ARRAY['current_attachment'::text, "
            "'history_attachment'::text, 'fetched_resource'::text, "
            "'published_artifact'::text])))"
        )

    async def test_bounded_operational_scans_use_their_ordered_indexes(
        self,
        store,
        pool,
    ) -> None:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            active_plan = await conn.fetch(
                "EXPLAIN (COSTS OFF) "
                "SELECT created_at, run_id, "
                "prepared_input_json ->> 'context_policy_revision', "
                "prepared_input_json -> 'pinned_models' "
                "FROM dlightrag_runs "
                "WHERE status IN ('queued', 'running') "
                "AND cancel_requested_at IS NULL "
                "AND NOT (status = 'running' AND lease_expires_at < NOW() "
                "         AND reclaims_without_progress >= $1) "
                "ORDER BY created_at, run_id LIMIT 200",
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )
            cancel_plan = await conn.fetch(
                "EXPLAIN (COSTS OFF) "
                "SELECT owner_id, run_id, created_at "
                "FROM dlightrag_runs "
                "WHERE cancel_requested_at IS NOT NULL "
                "AND status = 'running' AND lease_owner = $1 "
                "AND lease_expires_at > NOW() "
                "ORDER BY created_at, run_id LIMIT 200",
                _WORKER,
            )

        assert "idx_dlightrag_runs_claim" in "\n".join(
            str(row["QUERY PLAN"]) for row in active_plan
        )
        assert "idx_dlightrag_runs_cancel_pending" in "\n".join(
            str(row["QUERY PLAN"]) for row in cancel_plan
        )

    async def test_memory_settings_default_and_roundtrip(self, store, pool) -> None:
        """Enablement defaults on for absent rows and persists across updates."""
        from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore

        settings = PGMemorySettingsStore(pool=pool)

        assert (await settings.state(owner_id="alpha")).enabled is True
        disabled = await settings.set_enabled(owner_id="alpha", enabled=False)
        assert disabled.enabled is False
        assert disabled.epoch == 1
        assert (await settings.state(owner_id="beta")).enabled is True
        enabled = await settings.set_enabled(owner_id="alpha", enabled=True)
        assert enabled.enabled is True
        assert enabled.epoch == 1
        bumped = await settings.bump_epoch(owner_id="alpha")
        assert bumped.epoch == 2

    async def test_run_columns_match_the_contract(self, store, pool) -> None:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                "dlightrag_runs",
            )
        assert {str(row["column_name"]) for row in rows} == {
            "owner_id",
            "run_id",
            "run_kind",
            "lane",
            "submitted_by",
            "access_scope_kind",
            "submission_key",
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
            "agent_workspace_epoch",
            "retention_seconds",
            "purge_after",
            "next_attempt_at",
            "checkpoint_json",
            "handoff_started_at",
            "superseded_by_run_id",
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
                  AND c.conrelid::regclass::text LIKE 'dlightrag_%'
                """
            )
        actions = {
            (str(row["child"]), str(row["parent"])): _delete_action(row["on_delete"])
            for row in rows
        }
        assert actions[("dlightrag_run_events", "dlightrag_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_runs")] == "c"
        assert actions[("dlightrag_answer_run_artifacts", "dlightrag_blobs")] == "r"

    async def test_rejects_unknown_status_but_accepts_executor_owned_labels(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(
                    "UPDATE dlightrag_runs SET status = 'paused' WHERE run_id = $1", run_id
                )
            await conn.execute(
                "UPDATE dlightrag_runs SET phase = 'polishing' WHERE run_id = $1", run_id
            )
        claim = await _claimed(store)
        assert (
            await store.append_event(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                phase=None,
                event_type="thinking",
                payload={},
            )
        ) == 1

    async def test_allows_only_one_terminal_event_per_run(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        assert (
            await store.finish_success(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": "complete"},
            )
        ).committed
        run_id = uuid.UUID(creation.run.run_id)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs SET next_event_sequence = 3 WHERE run_id = $1",
                run_id,
            )
            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await conn.execute(
                    "INSERT INTO dlightrag_run_events "
                    "(owner_id, run_id, event_sequence, event_type, payload) "
                    "VALUES ($1, $2, 2, 'done', "
                    '\'{"status":"succeeded","result":{"answer":"complete"}}\'::jsonb)',
                    _OWNER,
                    run_id,
                )


# ---------------------------------------------------------------------------
# Creation, idempotency, owner scoping
# ---------------------------------------------------------------------------


class TestCreation:
    async def test_nonterminal_admission_limit_is_atomic_across_submitters(self, pool) -> None:
        first_store = FingerprintingRunStore(pool=pool, query_max_nonterminal_runs=1)
        second_store = FingerprintingRunStore(pool=pool, query_max_nonterminal_runs=1)
        await first_store.initialize()
        results = await asyncio.gather(
            first_store.create_run(owner_id=_OWNER, request=_request("a")),
            second_store.create_run(owner_id=_OTHER_OWNER, request=_request("b")),
            return_exceptions=True,
        )
        assert sum(isinstance(result, RunAdmissionLimitExceededError) for result in results) == 1
        assert sum(not isinstance(result, BaseException) for result in results) == 1

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
            idempotency_fingerprint=run_request_fingerprint(_request()),
            run_kind="answer",
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

    async def test_linked_and_direct_acceptance_replay_and_conflict_alike(
        self, store, pool
    ) -> None:
        """Both entry points are one acceptance: each replays or refuses the other's run."""
        async with pool.acquire() as conn, conn.transaction():
            linked = await store.create_run_in(
                conn, owner_id=_OWNER, request=_request(), idempotency_key="k1"
            )
        assert linked.replayed is False

        direct = await store.create_run(owner_id=_OWNER, request=_request(), idempotency_key="k1")
        assert direct.replayed is True
        assert direct.run.run_id == linked.run.run_id
        with pytest.raises(IdempotencyKeyConflict):
            await store.create_run(owner_id=_OWNER, request=_request("other"), idempotency_key="k1")

        async with pool.acquire() as conn, conn.transaction():
            relinked = await store.create_run_in(
                conn, owner_id=_OWNER, request=_request(), idempotency_key="k1"
            )
            assert relinked.replayed is True
            assert relinked.run.run_id == linked.run.run_id
            with pytest.raises(IdempotencyKeyConflict):
                await store.create_run_in(
                    conn, owner_id=_OWNER, request=_request("other"), idempotency_key="k1"
                )

    async def test_linked_acceptance_honours_the_lane_admission_limit(self, pool) -> None:
        limited = FingerprintingRunStore(pool=pool, query_max_nonterminal_runs=1)
        await limited.initialize()
        await limited.create_run(owner_id=_OWNER, request=_request("first"))

        async with pool.acquire() as conn, conn.transaction():
            with pytest.raises(RunAdmissionLimitExceededError):
                await limited.create_run_in(conn, owner_id=_OWNER, request=_request("second"))

    async def test_an_owners_active_limit_admits_exactly_that_many_of_concurrent_submissions(
        self, pool
    ) -> None:
        """Counted under the lane's acceptance lock, so submissions racing for the last places
        through several connections cannot all be admitted, and another owner is not touched."""
        stores = [FingerprintingRunStore(pool=pool) for _ in range(4)]
        await stores[0].initialize()
        owners = [_OWNER] * 8 + [_OTHER_OWNER] * 2

        results = await asyncio.gather(
            *(
                stores[index % len(stores)].create_run(
                    owner_id=owner,
                    request=_request(f"question {index}"),
                    idempotency_key=f"key-{index}",
                    active_run_limit=3,
                )
                for index, owner in enumerate(owners)
            ),
            return_exceptions=True,
        )

        mine, theirs = results[:8], results[8:]
        assert sum(not isinstance(result, BaseException) for result in mine) == 3
        assert sum(isinstance(result, ActiveRunLimitExceededError) for result in mine) == 5
        assert all(not isinstance(result, BaseException) for result in theirs)
        async with pool.acquire() as conn:
            held = {
                row["owner_id"]: row["held"]
                for row in await conn.fetch(
                    "SELECT owner_id, count(*) AS held FROM dlightrag_runs"
                    " WHERE status IN ('queued', 'running') GROUP BY owner_id"
                )
            }
        assert held == {_OWNER: 3, _OTHER_OWNER: 2}, "a refusal leaves no run behind"

    async def test_a_replay_at_the_limit_is_answered_and_only_a_run_that_ended_makes_room(
        self, store, pool
    ) -> None:
        async def submit(key: str, limit: int | None = 2) -> Any:
            return await store.create_run(
                owner_id=_OWNER, request=_request(key), idempotency_key=key, active_run_limit=limit
            )

        first, second = await submit("a"), await submit("b")

        # At the limit a retried key is still answered, and a new one is not.
        replay = await submit("a")
        assert (replay.replayed, replay.run.run_id) == (True, first.run.run_id)
        with pytest.raises(ActiveRunLimitExceededError) as refused:
            await submit("c")
        assert refused.value.limit == 2

        # A run waiting out a dependency is queued still, and keeps counting.
        claim = await _claimed(store)
        assert await store.defer(
            owner_id=_OWNER,
            run_id=claim.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            checkpoint={"phase": "deferred_dependency"},
            next_attempt_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        )
        with pytest.raises(ActiveRunLimitExceededError):
            await submit("c")

        # A limit lowered below what the owner holds refuses more and cancels nothing.
        with pytest.raises(ActiveRunLimitExceededError):
            await submit("c", limit=1)
        async with pool.acquire() as conn:
            statuses = await conn.fetchval(
                "SELECT array_agg(status ORDER BY created_at) FROM dlightrag_runs"
            )
        assert statuses == ["queued", "queued"]

        # The one that ended makes room for exactly one more, under the key that was refused.
        cancelled = await store.request_cancellation(owner_id=_OWNER, run_id=second.run.run_id)
        assert cancelled.outcome == "cancelled"
        admitted = await submit("c")
        assert admitted.replayed is False
        with pytest.raises(ActiveRunLimitExceededError):
            await submit("d")

    async def test_linked_acceptance_honours_the_owners_active_limit(self, store, pool) -> None:
        await store.create_run(
            owner_id=_OWNER, request=_request("first"), idempotency_key="first", active_run_limit=1
        )

        async with pool.acquire() as conn, conn.transaction():
            with pytest.raises(ActiveRunLimitExceededError):
                await store.create_run_in(
                    conn, owner_id=_OWNER, request=_request("second"), active_run_limit=1
                )

    @pytest.mark.parametrize("limit", [0, -1])
    async def test_a_limit_below_one_is_no_limit_the_store_will_count_by(
        self, store, limit: int
    ) -> None:
        with pytest.raises(ValueError, match="active run limit"):
            await store.create_run(
                owner_id=_OWNER, request=_request(), idempotency_key="k", active_run_limit=limit
            )

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

    async def test_queued_cancellation_racing_a_claim_reports_pending(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        async with pool.acquire() as claiming, claiming.transaction():
            await claiming.execute(
                "UPDATE dlightrag_runs SET status = 'running', lease_owner = 'claiming-worker', "
                "lease_expires_at = NOW() + INTERVAL '30 seconds', "
                "fencing_epoch = fencing_epoch + 1, started_at = NOW(), "
                "updated_at = NOW() "
                "WHERE owner_id = $1 AND run_id = $2",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
            cancellation = asyncio.create_task(
                store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
            )
            await asyncio.sleep(0.05)
            assert not cancellation.done()

        outcome = await asyncio.wait_for(cancellation, timeout=5)
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

    async def test_failure_yields_to_a_cancellation_that_won_the_row(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        accepted = await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        assert accepted.outcome == "pending"

        outcome, repeated = await asyncio.gather(
            store.finish_failure(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                error_kind="provider_error",
                error_message="provider failed",
                result={"partial": True},
            ),
            store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id),
        )

        assert repeated.outcome in {"pending", "already_terminal"}
        assert outcome.committed is True
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert record.error_kind is None
        assert record.error_message is None
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("done", {"status": "cancelled"})
        ]

    async def test_fast_terminal_yields_when_cancellation_commits_first(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        stage_id = StageIntentId.deterministic(
            run_id=creation.run.run_id,
            name="fast:final_generation:2",
        )

        await store.request_cancellation(owner_id=_OWNER, run_id=creation.run.run_id)
        outcome = await claim.execution.progress_store.settle_terminal(
            expected_progress_version=0,
            stage_intent_id=stage_id,
            state={"result": {"answer": "withheld"}},
            result={"answer": "withheld"},
        )

        assert isinstance(outcome, StageTerminalCommit)
        assert outcome.status == "cancelled"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "cancelled"
        assert record.result is None
        assert record.durable_progress_version == 0
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "done"
        assert events[0].payload == {"status": "cancelled"}

    async def test_fast_terminal_can_win_when_it_holds_the_row_first(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        stage_id = StageIntentId.deterministic(
            run_id=creation.run.run_id,
            name="fast:final_generation:2",
        )

        outcome = await claim.execution.progress_store.settle_terminal(
            expected_progress_version=0,
            stage_intent_id=stage_id,
            state={"result": {"answer": "winner"}},
            result={"answer": "winner"},
        )
        cancellation = await store.request_cancellation(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
        )

        assert isinstance(outcome, StageTerminalCommit)
        assert outcome.status == "succeeded"
        assert cancellation.outcome == "already_terminal"
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "succeeded"
        assert record.result == {"answer": "winner"}
        assert record.durable_progress_version == 1
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert len(events) == 1
        assert events[0].event_type == "done"
        assert events[0].payload == {
            "status": "succeeded",
            "result": {"answer": "winner"},
        }


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

    async def test_concurrent_query_claimers_claim_every_eligible_run_once(self, store) -> None:
        for index in range(24):
            await store.create_run(owner_id=_OWNER, request=_request(f"run-{index}"))

        attempts = await asyncio.gather(
            *(store.claim_next(worker_id=f"host-{index}") for index in range(24))
        )
        claimed = [claim for claim in attempts if claim is not None]
        assert len(claimed) == 24
        assert len({claim.run.run_id for claim in claimed}) == 24
        assert await store.claim_next(worker_id="no-work-left") is None

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

    async def test_claimed_workspace_handoff_persists_agent_workspace_epoch(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        workspace = PGAgentspaceStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        outcome = await workspace.handoff_epoch(
            expected_epoch=None,
            destination_epoch=claim.run.fencing_epoch,
            inventory=(),
        )

        assert isinstance(outcome, HandoffCommit)
        assert outcome.workspace_epoch == claim.run.fencing_epoch
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.agent_workspace_epoch == claim.run.fencing_epoch

    async def test_a_stale_workspace_handoff_changes_nothing_and_is_not_progress(
        self, store, pool
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        workspace = PGAgentspaceStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        epoch = claim.run.fencing_epoch
        handed_off = await workspace.handoff_epoch(
            expected_epoch=None, destination_epoch=epoch, inventory=()
        )
        assert handed_off == HandoffCommit(workspace_epoch=epoch)

        stale = await workspace.handoff_epoch(
            expected_epoch=None,
            destination_epoch=epoch + 1,
            inventory=(
                InventoryPathRecord(relative_path="notes/a.md", entry_type="file", size_bytes=1),
            ),
        )

        assert stale == HandoffConflict(expected_epoch=None, current_epoch=epoch)
        assert await workspace.load_inventory() == ()
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.agent_workspace_epoch == epoch
        # A handoff moves the workspace, never the Run's durable progress.
        assert record.durable_progress_version == 0

    async def test_handoff_and_rescan_write_the_whole_inventory(self, store, pool) -> None:
        """A handoff carries the copied inventory; a settlement's rescan replaces it exactly."""
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        run_id = uuid.UUID(creation.run.run_id)
        workspace = PGAgentspaceStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        def record(path: str, size: int, digest: str | None = None) -> InventoryPathRecord:
            return InventoryPathRecord(
                relative_path=path,
                entry_type="file",
                size_bytes=size,
                mode=0o100644,
                content_digest=digest,
            )

        def observed(records: Any) -> list[tuple[str, int, str | None]]:
            return [(item.relative_path, item.size_bytes, item.content_digest) for item in records]

        handed_off = (record("notes/a.md", 1, "a" * 64), record("src/b.py", 2), record("c.txt", 3))
        outcome = await workspace.handoff_epoch(
            expected_epoch=None,
            destination_epoch=claim.run.fencing_epoch,
            inventory=handed_off,
        )
        assert isinstance(outcome, HandoffCommit)
        assert observed(await workspace.load_inventory()) == [
            ("c.txt", 3, None),
            ("notes/a.md", 1, "a" * 64),
            ("src/b.py", 2, None),
        ]

        replaced = (record("notes/a.md", 10, "b" * 64), record("d.md", 4))
        async with pool.acquire() as conn, conn.transaction():
            await write_inventory(conn, _OWNER, run_id, upserts=replaced, replace_all=True)
        assert observed(await workspace.load_inventory()) == [
            ("d.md", 4, None),
            ("notes/a.md", 10, "b" * 64),
        ]

    async def test_recent_spills_are_newest_first_by_producing_intent(self, store, pool) -> None:
        """A summary's spill handles come from the row, ordered by its intent.

        The epoch-copy read walks every spill by resource id; this one has to name
        the newest few, and the producing effect intent is the only monotone mark
        the row carries.
        """
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        run_id = uuid.UUID(creation.run.run_id)
        workspace = PGAgentspaceStore(
            pool=pool,
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        async with pool.acquire() as conn, conn.transaction():
            for resource_id, ordinal in (("spill_z", 1), ("spill_m", 3), ("spill_a", 2)):
                await write_committed_spill(
                    conn,
                    _OWNER,
                    run_id,
                    CommittedSpillRecord(
                        resource_id=resource_id,
                        content_digest="b" * 64,
                        size_bytes=8,
                        session_id="01930000-0000-7000-8000-0000000000ff",
                        intent_id=f"01930000-0000-7000-8000-00000000000{ordinal}",
                    ),
                )

        newest = await workspace.load_recent_spills(limit=2)

        # The ids sort against the intents, so resource-id order cannot pass.
        assert [spill.resource_id for spill in newest] == ["spill_m", "spill_a"]
        with pytest.raises(ValueError, match="spill page limit"):
            await workspace.load_recent_spills(limit=0)

    async def test_reclaim_abandonment_persists_the_exact_public_error(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS - 1,
            )

        assert await store.claim_next(worker_id="abandoning-worker") is None
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == RUN_ABANDONED_ERROR_KIND
        assert record.error_message == _ABANDONED_ERROR_MESSAGE
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            (
                "error",
                {
                    "kind": RUN_ABANDONED_ERROR_KIND,
                    "message": _ABANDONED_ERROR_MESSAGE,
                },
            )
        ]

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

    async def test_a_lapsed_lease_is_renewed_by_the_worker_that_still_owns_it(
        self, store, pool
    ) -> None:
        """A lapse is reclaimability, not ownership loss: the epoch decides, not the clock.

        This test used to assert the opposite -- "an expired lease is never revived" -- which made a
        busy worker's lapse fatal to a Run it was still working on. What keeps that safe is the
        fencing epoch: a worker only renews while it is still the row's lease_owner at the epoch it
        was granted, so a reclaimed Run refuses it (``test_stale_worker_cannot_write_after_reclaim``
        and ``test_run_lease_renewal_pg.py`` pin that half).
        """
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await _expire_lease(pool, creation.run.run_id)
        renewal = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert renewal.renewed is True
        assert (
            await pool.fetchval(
                "SELECT lease_expires_at > NOW() FROM dlightrag_runs WHERE run_id = $1",
                creation.run.run_id,
            )
            is True
        )

        # And the worker that no longer owns it is still refused, which is what the old assertion
        # was protecting.
        await pool.execute(
            "UPDATE dlightrag_runs SET lease_owner = 'worker-2', "
            "fencing_epoch = fencing_epoch + 1 WHERE run_id = $1",
            creation.run.run_id,
        )
        refused = await store.heartbeat(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert refused.renewed is False

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
        assert (
            await store.append_event(
                **stale_args, phase=None, event_type="token", payload={"text": "stale"}
            )
            is None
        )
        assert (
            await store.append_event(**stale_args, phase=None, event_type="reset", payload={})
            is None
        )
        assert (
            await store.finish_success(**stale_args, result={"answer": "stale"})
        ).committed is False
        assert (
            await store.finish_failure(
                **stale_args, error_kind="provider_error", error_message="stale"
            )
        ).committed is False
        assert await store.release_for_shutdown(**stale_args) == "lease_lost"
        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
        )

        assert creation.run.prepared_input is not None
        stale_session = SessionId(str(creation.run.prepared_input["agent_session_id"]))
        from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository

        stale_repository = PGAgentSessionRepository(
            pool=pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=int(stale_args["fencing_epoch"]),
        )
        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        stale_append = await stale_repository.transact(
            session_id=stale_session,
            fencing_epoch=int(stale_args["fencing_epoch"]),
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
        assert stale_append.__class__.__name__ == "TransactionLeaseLost"

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

    async def test_executor_owned_event_labels_round_trip_without_runtime_registration(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        sequence = await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="thinking",
            payload={"step": 1},
        )
        assert sequence == 1
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("thinking", {"step": 1})
        ]

    async def test_tool_subjects_join_selects_one_non_empty_label_per_call(
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
        run_id = creation.run.run_id
        # A start event and a subject-less streaming update are not subject sources.
        await store.append_event(
            **args,
            phase=None,
            event_type="tool_start",
            payload={"call_id": "c1", "tool_name": "bash"},
        )
        await store.append_event(
            **args,
            phase=None,
            event_type="tool_progress",
            payload={"call_id": "c1", "text_chars": 0},
        )
        await store.append_event(
            **args,
            phase=None,
            event_type="tool_progress",
            payload={"call_id": "c1", "object_label": "rg -n TODO src"},
        )
        # A later update of the same call does not replace the first subject.
        await store.append_event(
            **args,
            phase=None,
            event_type="tool_progress",
            payload={"call_id": "c1", "object_label": "a later label"},
        )
        await store.append_event(
            **args,
            phase=None,
            event_type="tool_progress",
            payload={"call_id": "c2", "object_label": "quarterly revenue 2026"},
        )

        subjects = await store.load_tool_subjects(
            owner_id=_OWNER, run_id=run_id, call_ids=["c1", "c2", "missing"]
        )
        assert subjects == {"c1": "rg -n TODO src", "c2": "quarterly revenue 2026"}
        assert (
            await store.load_tool_subjects(owner_id=_OTHER_OWNER, run_id=run_id, call_ids=["c1"])
            == {}
        )
        assert (
            await store.load_tool_subjects(
                owner_id=_OWNER, run_id=str(uuid.uuid7()), call_ids=["c1"]
            )
            == {}
        )
        assert await store.load_tool_subjects(owner_id=_OWNER, run_id=run_id, call_ids=[]) == {}

        # A trimmed event log loses only the calls whose rows are gone.
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_run_events WHERE owner_id = $1 AND run_id = $2 "
                "AND payload->>'call_id' = 'c2'",
                _OWNER,
                uuid.UUID(run_id),
            )
        assert await store.load_tool_subjects(
            owner_id=_OWNER, run_id=run_id, call_ids=["c1", "c2"]
        ) == {"c1": "rg -n TODO src"}

    async def test_tool_call_arguments_read_the_run_transcript_in_call_order(
        self, store, pool
    ) -> None:
        session = "0f0f0f0f-0000-7000-8000-0000000000ab"
        creation = await store.create_run(
            owner_id=_OWNER, request=_request(agent_session_id=session)
        )
        run_id = str(creation.run.run_id)
        first, second = uuid.uuid4(), uuid.uuid4()
        now = datetime.datetime.now(datetime.UTC)
        command = "cd . && python scripts/validate_model.py case.draft.json"
        first_payload = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "bash", "arguments": {"command": command}},
                # A call the model shaped illegally is not executable evidence; the reader still
                # keeps its id out of the mapping rather than guessing at arguments.
                {"id": "bad", "name": "bash", "arguments": "not-an-object"},
            ],
        }
        second_payload = {
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "name": "write",
                    "arguments": {"path": "notes/x.md", "content": "x"},
                },
                # A later entry reusing a call id never replaces the call the Run recorded first.
                {"id": "c1", "name": "bash", "arguments": {"command": "a later duplicate"}},
            ],
        }
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
                "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
                _OWNER,
                uuid.UUID(session),
                uuid.UUID(run_id),
            )
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
                "entry_id, parent_entry_id, entry_type, schema_version, timestamp, payload_json, "
                "search_text, run_id) VALUES "
                "($1, $2, 1, $3, NULL, 'assistant_message', 1, $4, $6::jsonb, '', $8), "
                "($1, $2, 2, $5, $3, 'assistant_message', 1, $7, $9::jsonb, '', $8)",
                _OWNER,
                uuid.UUID(session),
                first,
                now,
                second,
                json.dumps(first_payload),
                now,
                uuid.UUID(run_id),
                json.dumps(second_payload),
            )

        calls = await store.load_tool_call_arguments(owner_id=_OWNER, run_id=run_id)

        assert [(call.call_id, call.tool_name) for call in calls] == [
            ("c1", "bash"),
            ("c2", "write"),
        ]
        assert calls[0].arguments == {"command": command}
        assert calls[1].arguments == {"path": "notes/x.md", "content": "x"}
        assert await store.load_tool_call_arguments(owner_id=_OTHER_OWNER, run_id=run_id) == ()
        assert await store.load_tool_call_arguments(owner_id=_OWNER, run_id=str(uuid.uuid7())) == ()

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
            *(
                store.append_event(
                    **args,
                    phase=None,
                    event_type="token",
                    payload={"text": f"chunk-{index}"},
                )
                for index in range(24)
            )
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
        await store.append_event(**args, phase=None, event_type="token", payload={"text": "one"})
        await store.append_event(**args, phase=None, event_type="reset", payload={})
        await store.append_event(**args, phase=None, event_type="token", payload={"text": "two"})
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
                "UPDATE dlightrag_runs "
                "SET lease_expires_at = NOW() + INTERVAL '1 second' WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        await store.append_event(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            phase=None,
            event_type="token",
            payload={"text": "hi"},
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
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            ("error", {"kind": "provider_error", "message": "boom"})
        ]

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
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

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
                "UPDATE dlightrag_runs "
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

        requirements = [
            requirement async for requirement in store.iter_active_run_requirements(page_size=1)
        ]

        assert {row["prepared_input"]["context_policy_revision"] for row in requirements} == {
            "queued",
            "recoverable",
        }
        assert all(row["run_kind"] == "answer" for row in requirements)
        assert all(row["prepared_input"]["pinned_models"] == [] for row in requirements)

    async def test_cancel_pending_rescan_keyset_pages_only_live_worker_leases(
        self,
        store,
        pool,
    ) -> None:
        first = await store.create_run(owner_id=_OWNER, request=_request())
        first_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=first.run.run_id)

        second = await store.create_run(owner_id=_OWNER, request=_request())
        second_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=second.run.run_id)

        other = await store.create_run(owner_id=_OWNER, request=_request())
        other_claim = await store.claim_next(worker_id="other-worker")
        assert other_claim is not None
        await store.request_cancellation(owner_id=_OWNER, run_id=other.run.run_id)

        expired = await store.create_run(owner_id=_OWNER, request=_request())
        expired_claim = await _claimed(store)
        await store.request_cancellation(owner_id=_OWNER, run_id=expired.run.run_id)
        await _expire_lease(pool, expired.run.run_id)

        pending = [
            item
            async for item in store.iter_cancel_pending(
                worker_id=_WORKER,
                page_size=1,
            )
        ]

        assert {run_id for _, run_id in pending} == {
            first_claim.run.run_id,
            second_claim.run.run_id,
        }
        assert other_claim.run.run_id not in {run_id for _, run_id in pending}
        assert expired_claim.run.run_id not in {run_id for _, run_id in pending}

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
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

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
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert events[0].payload == {"status": "cancelled"}

    async def test_sweeper_abandonment_persists_the_exact_public_error(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
                "SET reclaims_without_progress = $2, lease_expires_at = NOW() - INTERVAL '1 second' "
                "WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
                MAX_RECLAIMS_WITHOUT_PROGRESS,
            )

        sweep = await store.sweep_once()

        assert sweep.cancelled == 0
        assert sweep.abandoned == 1
        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error_kind == RUN_ABANDONED_ERROR_KIND
        assert record.error_message == _ABANDONED_ERROR_MESSAGE
        events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
        assert [(event.event_type, event.payload) for event in events] == [
            (
                "error",
                {
                    "kind": RUN_ABANDONED_ERROR_KIND,
                    "message": _ABANDONED_ERROR_MESSAGE,
                },
            )
        ]

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
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == (
            artifact.content
        )

    async def test_multichunk_and_empty_blobs_round_trip_exactly(self, store, pool) -> None:
        contents = (b"a" * BLOB_CHUNK_BYTES + b"middle" + b"z" * BLOB_CHUNK_BYTES, b"")
        artifacts = tuple(PendingArtifact(content=content) for content in contents)

        for index, artifact in enumerate(artifacts):
            await store.create_run(
                owner_id=_OWNER,
                request=_request(f"blob-{index}"),
                artifacts=[artifact],
                references=[_reference(artifact.digest, resource_id=f"res-{index}")],
            )
            assert (
                await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest)
                == contents[index]
            )

        async with pool.acquire() as conn:
            counts = [
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blob_chunks"
                    " WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                for artifact in artifacts
            ]
        assert counts == [3, 0]

    async def test_concurrent_same_digest_writers_converge_on_one_ordered_blob(
        self, store, pool
    ) -> None:
        content = b"first" + b"m" * (2 * BLOB_CHUNK_BYTES) + b"last"
        artifact = PendingArtifact(content=content)

        creations = await asyncio.gather(
            *(
                store.create_run(
                    owner_id=_OWNER,
                    request=_request(f"concurrent-{index}"),
                    artifacts=[artifact],
                    references=[_reference(artifact.digest, resource_id=f"res-{index}")],
                )
                for index in range(2)
            )
        )

        assert len({creation.run.run_id for creation in creations}) == 2
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == content
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                == 1
            )
            chunk_rows = await conn.fetch(
                "SELECT chunk_index FROM dlightrag_blob_chunks"
                " WHERE owner_id = $1 AND digest = $2 ORDER BY chunk_index",
                _OWNER,
                artifact.digest,
            )
        assert [int(row["chunk_index"]) for row in chunk_rows] == list(range(3))

    async def test_new_blobs_are_acquired_in_canonical_order_across_acceptances(
        self, store
    ) -> None:
        first = PendingArtifact(content=b"new-opposite-order-a")
        second = PendingArtifact(content=b"new-opposite-order-b")

        creations = await asyncio.wait_for(
            asyncio.gather(
                store.create_run(
                    owner_id=_OWNER,
                    request=_request("new-order-left"),
                    artifacts=[first, second],
                    references=[
                        _reference(first.digest, resource_id="left-first", ordinal=0),
                        _reference(second.digest, resource_id="left-second", ordinal=1),
                    ],
                ),
                store.create_run(
                    owner_id=_OWNER,
                    request=_request("new-order-right"),
                    artifacts=[second, first],
                    references=[
                        _reference(second.digest, resource_id="right-second", ordinal=0),
                        _reference(first.digest, resource_id="right-first", ordinal=1),
                    ],
                ),
            ),
            timeout=5,
        )

        assert len({creation.run.run_id for creation in creations}) == 2
        assert await _blob_store(store).read(owner_id=_OWNER, digest=first.digest) == first.content
        assert (
            await _blob_store(store).read(owner_id=_OWNER, digest=second.digest) == second.content
        )
        left_references = await store.list_run_artifacts(
            owner_id=_OWNER, run_id=creations[0].run.run_id
        )
        right_references = await store.list_run_artifacts(
            owner_id=_OWNER, run_id=creations[1].run.run_id
        )
        assert [reference.resource_id for reference in left_references] == [
            "left-first",
            "left-second",
        ]
        assert [reference.resource_id for reference in right_references] == [
            "right-second",
            "right-first",
        ]

    async def test_existing_blobs_can_be_reused_in_opposite_order_without_deadlock(
        self, store, pool
    ) -> None:
        first = PendingArtifact(content=b"opposite-order-a")
        second = PendingArtifact(content=b"opposite-order-b")
        await store.create_run(
            owner_id=_OWNER,
            request=_request("opposite-order-seed"),
            artifacts=[first, second],
            references=[
                _reference(first.digest, resource_id="first", ordinal=0),
                _reference(second.digest, resource_id="second", ordinal=1),
            ],
        )

        async with pool.acquire() as left, pool.acquire() as right:
            async with left.transaction(), right.transaction():
                await write_blob_content(
                    left,
                    owner_id=_OWNER,
                    digest=first.digest,
                    content=first.content,
                )
                await write_blob_content(
                    right,
                    owner_id=_OWNER,
                    digest=second.digest,
                    content=second.content,
                )
                await asyncio.wait_for(
                    asyncio.gather(
                        write_blob_content(
                            left,
                            owner_id=_OWNER,
                            digest=second.digest,
                            content=second.content,
                        ),
                        write_blob_content(
                            right,
                            owner_id=_OWNER,
                            digest=first.digest,
                            content=first.content,
                        ),
                    ),
                    timeout=5,
                )

    async def test_uncommitted_blob_winner_exposes_size_mismatch_after_wait(
        self, store, pool
    ) -> None:
        artifact = PendingArtifact(content=b"uncommitted winner")
        async with (
            pool.acquire() as winner,
            pool.acquire() as contender,
            pool.acquire() as observer,
        ):
            winner_pid = int(await winner.fetchval("SELECT pg_backend_pid()"))
            contender_pid = int(await contender.fetchval("SELECT pg_backend_pid()"))
            winner_tx = winner.transaction()
            contender_tx = contender.transaction()
            contender_write: asyncio.Task[None] | None = None
            contender_consumed = False
            try:
                await winner_tx.start()
                await contender_tx.start()
                await write_blob_content(
                    winner,
                    owner_id=_OWNER,
                    digest=artifact.digest,
                    content=artifact.content,
                )
                contender_write = asyncio.create_task(
                    write_complete_blob(
                        contender,
                        owner_id=_OWNER,
                        digest=artifact.digest,
                        total_bytes=len(artifact.content) + 1,
                        chunks=(artifact.content + b"!",),
                    )
                )
                for _attempt in range(100):
                    blockers = await observer.fetchval("SELECT pg_blocking_pids($1)", contender_pid)
                    if winner_pid in blockers:
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("contender never waited on the uncommitted blob winner")
                assert not contender_write.done()
                await winner_tx.commit()
                with pytest.raises(BlobSizeConflict):
                    await asyncio.wait_for(contender_write, timeout=5)
                contender_consumed = True
            finally:
                task_error: BaseException | None = None
                if contender_write is not None and not contender_consumed:
                    if not contender_write.done():
                        contender_write.cancel()
                    try:
                        await contender_write
                    except asyncio.CancelledError, BlobSizeConflict:
                        pass
                    except BaseException as exc:
                        task_error = exc
                rollbacks = []
                if winner.is_in_transaction():
                    rollbacks.append(winner_tx.rollback())
                if contender.is_in_transaction():
                    rollbacks.append(contender_tx.rollback())
                if rollbacks:
                    await asyncio.gather(*rollbacks, return_exceptions=True)
                if task_error is not None:
                    raise task_error

        assert (
            await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest)
            == artifact.content
        )

    async def test_blob_size_collision_rolls_back_acceptance_and_publication(
        self, store, pool
    ) -> None:
        content = b"collision payload"
        artifact = PendingArtifact(content=content)
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_blobs (owner_id, digest, byte_size) VALUES ($1, $2, $3)",
                _OWNER,
                artifact.digest,
                len(content) + 1,
            )

        with pytest.raises(ValueError, match="blob digest collision"):
            await store.create_run(
                owner_id=_OWNER,
                request=_request("collision-acceptance"),
                artifacts=[artifact],
                references=[_reference(artifact.digest)],
            )

        creation = await store.create_run(owner_id=_OWNER, request=_request("publication"))
        claim = await _claimed(store)
        assert claim.run.run_id == creation.run.run_id
        publication = PendingPublication(
            resource_id="published-1",
            reference_kind="published_artifact",
            filename="result.txt",
            mime_type="text/plain",
            content=content,
        )
        with pytest.raises(ValueError, match="blob digest collision"):
            await store.finish_success(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": "not committed"},
                publications=[publication],
            )

        record = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.status == "running"
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_blob_chunks"
                    " WHERE owner_id = $1 AND digest = $2",
                    _OWNER,
                    artifact.digest,
                )
                == 0
            )
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_run_artifacts") == 0

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
        assert await _blob_store(store).read(owner_id="ghost", digest=artifact.digest) is None
        assert first.run.run_id is not None

    async def test_deleting_one_run_keeps_bytes_another_run_still_links(self, store, pool) -> None:
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
        deletion = await delete_runs(pool, store, owner_id=_OWNER, run_ids=[first.run.run_id])
        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is not None

        final = await delete_runs(pool, store, owner_id=_OWNER, run_ids=[second.run.run_id])
        assert final.runs == 1
        assert final.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is None

    async def test_deleting_a_run_collects_its_fetched_resource_blob(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        digest = await _write_fetched_blob(pool, creation.run.run_id, b"fetched delete bytes")

        deletion = await delete_runs(pool, store, owner_id=_OWNER, run_ids=[creation.run.run_id])

        assert deletion.runs == 1
        assert deletion.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None

    async def test_deletion_is_owner_scoped(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        deletion = await delete_runs(
            pool, store, owner_id=_OTHER_OWNER, run_ids=[creation.run.run_id]
        )
        assert deletion.runs == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is not None

    async def test_unlinked_session_is_removed_only_after_its_final_run_route(
        self, store, pool
    ) -> None:
        session_id = str(uuid.uuid7())
        first = await store.create_run(
            owner_id=_OWNER,
            request=_request("first", agent_session_id=session_id),
        )
        second = await store.create_run(
            owner_id=_OWNER,
            request=_request("second", agent_session_id=session_id),
        )
        claimed = await _claimed(store)
        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
        )

        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        await claimed.execution.session_repository.transact(
            session_id=SessionId(session_id),
            fencing_epoch=claimed.execution.fencing_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )

        assert (
            await delete_runs(pool, store, owner_id=_OWNER, run_ids=[first.run.run_id])
        ).runs == 1
        async with pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM dlightrag_agent_sessions"
                " WHERE owner_id = $1 AND session_id = $2)",
                _OWNER,
                uuid.UUID(session_id),
            )

        assert (
            await delete_runs(pool, store, owner_id=_OWNER, run_ids=[second.run.run_id])
        ).runs == 1
        async with pool.acquire() as conn:
            assert not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM dlightrag_agent_sessions"
                " WHERE owner_id = $1 AND session_id = $2)",
                _OWNER,
                uuid.UUID(session_id),
            )

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
                delete_runs(pool, store, owner_id=_OWNER, run_ids=[first.run.run_id]), timeout=10
            )
        finally:
            await transaction.commit()
            await pool.release(adopting)

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) == (
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
        assert outcome.deleted == (
            DeletedRun(owner_id=_OWNER, run_id=creation.run.run_id, run_kind="answer"),
        )
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is None
        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "SELECT count(*) FROM dlightrag_run_events WHERE run_id = $1",
                uuid.UUID(creation.run.run_id),
            )
        assert int(remaining) == 0

    async def test_prune_collects_fetched_resource_blobs(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        digest = await _write_fetched_blob(pool, creation.run.run_id, b"fetched retention bytes")
        claim = await _claimed(store)
        await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "old"},
        )
        await _backdate_finish(pool, creation.run.run_id, days=370)

        outcome = await store.prune_expired_runs()

        assert outcome.runs == 1
        assert outcome.artifacts == 1
        assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None

    async def test_prune_leaves_unfinished_runs_alone(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs "
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


async def _succeed_and_expire(store: PGRunStore, pool: Any, run_id: str) -> None:
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

        deletion = await delete_runs(pool, store, owner_id=_OWNER, run_ids=[creation.run.run_id])

        assert deletion.runs == 1
        assert deletion.artifacts == 0
        assert await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id) is None
        assert await _blob_store(store).read(owner_id=_OWNER, digest=artifact.digest) is not None

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
            assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is not None

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
        assert await _blob_store(store).read(owner_id=_OWNER, digest=contended) is not None
        for digest in (digests[0], digests[2]):
            assert await _blob_store(store).read(owner_id=_OWNER, digest=digest) is None


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
            await store.append_event(
                **args,
                phase=None,
                event_type="token",
                payload={"text": f"chunk-{index}"},
            )
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


@dataclass(frozen=True)
class _LiveChild:
    """A claimed research Run with one running Child Session and one pending ask."""

    run_id: str
    fencing_epoch: int
    child_session_id: str
    request_id: str


async def _run_with_a_live_child(store: FingerprintingRunStore, query: str) -> _LiveChild:
    creation = await store.create_run(owner_id=_OWNER, request=_request(query, mode="research"))
    claim = await _claimed(store)
    run_id = creation.run.run_id
    assert claim.run.run_id == run_id
    parent_id = str(uuid.uuid7())
    child_id = str(uuid.uuid7())
    assert await store.upsert_child_session(
        owner_id=_OWNER,
        run_id=run_id,
        child_session_id=child_id,
        parent_session_id=parent_id,
        parent_call_id=f"{query}-call",
        parent_intent_id=str(uuid.uuid7()),
        objective=f"{query} objective",
        context_mode="isolated",
        model_role="query",
        tools=("search_knowledge_base",),
        depth=1,
        context_snapshot={
            "parent_session_id": parent_id,
            "parent_entry_id": str(uuid.uuid7()),
            "depth": 0,
            "messages": [],
            "evidence_state": {},
        },
        plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
        budget={"provider_attempt_limit": 2},
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
    )
    child_epoch = await store.claim_child_session(
        owner_id=_OWNER,
        run_id=run_id,
        child_session_id=child_id,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
    )
    child = await store.load_child_session(
        owner_id=_OWNER, run_id=run_id, child_session_id=child_id
    )
    assert child is not None and child_epoch is not None
    request_id = str(uuid.uuid7())
    assert await store.create_child_guidance(
        owner_id=_OWNER,
        run_id=run_id,
        request_id=request_id,
        child_session_id=child_id,
        child_operation_id=child["operation_id"],
        parent_session_id=parent_id,
        question=f"{query} question",
        expires_after_seconds=300,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
        child_fencing_epoch=child_epoch,
    )
    return _LiveChild(run_id, claim.run.fencing_epoch, child_id, request_id)


async def _child_states(store: PGRunStore, live: _LiveChild) -> tuple[str, str, str]:
    """The Child Session, its current Operation, and its ask, by status."""
    child = await store.load_child_session(
        owner_id=_OWNER, run_id=live.run_id, child_session_id=live.child_session_id
    )
    guidance = await store.load_child_guidance(
        owner_id=_OWNER, run_id=live.run_id, request_id=live.request_id
    )
    assert child is not None and guidance is not None
    return str(child["status"]), str(child["operation_status"]), str(guidance["status"])


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

    async def test_a_long_inbox_is_read_in_bounded_ordered_batches(self, store) -> None:
        """The parent inbox has no enqueue cap, so one read locks a bounded batch."""
        from dlightrag.adapters.postgres.runtime._child import PENDING_CONTROL_READ_LIMIT

        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        total = PENDING_CONTROL_READ_LIMIT + 5
        for index in range(total):
            queued = await store.enqueue_agent_control(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                kind="steer",
                content=f"steer {index}",
            )
            assert queued is not None
        claim = await _claimed(store)

        delivered: list[str] = []
        batches: list[int] = []
        while True:
            batch = await store.load_pending_agent_controls(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
            assert batch is not None
            if not batch:
                break
            batches.append(len(batch))
            delivered.extend(str(item["content"]) for item in batch)
            assert await store.acknowledge_agent_controls(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                control_sequences=tuple(int(item["control_sequence"]) for item in batch),
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )

        assert batches == [PENDING_CONTROL_READ_LIMIT, 5]
        assert delivered == [f"steer {index}" for index in range(total)]

    @pytest.mark.parametrize("model_role", ["query", "extract", "keyword", "vlm", "default"])
    async def test_one_spawn_call_persists_multiple_child_lineages(self, store, model_role) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        children = (str(uuid.uuid7()), str(uuid.uuid7()))
        parent_intent_id = str(uuid.uuid7())
        child_epochs: list[int] = []

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
                model_role=model_role,
                tools=("search_knowledge_base",),
                depth=1,
                context_snapshot={
                    "parent_session_id": parent_id,
                    "parent_entry_id": str(uuid.uuid7()),
                    "depth": 0,
                    "messages": [{"role": "user", "content": "parent"}],
                    "evidence_state": {},
                },
                plan={
                    "schema_version": 1,
                    "tools": ["search_knowledge_base"],
                    "model_role": model_role,
                    "model_identity": {"model": f"selected-{model_role}"},
                    "model_profile": {"supports_images": model_role == "vlm"},
                },
                budget={"provider_attempt_limit": 2},
                host_state={"evidence": {}},
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
            child_epoch = await store.claim_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
            assert child_epoch is not None
            child_epochs.append(child_epoch)
            from dlightrag.engine.agent.session.ids import LaneId, SessionId
            from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
            from dlightrag.engine.agent.session.transactions import (
                RegisterExpectation,
                SessionTransaction,
                TransactionCommit,
                TransactionLeaseLost,
            )

            child_store = claim.execution.session_repository.for_child(
                SessionId(child_id),
                fencing_epoch=child_epoch,
            )
            head = LaneHead(LaneId.main(), None)
            state = LaneState(LaneId.main())
            child_commit = await child_store.transact(
                session_id=SessionId(child_id),
                fencing_epoch=child_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[SetRegister(head), SetRegister(state)],
                    expectations=[
                        RegisterExpectation(head.ref, None),
                        RegisterExpectation(state.ref, None),
                    ],
                ),
            )
            assert isinstance(child_commit, TransactionCommit)
            parent_write = await claim.execution.session_repository.transact(
                session_id=SessionId(child_id),
                fencing_epoch=claim.run.fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[SetRegister(state)],
                    expectations=[RegisterExpectation(state.ref, child_commit.commit_sequence)],
                ),
            )
            assert isinstance(parent_write, TransactionLeaseLost)
            assert await store.finish_child_session(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                child_session_id=child_id,
                status="succeeded",
                summary="done",
                usage={"input_tokens": 10 + position},
                outcome={
                    "status": "succeeded",
                    "summary": "done",
                    "handles": [],
                    "usage": {"input_tokens": 10 + position},
                    "child_session_id": child_id,
                    "evidence_state": {"contexts": {}},
                },
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )

        roster = await store.list_child_sessions(owner_id=_OWNER, run_id=creation.run.run_id)
        assert {item["child_session_id"] for item in roster} == set(children)
        assert {item["parent_session_id"] for item in roster} == {parent_id}
        assert {item["parent_intent_id"] for item in roster} == {parent_intent_id}
        assert [item["objective"] for item in roster] == ["child 0", "child 1"]
        assert [item["usage"]["input_tokens"] for item in roster] == [10, 11]
        assert child_epochs == [1, 1]
        assert [item["depth"] for item in roster] == [1, 1]
        assert all(item["context_snapshot"]["messages"] for item in roster)
        assert all(item["plan"]["tools"] == ["search_knowledge_base"] for item in roster)
        assert all(item["model_role"] == model_role for item in roster)
        assert all(
            item["plan"]["model_identity"] == {"model": f"selected-{model_role}"} for item in roster
        )
        assert all(
            item["plan"]["model_profile"]["supports_images"] == (model_role == "vlm")
            for item in roster
        )
        assert all(item["budget"]["provider_attempt_limit"] == 2 for item in roster)
        assert [item["host_state"]["terminal_outcome"]["status"] for item in roster] == [
            "succeeded",
            "succeeded",
        ]
        assert all(
            item["host_state"]["terminal_outcome"]["evidence_state"] == {"contexts": {}}
            for item in roster
        )

    async def test_child_controls_continuation_and_correlated_guidance_are_durable(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        run_id = creation.run.run_id
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="interactive-call",
            parent_intent_id=str(uuid.uuid7()),
            objective="initial objective",
            context_mode="parent",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=1,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 0,
                "messages": [{"role": "user", "content": "parent context"}],
                "evidence_state": {},
            },
            plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
            budget={"provider_attempt_limit": 2},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert child_epoch == 1
        assert child is not None
        initial_operation_id = child["operation_id"]

        stale_parent = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            content="stale parent",
            submission_key="stale-parent",
            origin="parent",
            worker_id="stale-worker",
            fencing_epoch=claim.run.fencing_epoch,
        )
        queued = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            content="focus on primary sources",
            submission_key="steer-one",
        )
        duplicate = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            content="focus on primary sources",
            submission_key="steer-one",
        )
        conflict = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            content="different",
            submission_key="steer-one",
        )
        foreign = await store.enqueue_child_control(
            owner_id="another-owner",
            run_id=run_id,
            child_session_id=child_id,
            content="steal",
            submission_key="foreign",
        )
        parent_controls = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_controls = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            target_session_id=child_id,
            target_operation_id=initial_operation_id,
            child_fencing_epoch=child_epoch,
        )
        assert stale_parent is False
        assert queued["outcome"] == duplicate["outcome"] == "queued"
        assert queued["control_sequence"] == duplicate["control_sequence"]
        assert conflict["outcome"] == "idempotency_conflict"
        assert foreign["outcome"] == "unknown_child"
        assert [row["content"] for row in child_controls or ()] == ["focus on primary sources"]
        assert all(row["content"] != "focus on primary sources" for row in parent_controls or ())
        assert await store.acknowledge_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            control_sequences=(queued["control_sequence"],),
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            target_session_id=child_id,
            target_operation_id=initial_operation_id,
            child_fencing_epoch=child_epoch,
        )
        consumed = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="focus on primary sources",
            submission_key="steer-one",
        )
        assert consumed["outcome"] == "consumed"

        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            status="succeeded",
            summary="initial done",
            usage={"input_tokens": 3},
            outcome={
                "status": "succeeded",
                "summary": "initial done",
                "handles": [],
                "usage": {"input_tokens": 3},
                "child_session_id": child_id,
                "operation_id": initial_operation_id,
                "evidence_state": {"contexts": {}},
            },
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        late = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="too late",
            submission_key="late",
        )
        assert late["outcome"] == "terminal_child"

        continuation = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            content="follow up",
            submission_key="continue-one",
        )
        replay = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="follow up",
            submission_key="continue-one",
        )
        competing = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="competing",
            submission_key="continue-two",
        )
        continued = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert continuation["outcome"] == replay["outcome"] == "accepted"
        assert continuation["operation_id"] == replay["operation_id"]
        assert competing["outcome"] == "child_running"
        assert continued is not None
        assert continued["operation_sequence"] == 2
        assert continued["operation_input"] == "follow up"
        assert continued["plan"] == {"schema_version": 2, "tools": ["search_knowledge_base"]}
        assert continued["tools"] == ["search_knowledge_base"]
        assert continued["operation_usage"] == [{"input_tokens": 3}]

        next_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        request_id = str(uuid.uuid7())
        guidance = await store.create_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=request_id,
            child_session_id=child_id,
            child_operation_id=continuation["operation_id"],
            parent_session_id=parent_id,
            question="Which source should I prioritize?",
            expires_after_seconds=30,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=next_epoch,
        )
        pending = await store.list_pending_child_guidance(
            owner_id=_OWNER, run_id=run_id, parent_session_id=parent_id
        )
        reply = await store.reply_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=request_id,
            parent_session_id=parent_id,
            content="Use the official report.",
            submission_key="reply-one",
        )
        replied = await store.load_child_guidance(
            owner_id=_OWNER, run_id=run_id, request_id=request_id
        )
        assert guidance is not None and guidance["status"] == "pending"
        assert [row["request_id"] for row in pending] == [request_id]
        assert reply["outcome"] == "replied"
        assert replied is not None and replied["reply"] == "Use the official report."

        wake_request_id = str(uuid.uuid7())
        assert await store.create_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=wake_request_id,
            child_session_id=child_id,
            child_operation_id=continuation["operation_id"],
            parent_session_id=parent_id,
            question="Wake me with a reply",
            expires_after_seconds=30,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=next_epoch,
        )
        waiter = asyncio.create_task(
            store.wait_for_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=wake_request_id,
                timeout_seconds=5,
            )
        )
        await asyncio.sleep(0.05)
        assert (
            await store.reply_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=wake_request_id,
                content="wake now",
                submission_key="reply-wake",
            )
        )["outcome"] == "replied"
        woke = await asyncio.wait_for(waiter, timeout=2)
        assert woke is not None and woke["reply"] == "wake now"

        retiring_request_id = str(uuid.uuid7())
        assert await store.create_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=retiring_request_id,
            child_session_id=child_id,
            child_operation_id=continuation["operation_id"],
            parent_session_id=parent_id,
            question="This request will be retired",
            expires_after_seconds=30,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=next_epoch,
        )
        cancelled = await store.cancel_child_session_by_owner(
            submission_key="cancel-child",
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
        )
        assert cancelled["outcome"] == "cancellation_requested"
        retired = await store.load_child_guidance(
            owner_id=_OWNER, run_id=run_id, request_id=retiring_request_id
        )
        assert retired is not None and retired["status"] == "cancelled"
        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            status="cancelled",
            summary="cancelled",
            outcome={
                "status": "cancelled",
                "summary": "cancelled",
                "handles": [],
                "usage": {},
                "child_session_id": child_id,
                "operation_id": continuation["operation_id"],
                "evidence_state": None,
            },
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        denied = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="revive without permission",
            submission_key="continue-three",
        )
        parent_denied = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="parent cannot bypass cancellation",
            submission_key="continue-parent",
            origin="parent",
            reauthorize_user_cancelled=True,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        allowed = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="explicitly authorized",
            submission_key="continue-four",
            reauthorize_user_cancelled=True,
        )
        assert denied["outcome"] == "reauthorization_required"
        assert parent_denied["outcome"] == "reauthorization_required"
        assert allowed["outcome"] == "accepted"

    async def test_child_control_lease_expiry_and_terminal_run_races(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        run_id = creation.run.run_id
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="race-call",
            parent_intent_id=str(uuid.uuid7()),
            objective="race objective",
            context_mode="isolated",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=1,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 0,
                "messages": [],
                "evidence_state": {},
            },
            plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
            budget={"provider_attempt_limit": 2},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert child is not None and child_epoch == 1
        queued = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="keep the primary sources",
            submission_key="lease-steer",
        )
        lost_load = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            target_session_id=child_id,
            target_operation_id=child["operation_id"],
            child_fencing_epoch=child_epoch + 1,
        )
        lost_ack = await store.acknowledge_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            control_sequences=(queued["control_sequence"],),
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            target_session_id=child_id,
            target_operation_id=child["operation_id"],
            child_fencing_epoch=0,
        )
        held = await store.load_pending_agent_controls(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            target_session_id=child_id,
            target_operation_id=child["operation_id"],
            child_fencing_epoch=child_epoch,
        )
        assert queued["outcome"] == "queued"
        assert lost_load is None
        assert lost_ack is False
        assert [row["content"] for row in held or ()] == ["keep the primary sources"]

        request_id = str(uuid.uuid7())
        created = await store.create_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=request_id,
            child_session_id=child_id,
            child_operation_id=child["operation_id"],
            parent_session_id=parent_id,
            question="Expire this question",
            expires_after_seconds=1,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )
        assert created is not None and created["status"] == "pending"
        await asyncio.sleep(1.05)
        assert await store.expire_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=request_id,
            child_session_id=child_id,
            child_operation_id=child["operation_id"],
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )
        expired = await store.load_child_guidance(
            owner_id=_OWNER, run_id=run_id, request_id=request_id
        )
        late_reply = await store.reply_child_guidance(
            owner_id=_OWNER,
            run_id=run_id,
            request_id=request_id,
            content="too late",
            submission_key="expired-reply",
        )
        assert expired is not None and expired["status"] == "expired"
        assert late_reply["outcome"] == "expired"

        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            status="succeeded",
            summary="settled",
            usage={"input_tokens": 1},
            outcome={
                "status": "succeeded",
                "summary": "settled",
                "handles": [],
                "usage": {"input_tokens": 1},
                "child_session_id": child_id,
                "operation_id": child["operation_id"],
                "evidence_state": {"contexts": {}},
            },
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert (
            await store.finish_success(
                owner_id=_OWNER,
                run_id=run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": "parent done"},
            )
        ).committed is True
        late_continue = await store.continue_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="revive after parent terminal",
            submission_key="late-continue",
        )
        late_steer = await store.enqueue_child_control(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            content="steer after parent terminal",
            submission_key="late-steer",
        )
        assert late_continue["outcome"] == "run_terminal"
        assert late_steer["outcome"] == "terminal_child"

    async def test_sparse_precreate_promotes_to_full_reconstructible_child_envelope(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        parent_entry_id = str(uuid.uuid7())
        parent_intent_id = str(uuid.uuid7())

        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="sparse-first",
            parent_intent_id=parent_intent_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="sparse-first",
            parent_intent_id=parent_intent_id,
            objective="full objective",
            context_mode="parent",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=2,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": parent_entry_id,
                "depth": 1,
                "messages": [{"role": "user", "content": "full context"}],
                "evidence_state": {"contexts": {}},
            },
            plan={"schema_version": 2},
            budget={"provider_attempt_limit": 2},
            host_state={"dispatch_version": 3},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="sparse-first",
            parent_intent_id=parent_intent_id,
            objective="drifted objective",
            context_mode="isolated",
            model_role="extract",
            tools=(),
            depth=9,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 8,
                "messages": [],
                "evidence_state": {},
            },
            plan={"schema_version": 999},
            budget={"provider_attempt_limit": 999},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        child = await store.load_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
        )
        assert child is not None
        assert child["objective"] == "full objective"
        assert child["context"] == "parent"
        assert child["model_role"] == "query"
        assert child["tools"] == ["search_knowledge_base"]
        assert child["depth"] == 2
        assert child["context_snapshot"]["parent_entry_id"] == parent_entry_id
        assert child["plan"] == {"schema_version": 2}
        assert child["budget"] == {"provider_attempt_limit": 2}
        assert child["host_state"] == {"dispatch_version": 3}

    async def test_child_cancel_request_preserves_writer_until_operation_closes(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="cancel-call",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert child_epoch == 1
        assert await store.request_child_cancellation(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        from dlightrag.engine.agent.session.ids import LaneId, SessionId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
            TransactionCommit,
            TransactionLeaseLost,
        )

        child_store = claim.execution.session_repository.for_child(
            SessionId(child_id), fencing_epoch=child_epoch
        )
        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        closure_write = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
        assert isinstance(closure_write, TransactionCommit)
        child = await store.load_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
        )
        assert child is not None
        assert child["status"] == "running"
        assert child["cancel_requested_at"] is not None

        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            status="cancelled",
            summary="closed",
            outcome={
                "status": "cancelled",
                "summary": "closed",
                "handles": [],
                "usage": {},
                "child_session_id": child_id,
                "operation_id": str(uuid.uuid7()),
                "evidence_state": None,
            },
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        stale = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(state)],
                expectations=[RegisterExpectation(state.ref, closure_write.commit_sequence)],
            ),
        )
        assert isinstance(stale, TransactionLeaseLost)

    async def test_concurrent_guidance_waits_hold_no_pool_connection(self, store) -> None:
        """Waiting children listen through the notification hub, whatever their number.

        A connection per waiter would let enough waiting children exhaust the pool the
        replies they wait for must use; the hub listens on a connection of its own, so
        a waiting child holds none.
        """
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        run_id = creation.run.run_id
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="listen-pool",
            parent_intent_id=str(uuid.uuid7()),
            objective="wait without holding the pool",
            context_mode="isolated",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=1,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 0,
                "messages": [],
                "evidence_state": {},
            },
            plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
            budget={"provider_attempt_limit": 2},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert child is not None and child_epoch == 1
        request_ids = (str(uuid.uuid7()), str(uuid.uuid7()))
        for request_id in request_ids:
            assert await store.create_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=request_id,
                child_session_id=child_id,
                child_operation_id=child["operation_id"],
                parent_session_id=parent_id,
                question="Do not hold the domain pool",
                expires_after_seconds=30,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                child_fencing_epoch=child_epoch,
            )

        class _CountingPool:
            def __init__(self, inner: Any) -> None:
                self._inner = inner
                self.held = 0

            def acquire(self) -> Any:
                inner = self._inner
                pool = self

                class _Ctx:
                    def __init__(self) -> None:
                        self._cm = inner.acquire()

                    async def __aenter__(self) -> Any:
                        pool.held += 1
                        return await self._cm.__aenter__()

                    async def __aexit__(self, *args: Any) -> None:
                        try:
                            return await self._cm.__aexit__(*args)
                        finally:
                            pool.held -= 1

                return _Ctx()

        original_pool = store._operation_pool  # noqa: SLF001
        counting = _CountingPool(original_pool)
        store._operation_pool = counting  # noqa: SLF001
        try:
            waiters = [
                asyncio.create_task(
                    store.wait_for_child_guidance(
                        owner_id=_OWNER,
                        run_id=run_id,
                        request_id=request_id,
                        timeout_seconds=5,
                    )
                )
                for request_id in request_ids
            ]
            await asyncio.sleep(0.2)
            assert counting.held == 0
            assert not any(waiter.done() for waiter in waiters)
            for index, request_id in enumerate(request_ids):
                assert (
                    await store.reply_child_guidance(
                        owner_id=_OWNER,
                        run_id=run_id,
                        request_id=request_id,
                        content=f"reply {index}",
                        submission_key=f"pool-free-{index}",
                    )
                )["outcome"] == "replied"
            woke = await asyncio.wait_for(asyncio.gather(*waiters), timeout=2)
            for _ in range(200):
                if counting.held == 0:
                    break
                await asyncio.sleep(0.01)
            assert counting.held == 0
        finally:
            store._operation_pool = original_pool  # noqa: SLF001
        assert [row["reply"] for row in woke] == ["reply 0", "reply 1"]

    async def test_expire_child_guidance_false_after_reply_or_cancel_is_not_lease_loss(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        run_id = creation.run.run_id
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="expiry-race",
            parent_intent_id=str(uuid.uuid7()),
            objective="race expire",
            context_mode="isolated",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=1,
            context_snapshot={
                "parent_session_id": parent_id,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 0,
                "messages": [],
                "evidence_state": {},
            },
            plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
            budget={"provider_attempt_limit": 2},
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert child is not None and child_epoch == 1
        replied_id = str(uuid.uuid7())
        cancelled_id = str(uuid.uuid7())
        for request_id, question in (
            (replied_id, "Reply before expire"),
            (cancelled_id, "Cancel before expire"),
        ):
            assert await store.create_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=request_id,
                child_session_id=child_id,
                child_operation_id=child["operation_id"],
                parent_session_id=parent_id,
                question=question,
                expires_after_seconds=30,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                child_fencing_epoch=child_epoch,
            )
        assert (
            await store.reply_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=replied_id,
                content="already answered",
                submission_key="race-reply",
            )
        )["outcome"] == "replied"
        assert (
            await store.expire_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=replied_id,
                child_session_id=child_id,
                child_operation_id=child["operation_id"],
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                child_fencing_epoch=child_epoch,
            )
            is False
        )
        replied = await store.load_child_guidance(
            owner_id=_OWNER, run_id=run_id, request_id=replied_id
        )
        assert replied is not None and replied["status"] == "replied"
        assert (
            await store.cancel_child_session_by_owner(
                submission_key="cancel-child",
                owner_id=_OWNER,
                run_id=run_id,
                child_session_id=child_id,
                parent_session_id=parent_id,
            )
        )["outcome"] == "cancellation_requested"
        assert (
            await store.expire_child_guidance(
                owner_id=_OWNER,
                run_id=run_id,
                request_id=cancelled_id,
                child_session_id=child_id,
                child_operation_id=child["operation_id"],
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                child_fencing_epoch=child_epoch,
            )
            is False
        )
        cancelled = await store.load_child_guidance(
            owner_id=_OWNER, run_id=run_id, request_id=cancelled_id
        )
        assert cancelled is not None and cancelled["status"] == "cancelled"

    async def test_finish_child_session_fences_the_child_epoch(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        run_id = creation.run.run_id
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            parent_session_id=str(uuid.uuid7()),
            parent_call_id="child-epoch",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert child_epoch == 1
        outcome = {
            "status": "succeeded",
            "summary": "fenced",
            "handles": [],
            "usage": {},
            "child_session_id": child_id,
            "operation_id": str(uuid.uuid7()),
            "evidence_state": None,
        }
        assert (
            await store.finish_child_session(
                owner_id=_OWNER,
                run_id=run_id,
                child_session_id=child_id,
                status="succeeded",
                summary="fenced",
                outcome=outcome,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                child_fencing_epoch=child_epoch + 1,
            )
            is False
        )
        still_running = await store.load_child_session(
            owner_id=_OWNER, run_id=run_id, child_session_id=child_id
        )
        assert still_running is not None and still_running["status"] == "running"
        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=run_id,
            child_session_id=child_id,
            status="succeeded",
            summary="fenced",
            outcome=outcome,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )

    async def test_parent_success_is_blocked_until_children_settle(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=str(uuid.uuid7()),
            parent_call_id="barrier",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        blocked = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "too early"},
        )
        assert blocked.committed is False
        assert await store.finish_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            status="failed",
            summary="settled failure",
            outcome={
                "status": "failed",
                "summary": "settled failure",
                "handles": [],
                "usage": {},
                "child_session_id": child_id,
                "operation_id": str(uuid.uuid7()),
                "evidence_state": None,
            },
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        committed = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "after settlement"},
        )
        assert committed.committed is True

    async def test_parent_failure_cancels_any_remaining_children(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=str(uuid.uuid7()),
            parent_call_id="parent-failure",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        terminal = await store.finish_failure(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            error_kind="run_execution_failed",
            error_message="parent failed",
        )
        child = await store.load_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
        )

        assert terminal.committed is True
        assert child is not None and child["status"] == "cancelled"
        assert child["cancel_requested_at"] is not None
        assert child["host_state"]["terminal_outcome"]["child_session_id"] == child_id

    @pytest.mark.parametrize("settlement", ["fenced finish", "unleased finalizer"])
    async def test_settling_one_run_leaves_every_other_runs_children_alone(
        self, store, settlement: str
    ) -> None:
        """The settlement reaches only the Runs its own statement settled."""
        settled = await _run_with_a_live_child(store, "settled")
        bystander = await _run_with_a_live_child(store, "bystander")

        if settlement == "fenced finish":
            terminal = await store.finish_failure(
                owner_id=_OWNER,
                run_id=settled.run_id,
                worker_id=_WORKER,
                fencing_epoch=settled.fencing_epoch,
                error_kind="run_execution_failed",
                error_message="parent failed",
            )
            assert terminal.committed is True
        else:
            assert (
                await store.release_for_shutdown(
                    owner_id=_OWNER,
                    run_id=settled.run_id,
                    worker_id=_WORKER,
                    fencing_epoch=settled.fencing_epoch,
                )
                == "requeued"
            )
            cancellation = await store.request_cancellation(owner_id=_OWNER, run_id=settled.run_id)
            assert cancellation.outcome == "cancelled"

        assert await _child_states(store, settled) == ("cancelled", "cancelled", "cancelled")
        assert await _child_states(store, bystander) == ("running", "running", "pending")

    async def test_queued_parent_cancellation_atomically_cancels_accepted_children(
        self, store
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=str(uuid.uuid7()),
            parent_call_id="queued-parent-cancel",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        released = await store.release_for_shutdown(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert released == "requeued"

        cancellation = await store.request_cancellation(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
        )
        child = await store.load_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
        )

        assert cancellation.outcome == "cancelled"
        assert cancellation.run is not None and cancellation.run.status == "cancelled"
        assert child is not None and child["status"] == "cancelled"
        assert child["cancel_requested_at"] is not None
        assert child["host_state"]["terminal_outcome"]["status"] == "cancelled"

    async def test_child_lease_heartbeat_survives_original_window_and_fences_takeover(
        self, store: FingerprintingRunStore, pool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="lease-call",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        child_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert child_epoch == 1

        from dlightrag.engine.agent.session.ids import OperationId, SessionId
        from dlightrag.engine.answer.research.runtime import (
            _check_child_write,
            _drive_child_with_lease_renewal,
        )

        monkeypatch.setattr(
            "dlightrag.engine.answer.research.runtime._CHILD_LEASE_HEARTBEAT_SECONDS", 0.01
        )

        async def slow_drive(**_kwargs: Any) -> str:
            await asyncio.sleep(0.3)
            return "completed"

        # Exercise the actual runtime-to-adapter callback: a loose AsyncMock
        # would hide the missing parent owner/run arguments at this seam.
        renewed = _check_child_write(
            partial(
                store.heartbeat_child_session,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
            )
        )

        async with pool.acquire() as conn:
            original_expiry = await conn.fetchval(
                "UPDATE dlightrag_answer_child_sessions"
                " SET lease_expires_at = NOW() + INTERVAL '200 milliseconds'"
                " WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3"
                " RETURNING lease_expires_at",
                _OWNER,
                uuid.UUID(creation.run.run_id),
                uuid.UUID(child_id),
            )
        assert (
            await _drive_child_with_lease_renewal(
                cast(Any, SimpleNamespace(drive=slow_drive)),
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                session_id=SessionId(child_id),
                operation_id=OperationId.new(),
                child_fencing_epoch=child_epoch,
                renew_child=renewed,
            )
            == "completed"
        )
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT NOW() > $1", original_expiry)

        from dlightrag.engine.agent.session.ids import LaneId
        from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
        from dlightrag.engine.agent.session.transactions import (
            RegisterExpectation,
            SessionTransaction,
            TransactionCommit,
            TransactionLeaseLost,
        )

        child_store = claim.execution.session_repository.for_child(
            SessionId(child_id), fencing_epoch=child_epoch
        )
        head = LaneHead(LaneId.main(), None)
        state = LaneState(LaneId.main())
        commit = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )
        assert isinstance(commit, TransactionCommit)

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_answer_child_sessions"
                " SET lease_expires_at = NOW() - INTERVAL '1 second'"
                " WHERE owner_id = $1 AND run_id = $2 AND child_session_id = $3",
                _OWNER,
                uuid.UUID(creation.run.run_id),
                uuid.UUID(child_id),
            )
        next_epoch = await store.claim_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        assert next_epoch == child_epoch + 1
        assert not await store.heartbeat_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            child_fencing_epoch=child_epoch,
        )
        stale_write = await child_store.transact(
            session_id=SessionId(child_id),
            fencing_epoch=child_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(state)],
                expectations=[RegisterExpectation(state.ref, commit.commit_sequence)],
            ),
        )
        assert isinstance(stale_write, TransactionLeaseLost)

    async def test_child_observation_lists_are_owner_scoped(self, store) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
        claim = await _claimed(store)
        parent_id = str(uuid.uuid7())
        child_id = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            parent_session_id=parent_id,
            parent_call_id="observe-call",
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            objective="observe",
            context_mode="parent",
            model_role="extract",
            tools=("search_knowledge_base",),
            depth=1,
        )
        controls = await store.list_child_controls(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            limit=20,
        )
        questions = await store.list_child_guidance(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            limit=20,
        )
        foreign = await store.list_child_controls(
            owner_id=_OTHER_OWNER,
            run_id=creation.run.run_id,
            child_session_id=child_id,
            limit=20,
        )
        transcript = await store.load_agent_transcript(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            session_id=child_id,
            limit=20,
        )
        assert controls == ()
        assert questions == ()
        assert foreign == ()
        assert transcript == ()

    async def test_a_run_reads_only_the_messages_it_wrote(self, store, pool) -> None:
        """One conversation's agent is one Session across its Runs, and its lane head walks back
        through every earlier turn of it.

        Each Entry states the Run that wrote it, so the newest Run's trace reads its own messages:
        without that fact the lane head draws the Runs before it beside its own.
        """
        session = "0f0f0f0f-0000-7000-8000-0000000000aa"
        earlier = await store.create_run(
            owner_id=_OWNER, request=_request(agent_session_id=session)
        )
        latest = await store.create_run(owner_id=_OWNER, request=_request(agent_session_id=session))
        async with pool.acquire() as conn:
            # The Session an Entry hangs off, leased to the Run whose messages they are.
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
                "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
                _OWNER,
                uuid.UUID(session),
                earlier.run.run_id,
            )
            created = await conn.fetchval(
                "SELECT created_at FROM dlightrag_runs WHERE run_id = $1", latest.run.run_id
            )
            older, newer = uuid.uuid4(), uuid.uuid4()
            # Two Runs of one Session, with the boundary between them made explicit: the earlier
            # Run's message an hour before the newest Run started, its own a second after.
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
                "entry_id, parent_entry_id, entry_type, schema_version, timestamp, payload_json, "
                "search_text, run_id) VALUES "
                "($1, $2, 1, $3, NULL, 'user_message', 1, $4, '{}'::jsonb, '', $7), "
                "($1, $2, 2, $5, $3, 'assistant_message', 1, $6, '{}'::jsonb, '', $8)",
                _OWNER,
                uuid.UUID(session),
                older,
                created - datetime.timedelta(hours=1),
                newer,
                created + datetime.timedelta(seconds=1),
                earlier.run.run_id,
                latest.run.run_id,
            )
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_registers (owner_id, session_id, "
                "register_kind, register_key, sequence, payload_json) "
                "VALUES ($1, $2, 'lane_head', 'main', 2, $3::jsonb) "
                "ON CONFLICT (owner_id, session_id, register_kind, register_key) "
                "DO UPDATE SET sequence = 2, payload_json = $3::jsonb",
                _OWNER,
                uuid.UUID(session),
                json.dumps({"entry_id": str(newer)}),
            )

        page = await store.load_agent_transcript(
            owner_id=_OWNER, run_id=str(latest.run.run_id), session_id=session, limit=20
        )

        assert [row["sequence"] for row in page] == [2], "the earlier Run's message stays out"
        assert earlier.run.run_id != latest.run.run_id

        # A reader holding a transcript asks forward from what it holds, and gets only what landed.
        delta = await store.load_agent_transcript(
            owner_id=_OWNER,
            run_id=str(latest.run.run_id),
            session_id=session,
            limit=20,
            since=1,
        )
        assert [row["sequence"] for row in delta] == [2], "a delta reads forward from a sequence"
        assert (
            await store.load_agent_transcript(
                owner_id=_OWNER,
                run_id=str(latest.run.run_id),
                session_id=session,
                limit=20,
                since=2,
            )
            == ()
        ), "nothing landed after the last message, so the delta is empty rather than the page again"


async def test_run_timeline_reads_state_the_run_and_its_page_entries(store: Any, pool: Any) -> None:
    """The two Run-timeline reads against real PostgreSQL: step rows, then the page's entries."""
    session = "0f0f0f0f-0000-7000-8000-0000000000ac"
    creation = await store.create_run(owner_id=_OWNER, request=_request(agent_session_id=session))
    run_id = str(creation.run.run_id)
    now = datetime.datetime.now(datetime.UTC)
    invocation, tool, user, answer, compacted = (uuid.uuid4() for _ in range(5))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            _OWNER,
            uuid.UUID(session),
            creation.run.run_id,
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
            "entry_id, parent_entry_id, entry_type, schema_version, timestamp, payload_json, "
            "search_text, run_id) VALUES "
            "($1, $2, 1, $3, NULL, 'assistant_message', 1, $8, $9::jsonb, '', $13), "
            "($1, $2, 2, $4, $3, 'tool_result', 1, $10, $11::jsonb, 'two files', $13), "
            "($1, $2, 3, $5, $4, 'user_message', 1, $10, $12::jsonb, 'next', $13), "
            "($1, $2, 4, $6, $5, 'assistant_message', 1, $10, $14::jsonb, 'answer', $13), "
            "($1, $2, 5, $7, $6, 'compaction', 1, $10, '{}'::jsonb, '', $13)",
            _OWNER,
            uuid.UUID(session),
            invocation,
            tool,
            user,
            answer,
            compacted,
            now,
            json.dumps(
                {
                    "content": "",
                    "tool_calls": [{"id": "call-1", "name": "read", "arguments": {"path": "a.md"}}],
                }
            ),
            now + datetime.timedelta(seconds=1),
            json.dumps(
                {
                    "call_id": "call-1",
                    "tool_name": "read",
                    "outcome": "succeeded",
                    "content": [{"type": "text", "text": "two files"}],
                    "facts": {"path": "a.md", "hits": 2},
                }
            ),
            json.dumps({"content": "next"}),
            creation.run.run_id,
            json.dumps({"content": "answer", "tool_calls": []}),
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_registers (owner_id, session_id, "
            "register_kind, register_key, sequence, payload_json) "
            "VALUES ($1, $2, 'lane_head', 'main', 5, $3::jsonb) "
            "ON CONFLICT (owner_id, session_id, register_kind, register_key) "
            "DO UPDATE SET sequence = 5, payload_json = $3::jsonb",
            _OWNER,
            uuid.UUID(session),
            json.dumps({"entry_id": str(compacted)}),
        )

    steps = await store.load_run_timeline_steps(owner_id=_OWNER, run_id=run_id)

    assert [step["sequence"] for step in steps] == [1, 2, 3, 4], "compaction states no step"
    invocation_row, tool_row, user_row, answer_row = steps
    assert invocation_row["tool_call_ids"] == ("call-1",)
    assert invocation_row["tool_calls"] == 1 and invocation_row["has_content"] is False
    assert (tool_row["call_id"], tool_row["tool_name"]) == ("call-1", "read")
    assert tool_row["failed"] is False and tool_row["degraded"] is False
    assert user_row["entry_type"] == "user_message"
    assert answer_row["tool_calls"] == 0 and answer_row["has_content"] is True
    assert await store.load_run_timeline_steps(owner_id=_OTHER_OWNER, run_id=run_id) == ()
    assert await store.load_run_timeline_steps(owner_id=_OWNER, run_id=str(uuid.uuid7())) == ()

    entries = await store.load_timeline_entries(
        owner_id=_OWNER,
        run_id=run_id,
        keys=[(session, 2), (session, 4), (session, 99), (session, 2)],
    )

    assert [(row["session_id"], row["sequence"]) for row in entries] == [
        (session, 2),
        (session, 4),
    ], "a key names one entry, and a key the Run does not hold reads nothing"
    tool_message, answer_message = entries
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "two files"
    assert tool_message["facts"] == {"path": "a.md", "hits": 2}
    assert answer_message["role"] == "assistant"
    assert answer_message["content"] == "answer"
    assert (
        await store.load_timeline_entries(
            owner_id=_OWNER, run_id=str(uuid.uuid7()), keys=[(session, 2)]
        )
        == ()
    )


async def test_an_assistant_turns_whitespace_beside_its_calls_states_no_say(
    store: Any, pool: Any
) -> None:
    """The kind rule trims the words the way the old page read did, not only spaces."""
    session = "0f0f0f0f-0000-7000-8000-0000000000ad"
    creation = await store.create_run(owner_id=_OWNER, request=_request(agent_session_id=session))
    entry_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            _OWNER,
            uuid.UUID(session),
            creation.run.run_id,
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
            "entry_id, parent_entry_id, entry_type, schema_version, timestamp, payload_json, "
            "search_text, run_id) VALUES ($1, $2, 1, $3, NULL, 'assistant_message', 1, NOW(), "
            "$4::jsonb, ' ', $5)",
            _OWNER,
            uuid.UUID(session),
            entry_id,
            json.dumps(
                {
                    "content": " \n\t ",
                    "tool_calls": [{"id": "call-1", "name": "read"}],
                }
            ),
            creation.run.run_id,
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_registers (owner_id, session_id, "
            "register_kind, register_key, sequence, payload_json) "
            "VALUES ($1, $2, 'lane_head', 'main', 1, $3::jsonb) "
            "ON CONFLICT (owner_id, session_id, register_kind, register_key) "
            "DO UPDATE SET sequence = 1, payload_json = $3::jsonb",
            _OWNER,
            uuid.UUID(session),
            json.dumps({"entry_id": str(entry_id)}),
        )

    steps = await store.load_run_timeline_steps(owner_id=_OWNER, run_id=str(creation.run.run_id))

    assert len(steps) == 1
    assert steps[0]["has_content"] is False, "whitespace beside calls does not state a say"


async def test_a_delivery_warning_reads_as_a_delivery_message_and_step(
    store: Any, pool: Any
) -> None:
    session = "0f0f0f0f-0000-7000-8000-0000000000ae"
    creation = await store.create_run(owner_id=_OWNER, request=_request(agent_session_id=session))
    entry_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, "
            "fencing_epoch) VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            _OWNER,
            uuid.UUID(session),
            creation.run.run_id,
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_entries (owner_id, session_id, sequence, "
            "entry_id, parent_entry_id, entry_type, schema_version, timestamp, payload_json, "
            "search_text, run_id) VALUES ($1, $2, 1, $3, NULL, 'delivery_warning', 1, NOW(), "
            "$4::jsonb, 'findings', $5)",
            _OWNER,
            uuid.UUID(session),
            entry_id,
            json.dumps({"attempt": 2, "findings": ["first finding", "second finding"]}),
            creation.run.run_id,
        )
        await conn.execute(
            "INSERT INTO dlightrag_agent_session_registers (owner_id, session_id, "
            "register_kind, register_key, sequence, payload_json) "
            "VALUES ($1, $2, 'lane_head', 'main', 1, $3::jsonb) "
            "ON CONFLICT (owner_id, session_id, register_kind, register_key) "
            "DO UPDATE SET sequence = 1, payload_json = $3::jsonb",
            _OWNER,
            uuid.UUID(session),
            json.dumps({"entry_id": str(entry_id)}),
        )

    steps = await store.load_run_timeline_steps(owner_id=_OWNER, run_id=str(creation.run.run_id))
    entries = await store.load_timeline_entries(
        owner_id=_OWNER, run_id=str(creation.run.run_id), keys=[(session, 1)]
    )

    assert [step["entry_type"] for step in steps] == ["delivery_warning"]
    (message,) = entries
    assert message["role"] == "delivery"
    assert message["content"] == "first finding\nsecond finding"
    assert message["attempt"] == 2


async def _interactive_terminal_child(store: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    creation = await store.create_run(owner_id=_OWNER, request=_request(mode="research"))
    claim = await _claimed(store)
    identity: dict[str, Any] = dict(
        owner_id=_OWNER, run_id=creation.run.run_id, child_session_id=str(uuid.uuid4())
    )
    fence: dict[str, Any] = dict(worker_id=_WORKER, fencing_epoch=claim.run.fencing_epoch)
    parent_id = _request()["agent_session_id"]
    assert await store.upsert_child_session(
        **identity,
        **fence,
        parent_session_id=parent_id,
        parent_call_id="terminal-race",
        parent_intent_id=str(uuid.uuid4()),
        objective="check",
        context_mode="isolated",
        model_role="query",
        tools=("search_knowledge_base",),
        depth=1,
        context_snapshot={
            "parent_session_id": parent_id,
            "parent_entry_id": str(uuid.uuid4()),
            "depth": 0,
            "messages": [],
            "evidence_state": {},
        },
        plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
        budget={"provider_attempt_limit": 2},
    )
    await _settle_interactive_child(store, identity, fence, "succeeded")
    return identity, fence


async def _settle_interactive_child(store, identity, fence, status):
    child = await store.load_child_session(**identity)
    assert await store.finish_child_session(
        **identity,
        **fence,
        status=status,
        summary=status,
        outcome={
            "status": status,
            "summary": status,
            "handles": [],
            "usage": {},
            "child_session_id": identity["child_session_id"],
            "operation_id": child["operation_id"],
            "evidence_state": None,
        },
    )


async def _wait_for_run_lock_wait(pool, task) -> None:
    async with asyncio.timeout(8):
        while True:
            async with pool.acquire() as observer:
                blocked = await observer.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                    "AND wait_event_type='Lock' AND query LIKE '%dlightrag_runs%'"
                )
            if blocked:
                return
            assert not task.done(), "competing transition did not wait for Run lock"
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("continuation_first", [True, False])
async def test_success_serializes_snapshot_after_child_continuation(
    store: Any,
    pool: Any,
    continuation_first: bool,
) -> None:
    identity, fence = await _interactive_terminal_child(store)
    finish_args: dict[str, Any] = dict(
        owner_id=_OWNER, run_id=identity["run_id"], **fence, result={"answer": "done"}
    )
    continue_args: dict[str, Any] = dict(
        **identity, content="second operation", submission_key="terminal-race-continue"
    )
    async with pool.acquire() as held:
        tx = held.transaction()
        await tx.start()

        class HeldStore(FingerprintingRunStore):
            async def _run_write(self, operation):
                return await operation(held)

        serialized = HeldStore(pool=pool)
        continuation: Any = None
        terminal: Any = None
        task: asyncio.Task[Any]
        if continuation_first:
            continuation = await serialized.continue_child_session(**continue_args)
            task = asyncio.create_task(store.finish_success(**finish_args))
        else:
            terminal = await serialized.finish_success(**finish_args)
            task = asyncio.create_task(store.continue_child_session(**continue_args))
        try:
            await _wait_for_run_lock_wait(pool, task)
            await tx.commit()
            result = await asyncio.wait_for(task, 8)
        except BaseException:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await tx.rollback()
            raise
    child = await store.load_child_session(**identity)
    run = await store.get_run(owner_id=_OWNER, run_id=identity["run_id"])
    if continuation_first:
        assert continuation["outcome"] == "accepted"
        assert not result.committed
        assert run.status == child["status"] == child["operation_status"] == "running"
        assert child["operation_id"] == continuation["operation_id"]
        assert await store.claim_child_session(**identity, **fence)
    else:
        assert terminal.committed
        assert result["outcome"] == "run_terminal"
        assert run.status == "succeeded"
        assert child["operation_sequence"] == 1


async def test_owner_cancel_replays_original_operation_after_reauthorization_and_run_terminal(
    store: Any,
) -> None:
    from types import SimpleNamespace

    from dlightrag.application.answer_runs.service import AnswerService

    identity, fence = await _interactive_terminal_child(store)
    await store.continue_child_session(
        **identity, content="operation A", submission_key="operation-a"
    )
    service: Any = object.__new__(AnswerService)
    service._store = store
    service._coordinator = SimpleNamespace(wake=lambda: None)
    args = dict(**identity, action="cancel", idempotency_key="cancel-a")
    first = await service.control_child(**args)
    await _settle_interactive_child(store, identity, fence, "cancelled")
    denied = await store.continue_child_session(
        **identity,
        content="not authorized",
        submission_key="denied",
        origin="parent",
        reauthorize_user_cancelled=True,
        **fence,
    )
    assert denied["outcome"] == "reauthorization_required"
    continued = await service.control_child(
        **identity,
        action="continue",
        content="operation B",
        idempotency_key="operation-b",
        reauthorize_user_cancelled=True,
    )
    assert continued.outcome == "accepted"
    replay = await service.control_child(**args)
    assert replay == first
    child = await store.load_child_session(**identity)
    assert child["cancel_requested_at"] is None
    assert child["operation_id"] == continued.operation_id
    assert child["cancellation_origin"] is None
    # Same submission cannot become a steer with different intent.
    conflict = await service.control_child(
        **identity, action="steer", content="different", idempotency_key="cancel-a"
    )
    assert conflict.outcome == "idempotency_conflict"
    await _settle_interactive_child(store, identity, fence, "succeeded")
    assert (
        await store.finish_success(
            owner_id=_OWNER, run_id=identity["run_id"], **fence, result={"answer": "done"}
        )
    ).committed
    assert await service.control_child(**args) == first


@pytest.mark.parametrize("terminal_status", ["succeeded", "cancelled"])
async def test_rejected_owner_cancel_receipt_cannot_target_later_continuation(
    store: Any,
    terminal_status: str,
) -> None:
    identity, fence = await _interactive_terminal_child(store)
    if terminal_status == "cancelled":
        await store.continue_child_session(**identity, content="initial", submission_key="initial")
        await store.cancel_child_session_by_owner(**identity, submission_key="initial-cancel")
        await _settle_interactive_child(store, identity, fence, "cancelled")
    before = await store.load_child_session(**identity)
    first = await store.cancel_child_session_by_owner(**identity, submission_key="terminal-cancel")
    assert first["outcome"] == "terminal_child"
    continued = await store.continue_child_session(
        **identity,
        content="authorized continuation",
        submission_key="new-work",
        reauthorize_user_cancelled=True,
    )
    assert continued["outcome"] == "accepted"
    # Treat the first rejection response as lost; retransmit the exact submission.
    replay = await store.cancel_child_session_by_owner(**identity, submission_key="terminal-cancel")
    assert replay == first
    assert replay["operation_id"] == before["operation_id"]
    assert replay["status"] == terminal_status
    child = await store.load_child_session(**identity)
    assert child["cancel_requested_at"] is None
    assert child["operation_id"] == continued["operation_id"]
    controls = await store.list_child_controls(**identity)
    assert all(row["kind"] != "cancel" for row in controls)
    # A new submission is a new explicit intent and can cancel the current work.
    current = await store.cancel_child_session_by_owner(**identity, submission_key="new-cancel")
    assert current["operation_id"] == continued["operation_id"]
    assert current["outcome"] == "cancellation_requested"


class TestChildToolRefusals:
    """What the parent's control tools tell the Model when it names an id the Run never issued."""

    @staticmethod
    async def _parent_tools(store: PGRunStore, live: _LiveChild) -> tuple[dict[str, Any], str]:
        """The parent's tools over the real store, and the parent Session the Child belongs to."""
        guidance = await store.load_child_guidance(
            owner_id=_OWNER, run_id=live.run_id, request_id=live.request_id
        )
        assert guidance is not None
        parent_id = str(guidance["parent_session_id"])
        fence = {"worker_id": _WORKER, "fencing_epoch": live.fencing_epoch}
        host = SubagentHost(
            parent_session_id=SessionId(parent_id),
            run_id=live.run_id,
            owner_id=_OWNER,
            load_child=store.load_child_session,
            list_children=store.list_child_sessions,
            load_guidance=store.load_child_guidance,
            list_guidance=store.list_pending_child_guidance,
            steer_child=_check_child_write(partial(store.enqueue_child_control, **fence)),
            continue_child=_check_child_write(partial(store.continue_child_session, **fence)),
            reply_guidance=_check_child_write(partial(store.reply_child_guidance, **fence)),
        )
        return {tool.name: tool for tool in subagent_tools(host=host)}, parent_id

    @pytest.mark.parametrize("tool", ["steer_subagent", "continue_subagent"])
    @pytest.mark.parametrize("unknown", [str(uuid.uuid7()), "not-a-uuid"])
    async def test_a_child_id_the_run_never_issued_is_refused_not_raised(
        self, store, tool: str, unknown: str
    ) -> None:
        live = await _run_with_a_live_child(store, "unknown-child")
        tools, _parent = await self._parent_tools(store, live)

        result = await tools[tool].execute(
            tools[tool].input_model(child_session_id=unknown, content="look again"),
            tool_runtime(tool_name=tool),
        )

        assert result.is_error
        assert result.text_content == (
            f"Unknown child session {unknown}. This Run's children: {live.child_session_id}."
        )

    @pytest.mark.parametrize("tool", ["steer_subagent", "continue_subagent"])
    async def test_a_child_of_another_parent_is_refused_by_the_store_outcome(
        self, store, tool: str
    ) -> None:
        live = await _run_with_a_live_child(store, "foreign-child")
        tools, _parent = await self._parent_tools(store, live)
        stranger = str(uuid.uuid7())
        other_parent = str(uuid.uuid7())
        assert await store.upsert_child_session(
            owner_id=_OWNER,
            run_id=live.run_id,
            child_session_id=stranger,
            parent_session_id=other_parent,
            parent_call_id="other-call",
            parent_intent_id=str(uuid.uuid7()),
            objective="not ours",
            context_mode="isolated",
            model_role="query",
            tools=("search_knowledge_base",),
            depth=1,
            context_snapshot={
                "parent_session_id": other_parent,
                "parent_entry_id": str(uuid.uuid7()),
                "depth": 0,
                "messages": [],
                "evidence_state": {},
            },
            plan={"schema_version": 2, "tools": ["search_knowledge_base"]},
            budget={"provider_attempt_limit": 2},
            worker_id=_WORKER,
            fencing_epoch=live.fencing_epoch,
        )

        # The roster holds it, so only the store's own outcome says it is not this parent's.
        result = await tools[tool].execute(
            tools[tool].input_model(child_session_id=stranger, content="look again"),
            tool_runtime(tool_name=tool),
        )

        assert result.is_error
        assert result.text_content == (
            f"Unknown child session {stranger}. This Run's children: {live.child_session_id}."
        )

    @pytest.mark.parametrize("unknown", [str(uuid.uuid7()), "not-a-uuid"])
    async def test_a_request_nobody_asked_is_refused_with_the_questions_waiting(
        self, store, unknown: str
    ) -> None:
        live = await _run_with_a_live_child(store, "unknown-request")
        tools, _parent = await self._parent_tools(store, live)

        result = await tools["reply_subagent"].execute(
            tools["reply_subagent"].input_model(request_id=unknown, content="use the report"),
            tool_runtime(tool_name="reply_subagent"),
        )

        assert result.is_error
        assert result.text_content == (
            f"Unknown guidance request {unknown}. Questions waiting for a reply: {live.request_id}."
        )

    async def test_replaying_a_reply_that_was_taken_returns_the_original_receipt(
        self, store
    ) -> None:
        live = await _run_with_a_live_child(store, "replayed-reply")
        tools, _parent = await self._parent_tools(store, live)
        reply = tools["reply_subagent"]
        # One tool call, run again as a recovered Run runs it: the same submission key.
        runtime = tool_runtime(tool_name="reply_subagent")
        arguments = reply.input_model(request_id=live.request_id, content="use the report")

        first = await reply.execute(arguments, runtime)
        replay = await reply.execute(arguments, runtime)
        # No question is pending any more, and the request is still the Run's.
        refused = await reply.execute(
            reply.input_model(request_id=str(uuid.uuid7()), content="use the report"),
            tool_runtime(tool_name="reply_subagent"),
        )

        assert not first.is_error and '"outcome":"replied"' in first.text_content
        assert not replay.is_error
        assert replay.text_content == first.text_content
        assert refused.is_error
        assert refused.text_content.endswith("No child question is waiting for a reply.")


class TestForkPoints:
    async def test_records_the_settled_head_under_a_live_claim(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)

        written = await store.record_fork_point(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            entry_id="01930000-0000-7000-8000-000000000001",
            projection_id="01930000-0000-7000-8000-000000000002",
        )

        assert written is True
        record = await store.load_routing(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.fork_point_entry_id == "01930000-0000-7000-8000-000000000001"
        assert record.fork_point_projection_id == "01930000-0000-7000-8000-000000000002"

    async def test_a_reclaiming_worker_owns_the_state_the_run_settles_at(self, store, pool) -> None:
        """Only the claim that ends the Run may say where it ended."""
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        await _claimed(store)
        await store.record_fork_point(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=1,
            entry_id="01930000-0000-7000-8000-00000000000a",
            projection_id=None,
        )
        await _expire_lease(pool, creation.run.run_id)
        reclaim = await _claimed(store, worker_id="worker-2")

        written = await store.record_fork_point(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id="worker-2",
            fencing_epoch=reclaim.run.fencing_epoch,
            entry_id="01930000-0000-7000-8000-00000000000b",
            projection_id=None,
        )

        assert written is True
        # The fenced-out worker cannot revise the state it lost.
        assert (
            await store.record_fork_point(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=1,
                entry_id="01930000-0000-7000-8000-00000000000c",
                projection_id=None,
            )
            is False
        )
        record = await store.load_routing(owner_id=_OWNER, run_id=creation.run.run_id)
        assert record is not None
        assert record.fork_point_entry_id == "01930000-0000-7000-8000-00000000000b"

    async def test_a_terminal_run_cannot_record_one(self, store, pool) -> None:
        creation = await store.create_run(owner_id=_OWNER, request=_request())
        claim = await _claimed(store)
        await store.finish_cancelled(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )

        assert (
            await store.record_fork_point(
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                worker_id=_WORKER,
                fencing_epoch=claim.run.fencing_epoch,
                entry_id="01930000-0000-7000-8000-00000000000d",
                projection_id=None,
            )
            is False
        )


async def test_a_succeeded_run_commits_the_session_workspace_head_with_cas(
    store: FingerprintingRunStore, pool: Any
) -> None:
    from dlightrag.adapters.postgres.answer.agentspace import PGAgentspaceStore
    from dlightrag.engine.runtime.agentspace import AgentspaceHeadCommit

    session_id = "00000000-0000-7000-8000-000000000001"

    async def commit(*, expected: str | None, digest_character: str) -> str:
        creation = await store.create_run(owner_id=_OWNER, request=_request("head"))
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions"
                " (owner_id, session_id, lease_run_id, fencing_epoch)"
                " VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
                _OWNER,
                uuid.UUID(session_id),
                uuid.UUID(creation.run.run_id),
            )
        claim = await _claimed(store)
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "done"},
            workspace_head=AgentspaceHeadCommit(
                session_id=session_id,
                head_run_id=creation.run.run_id,
                manifest_digest=digest_character * 64,
                expected_head_run_id=expected,
            ),
        )
        assert outcome.committed
        return creation.run.run_id

    first = await commit(expected=None, digest_character="a")
    reader = PGAgentspaceStore(
        pool=pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(first),
        worker_id=_WORKER,
        lease_owner=_WORKER,
        fencing_epoch=0,
    )
    head = await reader.load_session_agentspace_head(session_id=session_id)
    assert head is not None
    assert (head.head_run_id, head.revision, head.manifest_digest) == (first, 1, "a" * 64)

    second = await commit(expected=first, digest_character="b")
    head = await reader.load_session_agentspace_head(session_id=session_id)
    assert head is not None
    assert (head.head_run_id, head.revision, head.manifest_digest) == (second, 2, "b" * 64)

    # A Run that read the first head cannot commit over the second: its CAS names a moved-on
    # head, and the terminal write still succeeds without advancing the pointer.
    await commit(expected=first, digest_character="c")
    head = await reader.load_session_agentspace_head(session_id=session_id)
    assert head is not None
    assert (head.head_run_id, head.revision) == (second, 2)

    # The pointer belongs to the Session: deleting it cascades the row away, so no
    # conversation can leave a dangling head behind.
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM dlightrag_agent_sessions WHERE owner_id = $1 AND session_id = $2::uuid",
            _OWNER,
            uuid.UUID(session_id),
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_session_workspace_heads"
                " WHERE owner_id = $1 AND session_id = $2::uuid",
                _OWNER,
                uuid.UUID(session_id),
            )
        ) == 0


async def test_a_failed_run_has_no_workspace_head_channel(
    store: FingerprintingRunStore, pool: Any
) -> None:
    from dlightrag.engine.runtime.agentspace import AgentspaceHeadCommit

    session_id = "00000000-0000-7000-8000-000000000001"
    creation = await store.create_run(owner_id=_OWNER, request=_request("head"))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, fencing_epoch)"
            " VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(creation.run.run_id),
        )
    claim = await _claimed(store)
    await store.finish_failure(
        owner_id=_OWNER,
        run_id=creation.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
        error_kind="ANSWER_STREAM_FAILED",
        error_message="failed",
    )
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_session_workspace_heads"
                " WHERE owner_id = $1 AND session_id = $2::uuid",
                _OWNER,
                uuid.UUID(session_id),
            )
        ) == 0
    # The finished protocol has no head parameter at all: a failed settlement cannot name one.
    import inspect

    assert "workspace_head" not in inspect.signature(store.finish_failure).parameters
    del AgentspaceHeadCommit


async def test_a_fork_branches_its_lane_head_and_seeds_from_the_fork_point_snapshot(
    store: FingerprintingRunStore, pool: Any, tmp_path: Any
) -> None:
    from dlightrag.adapters.postgres.answer.agentspace import PGAgentspaceStore
    from dlightrag.engine.answer.agentspace import (
        AgentspaceSeed,
        bind_run_agentspace,
        head_run_agentspace,
        seed_epoch_from_head,
        session_heads_root,
        stage_session_head,
    )
    from dlightrag.engine.runtime.agentspace import AgentspaceHeadCommit

    session_id = "00000000-0000-7000-8000-000000000001"
    root = tmp_path / "workspace"
    root.mkdir(parents=True)

    async def round(
        *, lane_id: str, expected: str | None, files: dict[str, str], parent: str | None
    ) -> tuple[str, Any]:
        creation = await store.create_run(owner_id=_OWNER, request=_request("phase"))
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions"
                " (owner_id, session_id, lease_run_id, fencing_epoch)"
                " VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
                _OWNER,
                uuid.UUID(session_id),
                uuid.UUID(creation.run.run_id),
            )
        claim = await _claimed(store)
        assert claim is not None
        bound = await bind_run_agentspace(
            workspace_root=root,
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            fencing_epoch=claim.run.fencing_epoch,
            recorded_epoch=None,
            store=None,
        )
        for relative, content in files.items():
            path = bound.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        staged = stage_session_head(
            workspace=bound.workspace,
            heads_root=session_heads_root(root, _OWNER, session_id),
            run_id=creation.run.run_id,
        )
        outcome = await store.finish_success(
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            worker_id=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "phase done"},
            workspace_head=AgentspaceHeadCommit(
                session_id=session_id,
                lane_id=lane_id,
                head_run_id=creation.run.run_id,
                manifest_digest=staged.manifest_digest,
                expected_head_run_id=expected,
                parent_run_id=parent,
                manifest=staged.files,
            ),
        )
        assert outcome.committed
        return creation.run.run_id, bound

    # The main line settles two phases.
    first, _ = await round(
        lane_id="main", expected=None, files={"scripts/model.py": "model = {}\n"}, parent=None
    )
    second, _ = await round(
        lane_id="main",
        expected=first,
        files={"scripts/model.py": "model = {}\n", "case.draft.json": "{}\n"},
        parent=first,
    )

    # A fork at the first phase reads that phase's snapshot, not the newer main head.
    reader = PGAgentspaceStore(
        pool=pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(first),
        worker_id=_WORKER,
        lease_owner=_WORKER,
        fencing_epoch=0,
    )
    snapshot = await reader.load_run_agentspace_snapshot(run_id=first)
    assert snapshot is not None
    assert (snapshot.run_id, snapshot.parent_run_id) == (first, None)
    fork_destination = tmp_path / "fork-epoch"
    fork_destination.mkdir()
    seed_epoch_from_head(
        AgentspaceSeed(
            head_run_id=snapshot.run_id,
            manifest_digest=snapshot.manifest_digest,
            workspace=head_run_agentspace(
                session_heads_root(root, _OWNER, session_id), snapshot.run_id
            ),
        ),
        fork_destination,
    )
    assert (fork_destination / "scripts/model.py").read_text() == "model = {}\n"
    assert not (fork_destination / "case.draft.json").exists(), "the fork is turn-accurate"

    # The fork commits its own lane head; the main line is untouched.
    fork_run, _ = await round(
        lane_id="fork-1",
        expected=None,
        files={"scripts/model.py": "model = {}\n", "branch.txt": "fork\n"},
        parent=first,
    )

    async with pool.acquire() as conn:
        heads = await conn.fetch(
            "SELECT lane_id, head_run_id::text AS head_run_id"
            " FROM dlightrag_session_workspace_heads"
            " WHERE owner_id = $1 AND session_id = $2::uuid ORDER BY lane_id",
            _OWNER,
            uuid.UUID(session_id),
        )
        snapshots = await conn.fetch(
            "SELECT run_id::text AS run_id, parent_run_id::text AS parent_run_id"
            " FROM dlightrag_session_workspace_snapshots"
            " WHERE owner_id = $1 AND session_id = $2::uuid ORDER BY created_at",
            _OWNER,
            uuid.UUID(session_id),
        )
    assert {str(row["lane_id"]): str(row["head_run_id"]) for row in heads} == {
        "fork-1": fork_run,
        "main": second,
    }
    assert {
        str(row["run_id"]): (str(row["parent_run_id"]) if row["parent_run_id"] else None)
        for row in snapshots
    } == {first: None, second: first, fork_run: first}


async def test_a_completed_goals_workspace_archive_is_published_as_a_resource(
    store: FingerprintingRunStore, pool: Any
) -> None:
    import gzip

    from dlightrag.engine.runtime.records import PendingPublication

    session_id = "00000000-0000-7000-8000-000000000001"
    archive = gzip.compress(b"an archived workspace tree")
    creation = await store.create_run(owner_id=_OWNER, request=_request("goal"))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, fencing_epoch)"
            " VALUES ($1, $2, $3, 1) ON CONFLICT DO NOTHING",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(creation.run.run_id),
        )
    claim = await _claimed(store)
    outcome = await store.finish_success(
        owner_id=_OWNER,
        run_id=creation.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=claim.run.fencing_epoch,
        result={"answer": "goal complete"},
        publications=(
            PendingPublication(
                resource_id="artifact-0123456789abcdef0123",
                reference_kind="published_artifact",
                filename="workspace-snapshot.tar.gz",
                mime_type="application/gzip",
                content=archive,
                session_id=session_id,
                relative_path="workspace-snapshot.tar.gz",
                presentation="download",
                label="Completed goal workspace",
            ),
        ),
    )
    assert outcome.committed
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT resource.kind, resource.safe_name, resource.media_type,"
            "       blob.byte_size"
            " FROM dlightrag_answer_resources AS resource"
            " JOIN dlightrag_blobs AS blob"
            "   ON blob.owner_id = resource.owner_id AND blob.digest = resource.blob_digest"
            " WHERE resource.owner_id = $1 AND resource.run_id = $2",
            _OWNER,
            uuid.UUID(creation.run.run_id),
        )
    assert row is not None
    assert (str(row["kind"]), str(row["media_type"])) == (
        "published_artifact",
        "application/gzip",
    )
    assert str(row["safe_name"]).endswith(".tar.gz")
    assert int(row["byte_size"]) == len(archive)
