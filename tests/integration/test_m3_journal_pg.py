# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the M3 journal, progress, evidence, resource, and blob
adapters against a dedicated test database."""

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.adapters.postgres.session_journal import PGJournalStore
from dlightrag.agent.session.effects import EffectIntent, EffectSettlement, ToolResultEntry
from dlightrag.agent.session.entries import (
    EffectIntentEntry,
    EffectResultEntry,
    UserMessageEntry,
)
from dlightrag.agent.session.ids import EntryId, IntentId, ProjectionId, SessionId, StageIntentId
from dlightrag.agent.session.projection import ContextProjection, TokenAnchor
from dlightrag.runtime.records import ClaimedRun, PendingArtifact, PendingArtifactReference
from dlightrag.runtime.settlements import (
    CommittedSpillUpdate,
    CompleteBlobDescriptor,
    EffectHostUpdate,
    FetchedResourceSettlementUpdate,
    InventoryPathRecord,
    MemoryOperationSettlement,
    OpaqueEvidenceResourceWrite,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
    WorkspaceInventoryUpdate,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_ADMIN: dict[str, Any] = dict(
    host="localhost", port=5432, user="dlightrag", password="dlightrag", database="dlightrag"
)
_TEST_DATABASE = "dlightrag_m3_journal_test"
_OWNER = "owner-alpha"
_WORKER = "worker-1"
_OTHER_WORKER = "worker-2"


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_ADMIN)
        await conn.close()
        return True
    except OSError, asyncpg.PostgresError:
        return False


@pytest.fixture(autouse=True)
async def pool():
    if not await _pg_available():
        pytest.skip("PostgreSQL is not reachable")
    admin = await asyncpg.connect(**_ADMIN)
    try:
        await admin.execute(f'CREATE DATABASE "{_TEST_DATABASE}"')
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await admin.close()

    created = await asyncpg.create_pool(
        **{**_ADMIN, "database": _TEST_DATABASE}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_ADMIN)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{_TEST_DATABASE}" WITH (FORCE)')
        finally:
            await admin.close()


async def _store(pool) -> PGAnswerRunStore:
    store = PGAnswerRunStore(pool=pool)
    await store.initialize()
    return store


def _prepared_input() -> dict[str, Any]:
    return {
        "session_id": str(uuid.uuid7()),
        "fingerprint": "f" * 64,
        "query": "question?",
        "workspaces": ["default"],
        "schema_version": 1,
    }


async def _accept(pool, *, owner: str = _OWNER) -> PGAnswerRunStore:
    store = await _store(pool)
    await store.accept_run(
        owner_id=owner,
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="f" * 64,
        prepared_input=_prepared_input(),
    )
    return store


async def _claim(pool, *, owner: str = _OWNER, worker: str = _WORKER) -> ClaimedRun:
    store = await _store(pool)
    await store.accept_run(
        owner_id=owner,
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="f" * 64,
        prepared_input=_prepared_input(),
    )
    claimed = await store.claim_next(worker_id=worker)
    assert claimed is not None
    return claimed


def _now() -> datetime:
    return datetime.now(UTC)


def _user(session_id: SessionId, content: str = "hello") -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content=content
    )


def _intent_entry(session_id: SessionId, intent_id: IntentId) -> EffectIntentEntry:
    return EffectIntentEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent=EffectIntent(
            intent_id=intent_id,
            tool_name="search_knowledge_base",
            replay_policy="replayable",
            contract_version=1,
            input_schema_digest="a" * 64,
            canonical_input='{"q":"x"}',
            source_call_id="c1",
        ),
    )


def _settlement(intent_id: IntentId, update) -> EffectSettlement:
    if isinstance(update, FetchedResourceSettlementUpdate):
        update = EffectHostUpdate(fetched=(update,))
    elif isinstance(update, WorkspaceInventoryUpdate):
        update = EffectHostUpdate(workspace_inventory=update)
    elif isinstance(update, CommittedSpillUpdate):
        update = EffectHostUpdate(committed_outputs=(update,))
    result = ToolResultEntry.text(
        tool_name="search_knowledge_base", call_id="c1", outcome="succeeded", text="found"
    )
    return EffectSettlement(outcome="succeeded", result=result, host_update=update)


def _result_entry(session_id: SessionId, intent_id: IntentId) -> EffectResultEntry:
    return EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent_id=intent_id,
        result=ToolResultEntry.text(
            tool_name="search_knowledge_base", call_id="c1", outcome="succeeded", text="found"
        ),
    )


def _projection() -> ContextProjection:
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
        token_anchors=(
            TokenAnchor(through_sequence=0, measured_input_tokens=4, measured_output_tokens=2),
        ),
    )


def _evidence_write(*, ordinal: int = 0, content: bytes = b"evidence") -> OpaqueEvidenceWrite:
    import hashlib

    return OpaqueEvidenceWrite(
        session_id=str(uuid.uuid4()),
        intent_id=str(uuid.uuid4()),
        result_ordinal=ordinal,
        content_digest=hashlib.sha256(content).hexdigest(),
        locator_digest="b" * 64,
        content=content,
        locator=b"locator",
    )


async def test_effect_settlement_commits_everything_atomically(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()

    first = await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    assert first.__class__.__name__ == "SessionCommit"

    update = EffectHostUpdate(evidence=(_evidence_write(),))
    settlement = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, update),
        entries=[_result_entry(session_id, intent_id)],
        projection=_projection(),
    )
    assert settlement.__class__.__name__ == "EffectCommit"
    assert settlement.version == 2  # type: ignore[attr-defined]
    assert settlement.appended_sequences == (2,)  # type: ignore[attr-defined]

    snapshot = await store.load(session_id)
    assert snapshot.version == 2
    assert snapshot.active_projection is not None
    assert [entry.sequence for entry in snapshot.entries] == [1, 2]
    assert isinstance(snapshot.entries[1], EffectResultEntry)

    # The settled intent is terminal and host facts are durable.
    conn = await pool.acquire()
    try:
        outcome = await conn.fetchval(
            "SELECT outcome FROM dlightrag_agent_effects"
            " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3 AND intent_id = $4",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
            session_id.value,
            intent_id.value,
        )
        assert outcome == "succeeded"
        evidence_count = await conn.fetchval("SELECT count(*) FROM dlightrag_answer_evidence")
        assert evidence_count == 1
    finally:
        await pool.release(conn)


async def test_version_conflict_rolls_back_every_write(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()

    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    conflict = await store.settle_effect(
        session_id=session_id,
        expected_version=0,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate(evidence=(_evidence_write(),))),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert conflict.__class__.__name__ == "VersionConflict"

    snapshot = await store.load(session_id)
    assert snapshot.version == 1
    assert len(snapshot.entries) == 1
    conn = await pool.acquire()
    try:
        count = await conn.fetchval("SELECT count(*) FROM dlightrag_answer_evidence")
        assert count == 0
        outcome = await conn.fetchval(
            "SELECT outcome FROM dlightrag_agent_effects"
            " WHERE owner_id = $1 AND run_id = $2 AND session_id = $3 AND intent_id = $4",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
            session_id.value,
            intent_id.value,
        )
        assert outcome is None
    finally:
        await pool.release(conn)


async def test_stale_epoch_writes_zero_rows(pool) -> None:
    claimed = await _claim(pool)
    run_uuid = uuid.UUID(claimed.run.run_id)
    stale = PGJournalStore(
        pool=pool,
        owner_id=_OWNER,
        run_id=run_uuid,
        worker_id=_WORKER,
        lease_owner=_WORKER,
        fencing_epoch=claimed.execution.fencing_epoch + 99,
    )
    session_id = SessionId.new()
    outcome = await stale.append(
        session_id=session_id, expected_version=0, entries=[_user(session_id)]
    )
    assert outcome.__class__.__name__ == "LeaseLost"

    conn = await pool.acquire()
    try:
        sessions = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_agent_sessions WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_uuid,
        )
        assert sessions == 0
    finally:
        await pool.release(conn)


async def test_duplicate_evidence_identity_is_idempotent_only_with_equal_digests(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    write = _evidence_write(ordinal=3, content=b"first")
    first = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate(evidence=(write,))),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert first.__class__.__name__ == "EffectCommit"

    # Same identity, same digests, new settlement attempt: already settled, not conflict.
    again = await store.settle_effect(
        session_id=session_id,
        expected_version=2,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate(evidence=(write,))),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert again.__class__.__name__ == "EffectAlreadySettled"

    # A different intent over the SAME evidence identity with different content
    # is a deterministic conflict, and the transaction rolls back.
    intent_two = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=2,
        entries=[_intent_entry(session_id, intent_two)],
    )
    import hashlib

    different = OpaqueEvidenceWrite(
        session_id=write.session_id,
        intent_id=write.intent_id,
        result_ordinal=write.result_ordinal,
        content_digest=hashlib.sha256(b"changed").hexdigest(),
        locator_digest=write.locator_digest,
        content=b"changed",
        locator=write.locator,
    )
    conflict = await store.settle_effect(
        session_id=session_id,
        expected_version=3,
        intent_id=intent_two,
        settlement=_settlement(intent_two, EffectHostUpdate(evidence=(different,))),
        entries=[_result_entry(session_id, intent_two)],
    )
    assert conflict.__class__.__name__ == "EvidenceConflict"
    snapshot = await store.load(session_id)
    assert snapshot.version == 3
    assert len(snapshot.entries) == 3  # intent appended, its settlement did not


async def test_fetched_resource_settlement_writes_complete_blob_and_resource(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )

    import hashlib

    from dlightrag.runtime.blob_chunks import BLOB_CHUNK_BYTES, plan_blob

    body = b"z" * (BLOB_CHUNK_BYTES + 7)
    plan = plan_blob(body)
    update = FetchedResourceSettlementUpdate(
        resource=OpaqueFetchedResourceWrite(
            resource_id="fetched-1",
            safe_name="page.html",
            media_type="text/html",
            capabilities={},
            blob_digest=plan.digest,
            source_locator_digest=hashlib.sha256(b"https://x").hexdigest(),
            source_locator=b"https://x",
            session_id=session_id.value,
            intent_id=intent_id.value,
        ),
        complete_blob=CompleteBlobDescriptor(
            digest=plan.digest,
            total_bytes=plan.total_bytes,
            chunks=tuple(plan.chunk(body, index) for index in range(plan.chunk_count)),
        ),
    )
    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, update),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert settled.__class__.__name__ == "EffectCommit"

    conn = await pool.acquire()
    try:
        chunks = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_blob_chunks WHERE owner_id = $1 AND digest = $2",
            _OWNER,
            plan.digest,
        )
        assert chunks == 2  # 1 MiB + 7 bytes
        size = await conn.fetchval(
            "SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
            _OWNER,
            plan.digest,
        )
        assert size == BLOB_CHUNK_BYTES + 7
        kind = await conn.fetchval(
            "SELECT kind FROM dlightrag_answer_resources"
            " WHERE owner_id = $1 AND run_id = $2 AND resource_id = $3",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
            "fetched-1",
        )
        assert kind == "fetched_blob"
    finally:
        await pool.release(conn)

    # The resource identity stays stable across reloads.
    snapshot = await store.load(session_id)
    last_entry = snapshot.entries[-1]
    assert last_entry.__class__.__name__ == "EffectResultEntry"
    assert last_entry.result.text_content == "found"  # type: ignore[attr-defined]


async def test_unequal_existing_resource_is_rejected(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )

    resource = OpaqueEvidenceResourceWrite(
        resource_id="res-1",
        safe_name="doc.pdf",
        media_type="application/pdf",
        capabilities={},
        session_id=session_id.value,
        intent_id=intent_id.value,
        result_ordinal=0,
        locator_digest="c" * 64,
    )
    first = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate(evidence=(), resources=(resource,))),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert first.__class__.__name__ == "EffectCommit"

    intent_two = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=2,
        entries=[_intent_entry(session_id, intent_two)],
    )
    unequal = OpaqueEvidenceResourceWrite(
        resource_id="res-1",
        safe_name="doc.pdf",
        media_type="application/pdf",
        capabilities={},
        session_id=session_id.value,
        intent_id=intent_two.value,
        result_ordinal=0,
        locator_digest="d" * 64,
    )
    conflict = await store.settle_effect(
        session_id=session_id,
        expected_version=3,
        intent_id=intent_two,
        settlement=_settlement(intent_two, EffectHostUpdate(evidence=(), resources=(unequal,))),
        entries=[_result_entry(session_id, intent_two)],
    )
    assert conflict.__class__.__name__ == "EvidenceConflict"


async def test_accepted_attachment_registration_commits_with_acceptance(pool) -> None:
    store = await _store(pool)
    content = b"%PDF-accepted"
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    creation = await store.accept_run(
        owner_id=_OWNER,
        run_id=str(uuid.uuid7()),
        idempotency_key="accept-1",
        fingerprint="f" * 64,
        prepared_input=_prepared_input(),
        resources=(
            {
                "resource_id": "accepted-1",
                "safe_name": "report.pdf",
                "media_type": "application/pdf",
                "capabilities": {},
                "ordinal": 0,
                "blob_digest": digest,
            },
        ),
        blobs=(PendingArtifact(content=content),),
        references=(
            PendingArtifactReference(
                resource_id="accepted-1",
                reference_kind="current_attachment",
                ordinal=0,
                digest=digest,
                filename="report.pdf",
                mime_type="application/pdf",
            ),
        ),
    )
    assert not creation.replayed

    conn = await pool.acquire()
    try:
        blob = await conn.fetchval(
            "SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
            _OWNER,
            digest,
        )
        assert blob == len(content)
        kind = await conn.fetchval(
            "SELECT kind FROM dlightrag_answer_resources WHERE owner_id = $1 AND resource_id = $2",
            _OWNER,
            "accepted-1",
        )
        assert kind == "accepted_blob"
        prepared = await conn.fetchval(
            "SELECT prepared_input_json IS NOT NULL FROM dlightrag_answer_runs WHERE owner_id = $1",
            _OWNER,
        )
        assert prepared is True
    finally:
        await pool.release(conn)

    # Replayed acceptance returns the same run without duplicating anything.
    replay = await store.replay_run(
        owner_id=_OWNER, idempotency_key="accept-1", idempotency_fingerprint="f" * 64
    )
    assert replay is not None and replay.replayed


async def test_fast_stage_settlement_never_creates_an_agent_session(pool) -> None:
    claimed = await _claim(pool)
    progress = claimed.execution.progress_store
    stage_id = StageIntentId.deterministic(run_id=claimed.run.run_id, name="fast:planner:0")

    settled = await progress.settle_stage(
        expected_progress_version=0,
        stage_intent_id=stage_id,
        stage_name="planner",
        state={"plan": "canonical"},
        evidence=(),
    )
    assert settled.__class__.__name__ == "StageCommit"
    assert settled.progress_version == 1  # type: ignore[attr-defined]

    conn = await pool.acquire()
    try:
        sessions = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_agent_sessions WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
        assert sessions == 0
        progress_version = await conn.fetchval(
            "SELECT durable_progress_version FROM dlightrag_answer_runs"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
        assert progress_version == 1
    finally:
        await pool.release(conn)

    # Re-settling the same stage with the same state is idempotent and does not
    # advance progress twice.
    again = await progress.settle_stage(
        expected_progress_version=1,
        stage_intent_id=stage_id,
        stage_name="planner",
        state={"plan": "canonical"},
        evidence=(),
    )
    assert again.__class__.__name__ == "StageCommit"


async def test_stage_conflict_on_different_state(pool) -> None:
    claimed = await _claim(pool)
    progress = claimed.execution.progress_store
    stage_id = StageIntentId.deterministic(run_id=claimed.run.run_id, name="fast:retrieval:1")

    await progress.settle_stage(
        expected_progress_version=0,
        stage_intent_id=stage_id,
        stage_name="retrieval",
        state={"evidence": [1]},
        evidence=(),
    )
    conflict = await progress.settle_stage(
        expected_progress_version=1,
        stage_intent_id=stage_id,
        stage_name="retrieval",
        state={"evidence": [2]},
        evidence=(),
    )
    assert conflict.__class__.__name__ == "StageConflict"


async def test_event_appends_do_not_advance_progress(pool) -> None:
    claimed = await _claim(pool)
    store = await _store(pool)
    run_uuid = uuid.UUID(claimed.run.run_id)

    await store.append_token_batch(
        owner_id=_OWNER,
        run_id=str(run_uuid),
        worker_id=_WORKER,
        fencing_epoch=claimed.execution.fencing_epoch,
        text="token stream",
    )
    conn = await pool.acquire()
    try:
        progress = await conn.fetchval(
            "SELECT durable_progress_version FROM dlightrag_answer_runs"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_uuid,
        )
        assert progress == 0
    finally:
        await pool.release(conn)


async def _progress(pool, run_id: str) -> int:
    conn = await pool.acquire()
    try:
        value = await conn.fetchval(
            "SELECT durable_progress_version FROM dlightrag_answer_runs"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(run_id),
        )
        return int(value)
    finally:
        await pool.release(conn)


async def _expire_lease(pool, run_id: str) -> None:
    conn = await pool.acquire()
    try:
        await conn.execute(
            "UPDATE dlightrag_answer_runs SET lease_expires_at = NOW() - INTERVAL '1 second'"
            " WHERE run_id = $1",
            uuid.UUID(run_id),
        )
    finally:
        await pool.release(conn)


async def test_live_settlement_advances_durable_progress(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    after_append = await _progress(pool, claimed.run.run_id)
    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate()),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert settled.__class__.__name__ == "EffectCommit"
    assert await _progress(pool, claimed.run.run_id) == after_append + 1


async def test_memory_operation_event_is_atomic_and_exactly_once(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    update = EffectHostUpdate(
        memory_operation=MemoryOperationSettlement(
            operation="remember",
            outcome="changed",
            change_id=str(uuid.uuid7()),
            memory_ids=(str(uuid.uuid7()),),
            kind="preference",
            body="Use Chinese.",
        )
    )
    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, update),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert settled.__class__.__name__ == "EffectCommit"
    replay = await store.settle_effect(
        session_id=session_id,
        expected_version=2,
        intent_id=intent_id,
        settlement=_settlement(intent_id, update),
        entries=[_result_entry(session_id, intent_id)],
    )
    assert replay.__class__.__name__ == "EffectAlreadySettled"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT payload FROM dlightrag_answer_run_events "
            "WHERE owner_id = $1 AND run_id = $2 "
            "AND event_type = 'memory_operation_settled'",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
    assert len(rows) == 1
    payload = rows[0]["payload"]
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    assert payload["intent_id"] == intent_id.value
    assert payload["body"] == "Use Chinese."


async def test_prelude_settlement_does_not_advance_durable_progress(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    after_append = await _progress(pool, claimed.run.run_id)
    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate()),
        entries=[_result_entry(session_id, intent_id)],
        progress="prelude",
    )
    assert settled.__class__.__name__ == "EffectCommit"
    assert await _progress(pool, claimed.run.run_id) == after_append
    snapshot = await store.load(session_id)
    assert snapshot.version == 2
    assert isinstance(snapshot.entries[-1], EffectResultEntry)


async def test_prelude_only_reclaims_still_abandon(pool) -> None:
    claimed = await _claim(pool)
    journal = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await journal.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    store = await _store(pool)
    await _expire_lease(pool, claimed.run.run_id)
    first = await store.claim_next(worker_id="reclaim-0")
    assert first is not None
    assert first.run.reclaims_without_progress == 1
    settled = await first.execution.session_store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(intent_id, EffectHostUpdate()),
        entries=[_result_entry(session_id, intent_id)],
        progress="prelude",
    )
    assert settled.__class__.__name__ == "EffectCommit"
    for index in range(1, 3):
        await _expire_lease(pool, claimed.run.run_id)
        reclaimed = await store.claim_next(worker_id=f"reclaim-{index}")
        assert reclaimed is not None
        assert reclaimed.run.reclaims_without_progress == index + 1
    await _expire_lease(pool, claimed.run.run_id)
    fourth = await store.claim_next(worker_id="reclaim-3")
    assert fourth is None
    record = await store.get_run(owner_id=_OWNER, run_id=claimed.run.run_id)
    assert record is not None
    assert record.status == "failed"
    assert record.error_kind == "run_abandoned"


async def test_writer_startup_adds_workspace_tables(pool) -> None:
    await _store(pool)
    conn = await pool.acquire()
    try:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            " AND tablename LIKE 'dlightrag_answer_%'"
        )
        names = {row["tablename"] for row in tables}
        assert "dlightrag_answer_workspace_inventory" in names
        assert "dlightrag_answer_committed_spills" in names
        column = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns"
            " WHERE table_name = 'dlightrag_answer_runs' AND column_name = 'workspace_epoch'"
        )
        assert column == 1
    finally:
        await pool.release(conn)


async def test_committed_spill_cannot_carry_a_blob_digest(pool) -> None:
    claimed = await _claim(pool)
    conn = await pool.acquire()
    try:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "INSERT INTO dlightrag_answer_resources ("
                " owner_id, run_id, resource_id, kind, safe_name, media_type, blob_digest)"
                " VALUES ($1, $2, 'res_bad', 'committed_spill', 'x', 'text/plain', $3)",
                _OWNER,
                uuid.UUID(claimed.run.run_id),
                "a" * 64,
            )
    finally:
        await pool.release(conn)


async def test_handoff_wrong_epoch_writes_zero_rows(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.workspace_store
    assert store is not None
    result = await store.handoff_epoch(expected_epoch=3, destination_epoch=4, inventory=())
    assert result.__class__.__name__ == "HandoffConflict"
    record = await (await _store(pool)).get_run(owner_id=_OWNER, run_id=claimed.run.run_id)
    assert record is not None
    assert record.workspace_epoch is None


async def test_settle_effect_writes_inventory_and_spill_without_prelude_progress(pool) -> None:
    from dlightrag.runtime.settlements import (
        CommittedSpillUpdate,
        InventoryPathRecord,
        WorkspaceInventoryUpdate,
    )

    claimed = await _claim(pool)
    journal = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await journal.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    after_append = await _progress(pool, claimed.run.run_id)
    settled = await journal.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=_settlement(
            intent_id,
            WorkspaceInventoryUpdate(
                upserts=(
                    InventoryPathRecord(
                        relative_path="notes/a.md", entry_type="file", size_bytes=4
                    ),
                )
            ),
        ),
        entries=[_result_entry(session_id, intent_id)],
        progress="prelude",
    )
    assert settled.__class__.__name__ == "EffectCommit"
    assert await _progress(pool, claimed.run.run_id) == after_append
    intent_two = IntentId.new()
    await journal.append(
        session_id=session_id,
        expected_version=2,
        entries=[_intent_entry(session_id, intent_two)],
    )
    await journal.settle_effect(
        session_id=session_id,
        expected_version=3,
        intent_id=intent_two,
        settlement=_settlement(
            intent_two,
            CommittedSpillUpdate(
                resource_id="res_spill",
                content_digest="c" * 64,
                size_bytes=12,
                session_id=str(session_id),
                intent_id=str(intent_two),
            ),
        ),
        entries=[_result_entry(session_id, intent_two)],
    )
    workspace = claimed.execution.workspace_store
    assert workspace is not None
    assert any(item.relative_path == "notes/a.md" for item in await workspace.load_inventory())
    assert any(item.resource_id == "res_spill" for item in await workspace.load_spills())


async def test_terminal_finish_deletes_spill_rows(pool) -> None:
    claimed = await _claim(pool)
    workspace = claimed.execution.workspace_store
    assert workspace is not None
    from dlightrag.runtime.workspace import CommittedSpillRecord

    await workspace.register_spill(
        CommittedSpillRecord(
            resource_id="res_done",
            content_digest="d" * 64,
            size_bytes=1,
            session_id=str(uuid.uuid4()),
            intent_id=str(uuid.uuid4()),
        )
    )
    store = await _store(pool)
    await store.finish_success(
        owner_id=_OWNER,
        run_id=claimed.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=claimed.execution.fencing_epoch,
        result={"answer": "ok"},
    )
    assert await workspace.load_spills() == ()


async def test_effect_host_update_commits_inventory_and_spill(pool) -> None:
    claimed = await _claim(pool)
    journal = claimed.execution.session_store
    session_id = SessionId.new()
    intent_id = IntentId.new()
    await journal.append(
        session_id=session_id,
        expected_version=0,
        entries=[_intent_entry(session_id, intent_id)],
    )
    result = EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent_id=intent_id,
        result=ToolResultEntry.text(
            tool_name="write",
            call_id="c1",
            outcome="succeeded",
            text="wrote notes.md",
        ),
    )
    settled = await journal.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent_id,
        settlement=EffectSettlement(
            outcome="succeeded",
            result=result.result,
            host_update=EffectHostUpdate(
                committed_outputs=(
                    CommittedSpillUpdate(
                        resource_id="spill_from_details",
                        content_digest="f" * 64,
                        size_bytes=12,
                        session_id=session_id.value,
                        intent_id=intent_id.value,
                    ),
                ),
                workspace_inventory=WorkspaceInventoryUpdate(
                    upserts=(
                        InventoryPathRecord(
                            relative_path="notes.md",
                            entry_type="file",
                            size_bytes=5,
                            mode=0o644,
                            content_digest="e" * 64,
                        ),
                    ),
                ),
            ),
        ),
        entries=[result],
    )
    assert settled.__class__.__name__ == "EffectCommit"
    workspace = claimed.execution.workspace_store
    assert workspace is not None
    inventory = await workspace.load_inventory()
    assert any(item.relative_path == "notes.md" for item in inventory)
    assert any(item.resource_id == "spill_from_details" for item in await workspace.load_spills())


async def test_list_runs_is_owner_scoped_and_cursorable(pool) -> None:
    store = await _store(pool)
    first = await store.accept_run(
        owner_id=_OWNER,
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="a" * 64,
        prepared_input=_prepared_input(),
    )
    second = await store.accept_run(
        owner_id=_OWNER,
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="b" * 64,
        prepared_input=_prepared_input(),
    )
    await store.accept_run(
        owner_id="other",
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="c" * 64,
        prepared_input=_prepared_input(),
    )
    page = await store.list_runs(owner_id=_OWNER, limit=1)
    assert len(page) == 1
    assert page[0].run_id == first.run.run_id
    rest = await store.list_runs(owner_id=_OWNER, after_run_id=page[0].run_id, limit=10)
    assert [item.run_id for item in rest] == [second.run.run_id]
