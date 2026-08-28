# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL parity tests for the canonical AgentSessionRuntime."""

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import asyncpg
import pytest
from pydantic import BaseModel

from dlightrag.adapters.postgres.answer.answer_runs import PGAnswerRunStore
from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
from dlightrag.application.answer_runs import AnswerService
from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    SessionEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    LaneId,
    ProjectionId,
    SessionId,
    StageIntentId,
)
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import OperationCompleted, ToolBatchItem
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.engine.agent.session.registers import (
    LaneHead,
    LaneState,
    RegisterRecord,
    RequestSnapshot,
    SetRegister,
)
from dlightrag.engine.agent.session.runtime import (
    AgentSessionRuntime,
    RuntimeContext,
    ToolEffectResult,
)
from dlightrag.engine.agent.session.transactions import (
    HostDeltaSettlement,
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)
from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.engine.agent.tools import AgentTool, ToolResult
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.answer.fast import FastSessionHost, ensure_session_lane
from dlightrag.engine.runtime.blob_chunks import BLOB_CHUNK_BYTES, plan_blob
from dlightrag.engine.runtime.records import ClaimedRun, PendingArtifact, PendingArtifactReference
from dlightrag.engine.runtime.settlements import (
    CommittedSpillUpdate,
    CompleteBlobDescriptor,
    EffectHostUpdate,
    FetchedResourceSettlementUpdate,
    InventoryPathRecord,
    MemoryOperationSettlement,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
    WorkspaceInventoryUpdate,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _no_settled_result() -> None:
    return None


_ADMIN: dict[str, Any] = dict(
    host="localhost", port=5432, user="dlightrag", password="dlightrag", database="dlightrag"
)
_TEST_DATABASE = "dlightrag_agent_session_test"
_OWNER = "owner-alpha"
_WORKER = "worker-1"


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


async def _claim(
    pool,
    *,
    session_id: SessionId | None = None,
    lane_id: LaneId = LaneId.main(),
    source_lane_id: LaneId | None = None,
) -> ClaimedRun:
    store = await _store(pool)
    await store.accept_run(
        owner_id=_OWNER,
        run_id=str(uuid.uuid7()),
        idempotency_key=None,
        fingerprint="f" * 64,
        prepared_input={
            "agent_session_id": (session_id or SessionId.new()).value,
            "agent_lane_id": lane_id.value,
            "source_lane_id": source_lane_id.value if source_lane_id else None,
            "fingerprint": "f" * 64,
            "query": "question?",
            "workspaces": ["default"],
            "schema_version": 1,
        },
    )
    claimed = await store.claim_next(worker_id=_WORKER)
    assert claimed is not None
    return claimed


class Args(BaseModel):
    value: str


async def _unused(_args, _runtime) -> ToolResult:
    return ToolResult.text("unused")


_TOOL = AgentTool(
    name="lookup",
    description="lookup",
    input_model=Args,
    execute=_unused,
    replay_policy="never",
)
_PLAN = AgentRunPlan.from_tools([_TOOL], model_role="query", context_policy_revision="context-v1")


class Effects:
    def __init__(self, *, session_id: SessionId) -> None:
        self._session_id = session_id
        self.turns = [
            AssistantTurn(
                text="",
                tool_calls=(ToolCall("c1", "lookup", {"value": "x"}),),
                stop_reason="tool_use",
            ),
            AssistantTurn(text="done", tool_calls=(), stop_reason="stop"),
        ]

    async def assemble_request(self, context: RuntimeContext) -> RequestSnapshot:
        return RequestSnapshot.from_values(
            operation_id=context.operation_id,
            turn_number=getattr(context.state, "turn_count", 0) + 1,
            plan_digest=context.meta.plan_digest,
            model_role="query",
            messages=[{"role": "user", "content": "exact"}],
            tools=[],
            tool_choice="auto",
            max_tokens=100,
        )

    async def call_provider(
        self,
        context: RuntimeContext,
        request: RequestSnapshot,
        attempt_id: AttemptId,
        emit_ephemeral,
    ) -> AssistantTurn:
        del context, request, attempt_id, emit_ephemeral
        return self.turns.pop(0)

    async def execute_tool(
        self,
        context: RuntimeContext,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
        attempt_id: AttemptId,
        emit_ephemeral,
    ) -> ToolEffectResult[EffectHostUpdate]:
        del context, arguments, attempt_id, emit_ephemeral
        assert item.intent_id is not None
        content = b"evidence"
        evidence = OpaqueEvidenceWrite(
            session_id=self._session_id.value,
            intent_id=item.intent_id.value,
            result_ordinal=0,
            content_digest=hashlib.sha256(content).hexdigest(),
            locator_digest=hashlib.sha256(b"locator").hexdigest(),
            content=content,
            locator=b"locator",
        )
        return ToolEffectResult(
            ToolResultEntry.text(
                tool_name=item.tool_name,
                call_id=item.call_id,
                outcome="succeeded",
                text="found",
            ),
            EffectHostUpdate(evidence=(evidence,)),
        )

    async def compact(self, context: RuntimeContext, attempt: int):
        del context, attempt
        raise AssertionError("compaction not expected")


async def _drive(adapter, *, session_id: SessionId, fencing_epoch: int):
    runtime = AgentSessionRuntime(
        transactions=adapter,
        load=adapter.load,
        effects=Effects(session_id=session_id),
        tools=[_TOOL],
        fencing_epoch=fencing_epoch,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key=f"parity:{session_id.value}",
        content="question",
        plan=_PLAN,
    )
    final = await runtime.drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    return await adapter.load(session_id)


async def _seed_transaction_session(
    store, session_id: SessionId, fencing_epoch: int
) -> TransactionCommit:
    root = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="question",
    )
    head = LaneHead(LaneId.main(), root.entry_id)
    state = LaneState(LaneId.main())
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=[root],
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionCommit)
    return outcome


async def _append_lane_entry(
    store,
    session_id: SessionId,
    lane_id: LaneId,
    expected_head: RegisterRecord,
    entry: SessionEntry,
    *,
    fencing_epoch: int,
):
    assert isinstance(expected_head.value, LaneHead)
    placed = replace(entry, parent_entry_id=expected_head.value.entry_id)
    return await store.transact(
        session_id=session_id,
        fencing_epoch=fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=[placed],
            register_writes=[SetRegister(LaneHead(lane_id, placed.entry_id))],
            expectations=[RegisterExpectation(expected_head.ref, expected_head.sequence)],
        ),
    )


async def _append_transaction_entry(
    store,
    session_id: SessionId,
    entry: SessionEntry,
    *,
    fencing_epoch: int,
    intent_id: IntentId | None = None,
    host_delta: EffectHostUpdate | None = None,
):
    snapshot = await store.load(session_id)
    head = snapshot.tree.lane().head
    placed = replace(entry, parent_entry_id=head.value.entry_id)
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=[placed],
            register_writes=[SetRegister(LaneHead(LaneId.main(), placed.entry_id))],
            expectations=[RegisterExpectation(head.ref, head.sequence)],
            host_delta=(
                HostDeltaSettlement(intent_id, host_delta)
                if intent_id is not None and host_delta is not None
                else None
            ),
        ),
    )
    return outcome


def _tool_result(
    session_id: SessionId,
    intent_id: IntentId,
    *,
    parts=None,
    entry_id: EntryId | None = None,
) -> ToolResultMessageEntry:
    return ToolResultMessageEntry(
        entry_id=entry_id or EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        result=ToolResultEntry(
            tool_name="lookup",
            call_id="c1",
            outcome="succeeded",
            parts=parts or (ToolTextPart("found"),),
        ),
        intent_id=intent_id,
        source_index=0,
        contract_version=1,
        input_schema_digest="a" * 64,
        replay_policy="never",
        attempt_id=AttemptId.new(),
        effective_input_digest="b" * 64,
    )


def _claimed_session(claimed: ClaimedRun) -> SessionId:
    assert claimed.run.prepared_input is not None
    return SessionId(str(claimed.run.prepared_input["agent_session_id"]))


async def _progress(pool, run_id: str) -> int:
    async with pool.acquire() as conn:
        value = await conn.fetchval(
            "SELECT durable_progress_version FROM dlightrag_answer_runs"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(run_id),
        )
    return int(value)


async def test_agent_session_runtime_memory_and_pg_have_atomic_state_parity(pool) -> None:
    claimed = await _claim(pool)
    pg = claimed.execution.session_repository
    memory = MemoryAgentSessionRepository[EffectHostUpdate](
        fencing_epoch=claimed.execution.fencing_epoch
    )
    memory_id = SessionId.new()
    pg_id = _claimed_session(claimed)
    memory_snapshot = await _drive(
        memory,
        session_id=memory_id,
        fencing_epoch=claimed.execution.fencing_epoch,
    )
    pg_snapshot = await _drive(
        pg,
        session_id=pg_id,
        fencing_epoch=claimed.execution.fencing_epoch,
    )
    assert (
        [entry.entry_type for entry in memory_snapshot.entries]
        == [entry.entry_type for entry in pg_snapshot.entries]
        == ["user_message", "assistant_message", "tool_result", "assistant_message"]
    )
    assert [record.ref.kind for record in memory_snapshot.registers] == [
        record.ref.kind for record in pg_snapshot.registers
    ]
    assert memory_snapshot.commit_sequence == pg_snapshot.commit_sequence
    assert len(memory.applied_host_deltas(memory_id)) == 1
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_evidence") == 1


async def test_host_delta_identity_conflict_rolls_back_entry_and_register(pool) -> None:
    from dlightrag.adapters.postgres.answer.session_repository import _EvidenceIdentityConflict

    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    session_id = _claimed_session(claimed)
    await _seed_transaction_session(store, session_id, epoch)
    intent_id = IntentId.new()
    identity_session = SessionId.new()
    content = b"first"
    write = OpaqueEvidenceWrite(
        session_id=identity_session.value,
        intent_id=intent_id.value,
        result_ordinal=3,
        content_digest=hashlib.sha256(content).hexdigest(),
        locator_digest=hashlib.sha256(b"locator").hexdigest(),
        content=content,
        locator=b"locator",
    )
    first = await _append_transaction_entry(
        store,
        session_id,
        _tool_result(session_id, intent_id),
        fencing_epoch=epoch,
        intent_id=intent_id,
        host_delta=EffectHostUpdate(evidence=(write,)),
    )
    assert isinstance(first, TransactionCommit)
    before = await store.load(session_id)

    changed_content = b"changed"
    different = replace(
        write,
        content=changed_content,
        content_digest=hashlib.sha256(changed_content).hexdigest(),
    )
    second_intent = IntentId.new()
    with pytest.raises(_EvidenceIdentityConflict):
        await _append_transaction_entry(
            store,
            session_id,
            _tool_result(session_id, second_intent),
            fencing_epoch=epoch,
            intent_id=second_intent,
            host_delta=EffectHostUpdate(evidence=(different,)),
        )
    after = await store.load(session_id)
    assert after.commit_sequence == before.commit_sequence
    assert after.entries == before.entries
    assert after.tree.lane().head == before.tree.lane().head
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content_digest, content FROM dlightrag_answer_evidence"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
    assert row is not None
    assert row["content_digest"] == write.content_digest
    assert bytes(row["content"]) == content


async def test_fetched_resource_host_delta_writes_complete_blob(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    session_id = _claimed_session(claimed)
    await _seed_transaction_session(store, session_id, epoch)
    intent_id = IntentId.new()
    body = b"z" * (BLOB_CHUNK_BYTES + 7)
    blob = plan_blob(body)
    fetched = FetchedResourceSettlementUpdate(
        resource=OpaqueFetchedResourceWrite(
            resource_id="fetched-1",
            safe_name="page.html",
            media_type="text/html",
            capabilities={},
            blob_digest=blob.digest,
            source_locator_digest=hashlib.sha256(b"https://example.test/page").hexdigest(),
            source_locator=b"https://example.test/page",
            session_id=session_id.value,
            intent_id=intent_id.value,
        ),
        complete_blob=CompleteBlobDescriptor(
            digest=blob.digest,
            total_bytes=blob.total_bytes,
            chunks=tuple(blob.chunk(body, index) for index in range(blob.chunk_count)),
        ),
    )
    outcome = await _append_transaction_entry(
        store,
        session_id,
        _tool_result(session_id, intent_id),
        fencing_epoch=epoch,
        intent_id=intent_id,
        host_delta=EffectHostUpdate(fetched=(fetched,)),
    )
    assert isinstance(outcome, TransactionCommit)
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_blob_chunks WHERE owner_id = $1 AND digest = $2",
                _OWNER,
                blob.digest,
            )
            == 2
        )
        assert await conn.fetchval(
            "SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
            _OWNER,
            blob.digest,
        ) == len(body)
        chunk_rows = await conn.fetch(
            "SELECT chunk_index, content FROM dlightrag_blob_chunks"
            " WHERE owner_id = $1 AND digest = $2 ORDER BY chunk_index",
            _OWNER,
            blob.digest,
        )
        assert [int(row["chunk_index"]) for row in chunk_rows] == [0, 1]
        assert b"".join(bytes(row["content"]) for row in chunk_rows) == body
        resource = await conn.fetchrow(
            "SELECT kind, blob_digest FROM dlightrag_answer_resources"
            " WHERE owner_id = $1 AND run_id = $2 AND resource_id = 'fetched-1'",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
    assert resource is not None
    assert (resource["kind"], resource["blob_digest"]) == ("fetched_blob", blob.digest)


async def test_fetched_blob_size_collision_is_an_evidence_identity_conflict(pool) -> None:
    from dlightrag.adapters.postgres.answer.session_repository import _EvidenceIdentityConflict

    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    session_id = _claimed_session(claimed)
    await _seed_transaction_session(store, session_id, epoch)
    before = await store.load(session_id)
    intent_id = IntentId.new()
    body = b"fetched collision"
    blob = plan_blob(body)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_blobs (owner_id, digest, byte_size) VALUES ($1, $2, $3)",
            _OWNER,
            blob.digest,
            blob.total_bytes + 1,
        )
    fetched = FetchedResourceSettlementUpdate(
        resource=OpaqueFetchedResourceWrite(
            resource_id="fetched-collision",
            safe_name="collision.txt",
            media_type="text/plain",
            capabilities={},
            blob_digest=blob.digest,
            source_locator_digest=hashlib.sha256(b"https://example.test/collision").hexdigest(),
            source_locator=b"https://example.test/collision",
            session_id=session_id.value,
            intent_id=intent_id.value,
        ),
        complete_blob=CompleteBlobDescriptor(
            digest=blob.digest,
            total_bytes=blob.total_bytes,
            chunks=(body,),
        ),
    )

    with pytest.raises(_EvidenceIdentityConflict):
        await _append_transaction_entry(
            store,
            session_id,
            _tool_result(session_id, intent_id),
            fencing_epoch=epoch,
            intent_id=intent_id,
            host_delta=EffectHostUpdate(fetched=(fetched,)),
        )

    after = await store.load(session_id)
    assert after.commit_sequence == before.commit_sequence
    assert after.entries == before.entries
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_blob_chunks WHERE owner_id = $1 AND digest = $2",
                _OWNER,
                blob.digest,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_answer_resources"
                " WHERE owner_id = $1 AND resource_id = 'fetched-collision'",
                _OWNER,
            )
            == 0
        )


async def test_acceptance_registers_attachment_blob_atomically(pool) -> None:
    store = await _store(pool)
    content = b"%PDF-accepted"
    digest = hashlib.sha256(content).hexdigest()
    creation = await store.accept_run(
        owner_id=_OWNER,
        run_id=str(uuid.uuid7()),
        idempotency_key="attachment-acceptance",
        fingerprint="f" * 64,
        prepared_input={
            "agent_session_id": SessionId.new().value,
            "agent_lane_id": "main",
            "fingerprint": "f" * 64,
            "query": "question?",
            "workspaces": ["default"],
            "schema_version": 1,
        },
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
    async with pool.acquire() as conn:
        assert await conn.fetchval(
            "SELECT byte_size FROM dlightrag_blobs WHERE owner_id = $1 AND digest = $2",
            _OWNER,
            digest,
        ) == len(content)
        assert (
            await conn.fetchval(
                "SELECT kind FROM dlightrag_answer_resources"
                " WHERE owner_id = $1 AND run_id = $2 AND resource_id = 'accepted-1'",
                _OWNER,
                uuid.UUID(creation.run.run_id),
            )
            == "accepted_blob"
        )


async def test_host_delta_commits_workspace_inventory_and_spill(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    session_id = _claimed_session(claimed)
    await _seed_transaction_session(store, session_id, epoch)
    intent_id = IntentId.new()
    update = EffectHostUpdate(
        committed_outputs=(
            CommittedSpillUpdate(
                resource_id="spill-1",
                content_digest="c" * 64,
                size_bytes=12,
                session_id=session_id.value,
                intent_id=intent_id.value,
            ),
        ),
        workspace_inventory=WorkspaceInventoryUpdate(
            upserts=(
                InventoryPathRecord(
                    relative_path="notes/a.md",
                    entry_type="file",
                    size_bytes=4,
                    mode=0o644,
                    content_digest="d" * 64,
                ),
            ),
        ),
    )
    outcome = await _append_transaction_entry(
        store,
        session_id,
        _tool_result(session_id, intent_id),
        fencing_epoch=epoch,
        intent_id=intent_id,
        host_delta=update,
    )
    assert isinstance(outcome, TransactionCommit)
    workspace = claimed.execution.workspace_store
    assert workspace is not None
    inventory = await workspace.load_inventory()
    spills = await workspace.load_spills()
    assert [(item.relative_path, item.content_digest) for item in inventory] == [
        ("notes/a.md", "d" * 64)
    ]
    assert [(item.resource_id, item.content_digest) for item in spills] == [("spill-1", "c" * 64)]


async def test_memory_operation_event_is_exactly_once_with_transaction(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    session_id = _claimed_session(claimed)
    await _seed_transaction_session(store, session_id, epoch)
    snapshot = await store.load(session_id)
    head = snapshot.tree.lane().head
    assert isinstance(head.value, LaneHead)
    intent_id = IntentId.new()
    entry = replace(
        _tool_result(session_id, intent_id),
        parent_entry_id=head.value.entry_id,
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
    transaction = SessionTransaction.from_parts(
        entries=[entry],
        register_writes=[SetRegister(LaneHead(LaneId.main(), entry.entry_id))],
        expectations=[RegisterExpectation(head.ref, head.sequence)],
        host_delta=HostDeltaSettlement(intent_id, update),
    )
    first = await store.transact(
        session_id=session_id,
        fencing_epoch=epoch,
        transaction=transaction,
    )
    replay = await store.transact(
        session_id=session_id,
        fencing_epoch=epoch,
        transaction=transaction,
    )
    assert isinstance(first, TransactionCommit)
    assert isinstance(replay, RegisterConflict)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT payload FROM dlightrag_answer_run_events"
            " WHERE owner_id = $1 AND run_id = $2"
            " AND event_type = 'memory_operation_settled'",
            _OWNER,
            uuid.UUID(claimed.run.run_id),
        )
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload"])
    assert payload["intent_id"] == intent_id.value
    assert payload["body"] == "Use Chinese."


async def test_live_session_progress_and_passive_events_are_distinct(pool) -> None:
    claimed = await _claim(pool)
    run_store = await _store(pool)
    assert await _progress(pool, claimed.run.run_id) == 0
    await run_store.append_token_batch(
        owner_id=_OWNER,
        run_id=claimed.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=claimed.execution.fencing_epoch,
        text="ephemeral progress",
    )
    assert await _progress(pool, claimed.run.run_id) == 0
    register_only_session = _claimed_session(claimed)
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    register_commit = await claimed.execution.session_repository.transact(
        session_id=register_only_session,
        fencing_epoch=claimed.execution.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(register_commit, TransactionCommit)
    assert await _progress(pool, claimed.run.run_id) == 0
    session_id = register_only_session
    root = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="question",
    )
    root_commit = await _append_transaction_entry(
        claimed.execution.session_repository,
        session_id,
        root,
        fencing_epoch=claimed.execution.fencing_epoch,
    )
    assert isinstance(root_commit, TransactionCommit)
    assert await _progress(pool, claimed.run.run_id) == 1
    snapshot = await claimed.execution.session_repository.load(session_id)
    head = snapshot.tree.lane().head
    assert isinstance(head.value, LaneHead)
    recovery_intent = IntentId.new()
    recovery_entry = replace(
        _tool_result(session_id, recovery_intent),
        parent_entry_id=head.value.entry_id,
    )
    recovery = await claimed.execution.session_repository.transact(
        session_id=session_id,
        fencing_epoch=claimed.execution.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=[recovery_entry],
            register_writes=[SetRegister(LaneHead(LaneId.main(), recovery_entry.entry_id))],
            expectations=[RegisterExpectation(head.ref, head.sequence)],
            advances_durable_progress=False,
        ),
    )
    assert isinstance(recovery, TransactionCommit)
    assert await _progress(pool, claimed.run.run_id) == 1
    live_intent = IntentId.new()
    await _append_transaction_entry(
        claimed.execution.session_repository,
        session_id,
        _tool_result(session_id, live_intent),
        fencing_epoch=claimed.execution.fencing_epoch,
        intent_id=live_intent,
        host_delta=EffectHostUpdate(),
    )
    assert await _progress(pool, claimed.run.run_id) == 2


async def test_fast_stage_progress_never_creates_agent_session(pool) -> None:
    claimed = await _claim(pool)
    stage_id = StageIntentId.deterministic(
        run_id=claimed.run.run_id,
        name="fast:planner:0",
    )
    settled = await claimed.execution.progress_store.settle_stage(
        expected_progress_version=0,
        stage_intent_id=stage_id,
        stage_name="planner",
        state={"plan": "canonical"},
        evidence=(),
    )
    assert settled.__class__.__name__ == "StageCommit"
    assert await _progress(pool, claimed.run.run_id) == 1
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM dlightrag_agent_sessions"
                " WHERE owner_id = $1 AND lease_run_id = $2",
                _OWNER,
                uuid.UUID(claimed.run.run_id),
            )
            == 0
        )


async def test_postgres_and_service_transcript_project_typed_tool_result_parts(pool) -> None:
    session_id = SessionId.new()
    claimed = await _claim(pool, session_id=session_id)
    store = claimed.execution.session_repository
    epoch = claimed.execution.fencing_epoch
    await _seed_transaction_session(store, session_id, epoch)
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="checking",
        stop_reason="tool_use",
        tool_calls=(ToolCall("c1", "lookup", {"value": "x"}),),
    )
    assistant_commit = await _append_transaction_entry(
        store,
        session_id,
        assistant,
        fencing_epoch=epoch,
    )
    assert isinstance(assistant_commit, TransactionCommit)
    intent_id = IntentId.new()
    digest = hashlib.sha256(b"resource").hexdigest()
    result = _tool_result(
        session_id,
        intent_id,
        parts=(
            ToolTextPart("found"),
            ToolResourceAttachmentPart(
                resource_id="resource-1",
                safe_name="report.txt",
                media_type="text/plain",
                content_digest=digest,
                size_bytes=8,
            ),
        ),
    )
    result_commit = await _append_transaction_entry(
        store,
        session_id,
        result,
        fencing_epoch=epoch,
        intent_id=intent_id,
        host_delta=EffectHostUpdate(),
    )
    assert isinstance(result_commit, TransactionCommit)

    run_store = await _store(pool)
    projected = await run_store.load_agent_transcript(
        owner_id=_OWNER,
        run_id=claimed.run.run_id,
        session_id=session_id.value,
        limit=20,
    )
    assert [message["role"] for message in projected] == ["user", "assistant", "tool"]
    assert projected[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "lookup",
        "content": "found",
        "attachments": [
            {
                "resource_id": "resource-1",
                "safe_name": "report.txt",
                "media_type": "text/plain",
                "content_digest": digest,
                "size_bytes": 8,
            }
        ],
        "is_error": False,
    }
    service = AnswerService(
        store=run_store,
        coordinator=cast(Any, None),
        retrieval=cast(Any, None),
        capabilities=cast(Any, None),
        capability_view=cast(Any, None),
        models=cast(Any, None),
        resources=cast(Any, None),
        model_fingerprint_for_role=cast(Any, None),
    )
    transcript = await service.transcript_tail(
        owner_id=_OWNER,
        run_id=claimed.run.run_id,
    )
    assert transcript is not None
    assert transcript.messages[-1] == projected[-1]


async def test_fast_compaction_recovers_between_projection_and_assistant(pool) -> None:
    session_id = SessionId.new()
    first = await _claim(pool, session_id=session_id)
    first_host = FastSessionHost(
        transactions=first.execution.session_repository,
        load=first.execution.session_repository.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=first.execution.fencing_epoch,
    )
    await first_host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=first.run.run_id,
        idempotency_key="first-compaction-turn",
        content="old question",
    )
    await first_host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=first.run.run_id,
        content="old answer",
    )
    run_store = await _store(pool)
    await run_store.finish_success(
        owner_id=_OWNER,
        run_id=first.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=first.execution.fencing_epoch,
        result={"answer": "old answer"},
    )

    second = await _claim(pool, session_id=session_id)
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    second_host = FastSessionHost(
        transactions=second.execution.session_repository,
        load=second.execution.session_repository.load,
        load_settled_result=load_settled_result,
        fencing_epoch=second.execution.fencing_epoch,
    )
    await second_host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=second.run.run_id,
        idempotency_key="second-compaction-turn",
        content="current question",
    )
    before = await second.execution.session_repository.load(session_id)
    ancestry = before.tree.ancestry()
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        covered_through_sequence=ancestry[1].sequence,
        first_retained_sequence=ancestry[2].sequence,
        summary='{"goal":"old turn"}',
        covered_through_entry_id=ancestry[1].entry_id,
        first_retained_entry_id=ancestry[2].entry_id,
        source_digest=projection_source_digest([entry.entry_id for entry in ancestry[:2]]),
    )
    await second_host.commit_compaction(
        snapshot=before,
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=second.run.run_id,
        projection=projection,
    )

    settled_result = {
        "answer": "recovered answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    replay = await second_host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=second.run.run_id,
        idempotency_key="second-compaction-turn",
        content="current question",
    )

    assert replay.settled_payload == settled_result
    recovered = await second.execution.session_repository.load(session_id)
    assert [entry.entry_type for entry in recovered.tree.ancestry()] == [
        "user_message",
        "assistant_message",
        "user_message",
        "compaction",
        "assistant_message",
    ]
    checkpoint = recovered.tree.ancestry()[-2]
    assistant = recovered.tree.ancestry()[-1]
    assert isinstance(checkpoint, CompactionEntry)
    assert isinstance(assistant, AssistantMessageEntry)
    assert assistant.parent_entry_id == checkpoint.entry_id
    assert recovered.active_projection == projection


async def test_product_session_spans_answer_runs_and_projects_selected_lane(pool) -> None:
    session_id = SessionId.new()
    first = await _claim(pool, session_id=session_id)
    first_host = FastSessionHost(
        transactions=first.execution.session_repository,
        load=first.execution.session_repository.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=first.execution.fencing_epoch,
    )
    await first_host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=first.run.run_id,
        idempotency_key="first",
        content="first question",
    )
    await first_host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=first.run.run_id,
        content="first answer",
    )
    run_store = await _store(pool)
    await run_store.finish_success(
        owner_id=_OWNER,
        run_id=first.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=first.execution.fencing_epoch,
        result={"answer": "first answer"},
    )

    second = await _claim(pool, session_id=session_id)
    second_host = FastSessionHost(
        transactions=second.execution.session_repository,
        load=second.execution.session_repository.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=second.execution.fencing_epoch,
    )
    await second_host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=second.run.run_id,
        idempotency_key="second",
        content="second question",
    )
    await second_host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id=second.run.run_id,
        content="second answer",
    )
    await run_store.finish_success(
        owner_id=_OWNER,
        run_id=second.run.run_id,
        worker_id=_WORKER,
        fencing_epoch=second.execution.fencing_epoch,
        result={"answer": "second answer"},
    )

    branch_id = LaneId.new()
    branch = await _claim(
        pool,
        session_id=session_id,
        lane_id=branch_id,
        source_lane_id=LaneId.main(),
    )
    await ensure_session_lane(
        transactions=branch.execution.session_repository,
        load=branch.execution.session_repository.load,
        fencing_epoch=branch.execution.fencing_epoch,
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=branch_id,
    )
    branch_host = FastSessionHost(
        transactions=branch.execution.session_repository,
        load=branch.execution.session_repository.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=branch.execution.fencing_epoch,
    )
    await branch_host.accept(
        session_id=session_id,
        lane_id=branch_id,
        reservation_id=branch.run.run_id,
        idempotency_key="branch",
        content="branch question",
    )
    await branch_host.complete(
        session_id=session_id,
        lane_id=branch_id,
        reservation_id=branch.run.run_id,
        content="branch answer",
    )
    snapshot = await branch.execution.session_repository.load(session_id)
    assert [
        entry.content
        for entry in snapshot.tree.ancestry(LaneId.main())
        if isinstance(entry, UserMessageEntry | AssistantMessageEntry)
    ] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
    assert [
        entry.content
        for entry in snapshot.tree.ancestry(branch_id)
        if isinstance(entry, UserMessageEntry | AssistantMessageEntry)
    ] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
        "branch question",
        "branch answer",
    ]
    transcript = await run_store.load_agent_transcript(
        owner_id=_OWNER,
        run_id=branch.run.run_id,
        session_id=session_id.value,
        limit=20,
    )
    assert [message["content"] for message in transcript] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
        "branch question",
        "branch answer",
    ]


async def test_lane_register_cas_does_not_conflict_across_branches(pool) -> None:
    claimed = await _claim(pool)
    store = claimed.execution.session_repository
    session_id = _claimed_session(claimed)
    await _drive(
        store,
        session_id=session_id,
        fencing_epoch=claimed.execution.fencing_epoch,
    )
    initial = await store.load(session_id)
    main = initial.tree.lane()
    branch_id = LaneId.new()
    await ensure_session_lane(
        transactions=store,
        load=store.load,
        fencing_epoch=claimed.execution.fencing_epoch,
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=branch_id,
    )
    branch = (await store.load(session_id)).tree.lane(branch_id)
    user_main = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="main future",
    )
    user_branch = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="branch future",
    )
    assert isinstance(
        await _append_lane_entry(
            store,
            session_id,
            LaneId.main(),
            main.head,
            user_main,
            fencing_epoch=claimed.execution.fencing_epoch,
        ),
        TransactionCommit,
    )
    assert isinstance(
        await _append_lane_entry(
            store,
            session_id,
            branch_id,
            branch.head,
            user_branch,
            fencing_epoch=claimed.execution.fencing_epoch,
        ),
        TransactionCommit,
    )


async def test_stale_epoch_writes_zero_rows(pool) -> None:
    claimed = await _claim(pool)
    stale = PGAgentSessionRepository(
        pool=pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(claimed.run.run_id),
        worker_id=_WORKER,
        lease_owner=_WORKER,
        fencing_epoch=claimed.execution.fencing_epoch + 99,
    )
    session_id = SessionId.new()
    from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.engine.agent.session.transactions import RegisterExpectation, SessionTransaction

    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    outcome = await stale.transact(
        session_id=session_id,
        fencing_epoch=claimed.execution.fencing_epoch + 99,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert outcome.__class__.__name__ == "TransactionLeaseLost"
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM dlightrag_agent_sessions") == 0
