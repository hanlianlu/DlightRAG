# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the durable Answer coordinator on PostgreSQL 18.

Exercises the coordinator against the real fenced repository: claim, durable
Session progress, process restart from committed Session state, gap-free event
replay across a reconnect, observed cancellation, graceful-shutdown requeue,
and Session Entry/register round trips through JSONB.

Every test runs inside a throwaway database created and dropped per test, so the
configured administrative database is never mutated. Connection settings come
from the shared integration-test PostgreSQL environment; skipped if unavailable.
"""

import asyncio
import base64
import datetime
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import asyncpg
import pytest

from dlightrag._compose import _compose
from dlightrag.adapters.postgres.answer import session_repository as pg_session_repository
from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.application import Application
from dlightrag.application.config import DlightragConfig, LaneRuntimeConfig, RuntimeConfig
from dlightrag.application.settings import answer_executor_settings, answer_resource_settings
from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.entries import ToolResultMessageEntry, UserMessageEntry
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.agent.session.fold import WorkingContextProjection as _RunWorking
from dlightrag.engine.agent.session.ids import EntryId, LaneId, SessionId, StageIntentId
from dlightrag.engine.agent.session.operation import OperationCancelled
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import (
    HostTurnReservation,
    LaneHead,
    LaneState,
    OperationStateRegister,
    SetRegister,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.citations.streaming import AnswerStream
from dlightrag.engine.answer.execution import (
    AnswerExecutor,
    AnswerResourceResolver,
    OrchestratorRun,
)
from dlightrag.engine.answer.execution import executor as answer_executor_module
from dlightrag.engine.answer.execution.input import AnswerRunInput, PinnedModelProfile
from dlightrag.engine.answer.fast import FastRunBoundaries
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.publication import ArtifactIssue, PublicationPlan
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools.subagents import SubagentHost
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.runtime.coordinator import (
    RunCoordinator,
    RunSession,
)
from dlightrag.engine.runtime.records import (
    RunExecutionOutcome,
    Succeeded,
    run_request_fingerprint,
)
from tests.conftest import FingerprintingRunStore
from tests.integration.pg_conn import PG_CONN_KWARGS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_OWNER = "owner-alpha"
_REQUEST: dict[str, Any] = {
    "query": "why",
    "workspaces": ["default"],
    "agent_session_id": "00000000-0000-7000-8000-000000000001",
    "agent_lane_id": "main",
}
_REQUEST_FINGERPRINT = run_request_fingerprint(_REQUEST)
_VISUAL_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-corpus-visual").decode("ascii")


def _episode() -> _RunWorking:
    return _RunWorking(retained_tail_tokens=20_000)


def _answer_run_input() -> AnswerRunInput:
    return AnswerRunInput(
        query="why",
        workspaces=("default",),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}-model", None),
                profile=ModelProfile(context_window_tokens=1_000_000),
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint="public-request-hash",
        agent_session_id="00000000-0000-7000-8000-000000000001",
        agent_lane_id="main",
    )


def _answer_run_request(
    *,
    mode: str = "fast",
    agent_run_plan: AgentRunPlan | None = None,
) -> dict[str, Any]:
    run_input = replace(_answer_run_input(), agent_run_plan=agent_run_plan)
    request = run_input.as_request()
    request["mode"] = mode
    return request


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def store() -> AsyncIterator[FingerprintingRunStore]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_runtime_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        assert pool is not None
        created = FingerprintingRunStore(pool=pool)
        await created.initialize()
        # Establish the complete operational schema exactly as a real process does.
        await PGWebConversationStore(pool=pool, run_store=created).initialize()
        yield created
    finally:
        if pool is not None:
            await pool.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


class _Executor:
    def __init__(self, body: Any) -> None:
        self._body = body

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        return await self._body(session)


async def _settle(predicate: Any, *, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


async def _wait_for_status(
    store: PGRunStore,
    *,
    owner_id: str,
    run_id: str,
    status: str,
    timeout: float = 10.0,
) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=owner_id, run_id=run_id)
        return run is not None and run.status == status

    await _settle(_check, timeout=timeout)
    run = await store.get_run(owner_id=owner_id, run_id=run_id)
    assert run is not None
    return run


def _status_is(store: PGRunStore, run_id: str, status: str) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=_OWNER, run_id=run_id)
        return run is not None and run.status == status

    return _check


class _AsyncChildProvider:
    """Controlled provider for the real parent/Child Agent composition."""

    def __init__(
        self,
        *,
        resumed_parent: bool = False,
        fail_children: bool = False,
        child_evidence: bool = False,
    ) -> None:
        self.resumed_parent = resumed_parent
        self.fail_children = fail_children
        self.child_evidence = child_evidence
        self.release_children = asyncio.Event()
        self.children_started = asyncio.Event()
        self.parent_progressed = asyncio.Event()
        self.parent_notification_started = asyncio.Event()
        self.parent_calls = 0
        self.child_calls = 0

    async def __call__(self, **kwargs: Any) -> AssistantTurn:
        tools = kwargs.get("tools") or ()
        names = {str(tool.name) for tool in tools}
        if "spawn_agent" not in names:
            self.child_calls += 1
            messages = kwargs.get("messages") or ()
            has_tool_result = any(
                isinstance(message, dict) and message.get("role") == "tool" for message in messages
            )
            if self.child_calls >= 2:
                self.children_started.set()
            if self.fail_children:
                raise RuntimeError("controlled child provider failure")
            if self.child_evidence and not has_tool_result:
                await self.release_children.wait()
                return AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id=f"child-search-{self.child_calls}",
                            name="search_knowledge_base",
                            arguments={"query": "durable child evidence"},
                        ),
                    ),
                    stop_reason="tool_use",
                    usage_details={"input_tokens": 2, "output_tokens": 1},
                )
            if not self.child_evidence:
                await self.release_children.wait()
            return AssistantTurn(
                text=f"child finding {self.child_calls}",
                tool_calls=(),
                stop_reason="stop",
                usage_details={"input_tokens": 2, "output_tokens": 1},
            )

        self.parent_calls += 1
        if not self.resumed_parent and self.parent_calls == 1:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="spawn-investigations",
                        name="spawn_agent",
                        arguments={
                            "children": [
                                {"objective": "investigation one"},
                                {"objective": "investigation two"},
                            ]
                        },
                    ),
                ),
                stop_reason="tool_use",
                usage_details={"input_tokens": 3, "output_tokens": 1},
            )
        if not self.resumed_parent and self.parent_calls == 2:
            self.parent_progressed.set()
            return AssistantTurn(
                text="parent completed useful independent analysis",
                tool_calls=(),
                stop_reason="stop",
                usage_details={"input_tokens": 4, "output_tokens": 2},
            )
        self.parent_notification_started.set()
        return AssistantTurn(
            text="parent synthesis after child settlement",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 5, "output_tokens": 2},
        )


class _CancelPendingChildProvider:
    def __init__(self) -> None:
        self.parent_calls = 0
        self.child_tool_started = asyncio.Event()
        self._block_tool = asyncio.Event()

    async def __call__(self, **kwargs: Any) -> AssistantTurn:
        tools = kwargs.get("tools") or ()
        names = {str(tool.name) for tool in tools}
        if "spawn_agent" not in names:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="pending-child-search",
                        name="search_knowledge_base",
                        arguments={"query": "pending child effect"},
                    ),
                ),
                stop_reason="tool_use",
            )

        self.parent_calls += 1
        if self.parent_calls == 1:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="spawn-pending-child",
                        name="spawn_agent",
                        arguments={"children": [{"objective": "pending child"}]},
                    ),
                ),
                stop_reason="tool_use",
            )
        if self.parent_calls == 2:
            await self.child_tool_started.wait()
            last_content = str(kwargs["messages"][-1].get("content", ""))
            child_id = next(
                token.strip(".,;:()[]{}")
                for token in last_content.split()
                if len(token.strip(".,;:()[]{}")) == 36
                and token.strip(".,;:()[]{}").count("-") == 4
            )
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="cancel-pending-child",
                        name="cancel_subagent",
                        arguments={"child_session_id": child_id},
                    ),
                ),
                stop_reason="tool_use",
            )
        return AssistantTurn(
            text="parent cancelled the active child coherently",
            tool_calls=(),
            stop_reason="stop",
        )

    async def retrieve(self, _query: str) -> RetrievalResult:
        self.child_tool_started.set()
        await self._block_tool.wait()
        raise AssertionError("blocked child retrieval should be cancelled")


def _async_child_orchestrator(
    provider: Any,
    *,
    retrieve: Any = None,
) -> AnswerOrchestrator:
    profile = ModelProfile(context_window_tokens=1_000_000)
    return AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=retrieve or _retrieve_visual,
        model_func=provider,
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="research",
        subagent_host=SubagentHost(),
    )


def _async_child_plan(orchestrator: AnswerOrchestrator) -> AgentRunPlan:
    return AgentRunPlan.from_tools(
        orchestrator.prepare_run("why").tools,
        model_role="query",
        context_policy_revision=CONTEXT_POLICY_REVISION,
    )


async def test_session_turn_survives_a_new_worker(store: FingerprintingRunStore) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    seen: list[int] = []

    from dlightrag.engine.agent.session.entries import AssistantMessageEntry
    from dlightrag.engine.agent.session.ids import EntryId, LaneId, SessionId
    from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.engine.agent.session.transactions import RegisterExpectation, SessionTransaction

    async def body(session: RunSession) -> RunExecutionOutcome:
        repository = session.execution.session_repository
        assert session.prepared_input is not None
        session_id = SessionId(str(session.prepared_input["agent_session_id"]))
        snapshot = await repository.load(session_id)
        seen.append(snapshot.commit_sequence)
        if snapshot.commit_sequence == 0:
            entry = AssistantMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=datetime.datetime.now(datetime.UTC),
                content="searched",
                stop_reason="stop",
            )
            head = LaneHead(LaneId.main(), entry.entry_id)
            state = LaneState(LaneId.main())
            committed = await repository.transact(
                session_id=session_id,
                fencing_epoch=session.execution.fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    entries=[entry],
                    register_writes=[SetRegister(head), SetRegister(state)],
                    expectations=[
                        RegisterExpectation(head.ref, None),
                        RegisterExpectation(state.ref, None),
                    ],
                ),
            )
            assert committed.__class__.__name__ == "TransactionCommit"
            await asyncio.sleep(30)
        return Succeeded({"answer": "second attempt", "turns": snapshot.commit_sequence})

    first = RunCoordinator(
        store=store, executors={"answer": _Executor(body)}, query_worker_concurrency=1
    )
    await first.start()
    await _settle(_session_committed(store, run_id))
    await first.aclose()

    second = RunCoordinator(
        store=store, executors={"answer": _Executor(body)}, query_worker_concurrency=1
    )
    await second.start()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        await second.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.durable_progress_version == 1
    assert run.result == {"answer": "second attempt", "turns": 1}
    assert seen == [0, 1]


def _session_committed(store: PGRunStore, run_id: str) -> Any:
    async def _check() -> bool:
        run = await store.get_run(owner_id=_OWNER, run_id=run_id)
        return run is not None and run.durable_progress_version == 1

    return _check


async def test_the_coordinator_applies_retention_without_an_execution_slot(
    store: FingerprintingRunStore,
) -> None:
    """Every run-owning process trims expired event logs and prunes expired runs."""
    expired = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    live_request = {**_REQUEST, "query": "recent"}
    live = await store.create_run(
        owner_id=_OWNER,
        request=live_request,
        idempotency_fingerprint=run_request_fingerprint(live_request),
    )
    for creation in (expired, live):
        claim = await store.claim_next(worker_id="retention-setup")
        assert claim is not None
        await store.finish_success(
            owner_id=_OWNER,
            run_id=claim.run.run_id,
            worker_id="retention-setup",
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": creation.run.run_id},
        )
    pool = store._operation_pool  # noqa: SLF001 - backdating is test-only setup
    assert pool is not None
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET finished_at = NOW() - INTERVAL '370 days', "
            "purge_after = NOW() - INTERVAL '5 days' WHERE run_id = $1",
            uuid.UUID(expired.run.run_id),
        )

    held = asyncio.Event()

    async def body(session: RunSession) -> RunExecutionOutcome:
        await held.wait()
        return Succeeded({"answer": "held"})

    coordinator = RunCoordinator(
        store=store,
        executors={"answer": _Executor(body)},
        query_worker_concurrency=1,
        maintenance_seconds=0.05,
    )
    await coordinator.start()
    try:

        async def _pruned() -> bool:
            return await store.get_run(owner_id=_OWNER, run_id=expired.run.run_id) is None

        await _settle(_pruned)
    finally:
        held.set()
        await coordinator.aclose()

    survivor = await store.get_run(owner_id=_OWNER, run_id=live.run.run_id)
    assert survivor is not None
    assert survivor.events_trimmed_at is None


async def test_graceful_shutdown_requeues_without_crash_recovery(
    store: FingerprintingRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    running = asyncio.Event()

    async def body(session: RunSession) -> RunExecutionOutcome:
        running.set()
        await asyncio.sleep(30)
        return Succeeded({"answer": "unreachable"})

    coordinator = RunCoordinator(
        store=store, executors={"answer": _Executor(body)}, query_worker_concurrency=1
    )
    await coordinator.start()
    await asyncio.wait_for(running.wait(), timeout=10)
    await coordinator.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.reclaims_without_progress == 0
    assert run.lease_owner is None


async def test_reconnecting_subscriber_replays_without_gaps_or_duplicates(
    store: FingerprintingRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id

    async def body(session: RunSession) -> RunExecutionOutcome:
        await session.enter_phase("generating")
        await session.emit_token("hello ")
        await session.flush_tokens()
        await session.emit_token("world")
        await session.flush_tokens()
        return Succeeded({"answer": "hello world"})

    coordinator = RunCoordinator(
        store=store, executors={"answer": _Executor(body)}, query_worker_concurrency=1
    )
    await coordinator.start()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
        first: list[int] = []
        async for event in coordinator.subscribe(owner_id=_OWNER, run_id=run_id):
            first.append(event.sequence)
            if len(first) == 2:
                break
        second = [
            event.sequence
            async for event in coordinator.subscribe(
                owner_id=_OWNER, run_id=run_id, after_sequence=first[-1]
            )
        ]
    finally:
        await coordinator.aclose()

    assert first == [1, 2]
    assert second == [3, 4]
    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    assert [event.event_type for event in events] == ["progress", "token", "token", "done"]


async def test_running_run_observes_cancellation_and_commits_cancelled(
    store: FingerprintingRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    started = asyncio.Event()

    async def body(session: RunSession) -> RunExecutionOutcome:
        started.set()
        for _ in range(2000):
            await session.check_cancelled()
            await asyncio.sleep(0.01)
        return Succeeded({"answer": "unreachable"})

    coordinator = RunCoordinator(
        store=store,
        executors={"answer": _Executor(body)},
        query_worker_concurrency=1,
        heartbeat_seconds=0.05,
    )
    await coordinator.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=run_id)
        assert outcome.outcome == "pending"
        await _settle(_status_is(store, run_id, "cancelled"))
    finally:
        await coordinator.aclose()

    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    assert events[-1].event_type == "done"
    assert events[-1].payload == {"status": "cancelled"}


async def test_accepted_cancellation_wins_coordinator_failure_settlement(
    store: FingerprintingRunStore,
) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    started = asyncio.Event()
    fail = asyncio.Event()

    async def body(session: RunSession) -> RunExecutionOutcome:
        started.set()
        await fail.wait()
        raise RuntimeError("provider internals must not win cancellation")

    coordinator = RunCoordinator(
        store=store,
        executors={"answer": _Executor(body)},
        query_worker_concurrency=1,
    )
    await coordinator.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        outcome = await store.request_cancellation(owner_id=_OWNER, run_id=run_id)
        assert outcome.outcome == "pending"
        fail.set()
        await _settle(_status_is(store, run_id, "cancelled"))
    finally:
        fail.set()
        await coordinator.aclose()

    record = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert record is not None
    assert record.error_kind is None
    assert record.error_message is None
    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    assert [(event.event_type, event.payload) for event in events] == [
        ("done", {"status": "cancelled"})
    ]


async def test_session_round_trips_through_jsonb(store: FingerprintingRunStore) -> None:
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_REQUEST,
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    repository = claimed.execution.session_repository

    from dlightrag.engine.agent.session.entries import AssistantMessageEntry, UserMessageEntry
    from dlightrag.engine.agent.session.fold import fold_entries
    from dlightrag.engine.agent.session.ids import EntryId, LaneId, SessionId
    from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.engine.agent.session.transactions import RegisterExpectation, SessionTransaction

    assert creation.run.prepared_input is not None
    session_id = SessionId(str(creation.run.prepared_input["agent_session_id"]))
    user_entry_id = EntryId.new()
    entries = [
        UserMessageEntry(
            entry_id=user_entry_id,
            session_id=session_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            content="question",
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            parent_entry_id=user_entry_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            content="searched",
            stop_reason="stop",
            provider_state={"native": True},
        ),
    ]
    head = LaneHead(LaneId.main(), entries[-1].entry_id)
    state = LaneState(LaneId.main())
    committed = await repository.transact(
        session_id=session_id,
        fencing_epoch=claimed.execution.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=entries,
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert committed.__class__.__name__ == "TransactionCommit"

    await store.release_for_shutdown(
        owner_id=_OWNER,
        run_id=run_id,
        worker_id="worker-1",
        fencing_epoch=claimed.run.fencing_epoch,
    )
    reclaimed = await store.claim_next(worker_id="worker-2")
    assert reclaimed is not None

    snapshot = await reclaimed.execution.session_repository.load(session_id)
    assert snapshot.commit_sequence == 1
    folded = fold_entries(snapshot.entries)
    assert [message["role"] for message in folded] == ["user", "assistant"]
    assert folded[1]["provider_state"] == {"native": True}


async def test_accepted_run_executes_and_stores_a_projected_result_without_a_subscriber(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A descriptor-only caller still gets a finished run and a safe canonical result."""
    repository_calls = {"load": 0, "refresh": 0}
    original_load = PGAgentSessionRepository.load
    original_refresh = PGAgentSessionRepository.refresh

    async def counted_load(repository: Any, session_id: SessionId) -> Any:
        repository_calls["load"] += 1
        return await original_load(repository, session_id)

    async def counted_refresh(
        repository: Any,
        session_id: SessionId,
        *,
        previous: Any,
    ) -> Any:
        repository_calls["refresh"] += 1
        return await original_refresh(repository, session_id, previous=previous)

    monkeypatch.setattr(PGAgentSessionRepository, "load", counted_load)
    monkeypatch.setattr(PGAgentSessionRepository, "refresh", counted_refresh)
    finish_success_calls = 0
    original_finish_success = store.finish_success

    async def tracked_finish_success(**kwargs: Any) -> Any:
        nonlocal finish_success_calls
        finish_success_calls += 1
        return await original_finish_success(**kwargs)

    store.finish_success = tracked_finish_success  # type: ignore[method-assign]
    application, coordinator = _answer_runtime(store)
    await coordinator.start()
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_answer_run_request(),
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    run_id = creation.run.run_id
    coordinator.wake()
    try:
        await _settle(_status_is(store, run_id, "succeeded"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=run_id)
    assert run is not None
    result = run.result
    assert result is not None
    session_snapshot = await store.load_routing(owner_id=_OWNER, run_id=run_id)
    assert session_snapshot is not None
    # One refresh establishes the Host-neutral boundary before routing; Fast
    # acceptance and completion retain their existing refreshes.
    assert repository_calls == {"load": 1, "refresh": 3}
    agent_snapshot = await store.claim_next(worker_id="unused")
    assert agent_snapshot is None
    session_id = SessionId(session_snapshot.agent_session_id)
    repository = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    canonical = await repository.load(session_id)
    assert [entry.entry_type for entry in canonical.tree.ancestry()] == [
        "user_message",
        "assistant_message",
    ]
    chunk = result["contexts"]["chunks"][0]
    assert "image_data" not in chunk
    assert "_evidence_key" not in chunk
    assert "source_uri" not in chunk["metadata"]
    assert "source_download_locator" not in chunk["metadata"]

    events = await store.read_event_page(owner_id=_OWNER, run_id=run_id)
    done = events[-1]
    assert done.event_type == "done"
    assert done.payload["status"] == "succeeded"
    assert done.payload["result"] == result

    serialized = json.dumps(dict(result))
    assert _VISUAL_B64 not in serialized
    assert "data:image" not in serialized
    assert "/srv/private/book.pdf" not in serialized
    assert result["sources"][0]["source_uri"] == "corpus://book.pdf"
    assert result["evidence_images"][0]["chunk_id"] == "c1"
    assert "answer_images" not in result
    assert result["trace"]["retrieval"] == "ok"
    assert finish_success_calls == 0


async def test_fast_post_stage_cancellation_replays_without_generation_or_lane_interleaving(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_calls = {"load": 0, "refresh": 0}
    original_load = PGAgentSessionRepository.load
    original_refresh = PGAgentSessionRepository.refresh

    async def counted_load(repository: Any, session_id: SessionId) -> Any:
        repository_calls["load"] += 1
        return await original_load(repository, session_id)

    async def counted_refresh(
        repository: Any,
        session_id: SessionId,
        *,
        previous: Any,
    ) -> Any:
        repository_calls["refresh"] += 1
        return await original_refresh(repository, session_id, previous=previous)

    monkeypatch.setattr(PGAgentSessionRepository, "load", counted_load)
    monkeypatch.setattr(PGAgentSessionRepository, "refresh", counted_refresh)
    finish_success_calls = 0
    finish_failure_calls = 0
    original_finish_success = store.finish_success
    original_finish_failure = store.finish_failure

    async def tracked_finish_success(**kwargs: Any) -> Any:
        nonlocal finish_success_calls
        finish_success_calls += 1
        return await original_finish_success(**kwargs)

    async def tracked_finish_failure(**kwargs: Any) -> Any:
        nonlocal finish_failure_calls
        finish_failure_calls += 1
        return await original_finish_failure(**kwargs)

    store.finish_success = tracked_finish_success  # type: ignore[method-assign]
    store.finish_failure = tracked_finish_failure  # type: ignore[method-assign]

    class CountingSynthesizer(_CitingSynthesizer):
        def __init__(self) -> None:
            self.calls = 0

        async def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return await super().generate_stream(*args, **kwargs)

    synthesizer = CountingSynthesizer()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, synthesizer),
        retrieve_knowledge_base=_retrieve_visual,
        model_profile=ModelProfile(context_window_tokens=1_000_000),
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="fast",
    )
    application, coordinator = _answer_runtime(store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id=_OWNER,
        request=_answer_run_request(),
        idempotency_fingerprint=_REQUEST_FINGERPRINT,
    )
    final_stage_id = StageIntentId.deterministic(
        run_id=creation.run.run_id,
        name="fast:final_generation:2",
    )
    original_observe = FastRunBoundaries._observe
    crashed = False

    async def cancel_at_stage_observation(
        self: FastRunBoundaries,
        committed: Any,
    ) -> Any:
        nonlocal crashed
        if not crashed and getattr(committed, "stage_intent_id", None) == final_stage_id:
            crashed = True
            raise asyncio.CancelledError
        return await original_observe(self, committed)

    monkeypatch.setattr(FastRunBoundaries, "_observe", cancel_at_stage_observation)
    await coordinator.start()
    coordinator.wake()
    try:
        await _settle(_status_is(store, creation.run.run_id, "succeeded"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
    assert run is not None
    assert run.result is not None
    assert crashed is True
    assert synthesizer.calls == 1
    assert finish_success_calls == 0
    assert finish_failure_calls == 0
    routing = await store.load_routing(owner_id=_OWNER, run_id=creation.run.run_id)
    assert routing is not None
    # Cancellation reclaims this run for two execution attempts. Each attempt
    # performs one bounded Host-neutral boundary refresh before routing.
    assert repository_calls == {"load": 2, "refresh": 5}
    reader = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(creation.run.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    snapshot = await reader.load(SessionId(routing.agent_session_id))
    ancestry = snapshot.tree.ancestry()
    assert [entry.entry_type for entry in ancestry] == [
        "user_message",
        "assistant_message",
    ]
    assert [getattr(entry, "acceptance_id", None) for entry in ancestry] == [
        creation.run.run_id,
        creation.run.run_id,
    ]
    assert ancestry[1].parent_entry_id == ancestry[0].entry_id
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)

    async with cast(Any, store)._operation_pool.acquire() as conn:
        state = await conn.fetchval(
            "SELECT state FROM dlightrag_answer_run_stages"
            " WHERE owner_id = $1 AND run_id = $2 AND stage_intent_id = $3",
            _OWNER,
            uuid.UUID(creation.run.run_id),
            uuid.UUID(final_stage_id.value),
        )
    staged = json.loads(state) if isinstance(state, str) else state
    assert staged["result"] == run.result
    assert staged["result_digest"] == canonical_json(run.result)
    assert finish_success_calls == 0


async def test_fast_failure_clears_reservation_and_keeps_unanswered_user(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_calls = {"load": 0, "refresh": 0}
    original_load = PGAgentSessionRepository.load
    original_refresh = PGAgentSessionRepository.refresh

    async def counted_load(repository: Any, session_id: SessionId) -> Any:
        repository_calls["load"] += 1
        return await original_load(repository, session_id)

    async def counted_refresh(
        repository: Any,
        session_id: SessionId,
        *,
        previous: Any,
    ) -> Any:
        repository_calls["refresh"] += 1
        return await original_refresh(repository, session_id, previous=previous)

    monkeypatch.setattr(PGAgentSessionRepository, "load", counted_load)
    monkeypatch.setattr(PGAgentSessionRepository, "refresh", counted_refresh)

    class FailingSynthesizer:
        async def generate_stream(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("generation failed")

    profile = ModelProfile(context_window_tokens=1_000_000)
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, FailingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="fast",
    )
    application, coordinator = _answer_runtime(store, orchestrator=orchestrator)
    await coordinator.start()
    request = _answer_run_request()
    creation = await store.create_run(
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )
    coordinator.wake()
    try:
        await _settle(_status_is(store, creation.run.run_id, "failed"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    routing = await store.load_routing(owner_id=_OWNER, run_id=creation.run.run_id)
    assert routing is not None
    # The routing boundary adds one refresh before Fast acceptance/failure.
    assert repository_calls == {"load": 1, "refresh": 3}
    reader = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(creation.run.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    snapshot = await reader.load(SessionId(routing.agent_session_id))
    assert [entry.entry_type for entry in snapshot.tree.ancestry()] == ["user_message"]
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)


async def test_async_children_run_while_parent_progresses_and_barrier_adopts_results(
    store: FingerprintingRunStore,
) -> None:
    provider = _AsyncChildProvider()
    orchestrator = _async_child_orchestrator(provider)
    plan = _async_child_plan(orchestrator)
    application, coordinator = _answer_runtime(store=store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id="owner-async-child",
        request=_answer_run_request(mode="research", agent_run_plan=plan),
    )
    try:
        await coordinator.start()
        coordinator.wake()
        await asyncio.wait_for(provider.children_started.wait(), timeout=10.0)
        await asyncio.wait_for(provider.parent_progressed.wait(), timeout=10.0)
        active = await store.get_run(owner_id="owner-async-child", run_id=creation.run.run_id)
        children = await store.list_child_sessions(
            owner_id="owner-async-child", run_id=creation.run.run_id
        )
        assert active is not None and active.status == "running"
        assert len(children) == 2
        assert {row["status"] for row in children} == {"running"}
        assert all(row["plan"]["model_identity"]["scope"] == "child" for row in children)
        assert all(row["budget"]["provider_attempt_limit"] == 2 for row in children)
        assert all("context_snapshot" in row for row in children)

        provider.release_children.set()
        run = await _wait_for_status(
            store,
            owner_id="owner-async-child",
            run_id=creation.run.run_id,
            status="succeeded",
            timeout=15.0,
        )
    finally:
        provider.release_children.set()
        await coordinator.aclose()
        await application.aclose()

    children = await store.list_child_sessions(
        owner_id="owner-async-child", run_id=creation.run.run_id
    )
    assert {row["status"] for row in children} == {"succeeded"}
    assert provider.parent_calls == 3
    assert run.result is not None
    assert run.result["answer"] == "parent synthesis after child settlement"
    assert [item["purpose"] for item in run.result["trace"]["agent_operations"]] == [
        "research",
        "child_result",
    ]
    assert run.result["trace"]["usage"]["child_usage_details"] == {
        "input_tokens": 4,
        "output_tokens": 2,
    }
    assert run.result["trace"]["usage"]["inclusive_usage_details"]["input_tokens"] >= 16


async def test_async_child_dispatch_reconstructs_after_parent_worker_detach(
    store: FingerprintingRunStore,
) -> None:
    first_provider = _AsyncChildProvider()
    first_orchestrator = _async_child_orchestrator(first_provider)
    plan = _async_child_plan(first_orchestrator)
    first_application, first = _answer_runtime(store=store, orchestrator=first_orchestrator)
    creation = await store.create_run(
        owner_id="owner-async-restart",
        request=_answer_run_request(mode="research", agent_run_plan=plan),
    )
    await first.start()
    first.wake()
    try:
        await asyncio.wait_for(first_provider.children_started.wait(), timeout=10.0)
        await asyncio.wait_for(first_provider.parent_progressed.wait(), timeout=10.0)
    finally:
        await first.aclose()
        await first_application.aclose()

    detached = await store.list_child_sessions(
        owner_id="owner-async-restart", run_id=creation.run.run_id
    )
    accepted_ids = {row["child_session_id"] for row in detached}
    assert len(accepted_ids) == 2
    assert {row["status"] for row in detached} == {"running"}
    assert {row["cancel_requested_at"] for row in detached} == {None}

    second_provider = _AsyncChildProvider(resumed_parent=True)
    second_provider.release_children.set()
    second_orchestrator = _async_child_orchestrator(second_provider)
    second_application, second = _answer_runtime(store=store, orchestrator=second_orchestrator)
    try:
        await second.start()
        second.wake()
        try:
            run = await _wait_for_status(
                store,
                owner_id="owner-async-restart",
                run_id=creation.run.run_id,
                status="succeeded",
                timeout=15.0,
            )
        except AssertionError as exc:
            current = await store.get_run(
                owner_id="owner-async-restart", run_id=creation.run.run_id
            )
            current_children = await store.list_child_sessions(
                owner_id="owner-async-restart", run_id=creation.run.run_id
            )
            raise AssertionError(
                (
                    current,
                    current_children,
                    second_provider.parent_calls,
                    second_provider.child_calls,
                )
            ) from exc
    finally:
        await second.aclose()
        await second_application.aclose()

    restored = await store.list_child_sessions(
        owner_id="owner-async-restart", run_id=creation.run.run_id
    )
    assert {row["child_session_id"] for row in restored} == accepted_ids
    assert {row["status"] for row in restored} == {"succeeded"}
    assert second_provider.child_calls == 2
    # The detach can land immediately before or after the parent's independent
    # second turn is durable; reclaim must correctly resume either boundary.
    assert second_provider.parent_calls in {1, 2}, (
        second_provider.child_calls,
        run.result,
    )
    assert run.result is not None
    assert run.result["answer"] == "parent synthesis after child settlement"


async def test_terminal_child_race_and_second_scan_survive_reclaim_exactly_once(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _AsyncChildProvider(child_evidence=True)
    orchestrator = _async_child_orchestrator(provider)
    plan = _async_child_plan(orchestrator)
    application, coordinator = _answer_runtime(store=store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id="owner-result-before-adoption",
        request=_answer_run_request(mode="research", agent_run_plan=plan),
    )
    host = orchestrator.subagent_host
    assert host is not None
    original_notifications = host.completed_dispatch_notifications
    original_finish = store.finish_child_session
    first_scan = asyncio.Event()
    terminalized = asyncio.Event()
    second_scan = asyncio.Event()
    allow_second_scan = asyncio.Event()
    terminal_count = 0
    forced_race = False
    second_scan_interrupted = False

    async def _counted_finish(**kwargs: Any) -> bool:
        nonlocal terminal_count
        finished = await original_finish(**kwargs)
        if finished and str(kwargs["status"]) != "running":
            terminal_count += 1
            if terminal_count == 2:
                terminalized.set()
        return finished

    async def _controlled_notifications(*, seen: set[str]) -> tuple[tuple[str, str], ...]:
        nonlocal forced_race, second_scan_interrupted
        notifications = await original_notifications(seen=seen)
        if not forced_race and not notifications:
            rows = await store.list_child_sessions(
                owner_id="owner-result-before-adoption",
                run_id=creation.run.run_id,
            )
            if len(rows) == 2 and any(row["status"] == "running" for row in rows):
                first_scan.set()
                provider.release_children.set()
                await terminalized.wait()
                forced_race = True
                return ()
        if forced_race and notifications and not second_scan_interrupted:
            second_scan.set()
            try:
                await allow_second_scan.wait()
            except asyncio.CancelledError:
                second_scan_interrupted = True
                raise
        return notifications

    monkeypatch.setattr(store, "finish_child_session", _counted_finish)
    monkeypatch.setattr(host, "completed_dispatch_notifications", _controlled_notifications)
    await coordinator.start()
    coordinator.wake()
    try:
        await asyncio.wait_for(provider.parent_progressed.wait(), timeout=10.0)
        await asyncio.wait_for(first_scan.wait(), timeout=10.0)
        await asyncio.wait_for(second_scan.wait(), timeout=10.0)
    finally:
        await coordinator.aclose()
        await application.aclose()

    assert provider.parent_calls == 2
    assert forced_race is True
    assert terminal_count == 2
    terminal = await store.list_child_sessions(
        owner_id="owner-result-before-adoption", run_id=creation.run.run_id
    )
    terminal_states: dict[str, list[str]] = {}
    if any(row["status"] != "succeeded" for row in terminal):
        diagnostic_reader = PGAgentSessionRepository(
            pool=cast(Any, store)._operation_pool,
            owner_id="owner-result-before-adoption",
            run_id=uuid.UUID(creation.run.run_id),
            worker_id="reader",
            lease_owner="reader",
            fencing_epoch=1,
        )
        for row in terminal:
            diagnostic_snapshot = await diagnostic_reader.load(SessionId(row["child_session_id"]))
            terminal_states[row["child_session_id"]] = [
                repr(record.value) for record in diagnostic_snapshot.registers
            ]
    assert {row["status"] for row in terminal} == {"succeeded"}, (
        [(row["status"], row["summary"], row["host_state"]) for row in terminal],
        terminal_states,
    )
    assert {tuple(sorted((row["usage"] or {}).items())) for row in terminal} == {
        (("input_tokens", 4), ("output_tokens", 2))
    }

    resumed_provider = _AsyncChildProvider(resumed_parent=True)
    resumed_orchestrator = _async_child_orchestrator(resumed_provider)
    resumed_application, resumed = _answer_runtime(store=store, orchestrator=resumed_orchestrator)
    try:
        await resumed.start()
        resumed.wake()
        run = await _wait_for_status(
            store,
            owner_id="owner-result-before-adoption",
            run_id=creation.run.run_id,
            status="succeeded",
            timeout=15.0,
        )
    finally:
        allow_second_scan.set()
        await resumed.aclose()
        await resumed_application.aclose()

    assert resumed_provider.child_calls == 0
    assert resumed_provider.parent_calls == 1
    assert run.result is not None
    assert run.result["answer"] == "parent synthesis after child settlement"
    assert [item["purpose"] for item in run.result["trace"]["agent_operations"]] == [
        "research",
        "child_result",
    ]
    assert len(run.result["contexts"]["chunks"]) == 1
    assert run.result["contexts"]["chunks"][0]["chunk_id"] == "c1"
    assert run.result["usage"]["child_usage_details"] == {
        "input_tokens": 8,
        "output_tokens": 4,
    }
    async with cast(Any, store)._operation_pool.acquire() as conn:
        evidence_rows = await conn.fetch(
            "SELECT session_id, intent_id, result_ordinal "
            "FROM dlightrag_answer_evidence WHERE owner_id = $1 AND run_id = $2",
            "owner-result-before-adoption",
            uuid.UUID(creation.run.run_id),
        )
    evidence_keys = {
        (str(row["session_id"]), str(row["intent_id"]), int(row["result_ordinal"]))
        for row in evidence_rows
    }
    assert len(evidence_rows) == 2
    assert len(evidence_keys) == 2


async def test_child_provider_exceptions_finalize_roster_and_notify_parent(
    store: FingerprintingRunStore,
) -> None:
    provider = _AsyncChildProvider(fail_children=True)
    orchestrator = _async_child_orchestrator(provider)
    application, coordinator = _answer_runtime(store=store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id="owner-child-exception",
        request=_answer_run_request(
            mode="research", agent_run_plan=_async_child_plan(orchestrator)
        ),
    )
    try:
        await coordinator.start()
        coordinator.wake()
        run = await _wait_for_status(
            store,
            owner_id="owner-child-exception",
            run_id=creation.run.run_id,
            status="succeeded",
            timeout=15.0,
        )
    finally:
        await coordinator.aclose()
        await application.aclose()

    children = await store.list_child_sessions(
        owner_id="owner-child-exception", run_id=creation.run.run_id
    )
    assert {row["status"] for row in children} == {"failed"}
    assert provider.parent_calls == 3
    assert run.result is not None
    assert run.result["answer"] == "parent synthesis after child settlement"


async def test_cancel_active_child_closes_pending_agent_effect_before_roster_terminal(
    store: FingerprintingRunStore,
) -> None:
    provider = _CancelPendingChildProvider()
    orchestrator = _async_child_orchestrator(provider, retrieve=provider.retrieve)
    application, coordinator = _answer_runtime(store=store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id="owner-cancel-pending-child",
        request=_answer_run_request(
            mode="research", agent_run_plan=_async_child_plan(orchestrator)
        ),
    )
    try:
        await coordinator.start()
        coordinator.wake()
        try:
            run = await _wait_for_status(
                store,
                owner_id="owner-cancel-pending-child",
                run_id=creation.run.run_id,
                status="succeeded",
                timeout=15.0,
            )
        except AssertionError as exc:
            current = await store.get_run(
                owner_id="owner-cancel-pending-child", run_id=creation.run.run_id
            )
            current_children = await store.list_child_sessions(
                owner_id="owner-cancel-pending-child", run_id=creation.run.run_id
            )
            raise AssertionError((current, current_children, provider.parent_calls)) from exc
    finally:
        await coordinator.aclose()
        await application.aclose()

    children = await store.list_child_sessions(
        owner_id="owner-cancel-pending-child", run_id=creation.run.run_id
    )
    assert len(children) == 1
    assert children[0]["status"] == "cancelled"
    assert children[0]["cancel_requested_at"] is not None
    assert provider.parent_calls >= 3
    assert run.result is not None
    assert run.result["answer"] == "parent cancelled the active child coherently"

    reader = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id="owner-cancel-pending-child",
        run_id=uuid.UUID(creation.run.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    child_snapshot = await reader.load(SessionId(children[0]["child_session_id"]))
    child_state = next(
        record.value.state
        for record in child_snapshot.registers
        if isinstance(record.value, OperationStateRegister)
    )
    assert isinstance(child_state, OperationCancelled)
    pending_result = next(
        entry for entry in child_snapshot.entries if isinstance(entry, ToolResultMessageEntry)
    )
    assert pending_result.result.call_id == "pending-child-search"
    assert pending_result.result.outcome == "outcome_unknown"


async def test_parent_cancellation_settles_active_children_before_terminal_run(
    store: FingerprintingRunStore,
) -> None:
    provider = _AsyncChildProvider()
    orchestrator = _async_child_orchestrator(provider)
    application, coordinator = _answer_runtime(store=store, orchestrator=orchestrator)
    creation = await store.create_run(
        owner_id="owner-parent-cancel",
        request=_answer_run_request(
            mode="research", agent_run_plan=_async_child_plan(orchestrator)
        ),
    )
    try:
        await coordinator.start()
        coordinator.wake()
        await asyncio.wait_for(provider.children_started.wait(), timeout=10.0)
        outcome = await store.request_cancellation(
            owner_id="owner-parent-cancel", run_id=creation.run.run_id
        )
        assert outcome.outcome == "pending"
        coordinator.cancel_local("owner-parent-cancel", creation.run.run_id)
        await _wait_for_status(
            store,
            owner_id="owner-parent-cancel",
            run_id=creation.run.run_id,
            status="cancelled",
            timeout=15.0,
        )
    finally:
        provider.release_children.set()
        await coordinator.aclose()
        await application.aclose()

    children = await store.list_child_sessions(
        owner_id="owner-parent-cancel", run_id=creation.run.run_id
    )
    assert len(children) == 2
    assert {row["status"] for row in children} == {"cancelled"}
    assert all(row["cancel_requested_at"] is not None for row in children)
    reader = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id="owner-parent-cancel",
        run_id=uuid.UUID(creation.run.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    for child in children:
        snapshot = await reader.load(SessionId(child["child_session_id"]))
        child_state = next(
            record.value.state
            for record in snapshot.registers
            if isinstance(record.value, OperationStateRegister)
        )
        assert isinstance(child_state, OperationCancelled)


async def test_publication_correction_is_one_linked_agent_operation(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_calls = {"load": 0, "refresh": 0}
    decoded_rows = 0
    original_load = PGAgentSessionRepository.load
    original_refresh = PGAgentSessionRepository.refresh
    original_decode = pg_session_repository._decode_entry

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal decoded_rows
        decoded_rows += 1
        return original_decode(*args, **kwargs)

    async def counted_load(
        repository: PGAgentSessionRepository,
        session_id: SessionId,
    ) -> Any:
        repository_calls["load"] += 1
        return await original_load(repository, session_id)

    async def counted_refresh(
        repository: PGAgentSessionRepository,
        session_id: SessionId,
        *,
        previous: Any,
    ) -> Any:
        repository_calls["refresh"] += 1
        return await original_refresh(repository, session_id, previous=previous)

    monkeypatch.setattr(PGAgentSessionRepository, "load", counted_load)
    monkeypatch.setattr(PGAgentSessionRepository, "refresh", counted_refresh)
    monkeypatch.setattr(pg_session_repository, "_decode_entry", counted_decode)
    turns = [
        AssistantTurn(
            text="Draft answer with broken Artifact.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 3, "output_tokens": 2},
        ),
        AssistantTurn(
            text="Answer after queued follow-up.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 4, "output_tokens": 2},
        ),
        AssistantTurn(
            text="Answer after terminal-race steer.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 5, "output_tokens": 2},
        ),
        AssistantTurn(
            text="Corrected answer.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 6, "output_tokens": 2},
        ),
    ]

    async def model(**_kwargs: Any) -> AssistantTurn:
        return turns.pop(0)

    profile = ModelProfile(context_window_tokens=1_000_000)
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        model_func=model,
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="research",
    )
    probe = orchestrator.prepare_run("why")
    plan = AgentRunPlan.from_tools(
        probe.tools,
        model_role="query",
        context_policy_revision=CONTEXT_POLICY_REVISION,
    )
    publication_calls = 0

    def publication_plan(
        _root: Any,
        *,
        answer: str,
        attachments: Any,
        limits: Any,
    ) -> PublicationPlan:
        nonlocal publication_calls
        del attachments, limits
        publication_calls += 1
        if publication_calls == 1:
            return PublicationPlan(
                answer=answer,
                issues=(
                    ArtifactIssue(
                        kind="missing_file",
                        description="report is missing",
                    ),
                ),
            )
        return PublicationPlan(
            answer=answer,
            issues=(
                ArtifactIssue(
                    kind="missing_file",
                    description="report is still missing after correction",
                ),
            ),
        )

    monkeypatch.setattr(
        "dlightrag.engine.answer.execution.executor._publication_plan", publication_plan
    )
    control_polls = 0
    acknowledged_controls: set[int] = set()

    async def pending_controls(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        nonlocal control_polls
        control_polls += 1
        if control_polls == 1 and 1 not in acknowledged_controls:
            return (
                {
                    "control_sequence": 1,
                    "kind": "follow_up",
                    "content": "queued follow-up",
                },
            )
        if control_polls == 6 and 2 not in acknowledged_controls:
            return (
                {
                    "control_sequence": 2,
                    "kind": "steer",
                    "content": "late steer becomes follow-up",
                },
            )
        return ()

    async def acknowledge_controls(**kwargs: Any) -> bool:
        acknowledged_controls.update(int(item) for item in kwargs["control_sequences"])
        return True

    store.load_pending_agent_controls = pending_controls  # type: ignore[method-assign]
    store.acknowledge_agent_controls = acknowledge_controls  # type: ignore[method-assign]
    application, coordinator = _answer_runtime(store, orchestrator=orchestrator)
    request = _answer_run_request(mode="research", agent_run_plan=plan)
    creation = await store.create_run(
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )
    claimed = await store.claim_next(worker_id="history-seed")
    assert claimed is not None
    session_id = SessionId(_answer_run_input().agent_session_id)
    history: list[UserMessageEntry] = []
    parent_id: EntryId | None = None
    for index in range(1000):
        entry = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.datetime.now(datetime.UTC),
            parent_entry_id=parent_id,
            content=f"historical question {index}",
        )
        history.append(entry)
        parent_id = entry.entry_id
    assert parent_id is not None
    head = LaneHead(LaneId.main(), parent_id)
    state = LaneState(LaneId.main())
    seeded = await claimed.execution.session_repository.transact(
        session_id=session_id,
        fencing_epoch=claimed.execution.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            entries=history,
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(seeded, TransactionCommit)
    await store.release_for_shutdown(
        owner_id=_OWNER,
        run_id=creation.run.run_id,
        worker_id="history-seed",
        fencing_epoch=claimed.run.fencing_epoch,
    )
    await coordinator.start()
    coordinator.wake()
    try:
        await _settle(_status_is(store, creation.run.run_id, "succeeded"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
    assert run is not None and run.result is not None
    assert run.result["answer"] == "Corrected answer."
    operations = run.result["trace"]["agent_operations"]
    assert [item["purpose"] for item in operations] == [
        "research",
        "follow_up",
        "follow_up",
        "publication_correction",
    ]
    assert len({item["operation_id"] for item in operations}) == 4
    assert [item["usage"] for item in operations] == [
        {"input_tokens": 3, "output_tokens": 2},
        {"input_tokens": 4, "output_tokens": 2},
        {"input_tokens": 5, "output_tokens": 2},
        {"input_tokens": 6, "output_tokens": 2},
    ]
    # The Host-neutral pre-routing boundary contributes one bounded refresh;
    # linked Research operations keep their previous refresh discipline.
    assert repository_calls == {"load": 1, "refresh": 29}
    assert decoded_rows == 1000
    # Detached-parent baseline: load=4, decoded_rows=4,000, refresh=26.
    assert publication_calls == 2
    residual_outcome = {
        "status": "failed",
        "issues": [
            {
                "kind": "missing_file",
                "description": "report is still missing after correction",
            }
        ],
    }
    assert run.result["artifact_outcome"] == residual_outcome
    assert operations[-1]["publication_outcome"] == residual_outcome
    assert run.result["trace"]["usage"]["usage_details"] == {
        "input_tokens": 18,
        "output_tokens": 8,
    }
    assert acknowledged_controls == {1, 2}
    events = await store.read_event_page(owner_id=_OWNER, run_id=creation.run.run_id)
    assert sum(event.event_type == "reset" for event in events) == 1
    routing = await store.load_routing(owner_id=_OWNER, run_id=creation.run.run_id)
    assert routing is not None
    reader = PGAgentSessionRepository(
        pool=cast(Any, store)._operation_pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(creation.run.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    snapshot = await reader.load(SessionId(routing.agent_session_id))
    assert [entry.entry_type for entry in snapshot.tree.ancestry()[-8:]] == [
        "user_message",
        "assistant_message",
        "user_message",
        "assistant_message",
        "user_message",
        "assistant_message",
        "user_message",
        "assistant_message",
    ]
    users = [entry for entry in snapshot.tree.ancestry() if isinstance(entry, UserMessageEntry)]
    assert users[-3].content == "queued follow-up"
    assert users[-2].content == "late steer becomes follow-up"


async def test_research_empty_canonical_uses_concurrently_advanced_refresh_for_restore(
    store: FingerprintingRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = PGAgentSessionRepository.load
    original_refresh = PGAgentSessionRepository.refresh
    original_transact = PGAgentSessionRepository.transact
    original_decode = pg_session_repository._decode_entry
    original_restore = answer_executor_module._restore_durable_evidence
    counters = {"load": 0, "refresh": 0, "decoded_rows": 0}
    injected = False
    restored: list[SessionId] = []

    async def counted_load(
        repository: PGAgentSessionRepository,
        session_id: SessionId,
    ) -> Any:
        counters["load"] += 1
        return await original_load(repository, session_id)

    async def counted_refresh(
        repository: PGAgentSessionRepository,
        session_id: SessionId,
        *,
        previous: Any,
    ) -> Any:
        counters["refresh"] += 1
        return await original_refresh(repository, session_id, previous=previous)

    async def concurrent_transact(
        repository: PGAgentSessionRepository,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[Any],
    ) -> Any:
        nonlocal injected
        if not injected:
            injected = True
            entry = UserMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=datetime.datetime.now(datetime.UTC),
                content="concurrently committed history",
            )
            head = LaneHead(LaneId.main(), entry.entry_id)
            state = LaneState(LaneId.main())
            committed = await original_transact(
                repository,
                session_id=session_id,
                fencing_epoch=fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    entries=[entry],
                    register_writes=[SetRegister(head), SetRegister(state)],
                    expectations=[
                        RegisterExpectation(head.ref, None),
                        RegisterExpectation(state.ref, None),
                    ],
                ),
            )
            assert isinstance(committed, TransactionCommit)
        return await original_transact(
            repository,
            session_id=session_id,
            fencing_epoch=fencing_epoch,
            transaction=transaction,
        )

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        counters["decoded_rows"] += 1
        return original_decode(*args, **kwargs)

    async def tracked_restore(prepared: Any, repository: Any, session_id: SessionId) -> None:
        restored.append(session_id)
        await original_restore(prepared, repository, session_id)

    monkeypatch.setattr(PGAgentSessionRepository, "load", counted_load)
    monkeypatch.setattr(PGAgentSessionRepository, "refresh", counted_refresh)
    monkeypatch.setattr(PGAgentSessionRepository, "transact", concurrent_transact)
    monkeypatch.setattr(pg_session_repository, "_decode_entry", counted_decode)
    monkeypatch.setattr(answer_executor_module, "_restore_durable_evidence", tracked_restore)

    async def model(**_kwargs: Any) -> AssistantTurn:
        return AssistantTurn(
            text="answer after concurrent history",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 2, "output_tokens": 1},
        )

    profile = ModelProfile(context_window_tokens=1_000_000)
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        model_func=model,
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="research",
    )
    plan = AgentRunPlan.from_tools(
        orchestrator.prepare_run("why").tools,
        model_role="query",
        context_policy_revision=CONTEXT_POLICY_REVISION,
    )
    application, coordinator = _answer_runtime(store, orchestrator=orchestrator)
    await coordinator.start()
    request = _answer_run_request(mode="research", agent_run_plan=plan)
    creation = await store.create_run(
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=run_request_fingerprint(request),
    )
    coordinator.wake()
    try:
        await _settle(_status_is(store, creation.run.run_id, "succeeded"))
    finally:
        await coordinator.aclose()
        await application.aclose()

    run = await store.get_run(owner_id=_OWNER, run_id=creation.run.run_id)
    assert run is not None and run.result is not None
    assert injected is True
    assert restored == [SessionId(_answer_run_input().agent_session_id)]
    assert counters["load"] == 1
    assert counters["decoded_rows"] == 1
    assert counters["refresh"] > 1


def _answer_runtime(
    store: FingerprintingRunStore,
    *,
    orchestrator: AnswerOrchestrator | None = None,
) -> tuple[Application, RunCoordinator]:
    """Compose the final executor and coordinator over the throwaway database."""
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        runtime=RuntimeConfig(
            query=LaneRuntimeConfig(
                worker_concurrency=1,
                max_nonterminal_runs=30_000,
            )
        ),
        answer={"agent": {"execution_environment": "disabled"}},
    )
    components = _compose(config)
    application = Application(config, components)
    orchestrator = orchestrator or AnswerOrchestrator(
        synthesizer=cast(AnswerSynthesizer, _CitingSynthesizer()),
        retrieve_knowledge_base=_retrieve_visual,
        model_profile=ModelProfile(context_window_tokens=1_000_000),
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="fast",
    )

    executor = AnswerExecutor(
        store=store,
        blob_store=PGRunBlobStore(),
        pool=components.pool,
        warm=components.retrieval.warm,
        retrieve=components.retrieval.retrieve_result,
        planner_history_input_measure=components.retrieval.planner_history_input_measure,
        models=components.models,
        capabilities=components.capabilities,
        resources=AnswerResourceResolver(
            settings=answer_resource_settings(config),
            models=components.models,
            capabilities=components.capabilities,
        ),
        settings=answer_executor_settings(config),
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=lambda role: ModelFingerprint(
            "openai", f"test-{role}-model", None
        ),
        execution_environment=config.answer.agent.execution_environment,
    )

    async def _prepare(**kwargs: Any) -> OrchestratorRun:
        return OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            fast_history_targets=(),
            current_image_count=0,
            workspaces=["default"],
            registry=None,
        )

    executor.prepare_orchestrated_run = _prepare  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        store=store,
        executors={"answer": executor},
        query_worker_concurrency=1,
    )
    return application, coordinator


class _CitingSynthesizer:
    async def generate_stream(
        self,
        query: str,
        contexts: Any,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
        current_images: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, AsyncIterator[str]]:
        del current_images, memory_text

        async def _stream() -> AsyncIterator[str]:
            yield "the drawing shows it [1]"

        return contexts, AnswerStream(_stream())


async def _retrieve_visual(query: str) -> RetrievalResult:
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "evidence",
                    "reference_id": "1",
                    "file_path": "book.pdf",
                    "_workspace": "default",
                    "_evidence_key": "search_knowledge_base:c1",
                    "image_data": _VISUAL_B64,
                    "metadata": {
                        "source_type": "corpus",
                        "source_uri": "corpus://book.pdf",
                        "source_download_locator": "/srv/private/book.pdf",
                        "title": "book.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieval": "ok"},
    )
