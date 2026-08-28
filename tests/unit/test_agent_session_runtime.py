# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Interface-level live, recovery, control, and crash tests for the Runtime."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    ControlMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.ids import AttemptId, EntryId, LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import (
    Cancelling,
    CompletionReady,
    OperationCancelled,
    OperationCompleted,
    OperationFailed,
    ToolEffectPending,
)
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import (
    LaneHead,
    LaneState,
    OperationStateRegister,
    RegisterRef,
    RequestSnapshot,
    SetRegister,
)
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.agent.session.runtime import (
    AgentSessionEvent,
    AgentSessionRuntime,
    CompactionRequired,
    OperationConflictError,
    ProviderAttemptFailed,
    RuntimeContext,
    SessionLeaseLostError,
    SteerCommand,
    ToolEffectResult,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionLeaseLost,
)
from dlightrag.engine.agent.tools import AgentTool, ToolResult
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall


class _Args(BaseModel):
    value: str


async def _tool(_args: BaseModel, _runtime: Any) -> ToolResult:
    return ToolResult.text("unused")


def _agent_tool(*, replayable: bool = False) -> AgentTool:
    return AgentTool(
        name="lookup",
        description="lookup",
        input_model=_Args,
        execute=_tool,
        replay_policy="replayable" if replayable else "never",
    )


def _plan(tool: AgentTool) -> AgentRunPlan:
    return AgentRunPlan.from_tools(
        [tool],
        model_role="query",
        context_policy_revision="context-v1",
    )


@dataclass
class _Effects:
    provider_turns: list[AssistantTurn]
    crash_provider: bool = False
    crash_tool: bool = False
    compaction_required: bool = False

    def __post_init__(self) -> None:
        self.provider_attempts: list[AttemptId] = []
        self.tool_attempts: list[AttemptId] = []
        self.executed_sources: list[int] = []
        self._compacted = False

    async def assemble_request(
        self, context: RuntimeContext
    ) -> RequestSnapshot | CompactionRequired:
        if self.compaction_required and not self._compacted:
            return CompactionRequired()
        return RequestSnapshot.from_values(
            operation_id=context.operation_id,
            turn_number=getattr(context.state, "turn_count", 0) + 1,
            plan_digest=context.meta.plan_digest,
            model_role="query",
            messages=[{"role": "user", "content": "exact"}],
            tools=[],
            tool_choice="auto",
            max_tokens=256,
        )

    async def call_provider(
        self,
        context: RuntimeContext,
        request: RequestSnapshot,
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> AssistantTurn:
        del context
        self.provider_attempts.append(attempt_id)
        await emit_ephemeral(
            AgentSessionEvent(
                kind="provider_delta",
                session_id=SessionId.new(),
                lane_id=LaneId.main(),
                operation_id=request.operation_id,
                commit_sequence=None,
                data={"text": "partial"},
                ephemeral=True,
            )
        )
        if self.crash_provider:
            raise __import__("asyncio").CancelledError
        return self.provider_turns.pop(0)

    async def execute_tool(
        self,
        context: RuntimeContext,
        item: Any,
        arguments: Mapping[str, Any],
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> ToolEffectResult[dict[str, Any]]:
        del context, emit_ephemeral
        self.tool_attempts.append(attempt_id)
        self.executed_sources.append(item.source_index)
        if self.crash_tool:
            raise __import__("asyncio").CancelledError
        return ToolEffectResult(
            result=ToolResultEntry.text(
                tool_name=item.tool_name,
                call_id=item.call_id,
                outcome="succeeded",
                text=f"result:{arguments['value']}",
            ),
            host_delta={"source_index": item.source_index},
        )

    async def compact(self, context: RuntimeContext, attempt: int) -> Any:
        from dlightrag.engine.agent.session.entries import CompactionEntry
        from dlightrag.engine.agent.session.ids import EntryId, ProjectionId
        from dlightrag.engine.agent.session.projection import ContextProjection

        del attempt
        self._compacted = True
        projection = ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=1,
            covered_through_sequence=0,
            summary=None,
        )
        return __import__(
            "dlightrag.engine.agent.session.runtime", fromlist=["CompactionResult"]
        ).CompactionResult(
            entry=CompactionEntry(
                entry_id=EntryId.new(),
                session_id=context.session_id,
                timestamp=__import__("datetime").datetime.now(__import__("datetime").UTC),
                projection_id=projection.projection_id,
                summary=None,
                covered_through_sequence=0,
                first_retained_sequence=1,
            ),
            projection=projection,
        )


def _assistant(*calls: ToolCall, text: str = "", stop: str | None = None) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=tuple(calls),
        stop_reason=stop or ("tool_use" if calls else "stop"),  # type: ignore[arg-type]
    )


def _runtime(
    store: MemoryAgentSessionRepository[dict[str, Any]],
    effects: _Effects,
    tool: AgentTool,
    *,
    events: list[AgentSessionEvent] | None = None,
    controls: Any = None,
) -> AgentSessionRuntime[dict[str, Any]]:
    async def collect(event: AgentSessionEvent) -> None:
        if events is not None:
            events.append(event)

    return AgentSessionRuntime(
        repository=store,
        effects=effects,
        tools=[tool],
        fencing_epoch=1,
        event_sink=collect,
        controls=controls,
    )


@pytest.mark.asyncio
async def test_live_runtime_commits_exact_request_ordered_batch_host_delta_and_terminal() -> None:
    tool = _agent_tool()
    effects = _Effects(
        [
            _assistant(
                ToolCall("bad", "missing", {}),
                ToolCall("ok", "lookup", {"value": "x"}),
            ),
            _assistant(text="done"),
        ]
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    events: list[AgentSessionEvent] = []
    runtime = _runtime(store, effects, tool, events=events)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="run-1",
        content="question",
        plan=_plan(tool),
    )

    final = await runtime.drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )

    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    assert [type(entry) for entry in snapshot.tree.ancestry()] == [
        UserMessageEntry,
        AssistantMessageEntry,
        ToolResultMessageEntry,
        ToolResultMessageEntry,
        AssistantMessageEntry,
    ]
    results = [entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry)]
    assert [entry.source_index for entry in results] == [0, 1]
    assert [entry.result.outcome for entry in results] == ["unknown_tool", "succeeded"]
    assert effects.executed_sources == [1]
    assert len(store.applied_host_deltas(session_id)) == 1
    assert all(event.commit_sequence is not None or event.ephemeral for event in events)
    assert not any(record.ref.kind == "request_snapshot" for record in snapshot.registers)
    assert not any(record.ref.kind == "tool_arguments" for record in snapshot.registers)


@pytest.mark.asyncio
async def test_never_tool_crash_recovers_as_outcome_unknown_without_reexecution() -> None:
    tool = _agent_tool(replayable=False)
    first_effects = _Effects(
        [_assistant(ToolCall("c1", "lookup", {"value": "x"}))],
        crash_tool=True,
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, first_effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="never-crash",
        content="question",
        plan=_plan(tool),
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    crashed = await runtime.restore(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(crashed.state, ToolEffectPending)

    recovered_effects = _Effects([_assistant(text="done")])
    final = await _runtime(store, recovered_effects, tool).drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    assert recovered_effects.executed_sources == []
    results = [
        entry
        for entry in (await store.load(session_id)).entries
        if isinstance(entry, ToolResultMessageEntry)
    ]
    assert results[0].result.outcome == "outcome_unknown"


@pytest.mark.asyncio
async def test_replayable_tool_recovery_uses_fresh_attempt_and_settles_once() -> None:
    tool = _agent_tool(replayable=True)
    first_effects = _Effects(
        [_assistant(ToolCall("c1", "lookup", {"value": "x"}))],
        crash_tool=True,
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, first_effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="replay-crash",
        content="question",
        plan=_plan(tool),
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    recovered_effects = _Effects([_assistant(text="done")])
    final = await _runtime(store, recovered_effects, tool).drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    assert first_effects.tool_attempts[0] != recovered_effects.tool_attempts[0]
    assert recovered_effects.executed_sources == [0]
    assert len(store.applied_host_deltas(session_id)) == 1


@pytest.mark.asyncio
async def test_provider_crash_reuses_exact_snapshot_with_fresh_attempt() -> None:
    tool = _agent_tool()
    first_effects = _Effects([_assistant(text="unused")], crash_provider=True)
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, first_effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="provider-crash",
        content="question",
        plan=_plan(tool),
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    snapshot = await store.load(session_id)
    request = next(
        record.value for record in snapshot.registers if isinstance(record.value, RequestSnapshot)
    )

    recovered_effects = _Effects([_assistant(text="done")])
    final = await _runtime(store, recovered_effects, tool).drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    assert first_effects.provider_attempts[0] != recovered_effects.provider_attempts[0]
    assert request.messages == [{"content": "exact", "role": "user"}]


@pytest.mark.asyncio
async def test_steer_is_consumed_at_terminal_checkpoint_and_cancel_closes_batch() -> None:
    tool = _agent_tool()
    effects = _Effects([_assistant(text="first"), _assistant(text="after steer")])
    store = MemoryAgentSessionRepository[dict[str, Any]]()

    class Controls:
        issued = False
        acknowledged: tuple[str, ...] = ()

        async def poll(self, context: RuntimeContext):
            if isinstance(context.state, CompletionReady) and not self.issued:
                self.issued = True
                return (SteerCommand("s1", "correct this"),)
            return ()

        async def acknowledge(self, command_ids: tuple[str, ...]) -> bool:
            self.acknowledged = command_ids
            return True

    controls = Controls()
    runtime = _runtime(store, effects, tool, controls=controls)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="steer",
        content="question",
        plan=_plan(tool),
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    assert any(
        isinstance(entry, ControlMessageEntry) for entry in (await store.load(session_id)).entries
    )
    assert controls.acknowledged == ("s1",)

    cancel_effects = _Effects(
        [_assistant(ToolCall("c1", "lookup", {"value": "x"}))],
        crash_tool=True,
    )
    second = await _runtime(store, cancel_effects, tool).accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="cancel",
        content="second",
        plan=_plan(tool),
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await _runtime(store, cancel_effects, tool).drive(
            session_id=session_id,
            operation_id=second.operation_id,
        )
    cancel_runtime = _runtime(store, _Effects([]), tool)
    await cancel_runtime.cancel(session_id=session_id, operation_id=second.operation_id)
    cancelled = await cancel_runtime.close(
        session_id=session_id,
        operation_id=second.operation_id,
    )
    assert isinstance(cancelled.state, OperationCancelled)


@pytest.mark.asyncio
async def test_follow_up_is_bounded_unaccepted_fifo_and_compaction_is_automatic() -> None:
    tool = _agent_tool()
    effects = _Effects([_assistant(text="done")], compaction_required=True)
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="compact",
        content="question",
        plan=_plan(tool),
    )
    await runtime.follow_up(
        session_id=session_id,
        lane_id=LaneId.main(),
        input_id="f1",
        idempotency_key="next",
        content="later",
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    pending = next(
        record.value for record in snapshot.registers if record.ref.kind == "pending_input"
    )
    assert pending.items[0].content == "later"  # type: ignore[union-attr]
    assert any(entry.entry_type == "compaction" for entry in snapshot.entries)
    assert snapshot.active_projection is not None
    assert any(record.ref.kind == "context_projection" for record in snapshot.registers)

    accepted_follow_up = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="next",
        content="later",
        plan=_plan(tool),
    )
    assert accepted_follow_up.created
    accepted_snapshot = await store.load(session_id)
    assert not any(record.ref.kind == "pending_input" for record in accepted_snapshot.registers)


@pytest.mark.asyncio
async def test_runtime_event_sink_failure_is_observe_only() -> None:
    tool = _agent_tool()
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    effects = _Effects([_assistant(text="done")])

    async def broken(_event: AgentSessionEvent) -> None:
        raise RuntimeError("telemetry unavailable")

    runtime = AgentSessionRuntime(
        repository=store,
        effects=effects,
        tools=[tool],
        fencing_epoch=1,
        event_sink=broken,
    )
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="broken-events",
        content="question",
        plan=_plan(tool),
    )
    final = await runtime.drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)


@pytest.mark.asyncio
async def test_runtime_invariant_faults_session_and_rejects_new_work() -> None:
    from dlightrag.engine.agent.session.operation import OperationFailed
    from dlightrag.engine.agent.session.runtime import AgentSessionRuntimeError

    tool = _agent_tool()

    class BrokenEffects(_Effects):
        async def assemble_request(self, context: RuntimeContext) -> RequestSnapshot:
            return RequestSnapshot.from_values(
                operation_id=context.operation_id,
                turn_number=1,
                plan_digest="f" * 64,
                model_role="query",
                messages=[],
                tools=[],
                tool_choice="auto",
                max_tokens=10,
            )

    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, BrokenEffects([]), tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="fault",
        content="question",
        plan=_plan(tool),
    )
    final = await runtime.drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationFailed)
    assert final.state.kind == "runtime_fault"
    assert any(
        record.ref.kind == "session_fault" for record in (await store.load(session_id)).registers
    )
    with pytest.raises(AgentSessionRuntimeError, match="faulted"):
        await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="after-fault",
            content="new",
            plan=_plan(tool),
        )


@pytest.mark.asyncio
async def test_recovery_contract_change_is_source_position_synthetic_result() -> None:
    original = _agent_tool(replayable=True)
    first_effects = _Effects(
        [_assistant(ToolCall("c1", "lookup", {"value": "x"}))],
        crash_tool=True,
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    first = _runtime(store, first_effects, original)
    session_id = SessionId.new()
    accepted = await first.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="contract-change",
        content="question",
        plan=_plan(original),
    )
    with pytest.raises(__import__("asyncio").CancelledError):
        await first.drive(session_id=session_id, operation_id=accepted.operation_id)

    changed = AgentTool(
        name="lookup",
        description="lookup",
        input_model=_Args,
        execute=_tool,
        replay_policy="replayable",
        contract_version=original.contract_version + 1,
    )
    recovered_effects = _Effects([_assistant(text="done")])
    final = await _runtime(store, recovered_effects, changed).drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    result = next(
        entry
        for entry in (await store.load(session_id)).entries
        if isinstance(entry, ToolResultMessageEntry)
    )
    assert result.result.outcome == "tool_contract_changed"
    assert recovered_effects.executed_sources == []


@pytest.mark.asyncio
async def test_steer_while_cancelling_is_rejected_without_faulting_session() -> None:
    tool = _agent_tool()
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, _Effects([]), tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="cancel-steer",
        content="question",
        plan=_plan(tool),
    )
    await runtime.cancel(session_id=session_id, operation_id=accepted.operation_id)
    cancelling = await runtime.restore(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(cancelling.state, Cancelling)
    with pytest.raises(OperationConflictError, match="cancelling"):
        await runtime.steer(
            session_id=session_id,
            operation_id=accepted.operation_id,
            control_id="late-steer",
            content="change direction",
        )
    final = await runtime.close(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCancelled)
    assert not any(
        record.ref.kind == "session_fault" for record in (await store.load(session_id)).registers
    )


@pytest.mark.asyncio
async def test_compaction_retries_exactly_the_plan_attempt_limit() -> None:
    tool = _agent_tool()

    class FailedCompaction(_Effects):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.compaction_attempts: list[int] = []

        async def compact(self, context: RuntimeContext, attempt: int) -> Any:
            del context
            self.compaction_attempts.append(attempt)
            raise RuntimeError("summary invalid")

    effects = FailedCompaction([_assistant(text="unreachable")], compaction_required=True)
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, effects, tool)
    session_id = SessionId.new()
    plan = replace(_plan(tool), compaction_attempt_limit=3)
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="compact-limit",
        content="question",
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationFailed)
    assert final.state.kind == "compaction_failed"
    assert effects.compaction_attempts == [1, 2, 3]


@pytest.mark.asyncio
async def test_provider_retry_exhaustion_is_typed_operation_failure() -> None:
    tool = _agent_tool()

    class UnavailableProvider(_Effects):
        async def call_provider(
            self,
            context: RuntimeContext,
            request: RequestSnapshot,
            attempt_id: AttemptId,
            emit_ephemeral: Any,
        ) -> AssistantTurn:
            del context, request, emit_ephemeral
            self.provider_attempts.append(attempt_id)
            raise ProviderAttemptFailed("provider unavailable", retryable=True)

    effects = UnavailableProvider([])
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=effects,
        tools=[tool],
        fencing_epoch=1,
        provider_attempt_limit=2,
    )
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="provider-exhausted",
        content="question",
        plan=replace(_plan(tool), provider_attempt_limit=2),
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationFailed)
    assert final.state.kind == "provider_unavailable"
    assert len(effects.provider_attempts) == 2
    assert not any(
        record.ref.kind == "session_fault" for record in (await store.load(session_id)).registers
    )


@pytest.mark.asyncio
async def test_steer_inbox_limit_and_invalid_arguments_are_typed() -> None:
    tool = _agent_tool()
    effects = _Effects(
        [
            _assistant(ToolCall("invalid", "lookup", {"other": "x"})),
            _assistant(text="done"),
        ]
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="steer-full",
        content="question",
        plan=replace(_plan(tool), max_pending_steers=1),
    )
    await runtime.steer(
        session_id=session_id,
        operation_id=accepted.operation_id,
        control_id="first",
        content="first steer",
    )
    with pytest.raises(OperationConflictError, match="inbox is full"):
        await runtime.steer(
            session_id=session_id,
            operation_id=accepted.operation_id,
            control_id="second",
            content="second steer",
        )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    result = next(
        entry
        for entry in (await store.load(session_id)).entries
        if isinstance(entry, ToolResultMessageEntry)
    )
    assert result.result.outcome == "invalid_arguments"
    assert effects.executed_sources == []


@pytest.mark.asyncio
async def test_plan_denied_tool_is_synthetic_even_when_runtime_can_resolve_it() -> None:
    allowed = _agent_tool()
    denied = AgentTool(
        name="restricted",
        description="restricted",
        input_model=_Args,
        execute=_tool,
    )
    effects = _Effects(
        [
            _assistant(ToolCall("denied", "restricted", {"value": "x"})),
            _assistant(text="done"),
        ]
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=effects,
        tools=[allowed, denied],
        fencing_epoch=1,
    )
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="plan-denied",
        content="question",
        plan=_plan(allowed),
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    result = next(
        entry
        for entry in (await store.load(session_id)).entries
        if isinstance(entry, ToolResultMessageEntry)
    )
    assert result.result.outcome == "plan_denied"
    assert effects.executed_sources == []


@pytest.mark.asyncio
async def test_length_stopped_tool_call_is_never_executed() -> None:
    tool = _agent_tool()
    effects = _Effects(
        [
            _assistant(
                ToolCall("truncated", "lookup", {"value": "x"}),
                stop="length",
            ),
            _assistant(text="done"),
        ]
    )
    store = MemoryAgentSessionRepository[dict[str, Any]]()
    runtime = _runtime(store, effects, tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="truncated",
        content="question",
        plan=_plan(tool),
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    result = next(
        entry
        for entry in (await store.load(session_id)).entries
        if isinstance(entry, ToolResultMessageEntry)
    )
    assert result.result.outcome == "truncated_arguments"
    assert effects.executed_sources == []


class _CountingSnapshotRepository(MemoryAgentSessionRepository[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.refresh_calls = 0
        self.decoded_rows = 0
        self.refresh_allocations = 0
        self.loaded_snapshots: list[AgentSessionSnapshot] = []

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        self.load_calls += 1
        snapshot = await super().load(session_id)
        self.decoded_rows += len(snapshot.entries)
        self.loaded_snapshots.append(snapshot)
        return snapshot

    async def refresh(
        self,
        session_id: SessionId,
        *,
        previous: AgentSessionSnapshot,
    ) -> AgentSessionSnapshot:
        self.refresh_calls += 1
        snapshot = await super().refresh(session_id, previous=previous)
        self.refresh_allocations += int(snapshot is not previous)
        self.decoded_rows += snapshot.last_entry_sequence - previous.last_entry_sequence
        return snapshot

    async def authoritative(self, session_id: SessionId) -> AgentSessionSnapshot:
        return await super().load(session_id)


async def test_runtime_cache_decodes_thousand_entry_history_once_across_many_register_actions() -> (
    None
):
    tool = _agent_tool()
    session_id = SessionId.new()
    repository = _CountingSnapshotRepository()
    entries: list[UserMessageEntry] = []
    parent: EntryId | None = None
    for index in range(1000):
        entry = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=parent,
            content=f"history {index}",
        )
        entries.append(entry)
        parent = entry.entry_id
    assert parent is not None
    head = LaneHead(LaneId.main(), parent)
    lane = LaneState(LaneId.main())
    await repository.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=entries,
            register_writes=[SetRegister(head), SetRegister(lane)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(lane.ref, None),
            ],
        ),
    )

    class UnavailableHistory(_Effects):
        async def call_provider(
            self,
            context: RuntimeContext,
            request: RequestSnapshot,
            attempt_id: AttemptId,
            emit_ephemeral: Any,
        ) -> AssistantTurn:
            del context, request, attempt_id, emit_ephemeral
            raise ProviderAttemptFailed("provider unavailable", retryable=True)

    attempts = 50
    runtime = AgentSessionRuntime(
        repository=repository,
        effects=UnavailableHistory([]),
        tools=[tool],
        fencing_epoch=1,
        provider_attempt_limit=attempts,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="long-history",
        content="new question",
        plan=replace(_plan(tool), provider_attempt_limit=attempts),
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    assert isinstance(final.state, OperationFailed)
    assert repository.load_calls == 1
    assert repository.refresh_calls == 102
    assert repository.decoded_rows == 1000
    assert repository.refresh_allocations == 0
    initial = repository.loaded_snapshots[0]
    assert all(
        final.context.snapshot.entries[index] is initial.entries[index] for index in range(1000)
    )
    assert final.context.snapshot.entries[1000].sequence == 1001
    assert not any(
        record.ref.kind == "request_snapshot" for record in final.context.snapshot.registers
    )
    authoritative = await repository.authoritative(session_id)
    assert final.context.snapshot.entries == authoritative.entries
    assert final.context.snapshot.registers == authoritative.registers

    # The former full-load callback would have decoded all 1,001 rows per refresh.
    legacy_decoded_rows = 1000 + repository.refresh_calls * 1001
    assert legacy_decoded_rows == 103_102


class _ForcedOutcomeRepository(MemoryAgentSessionRepository[dict[str, Any]]):
    forced: str | None = None

    async def transact(self, **kwargs: Any):
        if self.forced == "conflict":
            return RegisterConflict(RegisterRef("operation_state", "forced"), 1, 2)
        if self.forced == "lease":
            return TransactionLeaseLost()
        return await super().transact(**kwargs)


@pytest.mark.parametrize(
    ("forced", "error"),
    [("conflict", OperationConflictError), ("lease", SessionLeaseLostError)],
)
async def test_conflict_or_lease_loss_invalidates_without_speculative_cache_publish(
    forced: str,
    error: type[Exception],
) -> None:
    tool = _agent_tool()
    repository = _ForcedOutcomeRepository()
    runtime = _runtime(repository, _Effects([]), tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key=f"forced-{forced}",
        content="question",
        plan=_plan(tool),
    )
    cached = runtime._snapshots[session_id]  # pyright: ignore[reportPrivateUsage]
    repository.forced = forced

    with pytest.raises(error):
        await runtime.steer(
            session_id=session_id,
            operation_id=accepted.operation_id,
            control_id="not-committed",
            content="speculative",
        )

    assert session_id not in runtime._snapshots  # pyright: ignore[reportPrivateUsage]
    durable = await repository.load(session_id)
    state_record = next(
        record for record in durable.registers if record.ref.kind == "operation_state"
    )
    assert isinstance(state_record.value, OperationStateRegister)
    assert getattr(state_record.value.state, "steers", ()) == ()
    assert durable.commit_sequence == cached.commit_sequence


class _SlowTransactionRepository(MemoryAgentSessionRepository[dict[str, Any]]):
    slow = False
    active_transactions = 0
    max_active_transactions = 0

    async def transact(self, **kwargs: Any):
        if not self.slow:
            return await super().transact(**kwargs)
        self.active_transactions += 1
        self.max_active_transactions = max(
            self.max_active_transactions,
            self.active_transactions,
        )
        try:
            await asyncio.sleep(0.01)
            return await super().transact(**kwargs)
        finally:
            self.active_transactions -= 1


async def test_concurrent_controls_are_serialized_without_cache_regression() -> None:
    tool = _agent_tool()
    repository = _SlowTransactionRepository()
    runtime = _runtime(repository, _Effects([]), tool)
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="concurrent-controls",
        content="question",
        plan=_plan(tool),
    )
    repository.slow = True

    await asyncio.gather(
        runtime.steer(
            session_id=session_id,
            operation_id=accepted.operation_id,
            control_id="control-a",
            content="a",
        ),
        runtime.steer(
            session_id=session_id,
            operation_id=accepted.operation_id,
            control_id="control-b",
            content="b",
        ),
    )

    view = await runtime.restore(session_id=session_id, operation_id=accepted.operation_id)
    authoritative = await repository.load(session_id)
    assert repository.max_active_transactions == 1
    assert {steer.control_id for steer in getattr(view.state, "steers", ())} == {
        "control-a",
        "control-b",
    }
    assert view.context.snapshot.commit_sequence == authoritative.commit_sequence
    assert view.context.snapshot.registers == authoritative.registers
