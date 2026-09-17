# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Regressions for Child Operation recovery and concurrent host composition."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import ToolBatchItem
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime, OperationIdempotencyConflict
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.research.runtime import (
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _child_agent_plan,
    _durable_child_usage,
    run_child_session,
)
from dlightrag.engine.answer.tools.subagents import (
    ChildControlInput,
    ChildOutcome,
    ChildRequest,
    SubagentHost,
    subagent_tools,
)
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.tool_helpers import tool_runtime
from tests.unit.test_subagents import _child_orchestrator, _context_snapshot, _FakeSession


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "cancelled", "succeeded"])
@pytest.mark.parametrize("settled_tool_turn", [False, True])
async def test_child_continuation_terminal_usage_is_operation_local(
    terminal_status, settled_tool_turn
):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2 and settled_tool_turn:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="read-current", name="read", arguments={"resource_id": "generated"}
                    ),
                ),
                stop_reason="tool_use",
                usage_details={"input_tokens": 5, "output_tokens": 2},
            )
        if len(calls) > 1:
            if terminal_status == "failed":
                raise RuntimeError("controlled provider rejection")
            if terminal_status == "cancelled":
                raise asyncio.CancelledError
            return AssistantTurn(
                text="follow-up summary",
                tool_calls=(),
                stop_reason="stop",
                usage_details={"input_tokens": 1, "output_tokens": 1},
            )
        return AssistantTurn(
            text="initial evidence summary",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 7, "output_tokens": 3},
        )

    orchestrator = _child_orchestrator(model)

    async def read(_raw, _runtime):
        return ToolResult.text("read complete")

    orchestrator._resource_reader = read
    parent = SessionId.new()
    child = SessionId.new()
    row = {}

    async def persist(**kw):
        row.setdefault("plan", kw["plan"])

    async def load(**kw):
        return dict(row)

    async def run(key, objective):
        row.update(
            operation_key=key, operation_id=OperationId.deterministic(idempotency_key=key).value
        )
        return await run_child_session(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            repository=repo,
            session=cast(Any, _FakeSession(run_id=parent.value)),
            fetched_buffer=FetchedResourceBuffer(),
            child_id=child,
            request=ChildRequest(objective=objective),
            parent_call_id="call",
            parent_session_id=parent,
            context_snapshot=_context_snapshot(parent),
            persist_child_runtime=persist,
            claim_child=AsyncMock(return_value=1),
            load_child=load,
        )

    repo = MemoryAgentSessionRepository[EffectHostUpdate]()
    first = await run("initial", "initial objective")
    second = await run("followup", "follow-up objective")
    assert first.status == "succeeded" and second.status == terminal_status
    assert first.usage == {"input_tokens": 7, "output_tokens": 3}
    expected_usage = {"input_tokens": 5, "output_tokens": 2} if settled_tool_turn else {}
    if terminal_status == "succeeded":
        expected_usage = {
            "input_tokens": expected_usage.get("input_tokens", 0) + 1,
            "output_tokens": expected_usage.get("output_tokens", 0) + 1,
        }
    assert dict(second.usage or {}) == expected_usage
    assert second.summary != first.summary
    store = SimpleNamespace(
        list_child_sessions=AsyncMock(
            return_value=[{"operation_usage": [first.usage, second.usage]}]
        )
    )
    total = await _durable_child_usage(store, owner_id="owner", run_id=parent.value)
    assert total == {
        "input_tokens": 7 + expected_usage.get("input_tokens", 0),
        "output_tokens": 3 + expected_usage.get("output_tokens", 0),
    }
    replay = await run("followup", "follow-up objective")
    assert replay.usage == second.usage
    assert len(calls) == 2 + int(settled_tool_turn)
    assert "initial evidence summary" in str(calls[1]["messages"])


@pytest.mark.asyncio
async def test_recovery_preserves_current_pinned_tools():

    async def model(**kw):
        return AssistantTurn(text="recovered v2", tool_calls=(), stop_reason="stop")

    orchestrator = _child_orchestrator(model, environment=MagicMock())
    assert orchestrator.subagent_host is not None
    orchestrator.subagent_host.async_lifecycle = False
    orchestrator.subagent_host.interactive_controls = False
    request = ChildRequest(objective="continue pinned work")
    parent = SessionId.new()
    child = SessionId.new()
    context = _context_snapshot(parent)
    prepared = orchestrator.prepare_child_session(request, context_snapshot=context)
    from dlightrag.engine.answer.research.runtime import _child_agent_plan

    oldplan = _child_agent_plan(prepared, request)
    oldnames = {tool.name for tool in oldplan.tools}
    assert {"write", "edit", "bash"} <= oldnames
    # Reconstruct an actually accepted, not-yet-driven v2 Operation.
    explicit = orchestrator.prepare_child_session(
        ChildRequest(objective=request.objective, tools=tuple(oldnames)), context_snapshot=context
    )
    repo = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=repo, effects=MagicMock(), tools=explicit.tools, fencing_epoch=1
    )
    await runtime.accept(
        session_id=child,
        lane_id=LaneId.main(),
        idempotency_key=f"child-session:{child.value}",
        content=request.objective,
        plan=oldplan,
    )
    outcome = await run_child_session(
        telemetry=NOOP_TELEMETRY,
        orchestrator=orchestrator,
        repository=repo,
        session=cast(Any, _FakeSession(run_id=parent.value)),
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child,
        request=request,
        parent_call_id="call",
        parent_session_id=parent,
        context_snapshot=context,
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
        load_child=AsyncMock(return_value={"plan": oldplan.canonical_payload()}),
    )
    assert outcome.status == "succeeded"
    assert {tool.name for tool in prepared.tools} == oldnames


@pytest.mark.asyncio
async def test_notification_input_is_immutable_after_evidence_restore():
    async def model(**kw):
        return AssistantTurn(text="unused", tool_calls=(), stop_reason="stop")

    parent = SessionId.new()
    child = SessionId.new()
    evidence = EvidenceLedger()
    evidence.add_rows(
        [{"chunk_id": "c1", "content": "concrete evidence", "file_path": "source.txt"}]
    )
    outcome = ChildOutcome(
        status="succeeded",
        summary="child complete",
        child_session_id=child.value,
        operation_id=OperationId.new().value,
        evidence_state=evidence.durable_state(),
    )
    row = {
        "status": "succeeded",
        "child_session_id": child.value,
        "parent_intent_id": "intent",
        "parent_call_id": "call",
        "host_state": {"terminal_outcome": outcome.durable_payload()},
    }
    orchestrator = _child_orchestrator(model)
    host = orchestrator.subagent_host
    assert host is not None
    host.parent_session_id = parent
    host.list_children = AsyncMock(return_value=[row])
    prepared = orchestrator.prepare_run("parent objective")
    first = (await host.completed_dispatch_notifications(seen=set()))[0]
    # Durable parent tool settlement persists this ledger; reclaim restores it.
    restored = prepared.evidence.durable_state()
    resumed = _child_orchestrator(model)
    resumed_host = resumed.subagent_host
    assert resumed_host is not None
    resumed_host.parent_session_id = parent
    resumed_host.list_children = AsyncMock(return_value=[row])
    resumed_prepared = resumed.prepare_run("parent objective")
    resumed_prepared.evidence.restore_ledger_state(restored)
    second = (await resumed_host.completed_dispatch_notifications(seen=set()))[0]
    assert first == second
    assert len(resumed_prepared.evidence.contexts["chunks"]) == 1
    repo = MemoryAgentSessionRepository[EffectHostUpdate]()
    plan = _child_agent_plan(prepared, ChildRequest(objective="parent objective"))
    runtime = AgentSessionRuntime(
        repository=repo, effects=MagicMock(), tools=prepared.tools, fencing_epoch=1
    )
    await runtime.accept(
        session_id=parent,
        lane_id=LaneId.main(),
        idempotency_key=first[0],
        content=first[1],
        plan=plan,
    )
    replay = await runtime.accept(
        session_id=parent,
        lane_id=LaneId.main(),
        idempotency_key=second[0],
        content=second[1],
        plan=plan,
    )
    assert not replay.created
    with pytest.raises(OperationIdempotencyConflict):
        await runtime.accept(
            session_id=parent,
            lane_id=LaneId.main(),
            idempotency_key=second[0],
            content="changed intent",
            plan=plan,
        )


def _batch(name):
    return ToolBatchItem(
        source_index=0,
        call_id="call-" + name,
        tool_name=name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy="replayable",
        contract_version=1,
        input_schema_digest="0" * 64,
        effective_input_digest="0" * 64,
    )


@pytest.mark.asyncio
async def test_concurrent_child_tool_cannot_overwrite_parent_dispatch_context():
    async def model(**kw):
        return AssistantTurn(text="unused", tool_calls=(), stop_reason="stop")

    async def read(raw, runtime):
        return ToolResult.text("read complete")

    orchestrator = _child_orchestrator(model)
    orchestrator._resource_reader = read
    host = orchestrator.subagent_host
    assert host is not None
    parent = SessionId.new()
    active_child = SessionId.new()
    host.parent_session_id = parent
    host.run_id = parent.value
    host.owner_id = "owner"
    captured = []

    async def persist(**kw):
        captured.append(kw)

    host.persist = persist
    host.prepare_dispatch = lambda *args: {}
    host.run_child = AsyncMock(return_value=ChildOutcome(status="succeeded", summary="fake runner"))
    parent_prepared = orchestrator.prepare_run("parent")
    child_prepared = orchestrator.prepare_child_session(
        ChildRequest(objective="active child"), context_snapshot=_context_snapshot(parent)
    )
    repo = MemoryAgentSessionRepository[EffectHostUpdate]()
    session = _FakeSession(run_id=parent.value)

    async def context(session_id, prepared, objective):
        plan = _child_agent_plan(prepared, ChildRequest(objective=objective))
        runtime = AgentSessionRuntime(
            repository=repo, effects=MagicMock(), tools=prepared.tools, fencing_epoch=1
        )
        accepted = await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key=session_id.value,
            content=objective,
            plan=plan,
        )
        return (
            await runtime.restore(session_id=session_id, operation_id=accepted.operation_id)
        ).context

    pc = await context(parent, parent_prepared, "parent")
    cc = await context(active_child, child_prepared, "child")
    reached = asyncio.Event()
    release = asyncio.Event()

    async def sparse_precreate(**kw):
        reached.set()
        await release.wait()

    parent_effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=orchestrator,
        prepared=parent_prepared,
        session=cast(Any, session),
        session_id=parent,
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=sparse_precreate,
    )
    child_effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=orchestrator,
        prepared=child_prepared,
        session=cast(Any, session),
        session_id=active_child,
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    args = {"children": [{"objective": "second investigation"}]}
    task = asyncio.create_task(
        parent_effects.execute_tool(pc, _batch("spawn_agent"), args, AttemptId.new(), AsyncMock())
    )
    await reached.wait()
    await child_effects.execute_tool(
        cc, _batch("read"), {"resource_id": "generated"}, AttemptId.new(), AsyncMock()
    )
    release.set()
    await task
    await asyncio.gather(*host.tasks.values())
    assert captured[0]["parent_session_id"] == parent.value
    assert captured[0]["context_snapshot"]["parent_session_id"] == parent.value
    assert host.context_snapshot is not None
    result = await run_child_session(
        telemetry=NOOP_TELEMETRY,
        orchestrator=orchestrator,
        repository=repo,
        session=cast(Any, session),
        fetched_buffer=FetchedResourceBuffer(),
        child_id=SessionId(captured[0]["child_session_id"]),
        request=ChildRequest(objective="second investigation"),
        parent_call_id="call",
        parent_session_id=parent,
        context_snapshot=host.context_snapshot,
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
    )
    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_user_continuation_status_and_cancel_ignore_stale_task():
    child = SessionId.new()
    old = ChildOutcome(status="succeeded", summary="prior operation", child_session_id=child.value)
    task = asyncio.create_task(asyncio.sleep(0, result=old))
    await task
    current: dict[str, Any] = {"status": "running", "operation_id": "new-operation"}

    async def cancel_current(**kwargs):
        cancelled = ChildOutcome(
            status="cancelled",
            summary="current cancelled",
            child_session_id=child.value,
            operation_id="new-operation",
        )
        current.update(
            status="cancelled", host_state={"terminal_outcome": cancelled.durable_payload()}
        )
        return True

    load_child = AsyncMock(side_effect=lambda **kw: dict(current))
    request_cancel = AsyncMock(side_effect=cancel_current)
    host = SubagentHost(
        tasks={child.value: task},
        load_child=load_child,
        request_cancel=request_cancel,
    )
    tools = {tool.name: tool for tool in subagent_tools(host=host)}
    status = await tools["subagent_status"].execute(
        ChildControlInput(child_session_id=child.value), tool_runtime(tool_name="subagent_status")
    )
    assert status.details is not None
    assert status.details["children"][0]["status"] == "running"
    result = await tools["cancel_subagent"].execute(
        ChildControlInput(child_session_id=child.value), tool_runtime(tool_name="cancel_subagent")
    )
    assert result.details is not None
    assert result.details["children"][0]["status"] == "cancelled"
    request_cancel.assert_awaited_once()
    load_child.assert_awaited()


async def test_wait_reads_durable_terminal_before_waiting_on_stale_task() -> None:
    child = SessionId.new()
    outcome = ChildOutcome(
        status="succeeded",
        summary="current complete",
        child_session_id=child.value,
        operation_id=OperationId.new().value,
    )
    old_task = asyncio.create_task(asyncio.sleep(60, result=outcome))
    host = SubagentHost(
        tasks={child.value: old_task},
        load_child=AsyncMock(
            return_value={
                "status": "succeeded",
                "operation_id": outcome.operation_id,
                "host_state": {"terminal_outcome": outcome.durable_payload()},
            }
        ),
    )
    try:
        tools = {tool.name: tool for tool in subagent_tools(host=host)}
        result = await asyncio.wait_for(
            tools["wait_subagent"].execute(
                ChildControlInput(child_session_id=child.value),
                tool_runtime(tool_name="wait_subagent"),
            ),
            timeout=0.1,
        )
        assert result.details is not None
        assert result.details["children"][0]["status"] == "succeeded"
    finally:
        old_task.cancel()
        await asyncio.gather(old_task, return_exceptions=True)
