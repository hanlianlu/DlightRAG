# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Foreground subagent composition, controls, replay, and durable children."""

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dlightrag.agent.session.effects import EffectIntent
from dlightrag.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.agent.session.ids import IntentId, SessionId
from dlightrag.agent.tools import AgentTool, PreparedToolTurn, ToolPreflight, ToolResult
from dlightrag.agent.tools.context import bind_tool_call, reset_tool_call
from dlightrag.ai.capacity import CONTEXT_POLICY
from dlightrag.ai.messages import AssistantTurn
from dlightrag.ai.telemetry import NOOP_TELEMETRY
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.executor import (
    FetchedResourceBuffer,
    JournalRunBoundaries,
    _seed_child_session,
    run_child_session,
)
from dlightrag.answer.resources.models import TextWindowBudget
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools.composition import compose_research_tools
from dlightrag.answer.tools.subagents import (
    ChildControlInput,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
    SubagentHost,
    child_session_id,
    subagent_tools,
)
from dlightrag.runtime import RunCancelledError
from tests.in_memory_session_store import InMemoryAgentSessionStore
from tests.unit.conftest import answer_image_policy, answer_model_profile


def _spawn_input(objective: str) -> SpawnAgentInput:
    return SpawnAgentInput(children=(ChildRequest(objective=objective),))


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def test_parent_tools_include_spawn_and_child_omits_it() -> None:
    host = SubagentHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        subagent_host=host,
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        subagent_host=host,
        child=True,
    )
    controls = {"subagent_status", "wait_subagent", "cancel_subagent"}
    assert {"spawn_agent", *controls} <= {tool.name for tool in parent}
    assert {"search_knowledge_base", *controls} == {tool.name for tool in child}
    assert "spawn_agent" not in {tool.name for tool in child}


async def test_spawn_many_runs_in_parallel_and_aggregates_usage() -> None:
    started = 0
    both_started = asyncio.Event()

    async def run_child(child_id: SessionId, request: ChildRequest, _call_id: str) -> ChildOutcome:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return ChildOutcome(
            status="succeeded",
            summary=request.objective,
            usage={"input_tokens": 2},
            child_session_id=child_id.value,
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        run_child=run_child,
    )
    token = bind_tool_call("parallel", "spawn_agent")
    try:
        result = await subagent_tools(host=host)[0].execute(
            SpawnAgentInput(
                children=(
                    ChildRequest(objective="one"),
                    ChildRequest(objective="two"),
                )
            )
        )
    finally:
        reset_tool_call(token)

    assert result.details is not None
    assert len(result.details["children"]) == 2
    assert result.details["inclusive_usage"] == {"input_tokens": 4}
    assert not host.tasks


async def test_default_depth_one_rejects_recursive_spawn() -> None:
    runner = AsyncMock()
    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        depth=1,
        run_child=runner,
    )

    result = await subagent_tools(host=host)[0].execute(_spawn_input("too deep"))

    assert "depth limit" in result.content
    runner.assert_not_awaited()


async def test_cancel_tool_joins_a_known_foreground_child() -> None:
    started = asyncio.Event()

    async def sleeper() -> ChildOutcome:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(sleeper())
    await started.wait()
    host = SubagentHost(tasks={"child-1": task})

    result = await subagent_tools(host=host)[3].execute(
        ChildControlInput(child_session_id="child-1")
    )

    assert task.cancelled()
    assert "[cancelled]" in result.content
    assert host.outcomes["child-1"].status == "cancelled"


async def test_replay_returns_journal_outcome_not_sidecar_summary() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId, _request: ChildRequest, _call_id: str
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Journaled child finding.",
            handles=("[1] report.pdf",),
            usage={"input_tokens": 8},
            child_session_id="child",
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        load_child=AsyncMock(
            return_value={"status": "succeeded", "summary": "Sidecar-only summary."}
        ),
        persist=persist,
        finish_child=finish,
        run_child=run_child,
    )
    tool = subagent_tools(host=host)[0]
    token = bind_tool_call("call-1", "spawn_agent")
    try:
        result = await tool.execute(_spawn_input("what happened?"))
    finally:
        reset_tool_call(token)
    assert "Journaled child finding." in result.content
    assert "[1] report.pdf" in result.content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 8}
    assert "Sidecar-only summary." not in result.content


async def test_spawn_reports_child_outcome_and_usage() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId, _request: ChildRequest, _call_id: str
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            handles=("[1] Page A [resource: res-a]",),
            usage={"input_tokens": 12, "output_tokens": 4},
            child_session_id="child",
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=persist,
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
    )
    tool = subagent_tools(host=host)[0]
    token = bind_tool_call("call-9", "spawn_agent")
    try:
        result = await tool.execute(_spawn_input("summarize filings"))
    finally:
        reset_tool_call(token)
    assert "Child summary." in result.content
    assert "[1] Page A [resource: res-a]" in result.content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result.details["children"][0]["status"] == "succeeded"
    persist.assert_awaited()
    finish.assert_awaited()
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "succeeded"


async def test_spawn_adopts_child_evidence_before_returning_result() -> None:
    adopted = MagicMock(return_value=("[2] child source",))
    child_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    async def run_child(
        _child_id: SessionId, _request: ChildRequest, _call_id: str
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            child_session_id="child",
            evidence_state=child_state,
        )

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        run_child=run_child,
        adopt_evidence=adopted,
    )
    tool = subagent_tools(host=host)[0]
    token = bind_tool_call("call-adopt", "spawn_agent")
    try:
        result = await tool.execute(_spawn_input("find source"))
    finally:
        reset_tool_call(token)

    assert "[2] child source" in result.content
    adopted.assert_called_once()
    assert adopted.call_args.args[0] == child_state
    assert adopted.call_args.args[2] == "call-adopt"


async def test_failed_child_is_recorded_failed() -> None:
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId, _request: ChildRequest, _call_id: str
    ) -> ChildOutcome:
        return ChildOutcome(status="failed", summary="provider down", child_session_id="child")

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=AsyncMock(),
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
    )
    tool = subagent_tools(host=host)[0]
    token = bind_tool_call("call-err", "spawn_agent")
    try:
        result = await tool.execute(_spawn_input("x"))
    finally:
        reset_tool_call(token)
    assert result.details is not None
    assert result.details["children"][0]["status"] == "failed"
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "failed"


@dataclass
class _FakeSession:
    run_id: str
    owner_id: str = "owner"

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, _phase: str) -> None:
        return None


def _child_orchestrator(model_func: Any, *, environment: Any = None) -> AnswerOrchestrator:
    profile = answer_model_profile()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("child should not search in this test")

    return AnswerOrchestrator(
        synthesizer=AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=profile,
        ),
        retrieve_knowledge_base=retrieve,
        search_web=None,
        model_func=model_func,
        telemetry=NOOP_TELEMETRY,
        model_profile=profile,
        text_window_budget=TextWindowBudget(CONTEXT_POLICY.hard_input_limit(profile)),
        subagent_host=SubagentHost(),
        resolved_mode="research",
        environment=environment,
    )


async def test_child_session_journals_and_replays_without_rerun() -> None:
    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        return AssistantTurn(
            text="Journaled child summary.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 3, "output_tokens": 2},
        )

    orchestrator = _child_orchestrator(model)
    journal = InMemoryAgentSessionStore()
    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:test:1")
    session = _FakeSession(run_id=str(parent_id.value))

    first = await run_child_session(
        orchestrator=orchestrator,
        journal=journal,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
    )
    assert first.status == "succeeded"
    assert first.summary == "Journaled child summary."
    assert first.usage == {"input_tokens": 3, "output_tokens": 2}
    assert calls["n"] == 1
    snapshot = await journal.load(child_id)
    assert snapshot.version >= 2
    assert any(entry.entry_type == "session_terminal" for entry in snapshot.entries)

    second = await run_child_session(
        orchestrator=orchestrator,
        journal=journal,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
    )
    assert second.summary == first.summary
    assert second.usage == first.usage
    assert second.handles == first.handles
    assert second.status == first.status
    assert calls["n"] == 1


async def test_child_selects_parent_context_and_an_inherited_tool_subset() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _child_orchestrator(model)
    orchestrator.prepare_run(
        "parent question",
        conversation_history=PriorTurns(
            [
                {"role": "user", "content": "older question"},
                {"role": "assistant", "content": "older answer"},
            ]
        ),
    )
    child = orchestrator.prepare_child_session(
        ChildRequest(
            objective="focused",
            context="parent",
            tools=("search_knowledge_base",),
        )
    )

    messages = await child.context.control_turn(
        evidence=child.evidence,
        working=WorkingContextProjection(retained_tail_tokens=1000),
        tool_schema_tokens=0,
    )

    assert [tool.name for tool in child.tools] == ["search_knowledge_base"]
    assert any(message.get("content") == "older answer" for message in messages)
    assert any(message.get("content") == "parent question" for message in messages)


def test_child_inherits_parent_path_tools_except_spawn() -> None:
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        child=True,
    )
    names = {tool.name for tool in child}
    assert names >= {"search_knowledge_base", "read", "write", "edit", "grep", "bash"}
    assert "spawn_agent" not in names


async def test_parent_cancel_marks_the_child_cancelled() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        raise AssertionError("cancelled child must not call the model")

    class _CancelSession(_FakeSession):
        async def check_cancelled(self) -> None:
            raise RunCancelledError

    parent_id = SessionId.new()
    outcome = await run_child_session(
        orchestrator=_child_orchestrator(model),
        journal=InMemoryAgentSessionStore(),  # type: ignore[arg-type]
        session=_CancelSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=SessionId.deterministic(run_id=str(parent_id.value), name="c"),
        request=ChildRequest(objective="stop"),
        parent_call_id="call-c",
        parent_session_id=parent_id,
    )
    assert outcome.status == "cancelled"


async def test_reclaim_resumes_a_nonterminal_child() -> None:
    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        return AssistantTurn(text="Resumed child.", tool_calls=(), stop_reason="stop")

    parent_id = SessionId.new()
    run_id = str(parent_id.value)
    child_id = SessionId.deterministic(run_id=run_id, name="reclaim")
    journal = InMemoryAgentSessionStore()
    await _seed_child_session(
        journal,  # type: ignore[arg-type]
        child_id,
        objective="continue",
        parent_session_id=parent_id,
        parent_call_id="call-r",
    )
    outcome = await run_child_session(
        orchestrator=_child_orchestrator(model),
        journal=journal,  # type: ignore[arg-type]
        session=_FakeSession(run_id=run_id),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="continue"),
        parent_call_id="call-r",
        parent_session_id=parent_id,
    )
    assert outcome.status == "succeeded"
    assert outcome.summary == "Resumed child."
    assert calls["n"] == 1


async def test_recovery_backfills_child_effect_intent_before_spawn_replay() -> None:
    from pydantic import BaseModel

    from dlightrag.agent.session.entries import EffectIntentEntry
    from dlightrag.ai.messages import ToolCall

    seen: dict[str, str] = {}

    async def link(**kwargs: Any) -> None:
        seen.update({key: str(value) for key, value in kwargs.items()})

    async def execute(_raw: BaseModel) -> ToolResult:
        return ToolResult(content="child replayed")

    parent_id = SessionId.new()
    run_id = str(parent_id.value)
    journal = InMemoryAgentSessionStore()
    tool = AgentTool(
        "spawn_agent",
        "Replay spawn.",
        SpawnAgentInput,
        execute,
        replay_policy="safe",
    )
    call = ToolCall(
        id="call-recovery",
        name=tool.name,
        arguments=_spawn_input("recover lineage").model_dump(mode="json"),
    )
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name=tool.name,
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        canonical_input=_spawn_input("recover lineage").model_dump_json(),
        source_call_id=call.id,
    )
    initial = JournalRunBoundaries(
        session=_FakeSession(run_id=run_id),  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        session_id=parent_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id=run_id,
    )
    await initial.commit_intents(
        PreparedToolTurn(
            assistant=AssistantTurn(text="", tool_calls=(call,), stop_reason="tool_use"),
            preflight=ToolPreflight(intents=(intent,), validation_results=()),
            transcript=[],
        )
    )
    snapshot = await journal.load(parent_id)
    assert any(isinstance(entry, EffectIntentEntry) for entry in snapshot.entries)
    recovered = JournalRunBoundaries(
        session=_FakeSession(run_id=run_id),  # type: ignore[arg-type]
        journal=journal,  # type: ignore[arg-type]
        session_id=parent_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id=run_id,
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        entries=snapshot.entries,
        persist_child_intent=link,
    )

    await recovered.recover_pending_intents(snapshot)

    assert seen["parent_intent_id"] == intent.intent_id.value
    assert seen["objective"] == "recover lineage"


async def test_parent_turn_binds_child_effect_intent() -> None:
    seen: dict[str, str] = {}

    async def link(**kwargs: Any) -> None:
        seen.update({key: str(value) for key, value in kwargs.items()})

    parent_id = SessionId.new()
    run_id = str(parent_id.value)
    bounds = JournalRunBoundaries(
        session=_FakeSession(run_id=run_id),  # type: ignore[arg-type]
        journal=InMemoryAgentSessionStore(),  # type: ignore[arg-type]
        session_id=parent_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id=run_id,
        persist_child_intent=link,
    )
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="spawn_agent",
        replay_policy="safe",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input=_spawn_input("research").model_dump_json(),
        source_call_id="call-9",
    )
    await bounds._bind_subagents_parent_intents((intent,))
    assert seen["parent_intent_id"] == intent.intent_id.value
    assert seen["parent_session_id"] == parent_id.value
    assert seen["objective"] == "research"
    assert seen["context_mode"] == "isolated"
    assert (
        seen["child_session_id"]
        == child_session_id(run_id=run_id, parent_session_id=parent_id, call_id="call-9").value
    )
