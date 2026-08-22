# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""delegate_research composition, replay, and journal-backed children."""

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dlightrag.agent.environment.access import PathAccess
from dlightrag.agent.session.effects import EffectIntent
from dlightrag.agent.session.ids import IntentId, SessionId
from dlightrag.agent.session.memory import InMemoryAgentSessionStore
from dlightrag.agent.tools.context import bind_tool_call, reset_tool_call
from dlightrag.ai.capacity import CONTEXT_POLICY
from dlightrag.ai.messages import AssistantTurn
from dlightrag.ai.telemetry import NOOP_TELEMETRY
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.executor import JournalRunBoundaries, _seed_child_session, run_child_session
from dlightrag.answer.resources.models import TextWindowBudget
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools.composition import compose_research_tools
from dlightrag.answer.tools.delegate import (
    ChildOutcome,
    DelegateHost,
    DelegateInput,
    child_session_id,
    delegate_research_tool,
)
from dlightrag.runtime import RunCancelledError
from tests.unit.conftest import answer_image_policy, answer_model_profile


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def test_parent_tools_include_delegate_and_child_omits_it() -> None:
    host = DelegateHost()
    parent = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        delegate_host=host,
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        child=True,
    )
    assert "delegate_research" in {tool.name for tool in parent}
    assert {tool.name for tool in child} == {"search_knowledge_base"}
    assert "write" not in {tool.name for tool in child}
    assert "bash" not in {tool.name for tool in child}


async def test_replay_returns_journal_outcome_not_sidecar_summary() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(_child_id: SessionId, _objective: str, _call_id: str) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Journaled child finding.",
            handles=("[1] report.pdf",),
            usage={"input_tokens": 8},
            child_session_id="child",
        )

    host = DelegateHost(
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
    tool = delegate_research_tool(host=host)
    token = bind_tool_call("call-1", "delegate_research")
    try:
        result = await tool.execute(DelegateInput(objective="what happened?"))
    finally:
        reset_tool_call(token)
    assert "Journaled child finding." in result.content
    assert "[1] report.pdf" in result.content
    assert "Usage: input_tokens=8" in result.content
    assert "Sidecar-only summary." not in result.content


async def test_delegate_reports_child_outcome_and_usage() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(_child_id: SessionId, _objective: str, _call_id: str) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            handles=("[1] Page A [resource: res-a]",),
            usage={"input_tokens": 12, "output_tokens": 4},
            child_session_id="child",
        )

    host = DelegateHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=persist,
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
    )
    tool = delegate_research_tool(host=host)
    token = bind_tool_call("call-9", "delegate_research")
    try:
        result = await tool.execute(DelegateInput(objective="summarize filings"))
    finally:
        reset_tool_call(token)
    assert "Child summary." in result.content
    assert "[1] Page A [resource: res-a]" in result.content
    assert "Usage: input_tokens=12, output_tokens=4" in result.content
    assert result.details is not None
    assert result.details["status"] == "succeeded"
    persist.assert_awaited()
    finish.assert_awaited()
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "succeeded"


async def test_failed_child_is_recorded_failed() -> None:
    finish = AsyncMock()

    async def run_child(_child_id: SessionId, _objective: str, _call_id: str) -> ChildOutcome:
        return ChildOutcome(status="failed", summary="provider down", child_session_id="child")

    host = DelegateHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=AsyncMock(),
        load_child=AsyncMock(return_value=None),
        finish_child=finish,
        run_child=run_child,
    )
    tool = delegate_research_tool(host=host)
    token = bind_tool_call("call-err", "delegate_research")
    try:
        result = await tool.execute(DelegateInput(objective="x"))
    finally:
        reset_tool_call(token)
    assert result.details is not None
    assert result.details["status"] == "failed"
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


def _child_orchestrator(
    model_func: Any, *, environment: object | None = None
) -> AnswerOrchestrator:
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
        delegate_host=DelegateHost(),
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
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="delegate:test:1")
    session = _FakeSession(run_id=str(parent_id.value))

    first = await run_child_session(
        orchestrator=orchestrator,
        journal=journal,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=[],
        child_id=child_id,
        objective="summarize filings",
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
        fetched_buffer=[],
        child_id=child_id,
        objective="summarize filings",
        parent_call_id="call-1",
        parent_session_id=parent_id,
    )
    assert second.summary == first.summary
    assert second.usage == first.usage
    assert second.handles == first.handles
    assert second.status == first.status
    assert calls["n"] == 1


def test_child_with_environment_gets_path_read_and_grep() -> None:
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
    assert names >= {"search_knowledge_base", "read", "grep"}
    assert "write" not in names
    assert "bash" not in names
    assert "delegate_research" not in names


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
        fetched_buffer=[],
        child_id=SessionId.deterministic(run_id=str(parent_id.value), name="c"),
        objective="stop",
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
        fetched_buffer=[],
        child_id=child_id,
        objective="continue",
        parent_call_id="call-r",
        parent_session_id=parent_id,
    )
    assert outcome.status == "succeeded"
    assert outcome.summary == "Resumed child."
    assert calls["n"] == 1


async def test_child_workspace_hold_blocks_writes() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def model(**_kwargs: object) -> AssistantTurn:
        started.set()
        await release.wait()
        return AssistantTurn(text="held", tool_calls=(), stop_reason="stop")

    orchestrator = _child_orchestrator(model, environment=object())
    parent_id = SessionId.new()
    child = asyncio.create_task(
        run_child_session(
            orchestrator=orchestrator,
            journal=InMemoryAgentSessionStore(),  # type: ignore[arg-type]
            session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
            fetched_buffer=[],
            child_id=SessionId.deterministic(run_id=str(parent_id.value), name="hold"),
            objective="hold",
            parent_call_id="call-h",
            parent_session_id=parent_id,
        )
    )
    await started.wait()
    write_entered = asyncio.Event()

    async def writer() -> None:
        async with orchestrator._access.hold(PathAccess("x", kind="write")):
            write_entered.set()

    write_task = asyncio.create_task(writer())
    await asyncio.sleep(0.05)
    assert not write_entered.is_set()
    release.set()
    await child
    await asyncio.wait_for(write_task, timeout=1)
    assert write_entered.is_set()


async def test_parent_turn_binds_delegate_effect_intent() -> None:
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
        fetched_buffer=[],
        run_id=run_id,
        link_delegate_intent=link,
    )
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="delegate_research",
        replay_policy="safe",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="call-9",
    )
    await bounds._bind_delegate_parent_intents((intent,))
    assert seen["parent_intent_id"] == intent.intent_id.value
    assert (
        seen["child_session_id"]
        == child_session_id(run_id=run_id, parent_session_id=parent_id, call_id="call-9").value
    )
