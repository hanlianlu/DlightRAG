# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned foreground and durable asynchronous Child Agent lifecycle tests."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.session.ids import EntryId, IntentId, OperationId, SessionId
from dlightrag.engine.ai.capacity import CONTEXT_POLICY
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.research.runtime import (
    FetchedResourceBuffer,
    run_child_session,
)
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools.composition import compose_research_tools
from dlightrag.engine.answer.tools.subagents import (
    AskParentInput,
    ChildContextSnapshot,
    ChildControlInput,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
    SubagentHost,
    child_guidance_tools,
    child_session_id,
    subagent_tools,
)
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from tests.in_memory_session_repository import InMemoryAgentSessionRepository
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_image_policy, answer_model_profile


def _spawn_input(objective: str) -> SpawnAgentInput:
    return SpawnAgentInput(children=(ChildRequest(objective=objective),))


def _context_snapshot(parent_id: SessionId | None = None) -> ChildContextSnapshot:
    return ChildContextSnapshot.from_values(
        parent_session_id=parent_id or SessionId.new(),
        parent_entry_id=EntryId.new(),
        depth=0,
        messages=[{"role": "user", "content": "parent question"}],
    )


async def _retrieve(_query: str) -> object:
    raise RuntimeError("unused")


def _durable_dispatch(
    _child_id: SessionId,
    _request: ChildRequest,
    _snapshot: ChildContextSnapshot,
) -> dict[str, Any]:
    return {
        "plan": {"schema_version": 2, "tools": []},
        "budget": {"provider_attempt_limit": 2},
        "host_state": {"dispatch_version": 3},
    }


def test_child_identity_uses_durable_intent_not_provider_call_id() -> None:
    run_id = SessionId.new().value
    parent_id = SessionId.new()
    first = child_session_id(
        run_id=run_id,
        parent_session_id=parent_id,
        parent_intent_id=IntentId.new(),
    )
    second = child_session_id(
        run_id=run_id,
        parent_session_id=parent_id,
        parent_intent_id=IntentId.new(),
    )
    assert first != second


async def test_ask_parent_persists_correlates_and_waits_without_provider_polling() -> None:
    parent_id = SessionId.new()
    child_id = SessionId.new().value
    operation_id = SessionId.new().value
    stored: dict[str, Any] = {}

    async def load_child(**_kwargs: Any) -> dict[str, Any]:
        return {"operation_id": operation_id, "fencing_epoch": 7}

    async def create(**kwargs: Any) -> dict[str, Any]:
        stored.update(kwargs)
        return {
            **kwargs,
            "status": "pending",
            "expires_at": datetime.now(UTC) + timedelta(seconds=30),
        }

    async def wait(**_kwargs: Any) -> dict[str, Any]:
        return {**stored, "status": "replied", "reply": "Prioritize the official report."}

    async def expire(**_kwargs: Any) -> bool:
        raise AssertionError("a prompt reply must not expire")

    host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        load_child=load_child,
        create_guidance=create,
        wait_guidance=wait,
        load_guidance=lambda **_kwargs: wait(),
        expire_guidance=expire,
    )
    tool = child_guidance_tools(host=host)[0]
    result = await tool.execute(
        AskParentInput(question="Which source?"),
        tool_runtime(tool_name="ask_parent", execution_scope=child_id),
    )

    assert "Prioritize the official report" in result.text_content
    assert stored["child_session_id"] == child_id
    assert stored["child_operation_id"] == operation_id
    assert stored["parent_session_id"] == parent_id.value
    assert stored["child_fencing_epoch"] == 7


async def test_wait_subagent_wakes_on_pending_question_without_settling() -> None:
    parent_id = SessionId.new()
    request_id = SessionId.new().value
    parked = asyncio.Event()
    asked = asyncio.Event()
    known_child: dict[str, str] = {}

    async def run_child(
        restored_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        parked.set()
        await asyncio.Event().wait()
        return ChildOutcome(
            status="succeeded", summary="unused", child_session_id=restored_id.value
        )

    async def list_guidance(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        child_id = known_child.get("id")
        if not asked.is_set() or child_id is None:
            return ()
        return (
            {
                "request_id": request_id,
                "child_session_id": child_id,
                "question": "Which source?",
            },
        )

    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        persist=AsyncMock(return_value=True),
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        list_guidance=list_guidance,
        context_snapshot=_context_snapshot(parent_id),
    )
    tools = {tool.name: tool for tool in subagent_tools(host=host)}
    spawned = await tools["spawn_agent"].execute(
        _spawn_input("investigate"),
        tool_runtime(call_id="wait-wake", tool_name="spawn_agent"),
    )
    assert spawned.details is not None
    spawned_id = str(spawned.details["children"][0]["child_session_id"])
    known_child["id"] = spawned_id
    await parked.wait()

    async def wait_for_child() -> Any:
        return await tools["wait_subagent"].execute(
            ChildControlInput(child_session_id=spawned_id),
            tool_runtime(tool_name="wait_subagent"),
        )

    waiter = asyncio.create_task(wait_for_child())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    asked.set()
    host.notify_parent()
    waited = await asyncio.wait_for(waiter, timeout=2)

    assert "running" in waited.text_content.lower()
    assert request_id in waited.text_content
    assert "Which source?" in waited.text_content
    assert not host.tasks[spawned_id].done()
    await host.stop(cancel=False)


async def test_continue_subagent_cannot_reauthorize_cancelled_children() -> None:
    captured: dict[str, Any] = {}

    async def continue_child(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"outcome": "reauthorization_required"}

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        owner_id="owner",
        continue_child=continue_child,
    )
    tool = next(item for item in subagent_tools(host=host) if item.name == "continue_subagent")
    result = await tool.execute(
        tool.input_model(child_session_id=SessionId.new().value, content="retry cancelled work"),
        tool_runtime(tool_name="continue_subagent"),
    )

    assert captured["reauthorize_user_cancelled"] is False
    assert captured["origin"] == "parent"
    assert "reauthorization_required" in result.text_content


def test_subagent_cancel_is_never_replayed_without_durable_reconciliation() -> None:
    tools = {tool.name: tool for tool in subagent_tools(host=SubagentHost())}
    assert tools["cancel_subagent"].replay_policy == "never"


def test_child_outcome_durable_payload_round_trips_evidence_state() -> None:
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }
    outcome = ChildOutcome(
        status="succeeded",
        summary="finding",
        handles=("[1] report.pdf",),
        usage={"input_tokens": 8},
        child_session_id="child-1",
        evidence_state=evidence_state,
    )

    assert ChildOutcome.from_durable_payload(outcome.durable_payload()) == outcome
    invalid = outcome.durable_payload()
    invalid["evidence_state"] = []
    with pytest.raises(ValueError, match="evidence state"):
        ChildOutcome.from_durable_payload(invalid)


async def test_notification_identity_tracks_child_operation_not_only_session() -> None:
    ledger = EvidenceLedger()
    parent_id = SessionId.new()
    child_id = SessionId.new().value
    parent_intent_id = IntentId.new().value
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "e1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }
    operation = {"id": "operation-1"}

    async def list_children(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        outcome = ChildOutcome(
            status="succeeded",
            summary=operation["id"],
            child_session_id=child_id,
            operation_id=operation["id"],
            evidence_state=evidence_state,
            usage={"input_tokens": 3},
        )
        return (
            {
                "child_session_id": child_id,
                "parent_call_id": "call",
                "parent_intent_id": parent_intent_id,
                "status": "succeeded",
                "host_state": {"terminal_outcome": outcome.durable_payload()},
            },
        )

    def merge(state: Any, child: str, call: str) -> tuple[str, ...]:
        before = len(ledger.contexts["chunks"])
        ledger.merge_child_state(state, child_session_id=child, parent_call_id=call)
        return tuple(ledger.citation_handles(after_chunk_count=before))

    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        list_children=list_children,
        merge_evidence=merge,
    )
    seen: set[str] = set()
    first = await host.completed_dispatch_notifications(seen=seen)
    seen.add(first[0][0])
    replay = await host.completed_dispatch_notifications(seen=seen)
    operation["id"] = "operation-2"
    continued = await host.completed_dispatch_notifications(seen=seen)

    assert first and not replay and continued
    assert first[0][0] != continued[0][0]
    assert len(ledger.contexts["chunks"]) == 1


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
    controls = {
        "subagent_status",
        "wait_subagent",
        "cancel_subagent",
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
    }
    assert {"spawn_agent", *controls} <= {tool.name for tool in parent}
    assert {"search_knowledge_base", "ask_parent"} == {tool.name for tool in child}
    assert not ({"spawn_agent"} | controls) & {tool.name for tool in child}


async def test_spawn_many_runs_in_parallel_and_aggregates_usage() -> None:
    started = 0
    both_started = asyncio.Event()

    async def run_child(
        child_id: SessionId,
        request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
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
        context_snapshot=_context_snapshot(),
        async_lifecycle=False,
    )
    result = await subagent_tools(host=host)[0].execute(
        SpawnAgentInput(
            children=(
                ChildRequest(objective="one"),
                ChildRequest(objective="two"),
            )
        ),
        tool_runtime(call_id="parallel", tool_name="spawn_agent"),
    )

    assert result.details is not None
    assert len(result.details["children"]) == 2
    assert result.details["inclusive_usage"] == {"input_tokens": 4}
    assert not host.tasks


async def test_async_spawn_persists_full_envelope_before_returning_handles() -> None:
    release = asyncio.Event()
    persisted: list[dict[str, Any]] = []

    async def persist(**kwargs: Any) -> bool:
        persisted.append(kwargs)
        return True

    async def run_child(
        child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        await release.wait()
        return ChildOutcome(
            status="succeeded",
            summary="late result",
            child_session_id=child_id.value,
            operation_id="operation-1",
        )

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        persist=persist,
        finish_child=AsyncMock(return_value=True),
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    spawn = subagent_tools(host=host)[0]
    result = await spawn.execute(
        _spawn_input("investigate"),
        tool_runtime(call_id="async-call", tool_name="spawn_agent"),
    )

    assert result.details is not None
    child_id = result.details["children"][0]["child_session_id"]
    assert result.details["children"][0]["status"] == "running"
    assert child_id in host.tasks and not host.tasks[child_id].done()
    assert persisted[0]["parent_intent_id"]
    assert persisted[0]["context_snapshot"]["parent_entry_id"]
    assert persisted[0]["objective"] == "investigate"
    assert persisted[0]["plan"] == {"schema_version": 2, "tools": []}
    assert persisted[0]["budget"] == {"provider_attempt_limit": 2}

    release.set()
    waited = await subagent_tools(host=host)[2].execute(
        ChildControlInput(child_session_id=child_id),
        tool_runtime(tool_name="wait_subagent"),
    )
    assert "late result" in waited.text_content
    await host.stop(cancel=False)


async def test_async_restore_reconstructs_a_pending_child_from_durable_envelope() -> None:
    parent_id = SessionId.new()
    child_id = SessionId.new()
    snapshot = _context_snapshot(parent_id)
    row: dict[str, Any] = {
        "child_session_id": child_id.value,
        "parent_session_id": parent_id.value,
        "parent_call_id": "call-restart",
        "parent_intent_id": IntentId.new().value,
        "status": "running",
        "objective": "resume",
        "context": "isolated",
        "model_role": "query",
        "tools": None,
        "context_snapshot": snapshot.canonical_payload(),
    }

    async def list_children(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        return (row,)

    async def run_child(
        restored_id: SessionId,
        request: ChildRequest,
        call_id: str,
        restored_snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        assert restored_id == child_id
        assert request.objective == "resume"
        assert call_id == "call-restart"
        assert restored_snapshot == snapshot
        row["status"] = "succeeded"
        return ChildOutcome(
            status="succeeded",
            summary="recovered",
            child_session_id=restored_id.value,
            operation_id="operation-recovered",
        )

    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        list_children=list_children,
        load_child=AsyncMock(return_value=None),
        finish_child=AsyncMock(return_value=True),
        run_child=run_child,
    )

    await host.restore_pending()
    await host.wait_for_activity()

    assert host.outcomes[child_id.value].summary == "recovered"
    await host.stop(cancel=False)


async def test_async_child_exception_terminalizes_without_failing_sibling() -> None:
    finished: dict[str, str] = {}

    async def run_child(
        child_id: SessionId,
        request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        if request.objective == "fails":
            raise RuntimeError("provider exploded")
        return ChildOutcome(
            status="succeeded",
            summary="sibling survived",
            child_session_id=child_id.value,
            operation_id=f"operation-{request.objective}",
        )

    async def finish_child(**kwargs: Any) -> bool:
        finished[kwargs["child_session_id"]] = kwargs["status"]
        return True

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        persist=AsyncMock(return_value=True),
        finish_child=finish_child,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    result = await subagent_tools(host=host)[0].execute(
        SpawnAgentInput(
            children=(ChildRequest(objective="fails"), ChildRequest(objective="survives"))
        ),
        tool_runtime(call_id="siblings", tool_name="spawn_agent"),
    )
    assert result.details is not None
    child_ids = [item["child_session_id"] for item in result.details["children"]]
    await asyncio.gather(*(host.tasks[child_id] for child_id in child_ids))

    assert sorted(finished.values()) == ["failed", "succeeded"]
    assert any(outcome.summary == "sibling survived" for outcome in host.outcomes.values())
    await host.stop(cancel=False)


async def test_process_detach_does_not_terminalize_child_as_cancelled() -> None:
    started = asyncio.Event()

    async def run_child(
        child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        started.set()
        await asyncio.Event().wait()
        return ChildOutcome(status="succeeded", summary="unused", child_session_id=child_id.value)

    finish = AsyncMock(return_value=True)
    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        persist=AsyncMock(return_value=True),
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    await subagent_tools(host=host)[0].execute(
        _spawn_input("survive restart"),
        tool_runtime(call_id="detach", tool_name="spawn_agent"),
    )
    await started.wait()

    await host.stop(cancel=False)

    finish.assert_not_awaited()


def test_legacy_foreground_tool_contract_is_exactly_preserved() -> None:
    legacy = {tool.name: tool for tool in subagent_tools(host=SubagentHost(async_lifecycle=False))}
    slice1 = {
        tool.name: tool
        for tool in subagent_tools(
            host=SubagentHost(async_lifecycle=True, interactive_controls=False)
        )
    }
    current = {tool.name: tool for tool in subagent_tools(host=SubagentHost())}

    assert legacy["spawn_agent"].contract_version == 2
    assert legacy["spawn_agent"].input_schema_digest == (
        "758524c627795a4fdaa920302a8f58163e85641632d85d27880fc831bf26c13e"
    )
    assert legacy["spawn_agent"].description == (
        "Run one or many foreground child Agent Sessions and wait for all results."
    )
    assert slice1["spawn_agent"].contract_version == 3
    assert slice1["spawn_agent"].description == current["spawn_agent"].description
    assert (
        not {
            "steer_subagent",
            "continue_subagent",
            "reply_subagent",
        }
        & slice1.keys()
    )
    assert current["spawn_agent"].contract_version == 4
    assert {
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
    } <= current.keys()
    assert current["spawn_agent"].input_schema_digest == legacy["spawn_agent"].input_schema_digest


async def test_spawn_checks_parent_cancellation_before_starting_children() -> None:
    cancelled = AsyncMock(side_effect=RunCancellationObserved)
    runner = AsyncMock()
    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        check_cancelled=cancelled,
        run_child=runner,
    )

    with pytest.raises(asyncio.CancelledError):
        await subagent_tools(host=host)[0].execute(
            _spawn_input("must not start"), tool_runtime(tool_name="spawn_agent")
        )

    cancelled.assert_awaited()
    runner.assert_not_awaited()


async def test_spawn_propagates_parent_cancel_and_finishes_persisted_child() -> None:
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        raise RunCancellationObserved

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=SessionId.new().value,
        persist=AsyncMock(),
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
        async_lifecycle=False,
    )

    with pytest.raises(asyncio.CancelledError):
        await subagent_tools(host=host)[0].execute(
            _spawn_input("cancel in flight"), tool_runtime(tool_name="spawn_agent")
        )

    assert finish.await_args is not None
    assert finish.await_args.kwargs["status"] == "cancelled"
    assert not host.tasks


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
        ChildControlInput(child_session_id="child-1"),
        tool_runtime(tool_name="cancel_subagent"),
    )

    assert task.cancelled()
    assert "[cancelled]" in result.text_content
    assert host.outcomes["child-1"].status == "cancelled"


async def test_terminal_persisted_spawn_replay_never_reenters_child_execution() -> None:
    persist = AsyncMock()
    finish = AsyncMock()
    run_child = AsyncMock()
    ledger = EvidenceLedger()
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "persisted finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    def remerge_evidence(state: Any, child_id: str, call_id: str) -> tuple[str, ...]:
        before = len(ledger.contexts["chunks"])
        ledger.merge_child_state(
            state,
            child_session_id=child_id,
            parent_call_id=call_id,
        )
        return tuple(ledger.citation_handles(after_chunk_count=before))

    merge_evidence = MagicMock(side_effect=remerge_evidence)

    async def load_child(**kwargs: Any) -> dict[str, Any]:
        child_id = kwargs["child_session_id"]
        return {
            "status": "succeeded",
            "summary": "Persisted child finding.",
            "host_state": {
                "terminal_outcome": {
                    "status": "succeeded",
                    "summary": "Persisted child finding.",
                    "handles": ["[1] report.pdf"],
                    "usage": {"input_tokens": 8},
                    "child_session_id": child_id,
                    "evidence_state": evidence_state,
                }
            },
        }

    host = SubagentHost(
        parent_session_id=SessionId.new(),
        run_id=str(SessionId.new().value),
        owner_id="owner",
        load_child=load_child,
        persist=persist,
        finish_child=finish,
        run_child=run_child,
        context_snapshot=_context_snapshot(),
        merge_evidence=merge_evidence,
        async_lifecycle=False,
    )
    tool = subagent_tools(host=host)[0]
    runtime = tool_runtime(call_id="call-1", tool_name="spawn_agent")
    result = await tool.execute(_spawn_input("what happened?"), runtime)
    replayed = await tool.execute(_spawn_input("what happened?"), runtime)

    assert "Persisted child finding." in result.text_content
    assert "[1] report.pdf" in result.text_content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 8}
    assert result.details["children"][0]["evidence_handles"] == ["[1] report.pdf"]
    assert replayed.details is not None
    assert replayed.details["children"][0]["evidence_handles"] == ["[1] report.pdf"]
    assert len(ledger.contexts["chunks"]) == 1
    assert merge_evidence.call_count == 2
    assert all(
        call.args
        == (
            evidence_state,
            result.details["children"][0]["child_session_id"],
            "call-1",
        )
        for call in merge_evidence.call_args_list
    )
    persist.assert_not_awaited()
    finish.assert_not_awaited()
    run_child.assert_not_awaited()


async def test_spawn_reports_child_outcome_and_usage() -> None:
    persist = AsyncMock()
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
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
        context_snapshot=_context_snapshot(),
        async_lifecycle=False,
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("summarize filings"),
        tool_runtime(call_id="call-9", tool_name="spawn_agent"),
    )
    assert "Child summary." in result.text_content
    assert "[1] Page A [resource: res-a]" in result.text_content
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
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
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
        context_snapshot=_context_snapshot(),
        merge_evidence=adopted,
        async_lifecycle=False,
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("find source"),
        tool_runtime(call_id="call-adopt", tool_name="spawn_agent"),
    )

    assert "[2] child source" in result.text_content
    adopted.assert_called_once()
    assert adopted.call_args.args[0] == child_state
    assert adopted.call_args.args[2] == "call-adopt"


async def test_failed_child_is_recorded_failed() -> None:
    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
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
        context_snapshot=_context_snapshot(),
        async_lifecycle=False,
    )
    tool = subagent_tools(host=host)[0]
    result = await tool.execute(
        _spawn_input("x"),
        tool_runtime(call_id="call-err", tool_name="spawn_agent"),
    )
    assert result.details is not None
    assert result.details["children"][0]["status"] == "failed"
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "failed"


@dataclass
class _FakeSession:
    run_id: str
    owner_id: str = "owner"
    execution: Any = field(default_factory=lambda: SimpleNamespace(fencing_epoch=1))

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, _phase: str) -> None:
        return None

    async def emit_tool_event(self, _event_type: str, _payload: object) -> None:
        return None


def _child_orchestrator(
    model_func: Any,
    *,
    environment: Any = None,
    retrieve_func: Any = None,
) -> AnswerOrchestrator:
    profile = answer_model_profile()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("child should not search in this test")

    return AnswerOrchestrator(
        synthesizer=AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=profile,
        ),
        retrieve_knowledge_base=retrieve_func or retrieve,
        search_web=None,
        model_func=model_func,
        telemetry=NOOP_TELEMETRY,
        model_profile=profile,
        text_window_budget=TextWindowBudget(CONTEXT_POLICY.hard_input_limit(profile)),
        subagent_host=SubagentHost(),
        resolved_mode="research",
        environment=environment,
    )


async def test_child_session_persists_and_replays_without_rerun() -> None:
    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        return AssistantTurn(
            text="Persisted child summary.",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"input_tokens": 3, "output_tokens": 2},
        )

    orchestrator = _child_orchestrator(model)
    repository = InMemoryAgentSessionRepository()
    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:test:1")
    session = _FakeSession(run_id=str(parent_id.value))

    first = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
    )
    assert first.status == "succeeded"
    assert first.summary == "Persisted child summary."
    assert first.usage == {"input_tokens": 3, "output_tokens": 2}
    assert calls["n"] == 1
    snapshot = await repository.load(child_id)
    assert snapshot.commit_sequence >= 2
    assert [entry.entry_type for entry in snapshot.entries] == [
        "user_message",
        "assistant_message",
    ]
    assert any(record.ref.kind == "operation_state" for record in snapshot.registers)

    second = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="summarize filings"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
    )
    assert second.summary == first.summary
    assert second.usage == first.usage
    assert second.handles == first.handles
    assert second.status == first.status
    assert calls["n"] == 1


async def test_child_continuation_accepts_a_new_operation_in_the_same_session() -> None:
    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        return AssistantTurn(text=f"answer {calls['n']}", tool_calls=(), stop_reason="stop")

    orchestrator = _child_orchestrator(model)
    repository = InMemoryAgentSessionRepository()
    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=parent_id.value, name="child:continuation")
    snapshot = _context_snapshot(parent_id)
    row: dict[str, Any] = {}

    async def persist(**kwargs: Any) -> bool:
        row.setdefault("plan", kwargs["plan"])
        return True

    async def load(**_kwargs: Any) -> dict[str, Any]:
        return dict(row)

    initial_key = f"child-session:{child_id.value}"
    row.update(
        operation_key=initial_key,
        operation_id=OperationId.deterministic(idempotency_key=initial_key).value,
    )
    first = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=_FakeSession(run_id=parent_id.value),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="initial objective"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=snapshot,
        persist_child_runtime=persist,
        claim_child=AsyncMock(return_value=1),
        load_child=load,
    )

    continuation_key = "continuation-one"
    row.update(
        operation_key=continuation_key,
        operation_id=OperationId.deterministic(idempotency_key=continuation_key).value,
    )
    second = await run_child_session(
        orchestrator=orchestrator,
        repository=repository,  # type: ignore[arg-type]
        session=_FakeSession(run_id=parent_id.value),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="follow-up objective"),
        parent_call_id="call-1",
        parent_session_id=parent_id,
        context_snapshot=snapshot,
        persist_child_runtime=persist,
        claim_child=AsyncMock(return_value=1),
        load_child=load,
    )

    current = await repository.load(child_id)
    assert first.operation_id != second.operation_id
    assert second.operation_id == row["operation_id"]
    assert calls["n"] == 2
    assert [
        getattr(entry, "content", None)
        for entry in current.entries
        if entry.entry_type == "user_message"
    ] == ["initial objective", "follow-up objective"]
    assert row["plan"]


async def test_child_renews_its_lease_while_a_provider_call_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlightrag.engine.answer.research.runtime._CHILD_LEASE_HEARTBEAT_SECONDS", 0.005
    )

    async def model(**_kwargs: object) -> AssistantTurn:
        await asyncio.sleep(0.03)
        return AssistantTurn(text="renewed child", tool_calls=(), stop_reason="stop")

    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:renew")
    renew_child = AsyncMock(return_value=True)
    outcome = await run_child_session(
        orchestrator=_child_orchestrator(model),
        repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
        session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="wait for a slow provider"),
        parent_call_id="call-renew",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
        renew_child=renew_child,
    )

    assert outcome.status == "succeeded"
    assert renew_child.await_count >= 1
    assert renew_child.await_args is not None
    assert renew_child.await_args.kwargs == {
        "child_session_id": child_id.value,
        "child_fencing_epoch": 1,
    }


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
        ),
        context_snapshot=ChildContextSnapshot.from_values(
            parent_session_id=SessionId.new(),
            parent_entry_id=EntryId.new(),
            depth=0,
            messages=[
                {"role": "user", "content": "older question"},
                {"role": "assistant", "content": "older answer"},
                {"role": "user", "content": "parent question"},
            ],
        ),
    )

    messages = await child.context.control_turn(
        evidence=child.evidence,
        working=WorkingContextProjection(retained_tail_tokens=1000),
        tool_schema_tokens=0,
    )

    assert [tool.name for tool in child.tools] == ["search_knowledge_base", "ask_parent"]
    assert any(message.get("content") == "older answer" for message in messages)
    assert any(message.get("content") == "parent question" for message in messages)


def test_child_defaults_to_read_only_parent_tools() -> None:
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        child=True,
    )
    names = {tool.name for tool in child}
    assert names >= {"search_knowledge_base", "read", "grep", "find", "ls"}
    assert not {"spawn_agent", "attach_artifact", "write", "edit", "bash"} & names


def test_child_can_explicitly_narrow_to_host_permitted_side_effect_tools() -> None:
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        child=True,
        tool_names=("read", "write"),
    )

    assert {tool.name for tool in child} == {"read", "write"}


def test_interactive_child_keeps_ask_parent_on_an_explicit_tool_subset() -> None:
    host = SubagentHost()
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        subagent_host=host,
        child=True,
        tool_names=("read", "write"),
    )
    slice1 = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        subagent_host=SubagentHost(async_lifecycle=True, interactive_controls=False),
        child=True,
    )

    assert {tool.name for tool in child} == {"read", "write", "ask_parent"}
    assert "ask_parent" not in {tool.name for tool in slice1}


async def test_cancelled_child_closes_pending_intent_before_terminal() -> None:
    from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
    from dlightrag.engine.agent.session.operation import OperationCancelled
    from dlightrag.engine.agent.session.registers import OperationStateRegister
    from dlightrag.engine.ai.messages import ToolCall

    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(
                    id="search-cancelled",
                    name="search_knowledge_base",
                    arguments={"query": "cancel me"},
                ),
            ),
            stop_reason="tool_use",
        )

    async def cancel_during_search(_query: str) -> Any:
        raise asyncio.CancelledError

    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="pending-cancel")
    repository = InMemoryAgentSessionRepository()
    with pytest.raises(asyncio.CancelledError):
        await run_child_session(
            orchestrator=_child_orchestrator(model, retrieve_func=cancel_during_search),
            repository=repository,  # type: ignore[arg-type]
            session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
            fetched_buffer=FetchedResourceBuffer(),
            child_id=child_id,
            request=ChildRequest(objective="cancel while searching"),
            parent_call_id="call-pending",
            parent_session_id=parent_id,
            context_snapshot=_context_snapshot(parent_id),
            persist_child_runtime=AsyncMock(),
            claim_child=AsyncMock(return_value=1),
        )

    snapshot = await repository.load(child_id)
    result = next(entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry))
    assert result.result.call_id == "search-cancelled"
    assert result.result.outcome == "outcome_unknown"
    state = next(
        record.value.state
        for record in snapshot.registers
        if isinstance(record.value, OperationStateRegister)
    )
    assert isinstance(state, OperationCancelled)


async def test_parent_cancel_marks_the_child_cancelled() -> None:
    async def model(**_kwargs: object) -> AssistantTurn:
        raise AssertionError("cancelled child must not call the model")

    class _CancelSession(_FakeSession):
        async def check_cancelled(self) -> None:
            raise RunCancellationObserved

    parent_id = SessionId.new()
    with pytest.raises(RunCancellationObserved):
        await run_child_session(
            orchestrator=_child_orchestrator(model),
            repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
            session=_CancelSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
            fetched_buffer=FetchedResourceBuffer(),
            child_id=SessionId.deterministic(run_id=str(parent_id.value), name="c"),
            request=ChildRequest(objective="stop"),
            parent_call_id="call-c",
            parent_session_id=parent_id,
            context_snapshot=_context_snapshot(parent_id),
            persist_child_runtime=AsyncMock(),
            claim_child=AsyncMock(return_value=1),
        )
