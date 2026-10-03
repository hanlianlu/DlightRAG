# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable asynchronous Child Agent lifecycle tests."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.session.ids import EntryId, IntentId, OperationId, SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.ai.capacity import CONTEXT_POLICY
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.scheduler import ModelScheduler, model_call_scope
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
    child_guidance_declarations,
    child_guidance_tools,
    child_session_id,
    subagent_declarations,
    subagent_tools,
)
from dlightrag.engine.runtime.coordinator import LeaseLostError, RunCancellationObserved
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


async def test_ask_parent_expiry_race_with_reply_does_not_treat_false_as_lease_loss() -> None:
    parent_id = SessionId.new()
    child_id = SessionId.new().value
    operation_id = SessionId.new().value
    expired_at = datetime.now(UTC) - timedelta(seconds=1)

    async def load_child(**_kwargs: Any) -> dict[str, Any]:
        return {"operation_id": operation_id, "fencing_epoch": 3}

    async def create(**kwargs: Any) -> dict[str, Any]:
        return {
            **kwargs,
            "status": "pending",
            "expires_at": expired_at,
        }

    async def expire(**_kwargs: Any) -> bool:
        return False

    async def load_guidance(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "replied",
            "reply": "Use the report.",
            "reply_origin": "user",
            "expires_at": expired_at,
        }

    host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        load_child=load_child,
        create_guidance=create,
        wait_guidance=AsyncMock(side_effect=AssertionError("reply race must not wait")),
        load_guidance=load_guidance,
        expire_guidance=expire,
    )
    result = await child_guidance_tools(host=host)[0].execute(
        AskParentInput(question="Which source?"),
        tool_runtime(tool_name="ask_parent", execution_scope=child_id),
    )

    assert "Use the report" in result.text_content
    assert result.is_error is False


async def test_ask_parent_expiry_false_while_pending_is_lease_loss() -> None:
    parent_id = SessionId.new()
    child_id = SessionId.new().value
    operation_id = SessionId.new().value
    expired_at = datetime.now(UTC) - timedelta(seconds=1)

    async def load_child(**_kwargs: Any) -> dict[str, Any]:
        return {"operation_id": operation_id, "fencing_epoch": 3}

    async def create(**kwargs: Any) -> dict[str, Any]:
        return {**kwargs, "status": "pending", "expires_at": expired_at}

    async def expire(**_kwargs: Any) -> bool:
        return False

    async def load_guidance(**kwargs: Any) -> dict[str, Any]:
        return {
            "request_id": kwargs["request_id"],
            "status": "pending",
            "expires_at": expired_at,
        }

    host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        load_child=load_child,
        create_guidance=create,
        wait_guidance=AsyncMock(side_effect=AssertionError("lost fence must not wait")),
        load_guidance=load_guidance,
        expire_guidance=expire,
    )
    with pytest.raises(LeaseLostError):
        await child_guidance_tools(host=host)[0].execute(
            AskParentInput(question="Which source?"),
            tool_runtime(tool_name="ask_parent", execution_scope=child_id),
        )


async def test_ask_parent_wait_does_not_hold_model_scheduler_active() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    waiting = asyncio.Event()
    release = asyncio.Event()
    parent_id = SessionId.new()
    child_id = SessionId.new().value
    operation_id = SessionId.new().value

    async def load_child(**_kwargs: Any) -> dict[str, Any]:
        return {"operation_id": operation_id, "fencing_epoch": 1}

    async def create(**kwargs: Any) -> dict[str, Any]:
        return {
            **kwargs,
            "status": "pending",
            "expires_at": datetime.now(UTC) + timedelta(seconds=30),
        }

    async def wait(**_kwargs: Any) -> dict[str, Any]:
        waiting.set()
        await release.wait()
        return {
            "status": "replied",
            "reply": "ok",
            "reply_origin": "parent",
            "expires_at": datetime.now(UTC) + timedelta(seconds=30),
        }

    host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        load_child=load_child,
        create_guidance=create,
        wait_guidance=wait,
        load_guidance=wait,
        expire_guidance=AsyncMock(side_effect=AssertionError("wait path must not expire")),
    )

    async def _ask() -> Any:
        return await child_guidance_tools(host=host)[0].execute(
            AskParentInput(question="Which source?"),
            tool_runtime(tool_name="ask_parent", execution_scope=child_id),
        )

    asked = asyncio.create_task(_ask())
    await waiting.wait()
    assert scheduler._active == 0

    other_ran = asyncio.Event()

    async def other() -> str:
        other_ran.set()
        return "ok"

    with model_call_scope("other-run"):
        assert await asyncio.wait_for(scheduler.run(other), timeout=1) == "ok"
    assert other_ran.is_set()
    release.set()
    result = await asked
    assert "ok" in result.text_content


async def test_has_running_children_ignores_sparse_precreate_rows() -> None:
    parent_id = SessionId.new()
    sparse = {
        "child_session_id": SessionId.new().value,
        "status": "running",
        "context_snapshot": {},
        "parent_call_id": "sparse",
        "objective": "not yet reconstructible",
        "context": "isolated",
        "model_role": "query",
    }
    full = {
        "child_session_id": SessionId.new().value,
        "status": "running",
        "parent_call_id": "full",
        "objective": "inspect",
        "context": "isolated",
        "model_role": "query",
        "context_snapshot": {
            "parent_session_id": parent_id.value,
            "parent_entry_id": EntryId.new().value,
            "depth": 0,
            "messages": [],
        },
    }

    async def list_sparse(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        return (sparse,)

    async def list_mixed(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        return (sparse, full)

    sparse_host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        list_children=list_sparse,
    )
    mixed_host = SubagentHost(
        parent_session_id=parent_id,
        owner_id="owner",
        run_id=SessionId.new().value,
        list_children=list_mixed,
    )

    assert await sparse_host.has_running_children() is False
    assert await mixed_host.has_running_children() is True


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

    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        owner_id="owner",
        list_children=list_children,
    )
    ledger = _parent_ledger(host)
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
        injected_tools=[],
        register_web_source=None,
        subagent_host=host,
        search_toolchain=SearchToolchain(),
    )
    child = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        subagent_host=host,
        child=True,
        search_toolchain=SearchToolchain(),
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


async def test_spawn_many_runs_children_in_parallel() -> None:
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
            child_session_id=child_id.value,
        )

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
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
    child_ids = [item["child_session_id"] for item in result.details["children"]]
    outcomes = await asyncio.gather(*(host.tasks[child_id] for child_id in child_ids))
    assert [outcome.summary for outcome in outcomes] == ["one", "two"]
    await host.stop(cancel=False)


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


def test_child_contracts_keep_the_digests_accepted_plans_pinned() -> None:
    def digest(declarations: Any) -> str:
        return AgentRunPlan.from_tools(
            declarations, model_role="query", context_policy_revision="pin"
        ).digest

    tools = {tool.name: tool for tool in subagent_tools(host=SubagentHost())}

    assert set(tools) == {
        "spawn_agent",
        "subagent_status",
        "wait_subagent",
        "cancel_subagent",
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
    }
    assert all(tool.contract_version == 5 for tool in tools.values())
    assert tools["spawn_agent"].input_schema_digest == (
        "ea2283b72d9dce2e740dcc58abbb136c041256381956de11f00cd6aff1b65ccd"
    )
    # Accepted Plans pin every declaration byte for byte, including the
    # model-role guidance suffix and the Child's ask_parent contract.
    assert digest(subagent_declarations()) == (
        "8e561933d1575be1808de0aaad55dd79abe985daf7b379e4c1e39db4f53ebf10"
    )
    assert digest(
        subagent_declarations(model_guidance="Choose model_role query for most children.")
    ) == ("839cb61df3da40722b105ee24fa64d1c60f96d36ec91e1240102edd51af48037")
    assert digest(child_guidance_declarations()) == (
        "496953e5dc7bb851b75500a09fdb85213983f829aee2e046edc59d7c989e1ecb"
    )


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

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=SessionId.new().value,
        persist=AsyncMock(),
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    result = await subagent_tools(host=host)[0].execute(
        _spawn_input("cancel in flight"), tool_runtime(tool_name="spawn_agent")
    )
    assert result.details is not None
    child_id = result.details["children"][0]["child_session_id"]

    with pytest.raises(asyncio.CancelledError):
        await host.tasks[child_id]

    assert finish.await_args is not None
    assert finish.await_args.kwargs["status"] == "cancelled"


async def test_cancel_tool_joins_a_known_in_process_child() -> None:
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
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "persisted finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    async def load_child(**kwargs: Any) -> dict[str, Any]:
        child_id = kwargs["child_session_id"]
        return {
            "status": "succeeded",
            "summary": "Persisted child finding.",
            "parent_call_id": "call-1",
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

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=str(SessionId.new().value),
        owner_id="owner",
        load_child=load_child,
        persist=persist,
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    ledger = _parent_ledger(host)
    spawn, _status, wait = subagent_tools(host=host)[:3]
    runtime = tool_runtime(call_id="call-1", tool_name="spawn_agent")
    waits: list[Any] = []
    for _ in range(2):
        spawned = await spawn.execute(_spawn_input("what happened?"), runtime)
        assert spawned.details is not None
        child_id = spawned.details["children"][0]["child_session_id"]
        assert (await host.tasks[child_id]).summary == "Persisted child finding."
        waits.append(
            await wait.execute(
                ChildControlInput(child_session_id=child_id),
                tool_runtime(tool_name="wait_subagent"),
            )
        )

    handles: list[list[str]] = []
    for waited in waits:
        assert "Persisted child finding." in waited.text_content
        assert waited.details is not None
        assert waited.details["inclusive_usage"] == {"input_tokens": 8}
        handles.append(waited.details["children"][0]["evidence_handles"])
    # Adoption is idempotent: the replayed finding enters the parent ledger once,
    # labelled with the call that dispatched it, and both reads cite it identically.
    (row,) = ledger.contexts["chunks"]
    assert row["_child_lineage"]["parent_call_id"] == "call-1"
    assert handles[0] == handles[1] == ledger.citation_handles()
    run_child.assert_not_awaited()
    finish.assert_not_awaited()


@pytest.mark.parametrize("path", ["notification", "status", "wait", "cancel"])
async def test_every_adoption_path_labels_child_evidence_with_the_dispatching_call(
    path: str,
) -> None:
    parent_id = SessionId.new()
    child_id = child_session_id(
        run_id="run", parent_session_id=parent_id, parent_intent_id=IntentId.new()
    ).value
    evidence_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }
    outcome = ChildOutcome(
        status="succeeded",
        summary="Child finding.",
        child_session_id=child_id,
        evidence_state=evidence_state,
    )
    row = {
        "child_session_id": child_id,
        "parent_call_id": "call-dispatch",
        "parent_intent_id": "intent-1",
        "status": "succeeded",
        "host_state": {"terminal_outcome": outcome.durable_payload()},
    }
    merge_evidence = MagicMock(return_value=("[1] finding",))

    async def load_child(**_kwargs: Any) -> dict[str, Any]:
        return row

    async def list_children(**_kwargs: Any) -> tuple[dict[str, Any], ...]:
        return (row,)

    host = SubagentHost(
        parent_session_id=parent_id,
        run_id="run",
        owner_id="owner",
        load_child=load_child,
        list_children=list_children,
        merge_evidence=merge_evidence,
    )
    if path == "notification":
        assert await host.completed_dispatch_notifications(seen=set())
    else:
        tool = {"status": 1, "wait": 2, "cancel": 3}[path]
        await subagent_tools(host=host)[tool].execute(
            ChildControlInput(child_session_id=child_id),
            tool_runtime(call_id="call-control", tool_name=f"{path}_subagent"),
        )

    merge_evidence.assert_called_once_with(evidence_state, child_id, "call-dispatch")


async def test_wait_reports_child_outcome_and_usage() -> None:
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

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=persist,
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    spawn, _status, wait = subagent_tools(host=host)[:3]
    spawned = await spawn.execute(
        _spawn_input("summarize filings"),
        tool_runtime(call_id="call-9", tool_name="spawn_agent"),
    )
    assert spawned.details is not None
    child_id = spawned.details["children"][0]["child_session_id"]
    await host.tasks[child_id]
    result = await wait.execute(
        ChildControlInput(child_session_id=child_id), tool_runtime(tool_name="wait_subagent")
    )

    assert "Child summary." in result.text_content
    assert "[1] Page A [resource: res-a]" in result.text_content
    assert result.details is not None
    assert result.details["inclusive_usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result.details["children"][0]["status"] == "succeeded"
    persist.assert_awaited()
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "succeeded"


async def test_wait_adopts_child_evidence_into_the_parent() -> None:
    adopted = MagicMock(return_value=("[2] child source",))
    child_state = {
        "contexts": {
            "chunks": [{"chunk_id": "c1", "content": "finding"}],
            "entities": [],
            "relationships": [],
        }
    }

    async def run_child(
        child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return ChildOutcome(
            status="succeeded",
            summary="Child summary.",
            child_session_id=child_id.value,
            evidence_state=child_state,
        )

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=str(SessionId.new().value),
        owner_id="owner",
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
        merge_evidence=adopted,
    )
    spawn, _status, wait = subagent_tools(host=host)[:3]
    spawned = await spawn.execute(
        _spawn_input("find source"),
        tool_runtime(call_id="call-adopt", tool_name="spawn_agent"),
    )
    assert spawned.details is not None
    child_id = spawned.details["children"][0]["child_session_id"]
    await host.tasks[child_id]
    result = await wait.execute(
        ChildControlInput(child_session_id=child_id), tool_runtime(tool_name="wait_subagent")
    )

    assert "[2] child source" in result.text_content
    adopted.assert_called_once_with(child_state, child_id, "call-adopt")


async def test_failed_child_is_recorded_failed() -> None:
    finish = AsyncMock()

    async def run_child(
        child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return ChildOutcome(
            status="failed", summary="provider down", child_session_id=child_id.value
        )

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=AsyncMock(),
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    spawned = await subagent_tools(host=host)[0].execute(
        _spawn_input("x"),
        tool_runtime(call_id="call-err", tool_name="spawn_agent"),
    )
    assert spawned.details is not None
    child_id = spawned.details["children"][0]["child_session_id"]

    assert (await host.tasks[child_id]).status == "failed"
    assert finish.call_args is not None
    assert finish.call_args.kwargs["status"] == "failed"


async def test_a_child_that_names_impossible_tools_says_which() -> None:
    """The parent model named the Tools, so the failure has to name them back."""
    from dlightrag.engine.answer.errors import ChildToolNarrowingError

    finish = AsyncMock()

    async def run_child(
        _child_id: SessionId,
        _request: ChildRequest,
        _call_id: str,
        _snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        raise ChildToolNarrowingError(("no_such_tool",), reason="this Run offers no such Tool")

    parent_id = SessionId.new()
    host = SubagentHost(
        parent_session_id=parent_id,
        run_id=str(SessionId.new().value),
        owner_id="owner",
        persist=AsyncMock(),
        finish_child=finish,
        prepare_dispatch=_durable_dispatch,
        run_child=run_child,
        context_snapshot=_context_snapshot(parent_id),
    )
    spawn, _status, wait = subagent_tools(host=host)[:3]
    spawned = await spawn.execute(
        _spawn_input("x"),
        tool_runtime(call_id="call-narrow", tool_name="spawn_agent"),
    )
    assert spawned.details is not None
    child_id = spawned.details["children"][0]["child_session_id"]
    await host.tasks[child_id]
    result = await wait.execute(
        ChildControlInput(child_session_id=child_id), tool_runtime(tool_name="wait_subagent")
    )

    assert result.details is not None
    assert result.details["children"][0]["status"] == "failed"
    assert "no_such_tool" in result.text_content
    assert finish.call_args is not None
    assert "no_such_tool" in finish.call_args.kwargs["summary"]


@dataclass
class _FakeSession:
    fencing_epoch = 1
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
    retrieve_func: Any = None,
    subagent_host: SubagentHost | None = None,
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
        subagent_host=SubagentHost() if subagent_host is None else subagent_host,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )


def _parent_ledger(host: SubagentHost) -> EvidenceLedger:
    """The ledger of a parent Run whose orchestrator merges ``host``'s Child outcomes."""

    async def model(**_kwargs: object) -> AssistantTurn:
        raise AssertionError("the parent's model is not called")

    return _child_orchestrator(model, subagent_host=host).prepare_run("parent question").evidence


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
        telemetry=NOOP_TELEMETRY,
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
        telemetry=NOOP_TELEMETRY,
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
        telemetry=NOOP_TELEMETRY,
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
        telemetry=NOOP_TELEMETRY,
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

    renewed = asyncio.Event()

    async def heartbeat(
        *, owner_id: str, run_id: str, child_session_id: str, child_fencing_epoch: int
    ) -> bool:
        renewed.set()
        return True

    async def model(**_kwargs: object) -> AssistantTurn:
        await asyncio.wait_for(renewed.wait(), 1)
        return AssistantTurn(text="renewed child", tool_calls=(), stop_reason="stop")

    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:renew")
    renew_child = AsyncMock(side_effect=heartbeat)
    outcome = await run_child_session(
        telemetry=NOOP_TELEMETRY,
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
        "owner_id": "owner",
        "run_id": parent_id.value,
        "child_session_id": child_id.value,
        "child_fencing_epoch": 1,
    }


async def test_a_child_tool_settlement_promotes_the_parent_runs_notes() -> None:
    """A Child writes into the parent Run's working copy, so it promotes the same plane.

    ADR 0022 promotes at Tool settlement, and a Child's settlements are its own: without
    this, a note a Child writes reaches memory only if the parent happens to settle
    another Tool afterwards.
    """
    from dlightrag.engine.ai.messages import ToolCall

    class _Plane:
        def __init__(self) -> None:
            self.calls = 0

        async def reconcile(self) -> str | None:
            self.calls += 1
            return None

    calls = {"n": 0}

    async def model(**_kwargs: object) -> AssistantTurn:
        calls["n"] += 1
        if calls["n"] == 1:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        id="search-notes",
                        name="search_knowledge_base",
                        arguments={"query": "anything"},
                    ),
                ),
                stop_reason="tool_use",
            )
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    async def retrieve(_query: str) -> Any:
        return MagicMock(
            contexts={"chunks": [], "entities": [], "relationships": []},
            trace={},
        )

    plane = _Plane()
    parent_id = SessionId.new()
    child_id = SessionId.deterministic(run_id=str(parent_id.value), name="child:notes:1")
    outcome = await run_child_session(
        telemetry=NOOP_TELEMETRY,
        orchestrator=_child_orchestrator(model, retrieve_func=retrieve),
        repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
        session=_FakeSession(run_id=str(parent_id.value)),  # type: ignore[arg-type]
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=ChildRequest(objective="write a note"),
        parent_call_id="call-notes",
        parent_session_id=parent_id,
        context_snapshot=_context_snapshot(parent_id),
        persist_child_runtime=AsyncMock(),
        claim_child=AsyncMock(return_value=1),
        session_notes=plane,  # type: ignore[arg-type]
    )

    assert outcome.status == "succeeded"
    assert plane.calls == 1


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
        working=WorkingContextProjection(),
    )

    assert [tool.name for tool in child.tools] == ["search_knowledge_base", "ask_parent"]
    assert any(message.get("content") == "older answer" for message in messages)
    assert any(message.get("content") == "parent question" for message in messages)


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
    outcome = await run_child_session(
        telemetry=NOOP_TELEMETRY,
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
    assert outcome.status == "cancelled"

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
            telemetry=NOOP_TELEMETRY,
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
