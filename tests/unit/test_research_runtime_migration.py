# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research Host migration through the canonical AgentSessionRuntime."""

import asyncio
import base64
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import httpx2
import pytest
from pydantic import BaseModel

from dlightrag.adapters.observability import LangfuseTelemetry
from dlightrag.adapters.observability import langfuse as langfuse_state
from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.session.effects import EffectIntent
from dlightrag.engine.agent.session.entries import (
    CompactionEntry,
    ToolResultMessageEntry,
    decode_entry_payload,
)
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.operation import OperationCompleted, ToolBatchItem
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import RequestSnapshot
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.agent.session.runtime import (
    AgentOperationCancelled,
    AgentSessionEvent,
    AgentSessionRuntime,
)
from dlightrag.engine.agent.tool_content import ToolTextPart
from dlightrag.engine.agent.tools import (
    AgentTool,
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolResultCapacityError,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.contracts import already_in_source_order
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.orchestration.orchestrator import (
    _admit_durable_attachment_messages,
    _hydrate_attachment_messages,
)
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.research.runtime import (
    AnswerRuntimeControls,
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _answer_runtime_event_sink,
    _build_effect_host_update,
    provider_attempt_detail,
)
from dlightrag.engine.answer.resources.models import ResourceManifestEntry, TextWindowBudget
from dlightrag.engine.answer.resources.registry import (
    FetchedResourceBytes,
    ResourceEffectOwner,
)
from dlightrag.engine.answer.tools.artifacts import attach_artifact_tool
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.in_memory_session_repository import MemoryAgentSessionRepository
from tests.unit.conftest import RecordingLangfuse, answer_model_profile


class _Session:
    fencing_epoch = 1
    owner_id = "owner"
    run_id = "run"
    execution = SimpleNamespace(fencing_epoch=1)

    async def check_cancelled(self) -> None:
        return None

    async def emit_tool_event(self, _kind: str, _payload: object) -> None:
        return None


class _StreamingSession(_Session):
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.phases: list[str] = []
        self.resets = 0

    async def emit_token(self, token: str) -> None:
        self.tokens.append(token)

    async def enter_phase(self, phase: str) -> None:
        self.phases.append(phase)

    async def reset_output(self) -> None:
        self.resets += 1


class _EmptyToolInput(BaseModel):
    pass


async def _settle_bounded_research_tool(
    profile: ModelProfile,
    text: str,
    *,
    session_fencing_epoch: int | None = None,
) -> tuple[Any, Any]:
    async def execute(_input: BaseModel, _runtime: Any) -> ToolResult:
        assert _runtime.fencing_epoch == (session_fencing_epoch or 1)
        return ToolResult.text(text)

    tool = AgentTool("bounded", "Return bounded text.", _EmptyToolInput, execute=execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=profile,
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
        session_fencing_epoch=session_fencing_epoch,
    )
    item = ToolBatchItem(
        source_index=0,
        call_id="bounded-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
    )
    settled = await effects.execute_tool(
        cast(Any, context),
        item,
        {},
        AttemptId.new(),
        emit_ephemeral,
        already_in_source_order,
    )
    return settled, prepared


@pytest.mark.asyncio
async def test_only_spawn_agent_captures_the_parent_context() -> None:
    async def execute(_input: BaseModel, _runtime: Any) -> ToolResult:
        return ToolResult.text("done")

    tools = (
        AgentTool("read", "Read something.", _EmptyToolInput, execute=execute),
        AgentTool("spawn_agent", "Spawn a child.", _EmptyToolInput, execute=execute),
    )
    bind = Mock()
    prepared = SimpleNamespace(
        tools=tools,
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=bind)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    async def run(tool: AgentTool) -> None:
        item = ToolBatchItem(
            source_index=0,
            call_id=f"{tool.name}-call",
            tool_name=tool.name,
            disposition="executable",
            result_entry_id=EntryId.new(),
            intent_id=IntentId.new(),
            replay_policy=tool.replay_policy,
            contract_version=tool.contract_version,
            input_schema_digest=tool.input_schema_digest,
            effective_input_digest="0" * 64,
        )
        await effects.execute_tool(
            cast(Any, context), item, {}, AttemptId.new(), emit_ephemeral, already_in_source_order
        )

    # Capturing the parent context folds the whole transcript, so an ordinary
    # tool call must not pay for it.
    await run(tools[0])
    bind.assert_not_called()

    await run(tools[1])
    bind.assert_called_once_with(prepared, context)


@pytest.mark.asyncio
async def test_research_effect_touches_shared_run_state_only_in_source_order() -> None:
    """A call does its own work beside its neighbours, never their shared settlement.

    The evidence freeze, the trace, and the host update read state every call of a
    batch shares, so they wait until the calls before this one have returned.
    """
    worked = asyncio.Event()

    async def execute(_input: BaseModel, _runtime: Any) -> ToolResult:
        worked.set()
        return ToolResult.text("done")

    tool = AgentTool(
        "lookup", "Look something up.", _EmptyToolInput, execute=execute, read_only=True
    )
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    item = ToolBatchItem(
        source_index=1,
        call_id="lookup-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
        read_only=True,
    )
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
    )
    waiting = asyncio.Event()
    earlier_returned = asyncio.Event()

    async def in_source_order() -> None:
        waiting.set()
        await earlier_returned.wait()

    settling = asyncio.create_task(
        effects.execute_tool(
            cast(Any, context),
            item,
            {},
            AttemptId.new(),
            lambda _event: asyncio.sleep(0),
            in_source_order,
        )
    )
    await waiting.wait()
    assert worked.is_set()
    assert prepared.trace["tool_observations"] == []
    assert not settling.done()

    earlier_returned.set()
    settled = await settling

    assert settled.result.outcome == "succeeded"
    assert [row["call_id"] for row in prepared.trace["tool_observations"]] == ["lookup-call"]


@pytest.mark.asyncio
async def test_research_tool_settlement_uses_small_profile_dynamic_residual() -> None:
    # 17_460 keeps the 52-token dynamic residual this scenario pins.
    profile = ModelProfile(context_window_tokens=17_460)

    settled, prepared = await _settle_bounded_research_tool(profile, "x" * 400)

    assert estimate_tokens(settled.result.text_content) <= 52
    assert settled.result.text_content != "x" * 400
    assert prepared.trace["tool_observations"][0]["capacity_tokens"] == 52


@pytest.mark.asyncio
async def test_research_tool_settlement_preserves_40k_on_large_profile() -> None:
    settled, prepared = await _settle_bounded_research_tool(
        answer_model_profile(),
        "large profile result",
    )

    assert settled.result.text_content == "large profile result"
    assert prepared.trace["tool_observations"][0]["capacity_tokens"] == 40_000


@pytest.mark.asyncio
async def test_research_tool_settlement_rejects_nonempty_result_with_zero_residual() -> None:
    # 17_408 is the window where the dynamic residual is exactly zero.
    profile = ModelProfile(context_window_tokens=17_408)
    assert CONTEXT_POLICY.hard_input_limit(profile) == CONTEXT_POLICY.compaction_trigger(profile)

    with pytest.raises(ToolResultCapacityError, match="no residual"):
        await _settle_bounded_research_tool(profile, "cannot fit")


@pytest.mark.asyncio
async def test_provider_text_streams_optimistically_for_a_terminal_turn() -> None:
    session = _StreamingSession()
    prepared = SimpleNamespace(
        tools=(),
        model_profile=answer_model_profile(),
        streamed_terminal_text=None,
        model_func=None,
        stream_model_func=None,
        agent_turn_count=0,
        trace={
            "prompt_cache": {"turns": 0, "prompt_tokens": 0, "cache_hit_tokens": 0, "cold_turns": 0}
        },
    )

    class _Orchestrator:
        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            emit_text = kwargs["emit_text"]
            await emit_text("draft ")
            await emit_text("answer")
            return AssistantTurn(text="draft answer", tool_calls=(), stop_reason="stop")

    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, _Orchestrator()),
        prepared=prepared,
        session=session,  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
        publish_provider_text=True,
    )
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
        snapshot=SimpleNamespace(entries=()),
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    turn = await effects.call_provider(
        cast(Any, context), cast(Any, object()), AttemptId.new(), emit_ephemeral
    )

    assert turn.text == "draft answer"
    assert session.tokens == ["draft ", "answer"]
    assert session.phases == ["generating"]
    assert session.resets == 0
    assert prepared.streamed_terminal_text == "draft answer"


@pytest.mark.asyncio
async def test_cancellation_during_a_provider_delta_cancels_without_retry() -> None:
    class _CancellingSession(_StreamingSession):
        async def emit_token(self, token: str) -> None:
            self.tokens.append(token)
            raise RunCancellationObserved

    session = _CancellingSession()
    prepared = SimpleNamespace(
        tools=(),
        model_profile=answer_model_profile(),
        streamed_terminal_text=None,
        model_func=None,
        stream_model_func=None,
        trace={
            "prompt_cache": {"turns": 0, "prompt_tokens": 0, "cache_hit_tokens": 0, "cold_turns": 0}
        },
    )

    class _Orchestrator:
        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            await kwargs["emit_text"]("partial")
            raise AssertionError("cancelled callback returned")

    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, _Orchestrator()),
        prepared=prepared,
        session=session,  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
        publish_provider_text=True,
    )
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    with pytest.raises(AgentOperationCancelled):
        await effects.call_provider(
            cast(Any, context), cast(Any, object()), AttemptId.new(), emit_ephemeral
        )

    assert session.tokens == ["partial"]
    assert session.resets == 1


@pytest.mark.asyncio
async def test_provider_draft_is_reset_when_the_turn_contains_tool_calls() -> None:
    session = _StreamingSession()
    prepared = SimpleNamespace(
        tools=(),
        model_profile=answer_model_profile(),
        streamed_terminal_text=None,
        model_func=None,
        agent_turn_count=0,
        trace={
            "prompt_cache": {"turns": 0, "prompt_tokens": 0, "cache_hit_tokens": 0, "cold_turns": 0}
        },
    )

    class _Orchestrator:
        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            await kwargs["emit_text"]("working")
            return AssistantTurn(
                text="working",
                tool_calls=(ToolCall(id="call", name="read", arguments={}),),
                stop_reason="tool_use",
            )

    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, _Orchestrator()),
        prepared=prepared,
        session=session,  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
        publish_provider_text=True,
    )
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
        snapshot=SimpleNamespace(entries=()),
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    await effects.call_provider(
        cast(Any, context), cast(Any, object()), AttemptId.new(), emit_ephemeral
    )

    assert session.tokens == ["working"]
    assert session.phases == ["generating", "researching"]
    assert session.resets == 1
    assert prepared.streamed_terminal_text is None


@pytest.mark.asyncio
async def test_a_steer_after_a_streamed_completion_replaces_the_draft_it_continues_past() -> None:
    class _EventLog(_Session):
        def __init__(self) -> None:
            self.events: list[tuple[str, str]] = []

        async def emit_token(self, token: str) -> None:
            self.events.append(("token", token))

        async def enter_phase(self, phase: str) -> None:
            self.events.append(("progress", phase))

        async def reset_output(self) -> None:
            self.events.append(("reset", ""))

    class _Orchestrator:
        def __init__(self) -> None:
            self.answers = iter(("Answer A.", "Answer B."))
            self.answered: list[str] = []

        async def assemble_runtime_request(self, _prepared: object, context: Any, **_: Any):
            return RequestSnapshot.from_values(
                operation_id=context.operation_id,
                turn_number=getattr(context.state, "turn_count", 0) + 1,
                plan_digest=context.meta.plan_digest,
                model_role="query",
                messages=[{"role": "user", "content": "question"}],
                tools=[],
                tool_choice="auto",
                max_tokens=256,
            )

        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            text = next(self.answers)
            await kwargs["emit_text"](text[:4])
            await kwargs["emit_text"](text[4:])
            self.answered.append(text)
            return AssistantTurn(text=text, tool_calls=(), stop_reason="stop")

    session = _EventLog()
    orchestrator = _Orchestrator()
    prepared = SimpleNamespace(
        tools=(),
        model_profile=answer_model_profile(),
        streamed_terminal_text=None,
        model_func=None,
        agent_turn_count=0,
        trace={},
    )
    steers = [{"control_sequence": 1, "kind": "steer", "content": "Focus on X."}]

    async def read_controls() -> tuple[dict[str, Any], ...]:
        # The user pressed Steer while Answer A streamed; the runtime reads it at A's
        # completion and goes on in the same Operation.
        return (steers.pop(),) if steers and orchestrator.answered else ()

    async def acknowledge(_sequences: tuple[int, ...]) -> bool:
        return True

    repository = MemoryAgentSessionRepository[Any]()
    runtime = AgentSessionRuntime(
        repository=repository,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=cast(Any, orchestrator),
            prepared=prepared,
            session=cast(Any, session),
            session_id=SessionId.new(),
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
            publish_provider_text=True,
        ),
        tools=(),
        fencing_epoch=1,
        holder="run",
        event_sink=_answer_runtime_event_sink(cast(Any, session)),
        controls=AnswerRuntimeControls(reader=read_controls, acknowledge=acknowledge, run_id="run"),
    )
    session_id = SessionId.new()
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="answer-run:run",
        content="question",
        plan=AgentRunPlan.from_tools([], model_role="query", context_policy_revision="ctx"),
    )

    operation = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    assert isinstance(operation.state, OperationCompleted)
    assert orchestrator.answered == ["Answer A.", "Answer B."]
    # What a client shows is every token since the last reset.
    draft = ""
    for kind, text in session.events:
        draft = "" if kind == "reset" else draft + text if kind == "token" else draft
    assert draft == "Answer B."
    assert prepared.streamed_terminal_text == "Answer B."


@pytest.mark.asyncio
async def test_artifact_attachment_settles_as_a_typed_host_update(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("analysis", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    )
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    session_id = SessionId.new()
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=session_id,
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    item = ToolBatchItem(
        source_index=0,
        call_id="attach-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
    )

    async def emit_ephemeral(_event: object) -> None:
        return None

    settled = await effects.execute_tool(
        cast(
            Any,
            SimpleNamespace(
                session_id=session_id,
                lane_id=LaneId.main(),
                operation_id=OperationId.new(),
            ),
        ),
        item,
        {"path": "analysis.md", "label": "Open analysis"},
        AttemptId.new(),
        emit_ephemeral,
        already_in_source_order,
    )

    assert settled.host_delta is not None
    attachment = settled.host_delta.artifact_attachment
    assert attachment is not None
    assert attachment.relative_path == "analysis.md"
    assert attachment.label == "Open analysis"
    assert attachment.session_id == session_id.value
    assert item.intent_id is not None
    assert attachment.intent_id == item.intent_id.value


@pytest.mark.asyncio
async def test_research_runtime_projects_a_reported_subject_into_tool_updates() -> None:
    emitted: list[Any] = []

    async def execute(_input: BaseModel, runtime: ToolRuntime) -> ToolResult:
        await runtime.emit_update(ToolResult.text("", subject="quarterly revenue 2026"))
        return ToolResult.text("added 3 new passages.")

    tool = AgentTool("search_knowledge_base", "Search.", _EmptyToolInput, execute=execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    item = ToolBatchItem(
        source_index=0,
        call_id="search-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
    )

    async def emit_ephemeral(event: Any) -> None:
        emitted.append(event)

    await effects.execute_tool(
        cast(
            Any,
            SimpleNamespace(
                session_id=SessionId.new(),
                lane_id=LaneId.main(),
                operation_id=OperationId.new(),
            ),
        ),
        item,
        {},
        AttemptId.new(),
        emit_ephemeral,
        already_in_source_order,
    )

    updates = [event for event in emitted if getattr(event, "kind", None) == "tool_update"]
    assert len(updates) == 1
    assert updates[0].data["tool_name"] == "search_knowledge_base"
    assert updates[0].data["call_id"] == "search-call"
    assert updates[0].data["object_label"] == "quarterly revenue 2026"


@pytest.mark.asyncio
async def test_research_runtime_measures_one_tool_attempt_and_publishes_it_on_settlement() -> None:
    """Elapsed time is measured by the adapter that awaited the effect, then
    published by the settlement event -- including for a failed Tool, whose
    failure is a typed result rather than an exception."""

    async def execute(_input: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        await asyncio.sleep(0.03)
        return ToolResult.text("too slow", is_error=True)

    tool = AgentTool("mcp_connection_hash", "Call a remote tool.", _EmptyToolInput, execute=execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    item = ToolBatchItem(
        source_index=0,
        call_id="mcp-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
    )

    settled = await effects.execute_tool(
        cast(
            Any,
            SimpleNamespace(
                session_id=SessionId.new(),
                lane_id=LaneId.main(),
                operation_id=OperationId.new(),
            ),
        ),
        item,
        {},
        AttemptId.new(),
        lambda _event: asyncio.sleep(0),
        already_in_source_order,
    )

    assert settled.result.outcome == "failed"
    assert settled.duration_ms is not None
    assert settled.duration_ms >= 25


@pytest.mark.usefixtures("reset_langfuse_client")
@pytest.mark.asyncio
async def test_a_tool_call_is_recorded_under_the_run_that_requested_it() -> None:
    """A Tool call belongs to the agent observation that asked for it, not to the
    trace root, and its input is the arguments the model chose."""

    async def execute(_input: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        return ToolResult.text("done")

    class _SearchInput(BaseModel):
        query: str

    tool = AgentTool("mcp_connection_hash", "Call a remote tool.", _SearchInput, execute=execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=EvidenceLedger(),
    )
    effects = ResearchRuntimeEffects(
        telemetry=LangfuseTelemetry(),
        orchestrator=cast(Any, SimpleNamespace(bind_child_context=lambda *_args: None)),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    item = ToolBatchItem(
        source_index=0,
        call_id="mcp-call",
        tool_name=tool.name,
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        effective_input_digest="0" * 64,
    )
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)
    telemetry = LangfuseTelemetry()
    context = SimpleNamespace(
        session_id=SessionId.new(),
        lane_id=LaneId.main(),
        operation_id=OperationId.new(),
    )

    with telemetry.trace(session_id="sess-1", user_id="owner-1"):
        async with telemetry.observe("run-answer"):
            await effects.execute_tool(
                cast(Any, context),
                item,
                {"query": "which filings mention risk"},
                AttemptId.new(),
                lambda _event: asyncio.sleep(0),
                already_in_source_order,
            )

    assert [observation.kwargs["name"] for observation in client.observations] == [
        "run-answer",
        "execute-agent-tool",
    ]
    tool_span = client.observations[1]
    assert tool_span.kwargs["as_type"] == "tool"
    assert tool_span.parent is client.observations[0]
    assert tool_span.kwargs["input"]["tool"] == "mcp_connection_hash"
    assert "which filings mention risk" in tool_span.kwargs["input"]["arguments"]
    assert tool_span.kwargs["metadata"]["call_id"] == "mcp-call"


@pytest.mark.asyncio
async def test_answer_event_sink_publishes_tool_identity_outcome_and_elapsed() -> None:
    """The wire event a browser folds is produced here; nothing else renames or
    drops the settlement facts between the Session commit and the stream."""
    recorded: list[tuple[str, dict[str, Any]]] = []

    class _Recorder(_Session):
        async def emit_tool_event(self, kind: str, payload: object) -> None:
            recorded.append((kind, dict(cast(Any, payload))))

    sink = _answer_runtime_event_sink(cast(Any, _Recorder()))
    settlement = {
        "entry_id": "entry-1",
        "tool_name": "mcp_connection_hash",
        "call_id": "call-9",
        "source_index": 1,
        "outcome": "succeeded",
        "duration_ms": 1500,
    }
    await sink(
        AgentSessionEvent(
            kind="tool_result_committed",
            session_id=SessionId.new(),
            lane_id=LaneId.main(),
            operation_id=OperationId.new(),
            commit_sequence=7,
            data=settlement,
        )
    )

    assert recorded == [
        (
            "tool_end",
            {
                "tool_name": "mcp_connection_hash",
                "call_id": "call-9",
                "source_index": 1,
                "outcome": "succeeded",
                "duration_ms": 1500,
                "session_commit_sequence": 7,
            },
        )
    ]


@pytest.mark.asyncio
async def test_research_host_uses_runtime_instead_of_a_second_answer_interpreter() -> None:
    calls = 0

    async def model(**_kwargs) -> AssistantTurn:
        nonlocal calls
        calls += 1
        return AssistantTurn(text="runtime answer", tool_calls=(), stop_reason="stop")

    async def retrieve(_query: str):
        raise AssertionError("provider did not request a Tool")

    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),  # Fast-only collaborator is unused.
        retrieve_knowledge_base=retrieve,
        model_func=model,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run("question")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=orchestrator,
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=session_id,
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    runtime = AgentSessionRuntime(
        repository=store,
        effects=effects,
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="research-run",
        content="question",
        plan=plan,
    )
    final = await runtime.drive(
        session_id=session_id,
        operation_id=accepted.operation_id,
    )
    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    assert [entry.entry_type for entry in snapshot.entries] == [
        "user_message",
        "assistant_message",
    ]
    orchestrator.restore_runtime_snapshot(prepared, snapshot)
    assert prepared.last_turn is not None
    assert prepared.last_turn.assistant.text == "runtime answer"
    assert calls == 1


def test_web_image_effect_deduplicates_its_tool_attachment_settlement() -> None:
    intent_id = IntentId.new()
    intent = EffectIntent(
        intent_id=intent_id,
        tool_name="read",
        replay_policy="replayable",
        contract_version=3,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="read-1",
    )
    content = b"image-bytes"
    fetched = FetchedResourceBytes(
        resource_id="res-image",
        ordinal=2,
        filename="image.png",
        mime_type="image/png",
        url="https://example.com/image.png",
        content=content,
        admission_origin="agent",
        acquisition="direct_http",
    )
    buffer = FetchedResourceBuffer()
    buffer.append(fetched, ResourceEffectOwner("session", intent_id))

    update = _build_effect_host_update(
        session_id=SessionId.new(),
        intent=intent,
        ledger_state=lambda: "{}",
        fetched_buffer=buffer,
        execution_scope="session",
        tool_effects=ToolEffects(
            attached_resources=(
                ResourceAttachmentBytes(
                    resource_id="res-image",
                    filename="image.png",
                    mime_type="image/png",
                    source_locator="res-image",
                    content=content,
                ),
            )
        ),
    )

    assert len(update.fetched) == 1
    assert update.fetched[0].resource.capabilities["resource_kind"] == "web"


def test_recovery_fails_honestly_on_excess_visual_evidence() -> None:
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    digest = hashlib.sha256(image).hexdigest()
    messages = [
        {
            "role": "tool",
            "attachments": [
                {
                    "resource_id": resource_id,
                    "media_type": "image/png",
                    "content_digest": digest,
                    "size_bytes": len(image),
                }
                for resource_id in ("first", "second")
            ],
        }
    ]
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=len(image),
        max_bytes_per_image=len(image),
        max_pixels=100,
        max_px=10,
        min_px=1,
        quality=85,
        min_quality=70,
    )
    snapshots = {"first": image, "second": image}

    from dlightrag.engine.answer.errors import AnswerInputOverflowError

    with pytest.raises(AnswerInputOverflowError, match="image budget"):
        _admit_durable_attachment_messages(messages, snapshots, budget)
    assert budget.count == 1
    assert budget.used_bytes == len(image)


def test_durable_tool_attachment_is_hydrated_for_provider_projection() -> None:
    messages = [
        {
            "role": "tool",
            "attachments": [
                {
                    "resource_id": "res-image",
                    "media_type": "image/png",
                    "content_digest": (
                        "2c8648d103e3dd7ad87660da0f126a1443b6d21ac1bd3ec000c5e24e2373a90c"
                    ),
                    "size_bytes": 11,
                }
            ],
        }
    ]

    _hydrate_attachment_messages(messages, {"res-image": b"image-bytes"})

    assert messages[0]["attachments"][0]["data_url"] == ("data:image/png;base64,aW1hZ2UtYnl0ZXM=")


@pytest.mark.asyncio
async def test_research_runtime_effects_convert_one_resource_tool_to_host_delta() -> None:
    turns = [
        AssistantTurn(
            text="",
            tool_calls=(ToolCall("read-1", "read", {"resource_id": "attachment-1"}),),
            stop_reason="tool_use",
        ),
        AssistantTurn(text="done", tool_calls=(), stop_reason="stop"),
    ]

    async def model(**_kwargs) -> AssistantTurn:
        return turns.pop(0)

    async def retrieve(_query: str) -> RetrievalResult:
        raise AssertionError("knowledge retrieval was not requested")

    async def read_resource(request: Any, _runtime: Any) -> ToolResult:
        resource_id = request.resource_id
        assert resource_id == "attachment-1"
        return ToolResult.text(
            "bounded attachment text",
            effects=ToolEffects(
                evidence_sources=(
                    EvidenceSourceFact(
                        resource_id=resource_id,
                        source_type="web_attachment",
                        source_uri=resource_id,
                        title="notes.txt",
                    ),
                ),
                attached_resources=(
                    ResourceAttachmentBytes(
                        resource_id=resource_id,
                        filename="notes.txt",
                        mime_type="text/plain",
                        source_locator="attachment:attachment-1",
                        content=b"bounded attachment text",
                    ),
                ),
            ),
        )

    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=retrieve,
        model_func=model,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resource_reader=read_resource,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run("read the attachment")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="resource-tool",
        content="read the attachment",
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    result = next(entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry))
    # A resource Tool answers with the passage itself, and the Tool declares that, so
    # the admitted row is labelled where it stands instead of being carried twice:
    # the label is what the Citation Contract asks the model to reuse. The Tool knows
    # this, so no body length or substring guess decides it.
    assert result.result.text_content == "[1-1] notes.txt\n\nbounded attachment text"
    assert result.result.text_content.count("bounded attachment text") == 1
    [(intent_id, delta)] = store.applied_host_deltas(session_id)
    assert intent_id == result.intent_id
    assert len(delta.evidence) == 1
    assert delta.evidence[0].session_id == session_id.value
    assert len(delta.fetched) == 1
    assert delta.fetched[0].resource.resource_id == "attachment-1"
    assert delta.fetched[0].complete_blob.total_bytes == len(b"bounded attachment text")


@pytest.mark.asyncio
async def test_research_reads_run_at_once_yet_cite_and_settle_in_source_order() -> None:
    """Two reads of one turn overlap, and their evidence still follows the batch.

    The first read can only finish after the second one did, so run one at a time
    they would never finish. The second read's passage arrives first, yet it takes
    the second citation number and its result settles after the first read's.
    """
    turns = [
        AssistantTurn(
            text="",
            tool_calls=(
                ToolCall("read-1", "read", {"resource_id": "first"}),
                ToolCall("read-2", "read", {"resource_id": "second"}),
            ),
            stop_reason="tool_use",
        ),
        AssistantTurn(text="done", tool_calls=(), stop_reason="stop"),
    ]

    async def model(**_kwargs) -> AssistantTurn:
        return turns.pop(0)

    async def retrieve(_query: str) -> RetrievalResult:
        raise AssertionError("knowledge retrieval was not requested")

    second_read = asyncio.Event()

    async def read_resource(request: Any, _runtime: Any) -> ToolResult:
        resource_id = request.resource_id
        if resource_id == "first":
            await second_read.wait()
        else:
            second_read.set()
        return ToolResult.text(
            f"{resource_id} passage",
            effects=ToolEffects(
                evidence_sources=(
                    EvidenceSourceFact(
                        resource_id=resource_id,
                        source_type="web_attachment",
                        source_uri=resource_id,
                        title=f"{resource_id}.txt",
                    ),
                )
            ),
        )

    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=retrieve,
        model_func=model,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resource_reader=read_resource,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run("read both attachments")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="concurrent-reads",
        content="read both attachments",
        plan=plan,
    )

    final = await asyncio.wait_for(
        runtime.drive(session_id=session_id, operation_id=accepted.operation_id), timeout=5
    )

    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    results = [entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry)]
    assert [entry.result.call_id for entry in results] == ["read-1", "read-2"]
    assert [entry.result.text_content for entry in results] == [
        "[1-1] first.txt\n\nfirst passage",
        "[2-1] second.txt\n\nsecond passage",
    ]


@pytest.mark.asyncio
async def test_provider_overflow_compacts_shrinks_and_retries_through_host_effects() -> None:
    from dlightrag.engine.ai.capacity import CONTEXT_POLICY

    model_calls = 0
    summary_calls: list[dict[str, Any]] = []

    async def model(**_kwargs) -> AssistantTurn:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return AssistantTurn(
                text="",
                tool_calls=(
                    ToolCall(
                        "search-1",
                        "search_knowledge_base",
                        {"query": "one fact"},
                    ),
                ),
                stop_reason="tool_use",
            )
        if model_calls == 2:
            raise RuntimeError("prompt is too long: 300000 tokens > 200000 maximum")
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    def stream_model(**kwargs: Any):
        summary_calls.append(kwargs)

        async def stream():
            if len(summary_calls) < 3:
                yield "## Progress\nmissing required goal"
            else:
                yield (
                    "## Goal\nAnswer the question.\n\n"
                    "## Progress\nResearch started.\n\n"
                    "## Next Steps\nFinish the answer."
                )

        return stream()

    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "content": "one grounded fact",
                        "metadata": {"title": "Source"},
                    }
                ],
                "entities": [],
                "relationships": [],
            },
            trace={"retrieved": 1},
        )

    profile = answer_model_profile()
    question = "Large question " + "x" * 40_000
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=retrieve,
        model_func=model,
        stream_model_func=stream_model,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run(question)
    plan = replace(
        AgentRunPlan.from_tools(
            prepared.tools,
            model_role="query",
            context_policy_revision="context-v1",
            model_identity=asdict(
                ModelInvocationFingerprint("openai", "query", None, "chat_completion")
            ),
            model_profile=asdict(profile),
        ),
        compaction_attempt_limit=3,
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="overflow",
        content=question,
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    snapshot = await store.load(session_id)
    assert any(isinstance(entry, CompactionEntry) for entry in snapshot.entries)
    assert snapshot.active_projection is not None
    assert snapshot.active_projection.summary is not None
    assert model_calls == 3
    assert len(summary_calls) == plan.compaction_attempt_limit
    assert prepared.trace["compactions"][-1]["tail_target_tokens"] == (
        CONTEXT_POLICY.retained_tail_target(profile) // 4
    )


@pytest.mark.asyncio
async def test_research_child_tool_receives_child_session_fence_not_parent_fence():
    await _settle_bounded_research_tool(answer_model_profile(), "child", session_fencing_epoch=7)


def test_provider_failure_detail_names_the_http_status_without_provider_text() -> None:
    """A durable provider error classifies the failure and stores no prompt text."""

    class Rejected(Exception):
        def __init__(self) -> None:
            super().__init__("max_tokens is too large: 384000 > 65536 for prompt 'secret'")
            self.status_code = 400

    detail = provider_attempt_detail(Rejected(), retryable=False)

    assert detail == "Model provider rejected the request (HTTP 400)"
    assert "secret" not in detail
    assert "max_tokens" not in detail

    assert provider_attempt_detail(Rejected(), retryable=True) == (
        "Model provider is temporarily unavailable (HTTP 400)"
    )
    # Without a status there is nothing safe to add beyond the verdict.
    assert provider_attempt_detail(ValueError("no status"), retryable=False) == (
        "Model provider rejected the request"
    )


@pytest.mark.asyncio
async def test_attached_resources_pin_their_earlier_handles_for_recovery() -> None:
    """An adopted Resource's row records the earlier handle.

    The alias is what makes the model's printed handle resolvable after a resume, so
    it has to reach the durable row, not just the in-memory registry. Adoption and
    settlement describe a Resource through this one translation.
    """
    import hashlib

    from dlightrag.engine.agent.tools import ResourceAttachmentBytes
    from dlightrag.engine.answer.resource_settlement import attached_resource_update

    content = b"%PDF-1.7 adopted"
    fetched = attached_resource_update(
        ResourceAttachmentBytes(
            resource_id="res-adopted",
            filename="earlier.pdf",
            mime_type="application/pdf",
            source_locator="res-adopted",
            content=content,
            resource_kind="lineage_adoption",
            aliases=("res-earlier",),
        ),
        session_id=SessionId.new().value,
        intent_id=IntentId.new().value,
    )

    assert fetched.resource.capabilities["resource_aliases"] == ["res-earlier"]
    assert fetched.resource.capabilities["resource_kind"] == "lineage_adoption"
    assert fetched.resource.blob_digest == hashlib.sha256(content).hexdigest()
    assert fetched.complete_blob.digest == fetched.resource.blob_digest


@pytest.mark.asyncio
async def test_each_research_request_extends_the_previous_transcript_prefix() -> None:
    """A later request reuses the earlier one's bytes instead of re-rendering them.

    This is the property a provider prefix cache needs, and the reason evidence text
    is frozen into the Tool result that admitted it: the old shape re-rendered the
    accumulated corpus *after* the growing Session fold, so no earlier request could
    be a prefix of a later one and every turn paid full input price for the corpus.
    """
    requests: list[list[dict[str, Any]]] = []

    async def model(**kwargs: Any) -> AssistantTurn:
        requests.append([dict(message) for message in kwargs["messages"]])
        if len(requests) == 1:
            return AssistantTurn(
                text="",
                tool_calls=(ToolCall("search-1", "search_knowledge_base", {"query": "one fact"}),),
                stop_reason="tool_use",
                # A first turn has nothing to hit; the provider still reports it.
                usage_details={"prompt_tokens": 10_000, "prompt_cache_hit_tokens": 0},
            )
        return AssistantTurn(
            text="done",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"prompt_tokens": 12_000, "prompt_cache_hit_tokens": 0},
        )

    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "reference_id": "source-1",
                        "file_path": "doc.txt",
                        "content": "one grounded fact",
                        "metadata": {"source_type": "file", "title": "doc.txt"},
                    }
                ],
                "entities": [],
                "relationships": [],
            },
            trace={"retrieved": 1},
        )

    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=retrieve,
        model_func=model,
        stream_model_func=cast(Any, None),
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run("What changed?")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    runtime = AgentSessionRuntime(
        repository=MemoryAgentSessionRepository[EffectHostUpdate](),
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="prefix",
        content="What changed?",
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    assert isinstance(final.state, OperationCompleted)
    assert len(requests) == 2
    first, second = requests
    # This composition has no per-Run tail (no admitted evidence images, memory or
    # skill context), so the later request is a strict extension of the earlier one: the
    # earlier request is its prefix byte for byte, key order included.
    assert json.dumps(second[: len(first)]) == json.dumps(first)
    # The admitted passage arrived inside the Tool result, not as a re-rendered pack.
    tool_messages = [message for message in second if message.get("role") == "tool"]
    assert "one grounded fact" in str(tool_messages[-1]["content"])
    assert sum("one grounded fact" in str(message) for message in second) == 1
    # And each turn's billed prompt was aggregated for the operator.
    cache = prepared.trace["prompt_cache"]
    assert cache["turns"] == 2
    assert cache["prompt_tokens"] == 22_000
    assert cache["cache_hit_tokens"] == 0
    # The first turn has nothing cached yet and is never counted as a regression.
    assert cache["cold_turns"] == 1


async def _one_fact(_query: str) -> RetrievalResult:
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "reference_id": "source-1",
                    "file_path": "doc.txt",
                    "content": "one grounded fact",
                    "metadata": {"source_type": "file", "title": "doc.txt"},
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieved": 1},
    )


async def _drive_research_run(
    repository: MemoryAgentSessionRepository[EffectHostUpdate],
    session_id: SessionId,
    question: str,
    *,
    model: Any,
    resource_manifest: tuple[ResourceManifestEntry, ...] = (),
    query_images: list[dict[str, Any]] | None = None,
    injected_tools: tuple[AgentTool, ...] = (),
    memory_text: str = "",
) -> Any:
    """Drive one Research Run over a shared Session the way the executor does.

    Each Run composes its own orchestrator, memory and Runtime, accepts its question
    as a User Entry of the one durable Session, and projects the settled Session back
    into its result.
    """
    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=_one_fact,
        model_func=model,
        stream_model_func=cast(Any, None),
        injected_tools=list(injected_tools),
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        resource_manifest=resource_manifest,
        search_toolchain=SearchToolchain(),
    )
    orchestrator.bind_recall(memory_text)
    prepared = orchestrator.prepare_run(question, query_images=query_images)
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    runtime = AgentSessionRuntime(
        repository=repository,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key=f"answer-run:{question}",
        content=question,
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(final.state, OperationCompleted)
    orchestrator.restore_runtime_snapshot(prepared, final.context.snapshot)
    return prepared


def _search_first(
    requests: list[list[dict[str, Any]]], usage: list[dict[str, int]] | None = None
) -> Any:
    """A model that searches once on the Session's first call and answers after."""

    async def model(**kwargs: Any) -> AssistantTurn:
        requests.append([dict(message) for message in kwargs["messages"]])
        details = usage[len(requests) - 1] if usage else None
        if len(requests) == 1:
            return AssistantTurn(
                text="",
                tool_calls=(ToolCall("search-1", "search_knowledge_base", {"query": "one fact"}),),
                stop_reason="tool_use",
                usage_details=details,
            )
        return AssistantTurn(
            text=f"answer {len(requests)}",
            tool_calls=(),
            stop_reason="stop",
            usage_details=details,
        )

    return model


class _LookupArgs(BaseModel):
    figure: str
    year: int
    unit: str


def _jsonb(value: Any) -> Any:
    """Order object keys as PostgreSQL's jsonb stores them: shorter first, then bytewise."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        keys = sorted(mapping, key=lambda key: (len(key.encode()), key.encode()))
        return {key: _jsonb(mapping[key]) for key in keys}
    if isinstance(value, list):
        return [_jsonb(item) for item in cast(list[Any], value)]
    return value


class _JsonbSessionRepository(MemoryAgentSessionRepository[EffectHostUpdate]):
    """A Session read back the way PostgreSQL returns it.

    A Run keeps the Entries it committed in memory. The next Run loads the Session, and
    every payload comes back decoded from its jsonb column, whose object keys are stored
    shortest first rather than in the order they were written.
    """

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        snapshot = await super().load(session_id)
        return replace(
            snapshot,
            entries=tuple(
                decode_entry_payload(
                    entry_type=entry.entry_type,
                    entry_id=entry.entry_id,
                    session_id=entry.session_id,
                    sequence=entry.sequence,
                    timestamp=entry.timestamp,
                    payload=_jsonb(json.loads(json.dumps(entry.canonical_payload()))),
                    parent_entry_id=entry.parent_entry_id,
                )
                for entry in snapshot.entries
            ),
        )


class _ChatEndpoint:
    """An OpenAI-compatible endpoint that records each request body as it was sent."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._replies = replies

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        message = self._replies.pop(0)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                        "message": {"role": "assistant", **message},
                    }
                ],
            },
        )


@pytest.mark.asyncio
async def test_a_follow_up_run_extends_the_previous_runs_last_request_on_the_wire() -> None:
    """The next Run's first request body starts with the previous Run's last one.

    Measured on this deployment, a follow-up Run's first request reused 0.8k-7.8k
    cached tokens of a 70k-244k prompt: the question block led every request, in front
    of a Session fold that already held the question. A Session read back from
    PostgreSQL would still break the prefix at the first Tool call with several
    arguments, replayed in jsonb's key order instead of the order the earlier Run sent.
    The bodies recorded here are what the provider receives, and the second Run loads
    the Session through jsonb. Everything the earlier Run sent before its own
    statement (recalled memory) is the prefix; what is new follows it.
    """
    from dlightrag.engine.ai.scheduler import ModelScheduler
    from dlightrag.engine.ai.settings import ModelSettings
    from dlightrag.engine.ai.tool_model import ToolModel
    from tests.unit.test_provider_attachment_contract import bind_mock_http

    wire = _ChatEndpoint(
        [
            {
                "content": None,
                "reasoning_content": "Look the figure up first.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            # The model's own key order, which jsonb does not keep.
                            "arguments": '{"figure":"revenue","year":2023,"unit":"EUR"}',
                        },
                    }
                ],
            },
            {"content": "Revenue was 12 EUR.", "reasoning_content": "Answer it."},
            {"content": "Prices rose.", "reasoning_content": "Explain it."},
        ]
    )
    model = ToolModel(
        ModelSettings(
            provider="openai",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            max_retries=0,
        ),
        scheduler=ModelScheduler(max_concurrency=1),
        telemetry=NOOP_TELEMETRY,
    )
    bind_mock_http(model._provider, wire)  # pyright: ignore[reportPrivateUsage]

    async def lookup_figure(_args: BaseModel, _runtime: Any) -> ToolResult:
        return ToolResult.text("revenue 2023: 12 EUR")

    lookup = AgentTool(
        "lookup",
        "Look up one reported figure.",
        _LookupArgs,
        execute=lookup_figure,
    )
    memory = "Remembered about this owner (context only): reports in EUR."
    repository = _JsonbSessionRepository()
    session_id = SessionId.new()

    await _drive_research_run(
        repository,
        session_id,
        "What was the 2023 revenue?",
        model=model,
        injected_tools=(lookup,),
        memory_text=memory,
    )
    await _drive_research_run(
        repository,
        session_id,
        "Why did it change?",
        model=model,
        injected_tools=(lookup,),
        memory_text=memory,
        resource_manifest=(
            ResourceManifestEntry("res-1", "report.pdf", "application/pdf", "bytes", 1),
        ),
    )
    earlier, later = wire.bodies[1], wire.bodies[2]

    assert json.dumps(later["tools"]) == json.dumps(earlier["tools"])
    # The earlier Run's own statement, memory, closes its last request.
    assert earlier["messages"][-1]["content"] == memory
    cut = len(earlier["messages"]) - 1
    assert json.dumps(later["messages"][:cut]) == json.dumps(earlier["messages"][:cut])
    assert later["messages"][cut]["role"] == "assistant"
    assert later["messages"][cut]["content"] == "Revenue was 12 EUR."
    assert later["messages"][cut + 1] == {"role": "user", "content": "Why did it change?"}
    # The follow-up's first request states each question and the earlier answer once:
    # the Session fold is the only copy of the earlier turn, and what only the
    # follow-up registered follows it.
    for statement in ("What was the 2023 revenue?", "Revenue was 12 EUR.", "Why did it change?"):
        assert json.dumps(later["messages"]).count(statement) == 1
    assert "[resource: res-1] report.pdf" in json.dumps(later["messages"][cut + 2 :])


@pytest.mark.asyncio
async def test_a_compaction_that_covers_the_question_restates_it_after_the_summary() -> None:
    """The question survives a compaction verbatim, where it stood.

    A compaction keeps whole exchanges, so it covers the Run's own User Entry and leaves
    the question only in the summary's paraphrase. The request after it states the
    question again right after the summary and before the retained work, so a steer or
    follow-up retained after it still comes later.
    """
    requests: list[list[dict[str, Any]]] = []
    # Large enough that covering it shrinks the request, as a real compaction must.
    question = "What changed? " + "x" * 40_000

    async def model(**kwargs: Any) -> AssistantTurn:
        requests.append([dict(message) for message in kwargs["messages"]])
        if len(requests) == 1:
            return AssistantTurn(
                text="",
                tool_calls=(ToolCall("search-1", "search_knowledge_base", {"query": "one fact"}),),
                stop_reason="tool_use",
            )
        if len(requests) == 2:
            raise RuntimeError("prompt is too long: 300000 tokens > 200000 maximum")
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    def summarize(**_kwargs: Any) -> Any:
        async def stream() -> Any:
            yield (
                "## Goal\nAnswer the question.\n\n"
                "## Progress\nResearch started.\n\n"
                "## Next Steps\nFinish the answer."
            )

        return stream()

    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=_one_fact,
        model_func=model,
        stream_model_func=summarize,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        search_toolchain=SearchToolchain(),
    )
    prepared = orchestrator.prepare_run(question)
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(
            ModelInvocationFingerprint("openai", "query", None, "chat_completion")
        ),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    runtime = AgentSessionRuntime(
        repository=MemoryAgentSessionRepository[EffectHostUpdate](),
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
        holder="run",
        provider_attempt_limit=plan.provider_attempt_limit,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        idempotency_key="covered-question",
        content=question,
        plan=plan,
    )
    final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)

    assert isinstance(final.state, OperationCompleted)
    after = requests[2]
    summary_at = next(
        index
        for index, message in enumerate(after)
        if "Answer the question." in str(message["content"])
    )
    assert after[summary_at + 1] == {"role": "user", "content": question}
    assert after[summary_at + 2]["tool_calls"]
    assert sum(question in str(message["content"]) for message in after) == 1


@pytest.mark.asyncio
async def test_a_follow_up_run_reports_its_own_turns_and_its_cold_first_turn() -> None:
    """A Run's trace describes that Run, measured against the Session's last request.

    ``agent_turns`` counted every Assistant Entry of the Session, so one Run reported
    28 turns while it made 6; and a follow-up Run's first turn was exempt from the
    cold count, so ``cold_turns`` stayed 0 while those turns reused only the head.
    """
    requests: list[list[dict[str, Any]]] = []
    model = _search_first(
        requests,
        [
            # The Session's first request has nothing before it.
            {"prompt_tokens": 20_000, "prompt_cache_hit_tokens": 0},
            # Its second reuses the first.
            {"prompt_tokens": 22_000, "prompt_cache_hit_tokens": 19_500},
            # The follow-up's first turn hit its system prompt and Tools only.
            {"prompt_tokens": 23_000, "prompt_cache_hit_tokens": 800},
        ],
    )
    repository = MemoryAgentSessionRepository[EffectHostUpdate]()
    session_id = SessionId.new()

    earlier = await _drive_research_run(repository, session_id, "What changed?", model=model)
    follow_up = await _drive_research_run(repository, session_id, "Why?", model=model)

    assert earlier.trace["agent_turns"] == 2
    assert follow_up.trace["agent_turns"] == 1
    assert earlier.trace["prompt_cache"]["cold_turns"] == 0
    # 800 of the 22,000 tokens the earlier Run's last request billed.
    assert follow_up.trace["prompt_cache"]["cold_turns"] == 1


@pytest.mark.asyncio
async def test_a_run_whose_request_re_renders_pixels_is_not_judged_cold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attached images follow the transcript, so no later turn can reuse them.

    With a system prompt and Tools of about 8k and twelve images of about 2k each, the
    second turn reuses about 8k of about 33k while the cache works exactly as designed;
    judged against what the first turn billed, every turn would count as cold and warn.
    """
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    requests: list[list[dict[str, Any]]] = []
    model = _search_first(
        requests,
        [
            {"prompt_tokens": 33_000, "prompt_cache_hit_tokens": 0},
            {"prompt_tokens": 34_000, "prompt_cache_hit_tokens": 8_000},
        ],
    )

    with caplog.at_level("WARNING"):
        prepared = await _drive_research_run(
            MemoryAgentSessionRepository[EffectHostUpdate](),
            SessionId.new(),
            "What does the chart show?",
            model=model,
            query_images=[picture],
        )

    assert "image_url" in str(requests[-1][-1]["content"])
    assert prepared.trace["prompt_cache"]["turns"] == 2
    assert prepared.trace["prompt_cache"]["cold_turns"] == 0
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


def test_a_stale_or_pixel_bearing_request_has_no_cache_reference() -> None:
    """A turn is judged only against a request whose prefix the provider still keeps.

    A provider keeps a prefix for minutes, not for as long as a Session lasts, so a
    follow-up after a pause is not a regression; and pixels re-rendered after the
    transcript are billed but never reusable.
    """
    from dlightrag.engine.agent.session.entries import AssistantMessageEntry
    from dlightrag.engine.agent.session.registers import RequestSnapshot
    from dlightrag.engine.answer.research.runtime import _reusable_prompt_tokens

    now = datetime.now(UTC)

    def previous(age: timedelta) -> Any:
        return SimpleNamespace(
            entries=[
                AssistantMessageEntry(
                    entry_id=EntryId.new(),
                    session_id=SessionId.new(),
                    timestamp=now - age,
                    content="answer",
                    stop_reason="stop",
                    usage={"prompt_tokens": 33_000, "prompt_cache_hit_tokens": 0},
                )
            ]
        )

    def request(*content: Any) -> RequestSnapshot:
        return RequestSnapshot.from_values(
            operation_id=OperationId.new(),
            turn_number=2,
            plan_digest="a" * 64,
            model_role="query",
            messages=[{"role": "user", "content": list(content) or "why?"}],
            tools=[],
            tool_choice="auto",
            max_tokens=None,
        )

    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    recent = previous(timedelta(seconds=30))
    assert _reusable_prompt_tokens(recent, request(), requested_at=now) == 33_000
    assert _reusable_prompt_tokens(recent, request(picture), requested_at=now) is None
    assert (
        _reusable_prompt_tokens(previous(timedelta(minutes=10)), request(), requested_at=now)
        is None
    )


@pytest.mark.parametrize(
    ("input_key", "cache_key"),
    [
        ("prompt_tokens", "prompt_cache_hit_tokens"),
        ("input_tokens", "input_tokens_details.cached_tokens"),
    ],
)
def test_prompt_cache_counters_land_in_the_run_trace_and_warn_once_per_cold_turn(
    caplog: pytest.LogCaptureFixture, input_key: str, cache_key: str
) -> None:
    from dlightrag.engine.answer.research.runtime import _record_prompt_cache

    trace: dict[str, Any] = {}

    def turn(hit: int, billed: int) -> AssistantTurn:
        return AssistantTurn(
            text="x",
            tool_calls=(),
            stop_reason="stop",
            usage_details={input_key: billed, cache_key: hit},
        )

    with caplog.at_level("WARNING"):
        # The Session's first request has nothing before it to reuse.
        _record_prompt_cache(trace, turn(0, 60_000))
        # A large prompt that reuses nothing the previous request billed is exactly
        # the failure this counter exists to surface, and it is silent from the inside.
        _record_prompt_cache(trace, turn(0, 132_202), reusable_prompt_tokens=60_000)
        # A hit, and a miss too small to report, are both ordinary.
        _record_prompt_cache(trace, turn(120_000, 132_000), reusable_prompt_tokens=132_202)
        _record_prompt_cache(trace, turn(0, 512), reusable_prompt_tokens=132_000)

    cache = trace["prompt_cache"]
    assert cache == {
        "turns": 4,
        "prompt_tokens": 324_714,
        "cache_hit_tokens": 120_000,
        "cold_turns": 1,
    }
    (warning,) = [record for record in caplog.records if record.levelname == "WARNING"]
    fields = ("prompt_tokens", "reusable_prompt_tokens", "turn", "cache_hit_tokens")
    assert {name: getattr(warning, name) for name in fields} == {
        "prompt_tokens": 132_202,
        "reusable_prompt_tokens": 60_000,
        "turn": 2,
        "cache_hit_tokens": 0,
    }


def test_a_turn_that_reuses_only_the_system_prompt_and_tools_is_cold() -> None:
    """The measured miss: a follow-up Run's first turn hit, but only its head.

    Such a turn was exempt as a Run's first, and its hit was never zero, so neither
    half of the old rule fired while it reused 0.8k-7.8k of a 70k-244k prompt.
    """
    from dlightrag.engine.answer.research.runtime import _record_prompt_cache

    trace: dict[str, Any] = {}

    def turn(hit: int, billed: int) -> AssistantTurn:
        return AssistantTurn(
            text="x",
            tool_calls=(),
            stop_reason="stop",
            usage_details={"prompt_tokens": billed, "prompt_cache_hit_tokens": hit},
        )

    # The earlier Run's last request billed 240k; this one reused its system prompt and
    # Tool definitions only.
    _record_prompt_cache(trace, turn(7_800, 244_000), reusable_prompt_tokens=240_000)
    assert trace["prompt_cache"]["cold_turns"] == 1
    # Reusing the earlier Run's transcript is the healthy case.
    _record_prompt_cache(trace, turn(236_000, 246_000), reusable_prompt_tokens=244_000)
    assert trace["prompt_cache"]["cold_turns"] == 1


def test_a_provider_that_reports_no_usage_records_no_cache_turn() -> None:
    from dlightrag.engine.answer.research.runtime import _record_prompt_cache

    trace: dict[str, Any] = {"prompt_cache": {"turns": 0, "cold_turns": 0}}

    _record_prompt_cache(trace, AssistantTurn(text="x", tool_calls=(), stop_reason="stop"))

    assert trace["prompt_cache"]["turns"] == 0


def test_one_tool_result_gets_one_absolute_evidence_capacity() -> None:
    from dlightrag.engine.answer.research.runtime import _evidence_render_budget

    result = ToolResult.text("status")
    budget = _evidence_render_budget(capacity_tokens=40_000, result=result, pending_rows=4)

    # One absolute, model-aware room: the Tool's own text and the per-row framing come
    # out of it, and it is neither shared across a batch nor derived from the
    # compaction trigger. Freezing is what keeps a passage reusable, so shrinking it
    # to stay under a trigger would trade a permanent prefix for a turn compaction
    # can bound anyway.
    assert 0 < budget <= 40_000
    assert budget < _evidence_render_budget(capacity_tokens=40_000, result=result, pending_rows=1)
    # A Tool whose own text already fills the room gets none of it.
    assert (
        _evidence_render_budget(
            capacity_tokens=40_000,
            result=ToolResult.text("x" * 1_000_000),
            pending_rows=40,
        )
        == 0
    )


def test_the_freeze_splice_keeps_a_continuation_last() -> None:
    from dlightrag.engine.answer.research.runtime import _with_admitted_evidence

    ledger = EvidenceLedger()
    ledger.add_rows(
        [
            {
                "chunk_id": "c1",
                "reference_id": "source-1",
                "file_path": "doc.txt",
                "content": "a passage the tool did not carry",
                "metadata": {"source_type": "file", "title": "doc.txt"},
            }
        ]
    )
    continuation = "[more text available; cursor=abc]"
    result = ToolResult(
        parts=(ToolTextPart(f"tool body\n{continuation}"),),
        protected_text=continuation,
    )

    frozen = _with_admitted_evidence(
        result,
        evidence=ledger,
        budget_tokens=1_000_000,
        intent_key="intent-1",
    )

    text = frozen.text_content
    assert text.startswith("tool body")
    assert "a passage the tool did not carry" in text
    assert text.endswith(continuation)
    assert text.index("a passage the tool did not carry") < text.index(continuation)


def test_a_refused_freeze_returns_the_rows_to_the_next_tool_result() -> None:
    from dlightrag.engine.answer.research.runtime import _with_admitted_evidence

    ledger = EvidenceLedger()
    ledger.add_rows(
        [
            {
                "chunk_id": "c1",
                "reference_id": "source-1",
                "file_path": "doc.txt",
                "content": "passage behind a refused result",
                "metadata": {"source_type": "file"},
            }
        ]
    )
    # A Tool body the durable store cannot keep: the freeze is retracted so the
    # passage renders with a later Tool result instead of vanishing with this one.
    refused = _with_admitted_evidence(
        ToolResult.text("body with \x00 a NUL"),
        evidence=ledger,
        budget_tokens=1_000_000,
        intent_key="intent-1",
    )
    later = _with_admitted_evidence(
        ToolResult.text("later result"),
        evidence=ledger,
        budget_tokens=1_000_000,
        intent_key="intent-2",
    )

    assert refused.text_content == "body with \x00 a NUL"
    assert "passage behind a refused result" in later.text_content


def test_a_provider_without_cache_counters_is_not_a_cold_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.answer.research.runtime import _record_prompt_cache

    trace: dict[str, Any] = {}

    def turn(usage: dict[str, int]) -> AssistantTurn:
        return AssistantTurn(text="x", tool_calls=(), stop_reason="stop", usage_details=usage)

    with caplog.at_level("WARNING"):
        _record_prompt_cache(trace, turn({"input_tokens": 60_000, "output_tokens": 5}))
        # Reading an unreported hit as zero would make this turn cold.
        _record_prompt_cache(
            trace,
            turn({"input_tokens": 61_000, "output_tokens": 5}),
            reusable_prompt_tokens=60_000,
        )

    # A provider that reports no cache fields is a different fact from a reported
    # zero, and must not be counted as a regression.
    assert trace["prompt_cache"]["cold_turns"] == 0
    assert trace["prompt_cache"]["turns"] == 2
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


@pytest.mark.asyncio
async def test_a_note_settlement_promotes_the_working_copy_and_states_the_reason() -> None:
    """Memory is best effort: the Tool batch promotes, and a refusal lands on the trace."""
    from dlightrag.engine.answer.session_notes import SESSION_NOTES_DEGRADED_KEY

    class _Plane:
        def __init__(self, reason: str | None) -> None:
            self.reason = reason
            self.calls = 0

        async def reconcile(self) -> str | None:
            self.calls += 1
            return self.reason

    profile = ModelProfile(context_window_tokens=1_000_000)
    prepared = SimpleNamespace(tools=(), model_profile=profile, trace={})
    plane = _Plane("budget_refused")
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace()),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
        session_notes=cast(Any, plane),
    )

    await effects._promote_session_notes()

    assert plane.calls == 1
    assert prepared.trace[SESSION_NOTES_DEGRADED_KEY] == "budget_refused"


@pytest.mark.asyncio
async def test_a_run_without_a_plane_promotes_nothing() -> None:
    profile = ModelProfile(context_window_tokens=1_000_000)
    prepared = SimpleNamespace(tools=(), model_profile=profile, trace={})
    effects = ResearchRuntimeEffects(
        telemetry=NOOP_TELEMETRY,
        orchestrator=cast(Any, SimpleNamespace()),
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )

    await effects._promote_session_notes()

    assert prepared.trace == {}
