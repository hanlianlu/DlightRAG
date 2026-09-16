# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research Host migration through the canonical AgentSessionRuntime."""

import asyncio
import base64
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.session.effects import EffectIntent
from dlightrag.engine.agent.session.entries import CompactionEntry, ToolResultMessageEntry
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import OperationCompleted, ToolBatchItem
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.runtime import (
    AgentOperationCancelled,
    AgentSessionEvent,
    AgentSessionRuntime,
)
from dlightrag.engine.agent.tools import (
    AgentTool,
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolResultCapacityError,
    ToolRuntime,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.orchestration.orchestrator import (
    _admit_durable_attachment_messages,
    _hydrate_attachment_messages,
)
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.research.runtime import (
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _answer_runtime_event_sink,
    _build_effect_host_update,
    provider_attempt_detail,
)
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.resources.registry import (
    FetchedResourceBytes,
    ResourceEffectOwner,
)
from dlightrag.engine.answer.tools.artifacts import attach_artifact_tool
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.unit.conftest import answer_model_profile


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

    tool = AgentTool("bounded", "Return bounded text.", _EmptyToolInput, execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=profile,
        trace={"tool_observations": []},
        evidence=SimpleNamespace(ledger_state_json=lambda: "{}"),
    )
    effects = ResearchRuntimeEffects(
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
    )
    return settled, prepared


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
    )

    class _Orchestrator:
        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            emit_text = kwargs["emit_text"]
            await emit_text("draft ")
            await emit_text("answer")
            return AssistantTurn(text="draft answer", tool_calls=(), stop_reason="stop")

    effects = ResearchRuntimeEffects(
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
    )

    class _Orchestrator:
        async def call_runtime_provider(self, _request: object, **kwargs: Any) -> AssistantTurn:
            await kwargs["emit_text"]("partial")
            raise AssertionError("cancelled callback returned")

    effects = ResearchRuntimeEffects(
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
        streamed_terminal_text="older",
        model_func=None,
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

    await effects.call_provider(
        cast(Any, context), cast(Any, object()), AttemptId.new(), emit_ephemeral
    )

    assert session.tokens == ["working"]
    assert session.phases == ["generating", "researching"]
    assert session.resets == 1
    assert prepared.streamed_terminal_text is None


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
        evidence=SimpleNamespace(ledger_state_json=lambda: "{}"),
    )
    session_id = SessionId.new()
    effects = ResearchRuntimeEffects(
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
async def test_research_runtime_projects_live_object_label_into_tool_updates() -> None:
    emitted: list[Any] = []

    async def execute(_input: BaseModel, runtime: ToolRuntime) -> ToolResult:
        await runtime.emit_update(
            ToolResult.text("", details={"object_label": "quarterly revenue 2026"})
        )
        return ToolResult.text("added 3 new passages.")

    tool = AgentTool("search_knowledge_base", "Search.", _EmptyToolInput, execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=SimpleNamespace(ledger_state_json=lambda: "{}"),
    )
    effects = ResearchRuntimeEffects(
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

    tool = AgentTool("mcp_connection_hash", "Call a remote tool.", _EmptyToolInput, execute)
    prepared = SimpleNamespace(
        tools=(tool,),
        model_profile=answer_model_profile(),
        trace={"tool_observations": []},
        evidence=SimpleNamespace(ledger_state_json=lambda: "{}"),
    )
    effects = ResearchRuntimeEffects(
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
    )

    assert settled.result.outcome == "failed"
    assert settled.duration_ms is not None
    assert settled.duration_ms >= 25


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
    )
    prepared = orchestrator.prepare_run("question")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(ModelFingerprint("openai", "query", None)),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    effects = ResearchRuntimeEffects(
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
    )
    prepared = orchestrator.prepare_run("read the attachment")
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="context-v1",
        model_identity=asdict(ModelFingerprint("openai", "query", None)),
        model_profile=asdict(profile),
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=ResearchRuntimeEffects(
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
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
    assert result.result.text_content == "bounded attachment text"
    [(intent_id, delta)] = store.applied_host_deltas(session_id)
    assert intent_id == result.intent_id
    assert len(delta.evidence) == 1
    assert delta.evidence[0].session_id == session_id.value
    assert len(delta.fetched) == 1
    assert delta.fetched[0].resource.resource_id == "attachment-1"
    assert delta.fetched[0].complete_blob.total_bytes == len(b"bounded attachment text")


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
    )
    prepared = orchestrator.prepare_run(question)
    plan = replace(
        AgentRunPlan.from_tools(
            prepared.tools,
            model_role="query",
            context_policy_revision="context-v1",
            model_identity=asdict(ModelFingerprint("openai", "query", None)),
            model_profile=asdict(profile),
        ),
        compaction_attempt_limit=3,
    )
    session_id = SessionId.new()
    store = MemoryAgentSessionRepository[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        repository=store,
        effects=ResearchRuntimeEffects(
            orchestrator=orchestrator,
            prepared=prepared,
            session=_Session(),  # type: ignore[arg-type]
            session_id=session_id,
            fetched_buffer=FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=1,
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
    """An adopted Resource records the earlier handle at settlement.

    The alias is what makes the model's printed handle resolvable after a resume, so
    it has to reach the durable row, not just the in-memory registry.
    """
    from dlightrag.engine.agent.tools import ResourceAttachmentBytes
    from dlightrag.engine.answer.research.runtime import _build_effect_host_update

    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="read",
        replay_policy="never",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="call-1",
    )
    update = _build_effect_host_update(
        session_id=SessionId.new(),
        intent=intent,
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        execution_scope="scope",
        tool_effects=ToolEffects(
            attached_resources=(
                ResourceAttachmentBytes(
                    resource_id="res-adopted",
                    filename="earlier.pdf",
                    mime_type="application/pdf",
                    source_locator="res-earlier",
                    content=b"%PDF-1.7 adopted",
                    resource_kind="lineage_adoption",
                    aliases=("res-earlier",),
                ),
            )
        ),
    )

    (fetched,) = update.fetched
    assert fetched.resource.capabilities["resource_aliases"] == ["res-earlier"]
    assert fetched.resource.capabilities["resource_kind"] == "lineage_adoption"
