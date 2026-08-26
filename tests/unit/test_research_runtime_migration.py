# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research Host migration through the canonical AgentSessionRuntime."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.agent.session.entries import CompactionEntry, ToolResultMessageEntry
from dlightrag.agent.session.ids import LaneId, SessionId
from dlightrag.agent.session.memory import MemoryAgentSessionStore
from dlightrag.agent.session.operation import OperationCompleted
from dlightrag.agent.session.plan import AgentRunPlan
from dlightrag.agent.session.runtime import AgentSessionRuntime
from dlightrag.agent.tools import (
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
)
from dlightrag.ai.fingerprints import ModelFingerprint
from dlightrag.ai.messages import AssistantTurn, ToolCall
from dlightrag.ai.telemetry import NOOP_TELEMETRY
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.executor import FetchedResourceBuffer, ResearchRuntimeEffects
from dlightrag.answer.resources.models import TextWindowBudget
from dlightrag.rag.retrieval import RetrievalResult
from dlightrag.runtime.settlements import EffectHostUpdate
from tests.unit.conftest import answer_model_profile


class _Session:
    owner_id = "owner"
    run_id = "run"
    execution = SimpleNamespace(fencing_epoch=1)

    async def check_cancelled(self) -> None:
        return None

    async def emit_tool_event(self, _kind: str, _payload: object) -> None:
        return None


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
    store = MemoryAgentSessionStore[EffectHostUpdate]()
    effects = ResearchRuntimeEffects(
        orchestrator=orchestrator,
        prepared=prepared,
        session=_Session(),  # type: ignore[arg-type]
        session_id=session_id,
        fetched_buffer=FetchedResourceBuffer(),
        persist_child_intent=None,
    )
    runtime = AgentSessionRuntime(
        transactions=store,
        load=store.load,
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
    orchestrator.adopt_runtime_snapshot(prepared, snapshot)
    assert prepared.last_turn is not None
    assert prepared.last_turn.assistant.text == "runtime answer"
    assert calls == 1


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

    async def read_resource(
        resource_id: str,
        _focus: str | None,
        _cursor: str | None,
        _runtime: Any,
    ) -> ToolResult:
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
    store = MemoryAgentSessionStore[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        transactions=store,
        load=store.load,
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
    from dlightrag.ai.capacity import CONTEXT_POLICY

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
    store = MemoryAgentSessionStore[EffectHostUpdate]()
    runtime = AgentSessionRuntime(
        transactions=store,
        load=store.load,
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
