# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import io
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.agent.session.entries import ContextInjectionEntry
from dlightrag.agent.session.fold import PriorTurns
from dlightrag.ai.capacity import ModelProfile
from dlightrag.ai.fingerprints import ModelFingerprint
from dlightrag.ai.scheduler import ModelScheduler
from dlightrag.ai.telemetry import NOOP_TELEMETRY
from dlightrag.answer.capabilities import AnswerCapabilities
from dlightrag.answer.capability import AnswerImageCapability
from dlightrag.answer.errors import CurrentDocumentParseError
from dlightrag.answer.executor import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
    FetchedResourceBuffer,
    JournalRunBoundaries,
    OrchestratorRun,
    _close_execution_resources,
    _memory_recall_allowed,
)
from dlightrag.answer.highlights import SemanticHighlightSettings
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.runtime import RunExecutionError, RunSession, artifact_digest


def _executor() -> AnswerExecutor:
    return AnswerExecutor(
        store=MagicMock(),
        pool=MagicMock(),
        retrieve=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=AnswerExecutorSettings(
            default_top_k=10,
            default_chunk_top_k=20,
            semantic_highlights=SemanticHighlightSettings(
                enabled=True,
                timeout=10.0,
                max_concurrency=8,
                batch_size=8,
                max_input_chars=4096,
                cache_size=500,
            ),
        ),
        telemetry=NOOP_TELEMETRY,
    )


def test_acceptance_research_tools_include_every_configured_non_resource_surface() -> None:
    from pydantic import BaseModel

    from dlightrag.agent.tools import AgentTool, ToolResult

    class Args(BaseModel):
        value: str

    async def external(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("unused")

    executor = AnswerExecutor(
        store=MagicMock(),
        pool=MagicMock(),
        retrieve=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        execution_environment="trust",
        memory_store=MagicMock(),
        external_tools=(AgentTool("remote_lookup", "Remote lookup.", Args, external),),
    )

    names = {tool.name for tool in executor.acceptance_research_tools()}

    assert {
        "read",
        "write",
        "edit",
        "grep",
        "bash",
        "spawn_agent",
        "subagent_status",
        "wait_subagent",
        "cancel_subagent",
        "remember",
        "forget",
        "recall_memory",
        "load_skill",
        "remote_lookup",
    } <= names


def test_acceptance_plan_matches_runtime_tool_composition(tmp_path: Path) -> None:
    from dlightrag.agent.environment.local import LocalExecutionEnvironment
    from dlightrag.agent.session.plan import AgentRunPlan
    from dlightrag.agent.skills import SkillCatalog
    from dlightrag.answer.evidence import EvidenceLedger
    from dlightrag.answer.tools.composition import compose_research_tools
    from dlightrag.answer.tools.subagents import SubagentHost

    executor = AnswerExecutor(
        store=MagicMock(),
        pool=MagicMock(),
        retrieve=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        execution_environment="trust",
    )
    accepted = executor.acceptance_research_tools()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("tool definitions are never executed")

    runtime_tools = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=LocalExecutionEnvironment(tmp_path),
        subagent_host=SubagentHost(),
        skill_catalog=SkillCatalog(()),
    )
    runtime_by_name = {tool.name: tool for tool in runtime_tools}
    runtime_surface = tuple(runtime_by_name[tool.name] for tool in accepted)

    accepted_plan = AgentRunPlan.from_tools(
        accepted,
        model_role="query",
        context_policy_revision="policy-1",
    )
    runtime_plan = AgentRunPlan.from_tools(
        runtime_surface,
        model_role="query",
        context_policy_revision="policy-1",
    )

    assert runtime_plan.digest == accepted_plan.digest


def test_execution_rejects_tools_that_differ_from_the_accepted_agent_plan() -> None:
    from pydantic import BaseModel

    from dlightrag.agent.session.plan import AgentRunPlan
    from dlightrag.agent.tools import AgentTool, ToolResult
    from dlightrag.answer.executor import IncompatibleActiveRunError

    class Args(BaseModel):
        value: str

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("unused")

    accepted_tool = AgentTool("lookup", "Accepted description.", Args, execute)
    plan = AgentRunPlan.from_tools(
        (accepted_tool,),
        model_role="query",
        context_policy_revision="policy-1",
    )
    request = MagicMock(agent_run_plan=plan, context_policy_revision="policy-1")

    AnswerExecutor.validate_pinned_agent_run_plan(request, (accepted_tool,))
    with pytest.raises(IncompatibleActiveRunError, match="differs"):
        AnswerExecutor.validate_pinned_agent_run_plan(
            request,
            (AgentTool("lookup", "Changed description.", Args, execute),),
        )


def _resource_resolver() -> AnswerResourceResolver:
    capabilities = MagicMock()
    capabilities.refresh_answer = AsyncMock(
        return_value=AnswerCapabilities(
            answer=AnswerImageCapability(
                status="supported",
                configured_ceiling=3,
                effective_max_images=3,
                provider="test",
                base_url=None,
                model="test-model",
                failure_kind=None,
            ),
            vlm_status="unknown",
        )
    )
    return AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000_000,
            max_total_attachment_bytes=20_000_000,
            image_max_bytes=5_000_000,
            image_max_pixels=4_000_000,
        ),
        models=MagicMock(),
        capabilities=capabilities,
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


async def test_memory_recall_allowed_gating() -> None:
    """No settings checker keeps memory enabled; a false checker disables it."""
    assert await _memory_recall_allowed(None, owner_id="o") is True

    async def deny(**kwargs: Any) -> bool:
        del kwargs
        return False

    assert await _memory_recall_allowed(deny, owner_id="o") is False

    calls: list[str] = []

    async def allow(**kwargs: Any) -> bool:
        calls.append(kwargs["owner_id"])
        return True

    assert await _memory_recall_allowed(allow, owner_id="o") is True
    assert calls == ["o"]


async def test_child_model_calls_inherit_run_scheduler_ownership() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    second_queued = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def operation(label: str, *, block: bool = False) -> str:
        order.append(label)
        if block:
            first_started.set()
            await release_first.wait()
        return label

    async def execute(session: Any) -> Mapping[str, Any]:
        if session.run_id == "run-a":
            first = asyncio.create_task(scheduler.run(lambda: operation("a1", block=True)))
            await first_started.wait()
            second = asyncio.create_task(scheduler.run(lambda: operation("a2")))
            await asyncio.sleep(0)
            second_queued.set()
            await asyncio.gather(first, second)
            return {"run": "a"}
        await scheduler.run(lambda: operation("b1"))
        return {"run": "b"}

    executor = _executor()
    executor._execute = execute  # type: ignore[method-assign]
    run_a = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-a")))
    )
    await second_queued.wait()
    run_b = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-b")))
    )
    await asyncio.sleep(0)
    release_first.set()

    assert await asyncio.gather(run_a, run_b) == [{"run": "a"}, {"run": "b"}]
    assert order == ["a1", "b1", "a2"]


async def test_actionable_answer_errors_keep_their_public_message() -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=CurrentDocumentParseError("report.pdf")
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, MagicMock()))

    assert raised.value.kind == "CURRENT_DOCUMENT_PARSE_FAILED"
    assert "report.pdf" in raised.value.public_message


async def test_unknown_errors_map_to_generic_public_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("postgres://user:secret@host/db")
    )
    session = MagicMock(owner_id="owner", run_id="run-correlated")

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, session))

    assert raised.value.kind == "ANSWER_STREAM_FAILED"
    assert raised.value.public_message == "Answer run failed."
    assert "Answer run run-correlated execution failed" in caplog.text
    assert "postgres://user:secret@host/db" in caplog.text


async def test_url_current_image_is_pinned_once_for_durable_replay() -> None:
    resolver = _resource_resolver()
    image_bytes = _png_bytes()
    inline_bytes = b"notes"
    resolver.materialize_link_image = AsyncMock(return_value=image_bytes)  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png",
                filename="chart.png",
                ordinal=0,
                mime_type="image/png",
            ),
        ),
        attachments=(
            AttachmentReference(
                digest=artifact_digest(inline_bytes),
                filename="notes.txt",
                mime_type="text/plain",
                ordinal=0,
            ),
        ),
    )

    pinned, artifacts = await resolver.pin_current_image_links(request, (inline_bytes,))

    assert pinned.links == ()
    assert [item.filename for item in pinned.attachments] == ["chart.png", "notes.txt"]
    assert [item.ordinal for item in pinned.attachments] == [0, 1]
    assert artifacts == [image_bytes, inline_bytes]
    resources = await build_current_answer_resources(
        links=pinned.links,
        attachments=pinned.attachments,
        attachment_loaders=[
            in_memory_attachment_loader(image_bytes),
            in_memory_attachment_loader(inline_bytes),
        ],
    )
    resolver.materialize_link_image = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("durable replay must not refetch the URL")
    )
    images, _remaining, _image_resources = await resolver.prepare_current_images(resources)

    assert len(images) == 1
    resolver.materialize_link_image.assert_not_awaited()  # type: ignore[attr-defined]


async def test_unavailable_url_image_is_durably_demoted_to_an_ordinary_link() -> None:
    resolver = _resource_resolver()
    materialize = AsyncMock(return_value=None)
    resolver.materialize_link_image = materialize  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png",
                filename="chart.png",
                ordinal=0,
                mime_type="image/png",
            ),
        ),
    )

    pinned, artifacts = await resolver.pin_current_image_links(request, ())
    resources = await build_current_answer_resources(
        links=pinned.links,
        attachments=pinned.attachments,
        attachment_loaders=(),
    )
    images, _remaining, _image_resources = await resolver.prepare_current_images(resources)

    assert artifacts == []
    assert pinned.links[0].mime_type is None
    assert images == []
    materialize.assert_awaited_once()


async def test_stream_close_failure_does_not_skip_registry_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("stream close failed")

    registry = MagicMock(aclose=AsyncMock())

    await _close_execution_resources(Stream(), registry)

    registry.aclose.assert_awaited_once()
    assert "Failed to close Answer stream" in caplog.text


async def test_research_run_seeds_facts_without_duplicating_pinned_history() -> None:
    request = AnswerRunInput(
        query="resume",
        workspaces=("default",),
        history=(
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}", None),
                profile=ModelProfile(
                    context_window_tokens=262_144,
                    max_output_tokens=32_768,
                    supports_images=True,
                    supports_reasoning=True,
                ),
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision="m1-v1",
        model_catalog_revision="test",
        idempotency_fingerprint="request-hash",
        session_id=str(uuid.uuid7()),
    )
    prepared = MagicMock(tools=[], evidence=MagicMock(ledger_state_json=lambda: "{}"))
    orchestrator = MagicMock(resolved_mode="research")
    orchestrator.prepare_run.return_value = prepared
    orchestrator.staged_artifacts.return_value = ()
    orchestrator.answer_stream = AsyncMock(
        return_value=({"chunks": [], "entities": [], "relationships": []}, None)
    )
    registry = MagicMock(aclose=AsyncMock())
    executor = _executor()
    store = cast(Any, executor._store)
    store.load_routing = AsyncMock(
        return_value=MagicMock(
            resolved_mode="research",
            research_session_id=request.session_id,
            requested_mode="research",
            valid_modes=("fast", "research"),
        )
    )
    store.resolve = AsyncMock(return_value="research")
    executor.prepare_orchestrated_run = AsyncMock(  # type: ignore[method-assign]
        return_value=OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            workspaces=["default"],
            registry=registry,
        )
    )

    from dlightrag.agent.session.ids import SessionId
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        prepared_input=request.as_request(),
        enter_phase=AsyncMock(),
        flush_tokens=AsyncMock(),
        emit_token=AsyncMock(),
        execution=MagicMock(session_store=journal, progress_store=MagicMock()),
    )

    result = await executor.execute(cast(RunSession, session))

    assert result["answer"] == ""
    snapshot = await journal.load(SessionId(request.session_id))
    assert snapshot.version == 1
    kinds = {entry.__class__.__name__ for entry in snapshot.entries}
    assert "RunSegmentEntry" in kinds
    assert "UserMessageEntry" not in kinds  # pinned history is a contribution, not journal
    assert not any(
        str(getattr(entry, "key", "")).startswith("profile_memory_snapshot:")
        for entry in snapshot.entries
    )
    orchestrator.answer_stream.assert_awaited_once()


async def test_resumed_research_recovers_the_episode_from_the_folded_journal() -> None:
    request = AnswerRunInput(
        query="resume",
        workspaces=("default",),
        pinned_models=tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=ModelFingerprint("openai", f"test-{role}", None),
                profile=ModelProfile(
                    context_window_tokens=262_144,
                    max_output_tokens=32_768,
                    supports_images=True,
                    supports_reasoning=True,
                ),
            )
            for role in ("extract", "keyword", "query", "vlm")
        ),
        context_policy_revision="m1-v1",
        model_catalog_revision="test",
        idempotency_fingerprint="request-hash",
        session_id=str(uuid.uuid7()),
    )
    from datetime import UTC, datetime

    from dlightrag.agent.session.entries import (
        AssistantMessageEntry,
        RunSegmentEntry,
        UserMessageEntry,
    )
    from dlightrag.agent.session.ids import EntryId, SessionId
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    session_id = SessionId(request.session_id)
    now = datetime.now(UTC)
    journal = InMemoryAgentSessionStore()
    await journal.append(
        session_id=session_id,
        expected_version=0,
        entries=[
            UserMessageEntry(
                entry_id=EntryId.new(), session_id=session_id, timestamp=now, content="q"
            ),
            AssistantMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=now,
                content="done",
                stop_reason="stop",
            ),
        ],
    )
    recovered: list[str] = []
    prepared = MagicMock(
        tools=[],
        evidence=MagicMock(ledger_state_json=lambda: "{}"),
        working=MagicMock(
            record=MagicMock(side_effect=lambda exchange: recovered.append("record"))
        ),
    )
    orchestrator = MagicMock(resolved_mode="research")
    orchestrator.prepare_run.return_value = prepared
    orchestrator.staged_artifacts.return_value = ()
    orchestrator.answer_stream = AsyncMock(
        return_value=({"chunks": [], "entities": [], "relationships": []}, None)
    )
    orchestrator.recover_from_fold = AsyncMock(
        side_effect=lambda run, snapshot: recovered.append("recovered")
    )
    registry = MagicMock(aclose=AsyncMock())
    executor = _executor()
    store = cast(Any, executor._store)
    store.load_routing = AsyncMock(
        return_value=MagicMock(
            resolved_mode="research",
            research_session_id=request.session_id,
            requested_mode="research",
            valid_modes=("fast", "research"),
        )
    )
    store.resolve = AsyncMock(return_value="research")
    executor.prepare_orchestrated_run = AsyncMock(  # type: ignore[method-assign]
        return_value=OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            current_image_count=0,
            workspaces=["default"],
            registry=registry,
        )
    )
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        prepared_input=request.as_request(),
        enter_phase=AsyncMock(),
        flush_tokens=AsyncMock(),
        emit_token=AsyncMock(),
        execution=MagicMock(session_store=journal, progress_store=MagicMock()),
    )

    result = await executor.execute(cast(RunSession, session))

    assert result["answer"] == ""
    assert "recovered" in recovered
    resumed = await journal.load(session_id)
    segment = next(entry for entry in resumed.entries if isinstance(entry, RunSegmentEntry))
    assert segment.kind == "resume"
    assert segment.parent_head_id is not None
    orchestrator.answer_stream.assert_awaited_once()


async def test_journal_boundaries_apply_ordered_agent_controls() -> None:
    from dlightrag.agent.session.ids import SessionId
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    controls = (
        {"control_sequence": 1, "kind": "steer", "content": "first"},
        {"control_sequence": 2, "kind": "follow_up", "content": "second"},
    )
    reader = AsyncMock(side_effect=[controls, ()])
    acknowledge = AsyncMock(return_value=True)
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        control_reader=reader,
        control_ack=acknowledge,
    )

    assert await boundaries.apply_controls() is True
    assert await boundaries.apply_controls() is False
    snapshot = await journal.load(session_id)
    injections = [entry for entry in snapshot.entries if isinstance(entry, ContextInjectionEntry)]
    assert [entry.content for entry in injections] == ["first", "second"]
    assert [entry.label for entry in injections] == [
        "control:steer:1",
        "control:follow_up:2",
    ]
    acknowledge.assert_awaited_once_with((1, 2))


def test_fetched_resource_batches_are_atomic_and_session_scoped() -> None:
    from dlightrag.agent.session.effects import EffectIntent
    from dlightrag.agent.session.ids import IntentId, SessionId
    from dlightrag.agent.tools import ToolEffects
    from dlightrag.answer.resources.registry import (
        FetchedResourceBytes,
        ResourceEffectOwner,
    )
    from dlightrag.runtime.settlements import EffectHostUpdate

    buffer = FetchedResourceBuffer()
    parent = SessionId.new()
    child = SessionId.new()

    def append(scope: SessionId, resource_id: str) -> None:
        buffer.append(
            FetchedResourceBytes(
                resource_id=resource_id,
                ordinal=0,
                filename=f"{resource_id}.txt",
                mime_type="text/plain",
                url=f"https://example.com/{resource_id}",
                content=resource_id.encode(),
            ),
            ResourceEffectOwner(execution_scope=scope.value, call_id="call-1"),
        )

    append(parent, "parent-a")
    append(parent, "parent-b")
    append(child, "child-a")
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="read",
        replay_policy="replayable",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="call-1",
    )
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=MagicMock(),  # type: ignore[arg-type]
        session_id=parent,
        tools_by_name={},
        ledger_state=lambda: '{"chunks": []}',
        fetched_buffer=buffer,
        run_id="run-1",
    )

    update = boundaries._host_update(intent, ToolEffects())

    assert isinstance(update, EffectHostUpdate)
    assert len(update.evidence) == 1
    assert [item.resource.resource_id for item in update.fetched] == [
        "parent-a",
        "parent-b",
    ]
    assert [item.resource_id for item in buffer.drain(scope=child.value, call_id="call-1")] == [
        "child-a"
    ]


def test_memory_operation_details_become_a_typed_product_host_update() -> None:
    from dlightrag.agent.session.effects import EffectIntent
    from dlightrag.agent.session.ids import IntentId, SessionId
    from dlightrag.agent.tools import ToolEffects
    from dlightrag.answer.executor import FetchedResourceBuffer, JournalRunBoundaries

    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="remember",
        replay_policy="replayable",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="call-1",
    )
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=MagicMock(),  # type: ignore[arg-type]
        session_id=SessionId.new(),
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    update = boundaries._host_update(
        intent,
        ToolEffects(),
        {
            "memory_operation": {
                "operation": "remember",
                "outcome": "changed",
                "change_id": "change-1",
                "memory_ids": ["memory-1"],
                "kind": "preference",
                "body": "Use Chinese.",
            }
        },
    )

    assert update.memory_operation is not None
    assert update.memory_operation.change_id == "change-1"
    assert update.memory_operation.body == "Use Chinese."


async def test_non_last_spawn_settlement_carries_adopted_evidence() -> None:
    from dlightrag.agent.session.effects import EffectIntent
    from dlightrag.agent.session.ids import IntentId, SessionId
    from dlightrag.agent.session.store import EffectCommit
    from dlightrag.agent.tools import ToolExecution, ToolResult
    from dlightrag.agent.tools.contracts import ToolObservation
    from dlightrag.ai.messages import ToolCall
    from dlightrag.runtime.settlements import EffectHostUpdate

    session_id = SessionId.new()
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="spawn_agent",
        replay_policy="replayable",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id="spawn-call",
    )
    journal = AsyncMock()
    journal.settle_effect.return_value = EffectCommit(
        version=1,
        appended_sequences=(1,),
        intent_id=intent.intent_id,
        outcome="succeeded",
    )
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: '{"chunks": [{"child_session_id": "child-1"}]}',
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    execution = ToolExecution(
        call=ToolCall(id="spawn-call", name="spawn_agent", arguments={}),
        result=ToolResult.text("child complete"),
        observation=ToolObservation(
            tool="spawn_agent",
            call_id="spawn-call",
            outcome="ok",
            duration_ms=1,
            cached=False,
            is_error=False,
            content_chars=14,
        ),
    )

    await boundaries.settle_intent(intent, execution, turn_number=1, is_last=False)

    settlement = journal.settle_effect.await_args.kwargs["settlement"]
    assert isinstance(settlement.host_update, EffectHostUpdate)
    assert len(settlement.host_update.evidence) == 1
    assert b"child-1" in settlement.host_update.evidence[0].content


async def test_mixed_valid_invalid_tool_results_fold_in_source_order_live() -> None:
    from pydantic import BaseModel, ConfigDict

    from dlightrag.agent.session.fold import fold_entries
    from dlightrag.agent.session.ids import SessionId
    from dlightrag.agent.tools import AgentTool, ToolResult, ToolTurnExecutor
    from dlightrag.ai.messages import AssistantTurn, ToolCall
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    class SearchArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        query: str

    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(id="valid", name="search", arguments={"query": "q"}),
                ToolCall(id="invalid", name="missing", arguments={}),
            ),
            stop_reason="tool_use",
        )

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("found")

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    tool = AgentTool("search", "Search.", SearchArgs, execute, replay_policy="replayable")
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    executor = ToolTurnExecutor(model)
    prepared = await executor.prepare_turn([{"role": "user", "content": "q"}], [tool])
    await boundaries.commit_intents(prepared)
    await executor.execute_prepared(
        prepared,
        [tool],
        on_result=lambda intent, execution, is_last: boundaries.settle_intent(
            intent,
            execution,
            turn_number=1,
            is_last=is_last,
        ),
    )

    messages = fold_entries((await journal.load(session_id)).entries)
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "valid",
        "invalid",
    ]


async def test_mixed_valid_invalid_tool_results_fold_in_source_order_after_recovery() -> None:
    from pydantic import BaseModel, ConfigDict

    from dlightrag.agent.session.fold import fold_entries
    from dlightrag.agent.session.ids import SessionId
    from dlightrag.agent.tools import AgentTool, ToolResult, ToolTurnExecutor
    from dlightrag.ai.messages import AssistantTurn, ToolCall
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    class SearchArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        query: str

    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(
            text="",
            tool_calls=(
                ToolCall(id="valid", name="search", arguments={"query": "q"}),
                ToolCall(id="invalid", name="missing", arguments={}),
            ),
            stop_reason="tool_use",
        )

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("found")

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    tool = AgentTool("search", "Search.", SearchArgs, execute, replay_policy="replayable")
    initial = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    prepared = await ToolTurnExecutor(model).prepare_turn(
        [{"role": "user", "content": "q"}], [tool]
    )
    await initial.commit_intents(prepared)
    snapshot = await journal.load(session_id)
    recovered = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        entries=snapshot.entries,
    )

    await recovered.recover_pending_intents(snapshot)

    messages = fold_entries((await journal.load(session_id)).entries)
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
        "valid",
        "invalid",
    ]


async def test_durable_tool_error_folds_as_error_after_settlement() -> None:
    from dlightrag.agent.session.effects import EffectIntent
    from dlightrag.agent.session.fold import fold_entries
    from dlightrag.agent.session.ids import IntentId, SessionId
    from dlightrag.agent.tools import (
        PreparedToolTurn,
        ToolExecution,
        ToolPreflight,
        ToolResult,
    )
    from dlightrag.agent.tools.contracts import ToolObservation
    from dlightrag.ai.messages import AssistantTurn, ToolCall
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    call = ToolCall(id="call-error", name="broken", arguments={})
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name="broken",
        replay_policy="replayable",
        contract_version=1,
        input_schema_digest="a" * 64,
        canonical_input="{}",
        source_call_id=call.id,
    )
    boundaries = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    prepared = PreparedToolTurn(
        assistant=AssistantTurn(text="", tool_calls=(call,), stop_reason="tool_use"),
        preflight=ToolPreflight(intents=(intent,), validation_results=()),
        transcript=[],
    )
    await boundaries.commit_intents(prepared)
    execution = ToolExecution(
        call=call,
        result=ToolResult.text("Tool failed"),
        observation=ToolObservation(
            tool="broken",
            call_id=call.id,
            outcome="failed",
            duration_ms=1,
            cached=False,
            is_error=True,
            content_chars=11,
        ),
        is_error=True,
    )

    await boundaries.settle_intent(intent, execution, turn_number=1, is_last=True)

    messages = fold_entries((await journal.load(session_id)).entries)
    assert messages[-1]["is_error"] is True


async def test_never_effect_recovery_reports_unknown_without_reexecution() -> None:
    from pydantic import BaseModel

    from dlightrag.agent.session.entries import EffectResultEntry
    from dlightrag.agent.session.ids import SessionId
    from dlightrag.agent.tools import AgentTool, ToolResult, ToolTurnExecutor
    from dlightrag.ai.messages import AssistantTurn, ToolCall
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    class EmptyArgs(BaseModel):
        pass

    calls = 0

    async def model(**_kwargs: object) -> AssistantTurn:
        return AssistantTurn(
            text="",
            tool_calls=(ToolCall(id="call-never", name="mutate", arguments={}),),
            stop_reason="tool_use",
        )

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult.text("changed")

    tool = AgentTool("mutate", "Mutate.", EmptyArgs, execute)
    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    initial = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    prepared = await ToolTurnExecutor(model).prepare_turn(
        [{"role": "user", "content": "q"}], [tool]
    )
    await initial.commit_intents(prepared)
    snapshot = await journal.load(session_id)
    recovered = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        entries=snapshot.entries,
    )

    await recovered.recover_pending_intents(snapshot)

    result = next(
        entry
        for entry in (await journal.load(session_id)).entries
        if isinstance(entry, EffectResultEntry)
    )
    assert calls == 0
    assert result.result.outcome == "outcome_unknown"
    assert "may have happened" in result.result.text_content


async def test_replayable_effect_recovery_fits_oversized_observation() -> None:
    from pydantic import BaseModel

    from dlightrag.agent.session.effects import EffectIntent
    from dlightrag.agent.session.entries import EffectResultEntry
    from dlightrag.agent.session.ids import IntentId, SessionId
    from dlightrag.agent.tools import AgentTool, PreparedToolTurn, ToolPreflight, ToolResult
    from dlightrag.ai.capacity import CONTEXT_POLICY
    from dlightrag.ai.messages import AssistantTurn, ToolCall
    from dlightrag.ai.tokens import estimate_tokens
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    class EmptyArgs(BaseModel):
        pass

    payload = "x" * 200_000

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text(payload)

    tool = AgentTool(
        "large",
        "Large replayable result.",
        EmptyArgs,
        execute,
        replay_policy="replayable",
    )
    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    call = ToolCall(id="call-large", name=tool.name, arguments={})
    intent = EffectIntent(
        intent_id=IntentId.new(),
        tool_name=tool.name,
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        input_schema_digest=tool.input_schema_digest,
        canonical_input="{}",
        source_call_id=call.id,
    )
    initial = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
    )
    await initial.commit_intents(
        PreparedToolTurn(
            assistant=AssistantTurn(text="", tool_calls=(call,), stop_reason="tool_use"),
            preflight=ToolPreflight(intents=(intent,), validation_results=()),
            transcript=[],
        )
    )
    snapshot = await journal.load(session_id)
    recovered = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={tool.name: tool},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        entries=snapshot.entries,
        active_projection=snapshot.active_projection,
    )

    await recovered.recover_pending_intents(snapshot)

    result = next(
        entry
        for entry in (await journal.load(session_id)).entries
        if isinstance(entry, EffectResultEntry)
    )
    assert len(result.result.text_content) < len(payload)
    assert estimate_tokens(result.result.text_content) <= CONTEXT_POLICY.observation_reserve_tokens


async def test_durable_child_usage_aggregates_roster_rows() -> None:
    from dlightrag.answer.executor import _durable_child_usage

    store = MagicMock()
    store.list_child_sessions = AsyncMock(
        return_value=(
            {"usage": {"input_tokens": 3, "output_tokens": 2}},
            {"usage": {"input_tokens": 5, "output_tokens": 1}},
            {"usage": None},
        )
    )

    assert await _durable_child_usage(store, owner_id="owner", run_id="run-1") == {
        "input_tokens": 8,
        "output_tokens": 3,
    }


async def test_profile_memory_recall_snapshot_is_replay_stable() -> None:
    from dlightrag.agent.session.ids import SessionId
    from dlightrag.answer.executor import _resolve_profile_memory_snapshot
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    memory = AsyncMock()
    memory.recall.return_value = MagicMock(records=(), content_chars=0)
    snapshot = await journal.load(session_id)

    first = await _resolve_profile_memory_snapshot(
        memory=memory,
        journal=journal,  # type: ignore[arg-type]
        snapshot=snapshot,
        session_id=session_id,
        owner_id="owner",
        run_id="run-1",
        query="question",
    )
    second = await _resolve_profile_memory_snapshot(
        memory=memory,
        journal=journal,  # type: ignore[arg-type]
        snapshot=first[3],
        session_id=session_id,
        owner_id="owner",
        run_id="run-1",
        query="different replay query",
    )

    third = await _resolve_profile_memory_snapshot(
        memory=memory,
        journal=journal,  # type: ignore[arg-type]
        snapshot=second[3],
        session_id=session_id,
        owner_id="owner",
        run_id="run-2",
        query="new run query",
    )

    assert first[:3] == second[:3]
    assert third[:3] == first[:3]
    assert memory.recall.await_count == 2


async def test_control_replay_deduplicates_after_append_before_ack() -> None:
    from dlightrag.agent.session.ids import SessionId
    from dlightrag.runtime import LeaseLostError
    from tests.in_memory_session_store import InMemoryAgentSessionStore

    journal = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    control = ({"control_sequence": 7, "kind": "steer", "content": "stable"},)
    first = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        control_reader=AsyncMock(return_value=control),
        control_ack=AsyncMock(return_value=False),
    )
    with pytest.raises(LeaseLostError):
        await first.apply_controls()
    snapshot = await journal.load(session_id)
    replay_ack = AsyncMock(return_value=True)
    replay = JournalRunBoundaries(
        session=MagicMock(),
        journal=journal,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id="run-1",
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        entries=snapshot.entries,
        active_projection=snapshot.active_projection,
        control_reader=AsyncMock(return_value=control),
        control_ack=replay_ack,
    )

    assert await replay.apply_controls() is False
    assert len((await journal.load(session_id)).entries) == 1
    replay_ack.assert_awaited_once_with((7,))
