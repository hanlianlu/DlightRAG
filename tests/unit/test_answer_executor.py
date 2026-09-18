# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import datetime
import io
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from PIL import Image

from dlightrag.adapters.observability import LangfuseTelemetry
from dlightrag.adapters.observability import langfuse as langfuse_state
from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.engine.agent.session.ids import EntryId, LaneId, ProjectionId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import ContextProjectionRegister, SetRegister
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.reasoning import best_effort_reasoning_profile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.capabilities import (
    AnswerCapabilities,
    RequestModelContext,
)
from dlightrag.engine.answer.errors import (
    CurrentDocumentParseError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.execution import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
)
from dlightrag.engine.answer.execution.executor import (
    _child_lifecycle_for_plan,
    _close_execution_resources,
    _memory_recall_allowed,
    _stage_publications,
)
from dlightrag.engine.answer.execution.input import (
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.engine.answer.fast import FastSessionHost, ensure_session_lane
from dlightrag.engine.answer.highlights import SemanticHighlightSettings
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.publication import prepare_artifact_attachment, validate_publication
from dlightrag.engine.answer.resources import ResourceInput
from dlightrag.engine.dependencies import ProviderUnavailableError
from dlightrag.engine.runtime.coordinator import RunCancellationObserved, RunSession
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.records import (
    Deferred,
    RunExecutionOutcome,
    Succeeded,
    artifact_digest,
)
from tests.unit.conftest import RecordingLangfuse, answer_image_policy


@pytest.mark.asyncio
async def test_missing_fork_session_is_a_typed_run_conflict() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()

    with pytest.raises(RunExecutionError) as raised:
        await ensure_session_lane(
            repository=store,
            snapshot=await store.load(session_id),
            fencing_epoch=1,
            session_id=session_id,
            lane_id=LaneId.new(),
            source_lane_id=LaneId.main(),
        )

    assert raised.value.kind == "agent_session_conflict"


@pytest.mark.asyncio
async def test_a_fork_without_a_recorded_point_refuses_instead_of_using_the_tip() -> None:
    """Older Runs have no Fork Point; guessing the Lane tip is the divergence between a recorded Fork Point and a Lane tip."""
    executor = _executor()
    session_id = SessionId.new()
    snapshot = await MemoryAgentSessionRepository[None]().load(session_id)
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=_routing_record(session_id.value, fork_point_entry_id=None)
        )
    )
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id="parent",
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError, match="Fork from a Run that has one") as raised:
        await executor._resolve_fork_seed(cast(RunSession, session), cast(Any, request), snapshot)

    assert raised.value.kind == "fork_point_missing"


@pytest.mark.asyncio
async def test_a_fork_from_a_missing_parent_refuses() -> None:
    executor = _executor()
    session_id = SessionId.new()
    snapshot = await MemoryAgentSessionRepository[None]().load(session_id)
    executor._store = MagicMock(load_routing=AsyncMock(return_value=None))
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id="missing",
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError, match="Fork from a Run that still exists") as raised:
        await executor._resolve_fork_seed(cast(RunSession, session), cast(Any, request), snapshot)

    assert raised.value.kind == "fork_point_missing"


@pytest.mark.asyncio
async def test_a_fork_from_another_session_refuses() -> None:
    executor = _executor()
    session_id = SessionId.new()
    snapshot = await MemoryAgentSessionRepository[None]().load(session_id)
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=_routing_record(
                SessionId.new().value, fork_point_entry_id=EntryId.new().value
            )
        )
    )
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id="parent",
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError, match="Fork from a Run in this conversation") as raised:
        await executor._resolve_fork_seed(cast(RunSession, session), cast(Any, request), snapshot)

    assert raised.value.kind == "fork_point_missing"


@pytest.mark.asyncio
async def test_a_fork_whose_recorded_head_is_gone_refuses() -> None:
    executor = _executor()
    session_id = SessionId.new()
    snapshot = await MemoryAgentSessionRepository[None]().load(session_id)
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=_routing_record(session_id.value, fork_point_entry_id=EntryId.new().value)
        )
    )
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id="parent",
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError, match="whose head is still present") as raised:
        await executor._resolve_fork_seed(cast(RunSession, session), cast(Any, request), snapshot)

    assert raised.value.kind == "fork_point_stale"


def _routing_record(
    session_id: str,
    *,
    fork_point_entry_id: str | None,
    fork_point_projection_id: str | None = None,
    agent_lane_id: str = "main",
) -> Any:
    from dlightrag.engine.answer.runs.routing import RoutingRecord

    return RoutingRecord(
        requested_mode="fast",
        valid_modes=("fast",),
        resolved_mode="fast",
        agent_session_id=session_id,
        agent_lane_id=agent_lane_id,
        source_lane_id=None,
        fork_point_entry_id=fork_point_entry_id,
        fork_point_projection_id=fork_point_projection_id,
    )


def _fingerprint(role: str) -> ModelFingerprint:
    return ModelFingerprint("openai", f"test-{role}", None)


def _executor() -> AnswerExecutor:
    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
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
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
    )

    # These unit doubles replace execution; dedicated model-contract tests exercise preflight.
    executor.validate_active_prepared_input = Mock()
    return executor


def test_markdown_artifacts_keep_independent_citation_sources(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Primary fact [1-1].", encoding="utf-8")
    (root / "appendix.md").write_text("Appendix fact [2-1].", encoding="utf-8")
    plan = validate_publication(
        root,
        answer=("[Open analysis](artifact:analysis.md) [Open appendix](artifact:appendix.md)"),
        attachments=(
            prepare_artifact_attachment(root, path="analysis.md"),
            prepare_artifact_attachment(root, path="appendix.md"),
        ),
    )
    contexts = {
        "chunks": [
            {
                "chunk_id": "chunk-primary",
                "reference_id": "1",
                "file_path": "primary.pdf",
                "content": "Primary fact.",
                "_workspace": "default",
                "full_doc_id": "doc-primary",
                "metadata": {
                    "source_uri": "local://default/primary.pdf",
                    "source_download_locator": "/private/primary.pdf",
                    "source_file_name": "primary.pdf",
                },
            },
            {
                "chunk_id": "chunk-appendix",
                "reference_id": "2",
                "file_path": "appendix.pdf",
                "content": "Appendix fact.",
                "_workspace": "default",
                "full_doc_id": "doc-appendix",
                "metadata": {
                    "source_uri": "local://default/appendix.pdf",
                    "source_download_locator": "/private/appendix.pdf",
                    "source_file_name": "appendix.pdf",
                },
            },
        ]
    }

    publications, descriptors, artifact_sources = _stage_publications(
        plan=plan,
        answer=plan.answer,
        contexts=contexts,
        session_id="01930000-0000-7000-8000-000000000001",
    )

    resource_by_filename = {
        str(descriptor["filename"]): str(descriptor["resource_id"]) for descriptor in descriptors
    }
    assert [source.id for source in artifact_sources[resource_by_filename["analysis.md"]]] == ["1"]
    assert [source.id for source in artifact_sources[resource_by_filename["appendix.md"]]] == ["2"]
    content_by_filename = {
        publication.filename: publication.content for publication in publications
    }
    assert content_by_filename["analysis.md"] == b"Primary fact [1-1]."
    assert content_by_filename["appendix.md"] == b"Appendix fact [2-1]."


def test_acceptance_research_tools_include_every_configured_non_resource_surface() -> None:
    from dlightrag.engine.agent.skills import SkillsBundle

    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
        memory_store=MagicMock(),
        skills_bundle_factory=lambda owner_id, requested_skill=None: SkillsBundle(
            global_root=Path("/nonexistent-global-skills"),
        ),
    )

    names = {tool.name for tool in executor.acceptance_research_tools()}

    assert {
        "read",
        "write",
        "edit",
        "attach_artifact",
        "grep",
        "bash",
        "spawn_agent",
        "subagent_status",
        "wait_subagent",
        "cancel_subagent",
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
        "remember",
        "forget",
        "recall_memory",
        "load_skill",
    } <= names


def test_pinned_child_lifecycle_requires_current_contract() -> None:
    from pydantic import BaseModel

    from dlightrag.engine.agent.tools import AgentTool, ToolResult
    from dlightrag.engine.answer.tools.subagents import SubagentHost, subagent_tools
    from dlightrag.engine.runtime.errors import IncompatibleActiveRunError

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("unused")

    def _plan(*tools: AgentTool) -> AgentRunPlan:
        return AgentRunPlan.from_tools(
            tools, model_role="query", context_policy_revision="policy-1"
        )

    def _spawn(*, async_lifecycle: bool, interactive_controls: bool = True) -> AgentTool:
        return next(
            tool
            for tool in subagent_tools(
                host=SubagentHost(
                    async_lifecycle=async_lifecycle,
                    interactive_controls=interactive_controls,
                )
            )
            if tool.name == "spawn_agent"
        )

    supported = _spawn(async_lifecycle=True)
    unsupported = AgentTool(
        supported.name,
        supported.description,
        supported.input_model,
        execute,
        replay_policy=supported.replay_policy,
        contract_version=9,
    )
    from dataclasses import replace

    for version in (2, 3, 4):
        with pytest.raises(IncompatibleActiveRunError):
            _child_lifecycle_for_plan(_plan(replace(supported, contract_version=version)))
    assert _child_lifecycle_for_plan(_plan(supported)) == (True, True)
    with pytest.raises(IncompatibleActiveRunError):
        _child_lifecycle_for_plan(_plan(unsupported))


def test_acceptance_plan_matches_runtime_tool_composition(tmp_path: Path) -> None:
    from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
    from dlightrag.engine.answer.evidence import EvidenceLedger
    from dlightrag.engine.answer.tools.composition import compose_research_tools
    from dlightrag.engine.answer.tools.subagents import SubagentHost

    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
    )
    accepted = executor.acceptance_research_tools()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("tool definitions are never executed")

    async def read_resource(_request: Any, _runtime: Any) -> Any:
        raise RuntimeError("tool definitions are never executed")

    runtime_tools = compose_research_tools(
        injected_tools=[],
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=retrieve,  # type: ignore[arg-type]
        search_web=None,
        register_web_source=None,
        resource_reader=read_resource,
        environment=LocalExecutionEnvironment(tmp_path),
        artifacts_root=tmp_path / "artifacts",
        publication_limits=executor._settings.publication,
        subagent_host=SubagentHost(),
        skill_tools=[],
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

    from dlightrag.engine.agent.tools import AgentTool, ToolResult
    from dlightrag.engine.runtime.errors import IncompatibleActiveRunError

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
    with pytest.raises(IncompatibleActiveRunError, match="missing"):
        AnswerExecutor.validate_pinned_agent_run_plan(
            MagicMock(agent_run_plan=None),
            (accepted_tool,),
        )
    with pytest.raises(IncompatibleActiveRunError, match="differs"):
        AnswerExecutor.validate_pinned_agent_run_plan(
            request,
            (AgentTool("lookup", "Changed description.", Args, execute),),
        )


def test_pinned_model_profile_preserves_unverified_reasoning_semantics() -> None:
    pinned = PinnedModelProfile(
        role="query",
        fingerprint=_fingerprint("query"),
        profile=ModelProfile(
            context_window_tokens=10_000,
            max_output_tokens=8_000,
            reasoning=best_effort_reasoning_profile("openrouter"),
        ),
    )

    restored = PinnedModelProfile.from_json(pinned.as_json())

    assert restored == pinned
    assert restored.profile.reasoning is not None
    assert restored.profile.reasoning.best_effort is True


def test_execution_rejects_changed_context_or_model_pins() -> None:
    from dlightrag.engine.runtime.errors import IncompatibleActiveRunError

    pins = tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=_fingerprint(role),
            profile=ModelProfile(context_window_tokens=10_000),
            reasoning_settings={"ordinary": None, "agentic": None},
        )
        for role in ("extract", "keyword", "query", "vlm", "default")
    )
    executor = _executor()
    request = MagicMock(
        pinned_models=pins,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
    )
    executor._models.model_settings = lambda role: ModelSettings(model="test")
    executor.validate_pinned_model_profiles(request)

    request.model_catalog_revision = "stale-catalog"
    with pytest.raises(IncompatibleActiveRunError, match="model catalog"):
        executor.validate_pinned_model_profiles(request)

    request.model_catalog_revision = current_model_catalog_revision()
    request.context_policy_revision = "stale-policy"
    with pytest.raises(IncompatibleActiveRunError, match="context policy"):
        executor.validate_pinned_model_profiles(request)

    request.context_policy_revision = CONTEXT_POLICY_REVISION
    mismatched = _executor()
    mismatched._models.model_settings = lambda role: ModelSettings(model="test")
    mismatched._model_fingerprint_for_role = lambda role: ModelFingerprint(
        "other", f"test-{role}", None
    )
    with pytest.raises(IncompatibleActiveRunError, match="model endpoint"):
        mismatched.validate_pinned_model_profiles(request)


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
        resource_identity_secret=b"i" * 32,
        resource_cursor_secret=b"c" * 32,
    )


def test_resource_identity_is_stable_within_run_and_isolated_across_runs() -> None:
    resolver = _resource_resolver()
    resources = [ResourceInput(url="https://example.com/report")]

    first = resolver.build_resource_context(
        resources,
        resource_scope="owner-a\0run-1",
    )
    again = resolver.build_resource_context(
        resources,
        resource_scope="owner-a\0run-1",
    )
    other = resolver.build_resource_context(
        resources,
        resource_scope="owner-a\0run-2",
    )

    assert first is not None and again is not None and other is not None
    assert first.manifest()[0].resource_id == again.manifest()[0].resource_id
    assert first.manifest()[0].resource_id != other.manifest()[0].resource_id


def _png_bytes(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _multimodal_resolver(
    capability: AnswerImageCapability,
    *,
    policy_overrides: Mapping[str, int] | None = None,
) -> AnswerResourceResolver:
    models = MagicMock()
    models.web_sources.return_value = None
    models.vlm_func.return_value = AsyncMock()
    capabilities = MagicMock()
    overrides = dict(policy_overrides or {})
    capabilities.answer_image_policy.side_effect = lambda profile: answer_image_policy(
        max_images=capability.configured_ceiling if profile.supports_images else 0,
        **overrides,
    )
    capabilities.vlm_image_policy.side_effect = lambda profile: answer_image_policy(
        max_images=capability.configured_ceiling if profile.supports_images else 0,
        **overrides,
    )
    return AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000_000,
            max_total_attachment_bytes=20_000_000,
            image_max_bytes=5_000_000,
            image_max_pixels=4_000_000,
        ),
        models=models,
        capabilities=capabilities,
    )


def _image_capability(
    status: str,
    *,
    configured_ceiling: int = 3,
) -> AnswerImageCapability:
    return AnswerImageCapability(
        status=cast(Any, status),
        configured_ceiling=configured_ceiling,
        effective_max_images=configured_ceiling if status == "supported" else 0,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )


def _request_models(*, query_images: bool, vlm_images: bool) -> RequestModelContext:
    return RequestModelContext(
        extract=ModelProfile(context_window_tokens=100_000),
        query=ModelProfile(context_window_tokens=100_000, supports_images=query_images),
        vlm=ModelProfile(context_window_tokens=100_000, supports_images=vlm_images),
    )


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

    async def execute(session: Any, _run_trace: Any) -> RunExecutionOutcome:
        if session.run_id == "run-a":
            first = asyncio.create_task(scheduler.run(lambda: operation("a1", block=True)))
            await first_started.wait()
            second = asyncio.create_task(scheduler.run(lambda: operation("a2")))
            await asyncio.sleep(0)
            second_queued.set()
            await asyncio.gather(first, second)
            return Succeeded({"run": "a"})
        await scheduler.run(lambda: operation("b1"))
        return Succeeded({"run": "b"})

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

    assert await asyncio.gather(run_a, run_b) == [
        Succeeded({"run": "a"}),
        Succeeded({"run": "b"}),
    ]
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


@pytest.mark.parametrize(
    ("error", "checkpoint_key"),
    [
        (CorpusUnavailableError("offline"), "corpus_unavailable_attempt"),
        (ProviderUnavailableError(), "providers_unavailable_attempt"),
    ],
)
async def test_transient_answer_dependency_interruption_defers_same_run(
    error: Exception,
    checkpoint_key: str,
) -> None:
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    executor = _executor()
    executor._now = lambda: now
    executor._execute = AsyncMock(side_effect=error)  # type: ignore[method-assign]
    session = MagicMock(
        owner_id="owner",
        run_id="same-run",
        checkpoint={checkpoint_key: 2},
    )
    session.check_cancelled = AsyncMock()
    session.reset_output = AsyncMock()

    outcome = await executor.execute(cast(RunSession, session))

    assert isinstance(outcome, Deferred)
    assert outcome.checkpoint == {checkpoint_key: 3}
    assert (outcome.next_attempt_at - now).total_seconds() == 20
    session.check_cancelled.assert_awaited_once_with()
    session.reset_output.assert_awaited_once_with()


@pytest.mark.usefixtures("reset_langfuse_client")
async def test_the_run_trace_carries_question_attribution_and_hands_the_pipeline_its_root() -> None:
    """The pipeline that knows the answer writes the root; the root carries the question."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)
    executor = _executor()
    executor._telemetry = LangfuseTelemetry()
    executor._execute = AsyncMock(return_value=Succeeded({"answer": "the answer"}))  # type: ignore[method-assign]
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        prepared_input={
            "query": "why the sky is blue",
            "agent_session_id": "sess-1",
            "workspaces": ["default"],
        },
    )

    await executor.execute(cast(RunSession, session))

    root = client.observations[0]
    assert root.kwargs["name"] == "run-answer"
    assert root.kwargs["as_type"] == "agent"
    assert root.kwargs["input"] == {"query": "why the sky is blue"}
    assert root.kwargs["metadata"] == {
        "run_id": "run-1",
        "parent_run_id": None,
        "workspaces": ("default",),
    }

    calls = executor._execute.await_args_list if executor._execute.await_count else []
    handed = calls[0].args[1]
    handed.update(output={"answer": "the answer"})
    assert client.observations[0].updates == [{"output": {"answer": "the answer"}}]


@pytest.mark.usefixtures("reset_langfuse_client")
async def test_a_deferred_run_reports_why_it_has_no_answer() -> None:
    """One writer per path: the deferral path says which dependency deferred the attempt."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)
    executor = _executor()
    executor._telemetry = LangfuseTelemetry()
    executor._now = lambda: datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    executor._execute = AsyncMock(side_effect=ConnectionError("provider down"))  # type: ignore[method-assign]
    session = MagicMock(
        owner_id="owner",
        run_id="run-1",
        checkpoint={"providers": 2},
        prepared_input={"query": "q"},
    )
    session.check_cancelled = AsyncMock()
    session.reset_output = AsyncMock()

    outcome = await executor.execute(cast(RunSession, session))

    assert isinstance(outcome, Deferred)
    assert client.observations[0].updates == [
        {"output": {"outcome": "deferred", "component": "providers"}}
    ]


@pytest.mark.usefixtures("reset_langfuse_client")
async def test_a_cancelled_run_is_recorded_as_a_cancelled_outcome() -> None:
    """A user-requested stop is terminal, not a failure; the trace must not say ERROR."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)
    executor = _executor()
    executor._telemetry = LangfuseTelemetry()
    executor._execute = AsyncMock(side_effect=RunCancellationObserved())  # type: ignore[method-assign]
    session = MagicMock(owner_id="owner", run_id="run-1", prepared_input={"query": "q"})
    session.cancel_requested = True

    with pytest.raises(RunCancellationObserved):
        await executor.execute(cast(RunSession, session))

    assert client.observations[0].updates[-1] == {
        "level": "DEFAULT",
        "output": {"outcome": "cancelled"},
    }


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
    assert "postgres://user:secret@host/db" not in caplog.text


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
                mime_type=None,
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


async def test_research_multimodal_query_gets_all_raw_images_and_resource_handles() -> None:
    capability = _image_capability("supported", configured_ceiling=2)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=True, vlm_images=True)
    resources = [
        ResourceInput(filename="white.png", content=_png_bytes("white"), declared_mime="image/png"),
        ResourceInput(filename="black.png", content=_png_bytes("black"), declared_mime="image/png"),
    ]

    resolved = await resolver.resolve(
        resources,
        models=models,
        confirm_image_context=AsyncMock(return_value=(models, capability)),
        resolved_mode="research",
    )

    try:
        assert [block["type"] for block in resolved.query_images or ()] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert len(resolved.resource_manifest) == 2
        assert all(
            entry.resource_id in str(resolved.query_images) for entry in resolved.resource_manifest
        )
    finally:
        assert resolved.registry is not None
        await resolved.registry.aclose()


@pytest.mark.parametrize("query_status", ["unsupported", "unknown"])
async def test_research_text_only_query_rejects_images_without_visual_fallback(
    query_status: str,
) -> None:
    capability = _image_capability(query_status, configured_ceiling=2)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=False, vlm_images=True)
    resources = [
        ResourceInput(filename="white.png", content=_png_bytes("white"), declared_mime="image/png"),
        ResourceInput(filename="black.png", content=_png_bytes("black"), declared_mime="image/png"),
    ]

    from dlightrag.engine.answer.errors import AnswerImageError

    with pytest.raises(AnswerImageError):
        await resolver.resolve(
            resources,
            models=models,
            confirm_image_context=AsyncMock(return_value=(models, capability)),
            resolved_mode="research",
        )


async def test_research_image_admission_enforces_configured_count() -> None:
    capability = _image_capability("unsupported", configured_ceiling=1)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=False, vlm_images=True)

    with pytest.raises(CurrentImagePayloadError, match="at most 1"):
        await resolver.resolve(
            [
                ResourceInput(
                    filename="white.png",
                    content=_png_bytes("white"),
                    declared_mime="image/png",
                ),
                ResourceInput(
                    filename="black.png",
                    content=_png_bytes("black"),
                    declared_mime="image/png",
                ),
            ],
            models=models,
            confirm_image_context=AsyncMock(return_value=(models, capability)),
            resolved_mode="research",
        )


async def test_research_current_image_budget_is_all_or_error() -> None:
    first = _png_bytes("white")
    capability = _image_capability("supported", configured_ceiling=2)
    resolver = _multimodal_resolver(
        capability,
        policy_overrides={
            "max_total_bytes": len(first),
            "max_bytes_per_image": len(first),
        },
    )
    models = _request_models(query_images=True, vlm_images=True)

    with pytest.raises(CurrentImagePayloadError, match="query_image_2"):
        await resolver.resolve(
            [
                ResourceInput(filename="one.png", content=first, declared_mime="image/png"),
                ResourceInput(
                    filename="two.png",
                    content=_png_bytes("black"),
                    declared_mime="image/png",
                ),
            ],
            models=models,
            confirm_image_context=AsyncMock(return_value=(models, capability)),
            resolved_mode="research",
        )


async def test_unavailable_url_image_rejects_the_whole_current_request() -> None:
    resolver = _resource_resolver()
    materialize = AsyncMock(return_value=None)
    resolver.materialize_link_image = materialize  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png?version=1",
                filename=None,
                ordinal=0,
                mime_type=None,
            ),
        ),
    )

    with pytest.raises(CurrentImagePayloadError, match="could not be fetched and verified"):
        await resolver.pin_current_image_links(request, ())

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


async def test_durable_child_usage_aggregates_roster_rows() -> None:
    from dlightrag.engine.answer.research.runtime import _durable_child_usage

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


def test_public_document_citations_are_projected_into_the_published_artifact(
    tmp_path: Path,
) -> None:
    """A published file carries links for citations whose source has a public URL."""
    from dlightrag.engine.runtime.records import artifact_digest

    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text(
        "Web fact [1]. Local fact [2]. Web excerpt [1-1]. Local excerpt [2-1].",
        encoding="utf-8",
    )
    plan = validate_publication(
        root,
        answer="[Open report](artifact:report.md)",
        attachments=(prepare_artifact_attachment(root, path="report.md"),),
    )
    contexts = {
        "chunks": [
            {
                "chunk_id": "chunk-web",
                "reference_id": "1",
                "file_path": "沃尔沃汽车将在全球裁员近3000人",
                "content": "Web fact.",
                "_workspace": "__web_search__",
                "full_doc_id": "web-1",
                "metadata": {
                    "source_uri": "https://www.cls.cn/detail/2041214",
                    "source_download_locator": "https://www.cls.cn/detail/2041214",
                },
            },
            {
                "chunk_id": "chunk-local",
                "reference_id": "2",
                "file_path": "primary.pdf",
                "content": "Local fact.",
                "_workspace": "default",
                "full_doc_id": "doc-primary",
                "metadata": {
                    "source_uri": "local://default/primary.pdf",
                    "source_download_locator": "/private/primary.pdf",
                    "source_file_name": "primary.pdf",
                },
            },
        ]
    }

    publications, descriptors, _ = _stage_publications(
        plan=plan,
        answer=plan.answer,
        contexts=contexts,
        session_id="01930000-0000-7000-8000-000000000001",
    )

    (publication,) = publications
    (descriptor,) = descriptors
    assert (
        publication.content
        == (
            "Web fact [1](<https://www.cls.cn/detail/2041214>"
            ' "沃尔沃汽车将在全球裁员近3000人").'
            " Local fact [2]."
            " Web excerpt [1-1](<https://www.cls.cn/detail/2041214>"
            ' "沃尔沃汽车将在全球裁员近3000人").'
            " Local excerpt [2-1]."
        ).encode()
    )
    # The descriptor addresses the projected bytes, not the pre-projection ones.
    assert descriptor["byte_size"] == len(publication.content)
    assert descriptor["digest"] == artifact_digest(publication.content)


@pytest.mark.asyncio
async def test_recovery_restores_an_adopted_resource_under_its_own_handle() -> None:
    """An adopted Resource must survive a resume.

    A Run that adopted an earlier Run's document pins it as its own fetch, so
    recovery has to rebuild that Resource (and its earlier handle as alias) rather
    than treat the row as a Web catalog entry it never was.
    """
    import hashlib

    from dlightrag.engine.answer.resources.registry import ResourceRegistry
    from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
    from dlightrag.engine.runtime.records import RunFetchedResource

    document = b"%PDF-1.7 adopted earlier"
    snapshot = ConversionSnapshot(
        resource_id="res-earlier",
        input_digest=hashlib.sha256(document).hexdigest(),
        text="Adopted text.",
        visuals=(),
        extraction_status="complete",
        converter="fixture",
        converter_version="1",
    )
    effects = snapshot.effects()
    encoded = next(item.content for item in effects if item.resource_kind == "conversion_snapshot")
    blobs = {
        hashlib.sha256(document).hexdigest(): document,
        hashlib.sha256(encoded).hexdigest(): encoded,
    }
    rows = (
        RunFetchedResource(
            resource_id="res-adopted",
            ordinal=0,
            digest=hashlib.sha256(document).hexdigest(),
            filename="earlier.pdf",
            mime_type="application/pdf",
            source_locator=b"res-earlier",
            capabilities={"resource_kind": "lineage_adoption", "resource_aliases": ["res-earlier"]},
        ),
        RunFetchedResource(
            resource_id="res-earlier-conversion",
            ordinal=0,
            digest=hashlib.sha256(encoded).hexdigest(),
            filename="conversion.json",
            mime_type="application/json",
            source_locator=b"res-earlier",
            capabilities={"resource_kind": "conversion_snapshot"},
        ),
    )
    executor = _executor()
    executor._store.list_fetched_resources = AsyncMock(return_value=rows)

    async def stream(*, owner_id: str, digest: str, **kwargs: object):
        del owner_id, kwargs
        yield blobs[digest]

    executor._blob_store.stream = stream

    async with ResourceRegistry() as registry:
        await executor._restore_registry_fetches(registry, owner_id="owner", run_id="run")

        adopted = registry.canonical_resource_id("res-earlier")
        assert adopted.startswith("res-")
        assert registry.canonical_resource_id("res-adopted") == adopted
        assert registry.canonical_resource_id(adopted) == adopted
        read = await registry.read(adopted, max_window_tokens=1000)
        assert "Adopted text." in read.content


async def test_a_settled_run_records_the_state_it_ended_at() -> None:
    """A Fork branches from a recorded Fork Point, so the settlement has to write one.

    The head comes from the Lane the routing row fixed at acceptance, and the
    projection from that Lane's own register: a Run that compacted must hand a Fork
    the summary it was working from, not the whole transcript it had discarded.
    """
    from dlightrag.engine.agent.session.ids import ProjectionId
    from dlightrag.engine.agent.session.projection import ContextProjection
    from dlightrag.engine.answer.runs.routing import RoutingRecord

    executor = _executor()
    executor._execute = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    repository = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    lane_id = LaneId.main()
    snapshot = await repository.load(session_id)

    async def no_result() -> None:
        return None

    host = FastSessionHost(
        repository=repository,
        initial_snapshot=snapshot,
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=lane_id,
        reservation_id="one",
        idempotency_key="one-key",
        content="question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=lane_id,
        reservation_id="one",
        content="answer",
    )
    head = (await repository.load(session_id)).tree.lane(lane_id).head_entry_id
    assert head is not None
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
    )
    await repository.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(ContextProjectionRegister(lane_id, projection))],
            expectations=[
                RegisterExpectation(ContextProjectionRegister(lane_id, projection).ref, None)
            ],
        ),
    )
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=RoutingRecord(
                requested_mode="auto",
                valid_modes=("research",),
                resolved_mode="research",
                agent_session_id=session_id.value,
                agent_lane_id=lane_id.value,
                source_lane_id=None,
            )
        ),
        record_fork_point=AsyncMock(return_value=head.value),
    )
    session = MagicMock(
        owner_id="owner",
        run_id="run-fork-point",
        worker_id="worker-1",
        fencing_epoch=1,
    )
    session.execution.session_repository = repository

    await executor.execute(cast(RunSession, session))

    executor._store.record_fork_point.assert_awaited_once_with(
        owner_id="owner",
        run_id="run-fork-point",
        worker_id="worker-1",
        fencing_epoch=1,
        entry_id=head.value,
        projection_id=projection.projection_id.value,
    )


async def test_a_store_failure_costs_the_point_and_not_the_settled_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A settled Run is not failed by a convenience fact it could not write.

    The log carries the failure kind and nothing else: a store error's own text may
    hold a connection string, and this hook is not an operator diagnostic.
    """
    executor = _executor()
    executor._execute = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    executor._store = MagicMock(
        load_routing=AsyncMock(side_effect=RuntimeError("postgres://user:secret@host/db"))
    )
    session = MagicMock(owner_id="owner", run_id="run-no-fork-point")

    outcome = await executor.execute(cast(RunSession, session))

    assert outcome is not None
    assert "could not record its fork point (RuntimeError)" in caplog.text
    assert "postgres://user:secret@host/db" not in caplog.text


async def test_a_lost_claim_reports_an_unwritten_point_without_failing_the_run() -> None:
    """The claim that ends a Run owns the state it settles at; a fenced-out one says so."""
    from dlightrag.engine.answer.runs.routing import RoutingRecord

    executor = _executor()
    executor._execute = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=RoutingRecord(
                requested_mode="auto",
                valid_modes=("research",),
                resolved_mode="research",
                agent_session_id=SessionId.new().value,
                agent_lane_id=LaneId.main().value,
                source_lane_id=None,
            )
        ),
        record_fork_point=AsyncMock(return_value=False),
    )
    session = MagicMock(owner_id="owner", run_id="run-fenced-out", worker_id="w", fencing_epoch=9)
    snapshot = MagicMock()
    snapshot.tree.lane.return_value.head_entry_id = None
    session.execution.session_repository = MagicMock(load=AsyncMock(return_value=snapshot))
    run_trace = MagicMock()

    await executor._record_fork_point(cast(RunSession, session), run_trace)

    run_trace.update.assert_called_once_with(metadata={"fork_point": "unwritten"})


async def test_the_recorded_point_is_the_runs_lane_not_the_sessions_selected_one() -> None:
    """The Run's Lane comes from its routing row, so a drifted selection cannot lie.

    Here the Session's selected Lane is `main`, which holds an earlier head and a
    projection, while the Run ran on a forked Lane with a later head and none. A
    hook that read the selected Lane would record the wrong pair and pass every
    other test in this file.
    """
    from dlightrag.engine.answer.fast import ensure_session_lane as _ensure_session_lane
    from dlightrag.engine.answer.runs.routing import RoutingRecord

    executor = _executor()
    executor._execute = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    repository = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    fork_lane = LaneId.new()

    async def no_result() -> None:
        return None

    host = FastSessionHost(
        repository=repository,
        initial_snapshot=await repository.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="main-one",
        idempotency_key="main-one-key",
        content="parent question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="main-one",
        content="parent answer",
    )
    await _ensure_session_lane(
        repository=repository,
        snapshot=await repository.load(session_id),
        fencing_epoch=1,
        session_id=session_id,
        lane_id=fork_lane,
        source_lane_id=LaneId.main(),
    )
    await host.accept(
        session_id=session_id,
        lane_id=fork_lane,
        reservation_id="fork-one",
        idempotency_key="fork-one-key",
        content="branch question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=fork_lane,
        reservation_id="fork-one",
        content="branch answer",
    )
    fork_head = (await repository.load(session_id)).tree.lane(fork_lane).head_entry_id
    assert fork_head is not None
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=RoutingRecord(
                requested_mode="auto",
                valid_modes=("research",),
                resolved_mode="research",
                agent_session_id=session_id.value,
                agent_lane_id=fork_lane.value,
                source_lane_id=LaneId.main().value,
            )
        ),
        record_fork_point=AsyncMock(return_value=True),
    )
    session = MagicMock(
        owner_id="owner", run_id="run-forked", worker_id="worker-1", fencing_epoch=1
    )
    session.execution.session_repository = repository

    await executor.execute(cast(RunSession, session))

    # No projection was ever committed on the branch, and the parent's is not it.
    executor._store.record_fork_point.assert_awaited_once_with(
        owner_id="owner",
        run_id="run-forked",
        worker_id="worker-1",
        fencing_epoch=1,
        entry_id=fork_head.value,
        projection_id=None,
    )


async def test_a_fork_request_without_a_parent_refuses() -> None:
    """A Fork that names no parent has no state to branch from, and says so."""
    executor = _executor()
    executor._store = MagicMock(load_routing=AsyncMock(side_effect=AssertionError("no read")))
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id=None,
        agent_session_id=SessionId.new().value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor._resolve_fork_seed(
            cast(RunSession, session),
            cast(Any, request),
            await MemoryAgentSessionRepository[None]().load(SessionId.new()),
        )

    assert raised.value.kind == "fork_point_missing"
    assert "needs a parent Run" in raised.value.public_message


async def test_a_fork_whose_recorded_projection_is_not_the_one_there_refuses() -> None:
    """The recorded projection identity is checked, so another branch cannot be seeded."""
    executor = _executor()
    session_id = SessionId.new()
    repository = MemoryAgentSessionRepository[None]()
    snapshot = await repository.load(session_id)
    executor._store = MagicMock(
        load_routing=AsyncMock(
            return_value=_routing_record(
                session_id.value,
                fork_point_entry_id=EntryId.new().value,
                fork_point_projection_id=ProjectionId.new().value,
            )
        )
    )
    session = MagicMock(owner_id="owner", run_id="child")
    request = SimpleNamespace(
        parent_run_id="parent",
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor._resolve_fork_seed(cast(RunSession, session), cast(Any, request), snapshot)

    # The head is not on this Session at all, so the refusal names the stale point.
    assert raised.value.kind == "fork_point_stale"


@pytest.mark.asyncio
async def test_an_unreadable_plane_never_runs_the_one_last_carry(tmp_path: Path) -> None:
    """An unreadable plane is not an empty one.

    Promoting a legacy parent Run's notes into a plane whose state is unknown would
    replace memory that may already be there, so the migration stands down, the Run
    proceeds, and the reason is stated.
    """
    from dlightrag.engine.answer.session_notes import SESSION_NOTES_PLANE_UNREADABLE
    from dlightrag.engine.answer.workspace import epoch_paths, run_root
    from dlightrag.engine.runtime.settlements import InventoryPathRecord
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    parent_id = "01930000-0000-7000-8000-0000000000d1"
    session_id = "01930000-0000-7000-8000-0000000000d2"
    workspace_root = tmp_path / "ws"
    parent_root = run_root(workspace_root, "owner", parent_id)
    parent_workspace, _ = epoch_paths(parent_root, 1)
    (parent_workspace / "notes").mkdir(parents=True)
    (parent_workspace / "notes" / "plan.md").write_bytes(b"legacy")

    async def load(_owner: str, run_id: str) -> tuple[InventoryPathRecord, ...]:
        assert run_id == parent_id
        return (InventoryPathRecord("notes/plan.md", "file", 6),)

    store = InMemoryWorkspaceStore()

    async def unreadable(*, session_id: str):
        raise RuntimeError("plane is gone")

    store.load_session_notes = unreadable  # type: ignore[method-assign]
    executor = _executor()
    executor._workspace_inventory_loader = load
    executor._execution_environment = "trust"
    executor._workspace_root_setting = str(workspace_root)
    executor._working_dir = str(tmp_path / "corpus")
    session = MagicMock(
        owner_id="owner",
        run_id="01930000-0000-7000-8000-0000000000d3",
        workspace_epoch=None,
        execution=MagicMock(fencing_epoch=1, workspace_store=store),
        prepared_input={"agent_session_id": session_id},
    )

    bound, binding, _plane = await executor._claim_run_workspace(
        session=cast(RunSession, session),
        request=MagicMock(parent_run_id=parent_id),
        workspace_store=store,
        session_id=session_id,
    )

    assert bound is not None
    assert binding.records == ()
    assert binding.degraded_reason == SESSION_NOTES_PLANE_UNREADABLE
    assert store.session_notes == {}


@pytest.mark.asyncio
async def test_a_recovered_attempt_states_the_notes_its_own_epoch_holds(tmp_path: Path) -> None:
    """A re-claimed attempt names what it can open, not what the plane holds now.

    The working copy is the baseline on both paths: a recovery that reported the
    plane's current set would name a note this Run cannot read, and a change the Run
    made before it crashed would be reverted instead of promoted.
    """
    from dlightrag.engine.answer.workspace import bind_run_workspace, run_root
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    session_id = "01930000-0000-7000-8000-0000000000c1"
    run_id = "01930000-0000-7000-8000-0000000000c2"
    workspace_root = tmp_path / "ws"
    existing = await bind_run_workspace(
        workspace_root=workspace_root,
        owner_id="owner",
        run_id=run_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=InMemoryWorkspaceStore(),
    )
    (existing.workspace / "notes").mkdir(exist_ok=True)
    (existing.workspace / "notes" / "plan.md").write_bytes(b"mine")

    session = MagicMock(
        owner_id="owner",
        run_id=run_id,
        workspace_epoch=1,
        fencing_epoch=1,
        execution=MagicMock(fencing_epoch=1, workspace_store=InMemoryWorkspaceStore()),
        prepared_input={"agent_session_id": session_id},
    )
    executor = _executor()
    executor._execution_environment = "trust"
    executor._workspace_root_setting = str(workspace_root)
    executor._working_dir = str(tmp_path / "corpus")

    bound, binding, plane = await executor._claim_run_workspace(
        session=cast(RunSession, session),
        request=MagicMock(parent_run_id=None),
        workspace_store=session.execution.workspace_store,
        session_id=session_id,
    )

    assert bound is not None
    assert [note.relative_path for note in binding.records] == ["notes/plan.md"]
    assert binding.records[0].content == b"mine"
    assert binding.degraded_reason is None
    assert plane is not None
    assert (run_root(workspace_root, "owner", run_id) / "epochs" / "1").is_dir()


@pytest.mark.asyncio
async def test_the_one_last_carry_migrates_a_legacy_parent_runs_notes(
    tmp_path: Path,
) -> None:
    """A Session that predates the plane takes its memory from the parent Run, once."""
    import hashlib
    import uuid

    from dlightrag.engine.answer.workspace import epoch_paths, run_root
    from dlightrag.engine.runtime.settlements import InventoryPathRecord
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore, SessionNoteRecord

    parent_id = str(uuid.uuid4())
    session_id = "01930000-0000-7000-8000-000000000001"
    content = b"carried"
    record = InventoryPathRecord(
        relative_path="notes/plan.md",
        entry_type="file",
        size_bytes=len(content),
        content_digest=hashlib.sha256(content).hexdigest(),
    )
    workspace, _ = epoch_paths(run_root(tmp_path, "owner", parent_id), 1)
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(content)

    async def load(_owner: str, run_id: str) -> tuple[InventoryPathRecord, ...]:
        assert run_id == parent_id
        return (record,)

    executor = _executor()
    executor._workspace_inventory_loader = load
    store = InMemoryWorkspaceStore()

    notes = await executor._migrate_legacy_parent_notes(
        session=cast(RunSession, MagicMock(owner_id="owner")),
        request=MagicMock(parent_run_id=parent_id),
        workspace_store=store,
        session_id=session_id,
        workspace_root=tmp_path,
    )

    expected = (SessionNoteRecord(relative_path="notes/plan.md", content=content),)
    assert notes == expected
    assert await store.load_session_notes(session_id=session_id) == expected


@pytest.mark.asyncio
async def test_a_gone_legacy_tree_migrates_nothing_and_does_not_fail(tmp_path: Path) -> None:
    """The one last carry is best effort: a reclaimed parent tree leaves the note behind.

    Nothing here refuses the Run that happened to bind first, which is the failure
    the retired carry produced when a parent's Workspace had been reclaimed.
    """
    import uuid

    from dlightrag.engine.runtime.settlements import InventoryPathRecord
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    record = InventoryPathRecord(relative_path="notes/plan.md", entry_type="file", size_bytes=1)

    async def load(_owner: str, _run_id: str) -> tuple[InventoryPathRecord, ...]:
        return (record,)

    executor = _executor()
    executor._workspace_inventory_loader = load
    store = InMemoryWorkspaceStore()

    notes = await executor._migrate_legacy_parent_notes(
        session=cast(RunSession, MagicMock(owner_id="owner")),
        request=MagicMock(parent_run_id=str(uuid.uuid4())),
        workspace_store=store,
        session_id="01930000-0000-7000-8000-000000000001",
        workspace_root=tmp_path,
    )

    assert notes == ()
    assert store.session_notes == {}


@pytest.mark.asyncio
async def test_a_parent_with_no_registered_notes_migrates_nothing() -> None:
    from dlightrag.engine.runtime.settlements import InventoryPathRecord
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    async def load(_owner: str, _run_id: str) -> tuple[InventoryPathRecord, ...]:
        return (
            InventoryPathRecord(
                relative_path="artifacts/report.md", entry_type="file", size_bytes=8
            ),
        )

    executor = _executor()
    executor._workspace_inventory_loader = load
    store = InMemoryWorkspaceStore()

    notes = await executor._migrate_legacy_parent_notes(
        session=cast(RunSession, MagicMock(owner_id="owner")),
        request=MagicMock(parent_run_id="parent"),
        workspace_store=store,
        session_id="01930000-0000-7000-8000-000000000001",
        workspace_root=Path("/tmp"),
    )

    assert notes == ()
    assert store.session_notes == {}


def _pinned_models() -> tuple[Any, ...]:
    from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ModelSettings
    from dlightrag.engine.answer.execution.input import (
        PinnedModelProfile,
        model_reasoning_settings,
    )

    return tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=_fingerprint(role),
            profile=ModelProfile(context_window_tokens=1_000_000),
            reasoning_settings=model_reasoning_settings(ModelSettings(model=f"test-{role}")),
        )
        for role in CHAT_MODEL_SELECTORS
    )


def _fast_prepared_input(
    *,
    session_id: str,
    parent_run_id: str | None = None,
    profile_memory_enabled: bool = True,
) -> dict[str, Any]:
    from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION
    from dlightrag.engine.ai.catalog import current_model_catalog_revision
    from dlightrag.engine.answer.execution.input import AnswerRunInput

    run_input = AnswerRunInput(
        query="try that again",
        workspaces=("default",),
        pinned_models=_pinned_models(),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint="fast-slice-9",
        agent_session_id=session_id,
        agent_lane_id="main",
        parent_run_id=parent_run_id,
        continuation_kind="follow_up" if parent_run_id else None,
    )
    return {
        **run_input.as_request(),
        "profile_memory_enabled": profile_memory_enabled,
        "profile_memory_epoch": 1,
        "auth_mode": "jwt",
    }


async def _drive_fast_execute(
    *,
    tmp_path: Path,
    parent_run_id: str | None = None,
    parent_inventory: tuple[Any, ...] = (),
    memory: Any = None,
    memory_capability_current: Any = None,
    synthesizer: Any,
    compose_tools: Any = None,
) -> tuple[Any, Any, Any]:
    import uuid

    from dlightrag.engine.agent.session.fold import PriorTurns
    from dlightrag.engine.answer.execution.executor import OrchestratorRun
    from dlightrag.engine.answer.orchestration import AnswerOrchestrator
    from dlightrag.engine.answer.resources.models import TextWindowBudget
    from dlightrag.engine.answer.workspace import bind_run_workspace
    from dlightrag.engine.runtime.progress import StageCommit
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    session_id = SessionId.new()
    repository = MemoryAgentSessionRepository[Any](fencing_epoch=1)
    workspace_store = InMemoryWorkspaceStore()
    child_id = str(uuid.uuid4())
    owner = "owner"
    workspace_root = tmp_path / "ws"
    (tmp_path / "corpus").mkdir()
    if parent_run_id is not None:
        parent_store = InMemoryWorkspaceStore()
        parent = await bind_run_workspace(
            workspace_root=workspace_root,
            owner_id=owner,
            run_id=parent_run_id,
            fencing_epoch=1,
            recorded_epoch=None,
            store=parent_store,
        )
        if parent_inventory:
            (parent.workspace / "notes").mkdir(exist_ok=True)
            for record in parent_inventory:
                path = parent.workspace / record.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"the error was ECONNRESET on shard 4")
            await parent_store.replace_inventory(parent_inventory)

    async def load_parent_inventory(_owner: str, run_id: str) -> tuple[Any, ...]:
        if parent_run_id is None:
            return ()
        assert run_id == parent_run_id
        return parent_inventory

    orchestrator = AnswerOrchestrator(
        synthesizer=synthesizer,
        retrieve_knowledge_base=AsyncMock(
            return_value=MagicMock(
                contexts={"chunks": [], "entities": [], "relationships": []},
                trace={},
            )
        ),
        model_profile=ModelProfile(context_window_tokens=1_000_000),
        telemetry=NOOP_TELEMETRY,
        text_window_budget=TextWindowBudget(tokens=850_000),
        resolved_mode="fast",
    )

    async def prepare(**_kwargs: Any) -> OrchestratorRun:
        return OrchestratorRun(
            orchestrator=orchestrator,
            image_descriptions=[],
            query_images=None,
            history=PriorTurns(),
            fast_history_targets=(),
            current_image_count=0,
            workspaces=["default"],
            registry=None,
        )

    executor = _executor()
    executor._execution_environment = "trust"
    executor._workspace_root_setting = str(workspace_root)
    executor._working_dir = str(tmp_path / "corpus")
    executor._workspace_inventory_loader = load_parent_inventory
    executor._memory = memory
    executor._memory_capability_current = memory_capability_current
    executor.prepare_orchestrated_run = prepare  # type: ignore[method-assign]
    payload = _fast_prepared_input(session_id=session_id.value, parent_run_id=parent_run_id)
    executor.validate_pinned_model_profiles = MagicMock(
        return_value={item.role: item.profile for item in _pinned_models()}
    )
    executor._store.load_routing = AsyncMock(
        return_value=_routing_record(session_id.value, fork_point_entry_id=None)
    )
    progress = MagicMock()
    progress.load_stage = AsyncMock(return_value=None)
    progress.settle_stage = AsyncMock(
        return_value=StageCommit(
            progress_version=1,
            stage_intent_id=MagicMock(),
            evidence_count=0,
        )
    )
    session = MagicMock(
        owner_id=owner,
        run_id=child_id,
        worker_id="worker-1",
        fencing_epoch=1,
        durable_progress_version=0,
        prepared_input=payload,
        workspace_epoch=None,
        checkpoint=None,
    )
    session.check_cancelled = AsyncMock()
    session.enter_phase = AsyncMock()
    session.emit_token = AsyncMock()
    session.flush_tokens = AsyncMock()
    session.reset_output = AsyncMock()
    session.execution.session_repository = repository
    session.execution.progress_store = progress
    session.execution.workspace_store = workspace_store
    session.execution.fencing_epoch = 1
    if compose_tools is not None:
        import dlightrag.engine.answer.tools.composition as composition

        composition.compose_research_tools = compose_tools  # type: ignore[method-assign]
    return executor, session, workspace_store


@pytest.mark.asyncio
async def test_a_fast_continuation_binds_the_parent_note_into_its_own_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Research writes a note; Fast execute carries it; Fast's Inventory names it."""
    import hashlib
    import uuid

    from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
    from dlightrag.engine.answer.tools import composition as composition_module
    from dlightrag.engine.answer.workspace import run_root
    from dlightrag.engine.runtime.settlements import InventoryPathRecord

    parent_id = str(uuid.uuid4())
    payload = b"the error was ECONNRESET on shard 4"
    digest = hashlib.sha256(payload).hexdigest()
    record = InventoryPathRecord(
        relative_path="notes/plan.md",
        entry_type="file",
        size_bytes=len(payload),
        content_digest=digest,
    )
    captured: dict[str, Any] = {}

    class _Synthesizer:
        async def generate_stream(self, *_args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RuntimeError("stop after Fast generation")

    composed = MagicMock(side_effect=AssertionError("Fast must not compose tools"))
    monkeypatch.setattr(composition_module, "compose_research_tools", composed)
    executor, session, workspace_store = await _drive_fast_execute(
        tmp_path=tmp_path,
        parent_run_id=parent_id,
        parent_inventory=(record,),
        synthesizer=cast(AnswerSynthesizer, _Synthesizer()),
    )
    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))

    inventory = await workspace_store.load_inventory()
    assert [item.relative_path for item in inventory] == ["notes/plan.md"]
    assert inventory[0].content_digest == digest
    child_root = run_root(tmp_path / "ws", "owner", session.run_id)
    note = child_root / "epochs" / "1" / "workspace" / "notes" / "plan.md"
    assert note.read_bytes() == payload
    composed.assert_not_called()
    from dlightrag.engine.answer.workspace import bind_run_workspace
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    # The next turn of the same Session binds the plane, not the parent's tree: the
    # Fast Run migrated the legacy note into it, and the writer is gone by now.
    session_id = str(session.prepared_input["agent_session_id"])
    grandchild_store = InMemoryWorkspaceStore()
    grandchild_store.session_notes = workspace_store.session_notes
    grandchild = await bind_run_workspace(
        workspace_root=tmp_path / "ws",
        owner_id="owner",
        run_id=str(uuid.uuid4()),
        fencing_epoch=1,
        recorded_epoch=None,
        store=grandchild_store,
        notes=await grandchild_store.load_session_notes(session_id=session_id),
    )
    assert (grandchild.workspace / "notes" / "plan.md").read_bytes() == payload
    assert (await grandchild_store.load_inventory())[0].content_digest == digest


@pytest.mark.asyncio
async def test_fast_receives_recalled_profile_memory(tmp_path: Path) -> None:
    from dlightrag_memory.memory import RecallResult
    from dlightrag_memory.models import MemoryProvenance, MemoryRecord

    from dlightrag.engine.answer.memory import render_auto_recall
    from dlightrag.engine.answer.synthesizer import AnswerSynthesizer

    record = MemoryRecord(
        owner_id="owner",
        memory_id="m1",
        kind="fact",
        body="prefers short answers",
        provenance=MemoryProvenance(origin_kind="answer_run", origin_id="origin", run_id="origin"),
    )
    captured: dict[str, Any] = {}

    class _Memory:
        async def recall(self, **_kwargs: Any) -> RecallResult:
            return RecallResult(records=(record,), strategy="test", content_chars=len(record.body))

    class _Synthesizer:
        async def generate_stream(self, *_args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RuntimeError("stop after Fast generation")

    executor, session, _store = await _drive_fast_execute(
        tmp_path=tmp_path,
        memory=_Memory(),
        synthesizer=cast(AnswerSynthesizer, _Synthesizer()),
    )
    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))
    assert captured["memory_text"] == render_auto_recall((record,))


@pytest.mark.asyncio
async def test_fast_recall_is_suppressed_when_memory_capability_is_disabled(
    tmp_path: Path,
) -> None:
    from dlightrag.engine.answer.synthesizer import AnswerSynthesizer

    captured: dict[str, Any] = {}

    class _Memory:
        async def recall(self, **_kwargs: Any) -> Any:
            raise AssertionError("disabled capability must not recall")

    class _Synthesizer:
        async def generate_stream(self, *_args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RuntimeError("stop after Fast generation")

    async def disabled(**_kwargs: Any) -> bool:
        return False

    executor, session, _store = await _drive_fast_execute(
        tmp_path=tmp_path,
        memory=_Memory(),
        memory_capability_current=disabled,
        synthesizer=cast(AnswerSynthesizer, _Synthesizer()),
    )
    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))
    assert captured.get("memory_text") == ""


def test_the_reserved_recall_block_mirrors_the_gates_that_allow_recall() -> None:
    """The worst case a Fast measure reserves is bounded by the same facts Research uses."""
    from dlightrag.engine.answer.execution.executor import _worst_case_recall_block

    assert _worst_case_recall_block(None) != ""
    assert _worst_case_recall_block({"auth_mode": "jwt"}) != ""
    assert _worst_case_recall_block({"auth_mode": "jwt", "profile_memory_enabled": False}) == ""
    # A shared simple-auth caller owns no memory, so nothing is reserved for it.
    assert _worst_case_recall_block({"auth_mode": "simple"}) == ""
