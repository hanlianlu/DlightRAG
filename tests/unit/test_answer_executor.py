# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import datetime
import errno
import io
import shutil
import threading
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
from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.environment.confinement import ConfinementPolicy
from dlightrag.engine.agent.session.ids import EntryId, LaneId, ProjectionId, SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import (
    ContextProjectionRegister,
    HostTurnReservation,
    SetRegister,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.reasoning import best_effort_reasoning_profile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserSettings,
    RenderedPage,
    browser_failure,
)
from dlightrag.engine.answer.capabilities import (
    AnswerCapabilities,
    RequestModelContext,
)
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.execution import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
)
from dlightrag.engine.answer.execution.executor import (
    _close_execution_resources,
    _memory_recall_allowed,
    _publication_plan,
    _require_current_child_lifecycle,
    _stage_publications,
)
from dlightrag.engine.answer.execution.input import (
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
    new_resource_identity,
)
from dlightrag.engine.answer.fast import FastSessionHost, ensure_session_lane
from dlightrag.engine.answer.highlights import SemanticHighlightSettings
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.publication import (
    PublicationLimits,
    prepare_artifact_attachment,
)
from dlightrag.engine.answer.resources import ResourceInput
from dlightrag.engine.answer.resources.models import ResourceCursorError
from dlightrag.engine.dependencies import DependencyRetriesExhausted, ProviderUnavailableError
from dlightrag.engine.runtime.coordinator import RunCancellationObserved, RunSession
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.records import (
    Deferred,
    RunExecutionOutcome,
    Succeeded,
    artifact_digest,
)
from tests.in_memory_session_repository import MemoryAgentSessionRepository
from tests.support.agent_browser import FakeProvider
from tests.support.dns import public_dns
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
@pytest.mark.parametrize(
    ("parent_run_id", "recorded", "kind", "remedy"),
    [
        pytest.param(None, None, "fork_point_missing", "needs a parent Run", id="no-parent"),
        pytest.param("missing", None, "fork_point_missing", "that still exists", id="parent-gone"),
        pytest.param(
            "parent", "unrecorded", "fork_point_missing", "that has one", id="no-recorded-point"
        ),
        pytest.param(
            "parent", "foreign", "fork_point_missing", "in this conversation", id="other-session"
        ),
        pytest.param(
            "parent", "head-gone", "fork_point_stale", "head is still present", id="head-gone"
        ),
    ],
)
async def test_a_fork_whose_point_cannot_be_resolved_refuses_instead_of_using_the_tip(
    parent_run_id: str | None, recorded: str | None, kind: str, remedy: str
) -> None:
    """A Fork branches only at its parent's recorded point, and names the remedy otherwise.

    Guessing the Lane tip instead is the divergence between a recorded Fork Point and a
    Lane tip.
    """
    executor = _executor()
    session_id = SessionId.new()
    routing = {
        None: None,
        "unrecorded": _routing_record(session_id.value, fork_point_entry_id=None),
        "foreign": _routing_record(SessionId.new().value, fork_point_entry_id=EntryId.new().value),
        "head-gone": _routing_record(session_id.value, fork_point_entry_id=EntryId.new().value),
    }[recorded]
    executor._store = MagicMock(load_routing=AsyncMock(return_value=routing))
    request = SimpleNamespace(
        parent_run_id=parent_run_id,
        agent_session_id=session_id.value,
        source_lane_id="main",
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor._resolve_fork_seed(
            cast(RunSession, MagicMock(owner_id="owner", run_id="child")),
            cast(Any, request),
            await MemoryAgentSessionRepository[None]().load(session_id),
        )

    assert raised.value.kind == kind
    assert remedy in raised.value.public_message


@pytest.mark.parametrize("holder_live", [True, False])
async def test_a_fast_turn_releases_its_lane_only_once_its_run_has_ended(holder_live: bool) -> None:
    """A reservation whose Run ended is released for the next Run; a live one stays."""
    executor = _executor()
    executor._store = MagicMock(
        get_run=AsyncMock(return_value=SimpleNamespace(terminal=not holder_live))
    )
    repository = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()

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
        reservation_id="run-earlier",
        idempotency_key="earlier-key",
        content="earlier question",
    )
    session = MagicMock(owner_id="owner", run_id="run-next")
    session.execution.session_repository = repository
    session.execution.fencing_epoch = 1

    reclaimed = await executor._reclaim_lane(
        cast(RunSession, session), await repository.load(session_id), LaneId.main()
    )

    held = any(isinstance(record.value, HostTurnReservation) for record in reclaimed.registers)
    assert held is holder_live
    # The question the ended Run never answered stays on the Lane either way.
    assert [entry.entry_type for entry in reclaimed.tree.ancestry()] == ["user_message"]


async def _compact_lane(
    repository: MemoryAgentSessionRepository[None],
    session_id: SessionId,
    lane_id: LaneId,
    goal: str,
) -> Any:
    """Commit one compaction covering a Lane's head, as Fast and Research commit one."""
    from datetime import UTC
    from datetime import datetime as moment

    from dlightrag.engine.agent.session.entries import CompactionEntry
    from dlightrag.engine.agent.session.projection import (
        CompactionSummary,
        ContextProjection,
        projection_source_digest,
    )
    from dlightrag.engine.agent.session.registers import LaneHead

    snapshot = await repository.load(session_id)
    ancestry = snapshot.tree.ancestry(lane_id)
    last = ancestry[-1]
    head = snapshot.tree.lane(lane_id).head
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=last.sequence + 1,
        covered_through_sequence=last.sequence,
        summary=CompactionSummary(goal=goal).canonical_json(),
        covered_through_entry_id=last.entry_id,
        first_retained_entry_id=None,
        source_digest=projection_source_digest(
            [entry.entry_id for entry in ancestry if not isinstance(entry, CompactionEntry)]
        ),
    )
    entry = CompactionEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=moment.now(UTC),
        parent_entry_id=last.entry_id,
        projection_id=projection.projection_id,
        summary=projection.summary,
        covered_through_sequence=projection.covered_through_sequence,
        first_retained_sequence=projection.first_retained_sequence,
        covered_through_entry_id=projection.covered_through_entry_id,
        first_retained_entry_id=None,
        source_digest=projection.source_digest,
    )
    register = ContextProjectionRegister(lane_id, projection)
    await repository.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=[entry],
            register_writes=[SetRegister(LaneHead(lane_id, entry.entry_id)), SetRegister(register)],
            expectations=[
                RegisterExpectation(head.ref, head.sequence),
                RegisterExpectation(register.ref, None),
            ],
        ),
    )
    return projection


@pytest.mark.asyncio
async def test_a_fork_refuses_a_projection_its_point_never_had() -> None:
    """The recorded projection must be the one in force at the recorded head.

    A point naming another branch's projection would seed the Fork with a summary of
    turns its ancestry never held, so it is refused; the point's own one resolves.
    """
    repository = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()

    async def no_result() -> None:
        return None

    host = FastSessionHost(
        repository=repository,
        initial_snapshot=await repository.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )

    async def turn(name: str) -> None:
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id=name,
            idempotency_key=f"{name}-key",
            content=f"{name} question",
        )
        await host.complete(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id=name,
            content=f"{name} answer",
        )

    await turn("one")
    other = LaneId.new()
    await ensure_session_lane(
        repository=repository,
        snapshot=await repository.load(session_id),
        fencing_epoch=1,
        session_id=session_id,
        lane_id=other,
        source_lane_id=LaneId.main(),
    )
    foreign = await _compact_lane(repository, session_id, other, "Another branch.")
    own = await _compact_lane(repository, session_id, LaneId.main(), "This branch.")
    await turn("two")
    snapshot = await repository.load(session_id)
    head = snapshot.tree.lane(LaneId.main()).head_entry_id
    assert head is not None

    async def resolve(recorded: Any) -> Any:
        executor = _executor()
        executor._store = MagicMock(
            load_routing=AsyncMock(
                return_value=_routing_record(
                    session_id.value,
                    fork_point_entry_id=head.value,
                    fork_point_projection_id=recorded.projection_id.value,
                )
            )
        )
        request = SimpleNamespace(
            parent_run_id="parent",
            agent_session_id=session_id.value,
            source_lane_id="main",
        )
        return await executor._resolve_fork_seed(
            cast(RunSession, MagicMock(owner_id="owner", run_id="child")),
            cast(Any, request),
            snapshot,
        )

    seeded_head, seeded = await resolve(own)
    assert seeded_head == head
    assert seeded is not None and seeded.projection_id == own.projection_id
    with pytest.raises(RunExecutionError) as raised:
        await resolve(foreign)
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


def _fingerprint(role: str) -> ModelInvocationFingerprint:
    return ModelInvocationFingerprint("openai", f"test-{role}", None, "chat_completion")


def _executor(**overrides: Any) -> AnswerExecutor:
    executor = AnswerExecutor(
        **{
            "store": MagicMock(),
            "blob_store": MagicMock(),
            "pool": MagicMock(),
            "warm": Mock(),
            "retrieve": AsyncMock(),
            "planning": MagicMock(),
            "models": MagicMock(),
            "capabilities": MagicMock(),
            "resources": MagicMock(),
            "settings": AnswerExecutorSettings(
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
            "telemetry": NOOP_TELEMETRY,
            "model_invocation_fingerprint_for_role": _fingerprint,
            "shell_confinement": ConfinementPolicy(),
            "search_toolchain": SearchToolchain(),
            **overrides,
        }
    )

    # These unit doubles replace execution; dedicated model-contract tests exercise preflight.
    executor.validate_active_prepared_input = Mock()
    return executor


async def test_markdown_artifacts_keep_independent_citation_sources(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Primary fact [1-1].", encoding="utf-8")
    (root / "appendix.md").write_text("Appendix fact [2-1].", encoding="utf-8")
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

    plan = await _publication_plan(
        root,
        answer=("[Open analysis](artifact:analysis.md) [Open appendix](artifact:appendix.md)"),
        attachments=(
            prepare_artifact_attachment(root, path="analysis.md"),
            prepare_artifact_attachment(root, path="appendix.md"),
        ),
        limits=PublicationLimits(),
        contexts=contexts,
    )

    publications, descriptors, artifact_sources = _stage_publications(
        plan=plan,
        answer=plan.answer,
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


def test_research_declarations_include_every_configured_surface_without_binding(
    monkeypatch,
) -> None:
    from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
    from dlightrag.engine.agent.skills import SkillsBundle, SkillsBundleFactory
    from dlightrag.engine.agent.tools import ToolDeclaration

    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planning=MagicMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_invocation_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
        shell_confinement=ConfinementPolicy(),
        memory_store=MagicMock(),
        skills_bundle_factory=SkillsBundleFactory(
            global_root=Path("/nonexistent-global-skills"),
        ),
        search_toolchain=SearchToolchain(),
    )

    def forbid_execution_setup(*_args, **_kwargs):
        raise AssertionError("acceptance must use declarations without runtime setup")

    monkeypatch.setattr(LocalExecutionEnvironment, "__init__", forbid_execution_setup)
    monkeypatch.setattr(SkillsBundleFactory, "__call__", forbid_execution_setup)
    monkeypatch.setattr(SkillsBundle, "catalog", forbid_execution_setup)
    declarations = executor.research_tool_declarations(
        web_search=False, memory=True, model_guidance="", injected=()
    )
    assert all(type(tool) is ToolDeclaration for tool in declarations)
    names = {tool.name for tool in declarations}

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
    without_memory = executor.research_tool_declarations(
        web_search=False, memory=False, model_guidance="", injected=()
    )
    assert not {"remember", "forget", "recall_memory"} & {tool.name for tool in without_memory}


_BROWSER_SETTINGS = AgentBrowserSettings(
    lease_wait_seconds=1.0,
    navigation_timeout_seconds=5.0,
    settle_timeout_seconds=1.0,
    idle_release_seconds=1.0,
    max_page_bytes=1000,
)


def test_acceptance_offers_a_rendered_read_exactly_when_a_browser_is_composed() -> None:
    def read_properties(executor: AnswerExecutor) -> dict[str, Any]:
        declarations = executor.research_tool_declarations(
            web_search=False, memory=False, model_guidance="", injected=()
        )
        return {tool.name: tool for tool in declarations}["read"].definition.parameters[
            "properties"
        ]

    composed = _executor(browser_provider=FakeProvider(), browser_settings=_BROWSER_SETTINGS)

    assert "rendered" in read_properties(composed)
    assert "rendered" not in read_properties(_executor())


def test_an_executor_takes_a_browser_provider_and_its_settings_together() -> None:
    with pytest.raises(ValueError, match="provider and its settings"):
        _executor(browser_provider=FakeProvider())
    with pytest.raises(ValueError, match="provider and its settings"):
        _executor(browser_settings=_BROWSER_SETTINGS)


async def test_closing_the_executor_closes_its_browser_provider_even_if_the_adapter_fails() -> None:
    provider = FakeProvider()
    executor = _executor(browser_provider=provider, browser_settings=_BROWSER_SETTINGS)
    executor._execution_adapter = MagicMock(aclose=AsyncMock(side_effect=RuntimeError("closing")))

    with pytest.raises(RuntimeError, match="closing"):
        await executor.aclose()

    assert provider.closed is True


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

    tools = subagent_tools(host=SubagentHost())
    supported = next(tool for tool in tools if tool.name == "spawn_agent")
    unsupported = AgentTool(
        supported.name,
        supported.description,
        supported.input_model,
        execute=execute,
        replay_policy=supported.replay_policy,
        contract_version=9,
    )
    from dataclasses import replace

    for version in (2, 3, 4):
        with pytest.raises(IncompatibleActiveRunError):
            _require_current_child_lifecycle(_plan(replace(supported, contract_version=version)))
    with pytest.raises(IncompatibleActiveRunError):
        _require_current_child_lifecycle(_plan(unsupported))
    with pytest.raises(IncompatibleActiveRunError, match="missing its accepted Agent Plan"):
        _require_current_child_lifecycle(None)
    # The current contract, and a plan that offers no Children at all, both run.
    _require_current_child_lifecycle(_plan(*tools))
    _require_current_child_lifecycle(_plan())


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
        planning=MagicMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_invocation_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
        shell_confinement=ConfinementPolicy(),
        search_toolchain=SearchToolchain(),
    )
    accepted = executor.research_tool_declarations(
        web_search=False, memory=True, model_guidance="", injected=()
    )

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
        subagent_host=SubagentHost(model_guidance=""),
        skill_tools=[],
        search_toolchain=SearchToolchain(),
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

    accepted_tool = AgentTool("lookup", "Accepted description.", Args, execute=execute)
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
            (AgentTool("lookup", "Changed description.", Args, execute=execute),),
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

    serialized = pinned.as_json()
    restored = PinnedModelProfile.from_json(serialized)

    assert serialized["fingerprint"]["api_family"] == "chat_completion"
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
    mismatched._model_invocation_fingerprint_for_role = lambda role: ModelInvocationFingerprint(
        "other", f"test-{role}", None, "chat_completion"
    )
    with pytest.raises(IncompatibleActiveRunError, match="model invocation"):
        mismatched.validate_pinned_model_profiles(request)


def _resource_resolver(models: Any = None) -> AnswerResourceResolver:
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
        models=models or MagicMock(),
        capabilities=capabilities,
    )


async def test_a_run_mints_its_handles_and_cursors_from_its_own_identity() -> None:
    identity = new_resource_identity()
    resources = [
        ResourceInput(url="https://example.com/report"),
        ResourceInput(
            filename="notes.txt", content="".join(f"line {n}\n" for n in range(400)).encode()
        ),
    ]

    # Two resolvers stand for the process that accepted the Run and the one that
    # resumes it: nothing either one holds takes part in a handle or a cursor.
    first = _resource_resolver().build_resource_context(resources, resource_identity=identity)
    resumed = _resource_resolver().build_resource_context(resources, resource_identity=identity)
    other = _resource_resolver().build_resource_context(
        resources, resource_identity=new_resource_identity()
    )

    handles = [entry.resource_id for entry in first.manifest()]
    assert handles == [entry.resource_id for entry in resumed.manifest()]
    assert set(handles).isdisjoint(entry.resource_id for entry in other.manifest())
    page = await first.read(handles[1], max_window_tokens=200)
    assert page.next_cursor is not None
    continued = await resumed.read(handles[1], cursor=page.next_cursor, max_window_tokens=200)
    assert continued.content
    with pytest.raises(ResourceCursorError):
        await other.read(
            other.manifest()[1].resource_id, cursor=page.next_cursor, max_window_tokens=200
        )


def test_a_resume_under_rotated_database_credentials_mints_the_same_handles(
    test_config: Any,
) -> None:
    from dlightrag._compose import _compose

    postgres = test_config.storage.postgres
    rotated = test_config.model_copy(
        update={
            "storage": test_config.storage.model_copy(
                update={"postgres": postgres.model_copy(update={"password": "rotated"})}
            )
        }
    )
    identity = new_resource_identity()
    resources = [ResourceInput(url="https://example.com/report")]

    handles = [
        _compose(config)
        .coordinator._executors["answer"]
        ._resources.build_resource_context(resources, resource_identity=identity)
        .manifest()[0]
        .resource_id
        for config in (test_config, rotated)
    ]

    assert handles[0] == handles[1]


@pytest.mark.parametrize(
    ("order", "renderer", "asked"),
    [
        pytest.param(
            ("exa", "browser", "tavily"), True, ["exa", "browser", "tavily"], id="between"
        ),
        pytest.param(("exa", "tavily", "browser"), True, ["exa+tavily", "browser"], id="last"),
        pytest.param(("browser", "exa"), True, ["browser", "exa"], id="first"),
        pytest.param(("exa", "browser", "tavily"), False, ["exa+tavily"], id="no-browser"),
    ],
)
async def test_a_run_tries_hosted_providers_and_its_browser_in_the_configured_order(
    monkeypatch: pytest.MonkeyPatch,
    order: tuple[str, ...],
    renderer: bool,
    asked: list[str],
) -> None:
    async def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", blocked)
    calls: list[str] = []

    async def extract(url: str, *, providers: tuple[str, ...]) -> Any:
        calls.append("+".join(providers))
        raise RuntimeError("no text")

    async def render(url: str) -> RenderedPage:
        calls.append("browser")
        raise browser_failure("unreachable")

    resolver = _resource_resolver(MagicMock(extract_order=Mock(return_value=order)))
    registry = resolver.build_resource_context(
        [ResourceInput(url="https://example.com/report")],
        web_sources=cast(Any, SimpleNamespace(extract=extract)),
        page_renderer=render if renderer else None,
        resource_identity=new_resource_identity(),
    )

    await registry.read(registry.manifest()[0].resource_id, max_window_tokens=2000)

    assert calls == asked


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
        side_effect=AnswerInputOverflowError("Attached documents exceed the context window.")
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, MagicMock()))

    assert raised.value.kind == "ANSWER_INPUT_OVERFLOW"
    assert raised.value.public_message == "Attached documents exceed the context window."


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
    assert outcome.checkpoint == {checkpoint_key: 3, "dependency_deferrals": 1}
    assert (outcome.next_attempt_at - now).total_seconds() == 20
    session.check_cancelled.assert_awaited_once_with()
    session.reset_output.assert_awaited_once_with()


async def test_an_answer_run_whose_outages_spent_its_deferrals_stops() -> None:
    executor = _executor()
    executor._execute = AsyncMock(side_effect=ProviderUnavailableError())  # type: ignore[method-assign]
    session = MagicMock(
        owner_id="owner",
        run_id="same-run",
        checkpoint={"providers_unavailable_attempt": 10, "dependency_deferrals": 10},
    )
    session.check_cancelled = AsyncMock()
    session.reset_output = AsyncMock()

    with pytest.raises(DependencyRetriesExhausted) as raised:
        await executor.execute(cast(RunSession, session))

    assert raised.value.kind == "dependency_unavailable"
    assert raised.value.component == "providers"


async def test_a_recovery_copy_the_volume_fails_defers_and_the_next_claim_copies_it_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EIO while copying the recorded epoch is the volume's outage, not the Run's.

    The file it names is ``schema.sql``, text that would veto a deferral had it reached
    classification. Nothing recorded changed, so the next fencing copies the whole tree.
    """
    from dlightrag.engine.answer.workspace import bind_run_workspace, epoch_paths, run_root
    from tests.support.workspace_store import InMemoryWorkspaceStore

    run_id = "01930000-0000-7000-8000-0000000000e1"
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
    store = InMemoryWorkspaceStore()
    first = await bind_run_workspace(
        workspace_root=workspace_root,
        owner_id="owner",
        run_id=run_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=store,
    )
    (first.workspace / "schema.sql").write_text("create table t ();", encoding="utf-8")
    executor = _executor()
    executor._execution_environment = "trust"
    executor._workspace_root_setting = str(workspace_root)
    executor._working_dir = str(tmp_path / "corpus")

    async def claim(session: RunSession, _run_trace: object) -> Succeeded:
        await executor._claim_run_workspace(session=session, workspace_store=store, session_id=None)
        return Succeeded({})

    executor._execute = claim  # type: ignore[method-assign]

    def attempt(fencing_epoch: int, checkpoint: Mapping[str, Any]) -> RunSession:
        session = MagicMock(
            owner_id="owner",
            run_id=run_id,
            workspace_epoch=store.workspace_epoch,
            checkpoint=checkpoint,
            execution=MagicMock(fencing_epoch=fencing_epoch),
        )
        session.check_cancelled = AsyncMock()
        session.reset_output = AsyncMock()
        return cast(RunSession, session)

    copy = shutil.copy2

    def copy_until_schema(source: Path, destination: Path) -> object:
        if source.name == "schema.sql":
            raise OSError(errno.EIO, "Input/output error", str(source))
        return copy(source, destination)

    monkeypatch.setattr(shutil, "copy2", copy_until_schema)
    deferred = await executor.execute(attempt(2, {}))

    assert isinstance(deferred, Deferred)
    assert deferred.checkpoint == {
        "agent_workspace_unavailable_attempt": 1,
        "dependency_deferrals": 1,
    }
    assert store.workspace_epoch == 1

    monkeypatch.undo()
    assert await executor.execute(attempt(3, deferred.checkpoint)) == Succeeded({})
    recovered, _ = epoch_paths(run_root(workspace_root, "owner", run_id), 3)
    assert (recovered / "schema.sql").read_text(encoding="utf-8") == "create table t ();"
    assert store.workspace_epoch == 3


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

    assert await _durable_child_usage(
        store.list_child_sessions, owner_id="owner", run_id="run-1"
    ) == {
        "input_tokens": 8,
        "output_tokens": 3,
    }


async def test_public_document_citations_are_projected_into_the_published_artifact(
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

    plan = await _publication_plan(
        root,
        answer="[Open report](artifact:report.md)",
        attachments=(prepare_artifact_attachment(root, path="report.md"),),
        limits=PublicationLimits(),
        contexts=contexts,
    )

    publications, descriptors, _ = _stage_publications(
        plan=plan,
        answer=plan.answer,
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

    A Run that adopted an earlier Run's document records it as its own Resource,
    with the view it brought named by its own handle, so recovery has to rebuild
    that Resource (and its earlier handle as alias) rather than treat the row as a
    Web catalog entry it never was.
    """
    import hashlib

    from dlightrag.engine.answer.resources.registry import ResourceRegistry
    from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
    from dlightrag.engine.runtime.records import RunFetchedResource

    document = b"%PDF-1.7 adopted earlier"
    snapshot = ConversionSnapshot(
        resource_id="res-adopted",
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
            source_locator=b"res-adopted",
            capabilities={"resource_kind": "lineage_adoption", "resource_aliases": ["res-earlier"]},
        ),
        RunFetchedResource(
            resource_id="res-adopted-conversion",
            ordinal=0,
            digest=hashlib.sha256(encoded).hexdigest(),
            filename="conversion.json",
            mime_type="application/json",
            source_locator=b"res-adopted",
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

        for handle in ("res-earlier", "res-adopted"):
            read = await registry.read(handle, max_window_tokens=1000)
            assert read.resource_id == "res-adopted"
            assert "Adopted text." in read.content


@pytest.mark.asyncio
async def test_recovery_keeps_an_adoption_without_its_view_unconverted(monkeypatch) -> None:
    """A resumed Run must not convert what the Run before the resume refused to.

    An adoption that came without a stored view reads its text only through a view
    the earlier Run never had, so after a resume its text still refuses.
    """
    import hashlib

    from dlightrag.engine.answer.resources.models import ResourceNotConvertedError
    from dlightrag.engine.answer.resources.registry import ResourceRegistry
    from dlightrag.engine.runtime.records import RunFetchedResource

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a restored adoption is never converted")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    document = b"%PDF-1.7 adopted earlier, never converted"
    digest = hashlib.sha256(document).hexdigest()
    executor = _executor()
    executor._store.list_fetched_resources = AsyncMock(
        return_value=(
            RunFetchedResource(
                resource_id="res-adopted",
                ordinal=0,
                digest=digest,
                filename="earlier.pdf",
                mime_type="application/pdf",
                source_locator=b"res-adopted",
                capabilities={
                    "resource_kind": "lineage_adoption",
                    "resource_aliases": ["res-earlier"],
                },
            ),
        )
    )

    async def stream(*, owner_id: str, digest: str, **kwargs: object):
        del owner_id, digest, kwargs
        yield document

    executor._blob_store.stream = stream

    async with ResourceRegistry() as registry:
        await executor._restore_registry_fetches(registry, owner_id="owner", run_id="run")

        with pytest.raises(ResourceNotConvertedError):
            await registry.read("res-earlier", max_window_tokens=1000)


async def test_a_settled_run_records_the_state_it_ended_at() -> None:
    """A Fork branches from a recorded Fork Point, so the settlement has to write one.

    The head comes from the Lane the routing row fixed at acceptance, and the
    projection from that Lane's own register: a Run that compacted must hand a Fork
    the summary it was working from, not the whole transcript it had discarded.
    """
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


@pytest.mark.asyncio
async def test_an_unreadable_plane_costs_the_run_its_notes_and_states_why(tmp_path: Path) -> None:
    """An unreadable plane is a degradation: the Run still binds its workspace."""
    from dlightrag.engine.answer.session_notes import SESSION_NOTES_PLANE_UNREADABLE
    from tests.support.workspace_store import InMemoryWorkspaceStore

    session_id = "01930000-0000-7000-8000-0000000000d2"
    store = InMemoryWorkspaceStore()

    async def unreadable(*, session_id: str):
        raise RuntimeError("plane is gone")

    store.load_session_notes = unreadable  # type: ignore[method-assign]
    executor = _executor()
    executor._execution_environment = "trust"
    executor._workspace_root_setting = str(tmp_path / "ws")
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
        workspace_store=store,
        session_id=session_id,
    )

    assert bound is not None
    assert binding.records == ()
    assert binding.degraded_reason == SESSION_NOTES_PLANE_UNREADABLE


@pytest.mark.asyncio
async def test_a_recovered_attempt_states_the_notes_its_own_epoch_holds(tmp_path: Path) -> None:
    """A re-claimed attempt names what it can open, not what the plane holds now.

    The working copy is the baseline on both paths: a recovery that reported the
    plane's current set would name a note this Run cannot read, and a change the Run
    made before it crashed would be reverted instead of promoted.
    """
    from dlightrag.engine.answer.workspace import bind_run_workspace, run_root
    from tests.support.workspace_store import InMemoryWorkspaceStore

    session_id = "01930000-0000-7000-8000-0000000000c1"
    run_id = "01930000-0000-7000-8000-0000000000c2"
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
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
        workspace_store=session.execution.workspace_store,
        session_id=session_id,
    )

    assert bound is not None
    assert [note.relative_path for note in binding.records] == ["notes/plan.md"]
    assert binding.records[0].content == b"mine"
    assert binding.degraded_reason is None
    assert plane is not None
    assert (run_root(workspace_root, "owner", run_id) / "epochs" / "1").is_dir()


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


def _fast_prepared_input(*, session_id: str) -> dict[str, Any]:
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
        resource_identity=new_resource_identity(),
        agent_session_id=session_id,
        agent_lane_id="main",
    )
    return {
        **run_input.as_request(),
        "profile_memory_enabled": True,
        "profile_memory_epoch": 1,
    }


async def _drive_fast_execute(
    *,
    tmp_path: Path,
    memory: Any = None,
    memory_capability_current: Any = None,
    contexts: Mapping[str, Any] | None = None,
    answer: str | None = None,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Execute one Fast Run and return the requests its answer model received.

    Retrieval returns ``contexts``. The answer model records each request; given an
    ``answer`` it streams it and the Run settles its result, and without one it stops
    the Run, so the assertion is on the request a provider would have been sent.
    """
    import uuid
    from collections.abc import AsyncIterator

    from dlightrag.engine.agent.session.fold import PriorTurns
    from dlightrag.engine.answer.execution.executor import OrchestratorRun
    from dlightrag.engine.answer.orchestration import AnswerOrchestrator
    from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
    from dlightrag.engine.runtime.progress import StageCommit, StageTerminalCommit
    from tests.support.workspace_store import InMemoryWorkspaceStore

    session_id = SessionId.new()
    repository = MemoryAgentSessionRepository[Any](fencing_epoch=1)
    workspace_store = InMemoryWorkspaceStore()
    owner = "owner"
    workspace_root = tmp_path / "ws"
    (tmp_path / "corpus").mkdir()
    sent: list[dict[str, Any]] = []

    async def answer_model(**kwargs: Any) -> Any:
        sent.append(kwargs)
        if answer is None:
            raise RuntimeError("stop after the Fast request")
        text = answer

        async def tokens() -> AsyncIterator[str]:
            yield text

        return tokens()

    orchestrator = AnswerOrchestrator(
        synthesizer=AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=ModelProfile(context_window_tokens=1_000_000),
            model_func=answer_model,
        ),
        retrieve_knowledge_base=AsyncMock(
            return_value=MagicMock(
                contexts=contexts or {"chunks": [], "entities": [], "relationships": []},
                trace={},
            )
        ),
        model_profile=ModelProfile(context_window_tokens=1_000_000),
        telemetry=NOOP_TELEMETRY,
        resolved_mode="fast",
        search_toolchain=SearchToolchain(),
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
    executor._memory = memory
    executor._memory_capability_current = memory_capability_current
    executor.prepare_orchestrated_run = prepare  # type: ignore[method-assign]
    payload = _fast_prepared_input(session_id=session_id.value)
    executor.validate_pinned_model_profiles = MagicMock(
        return_value={item.role: item.profile for item in _pinned_models()}
    )
    executor._store.load_routing = AsyncMock(
        return_value=_routing_record(session_id.value, fork_point_entry_id=None)
    )
    executor._store.list_artifact_attachments = AsyncMock(return_value=[])
    executor._store.record_fork_point = AsyncMock(return_value=True)
    progress = MagicMock()
    progress.load_stage = AsyncMock(return_value=None)
    progress.settle_stage = AsyncMock(
        return_value=StageCommit(
            progress_version=1,
            stage_intent_id=MagicMock(),
            evidence_count=0,
        )
    )
    progress.settle_terminal = AsyncMock(
        return_value=StageTerminalCommit(
            progress_version=2,
            stage_intent_id=MagicMock(),
            status="succeeded",
            terminal_event_sequence=1,
        )
    )
    session = MagicMock(
        owner_id=owner,
        run_id=str(uuid.uuid4()),
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
    return executor, session, sent


@pytest.mark.asyncio
async def test_fast_sends_recalled_profile_memory_and_no_tools(tmp_path: Path) -> None:
    from dlightrag_memory.memory import RecallResult
    from dlightrag_memory.models import MemoryProvenance, MemoryRecord

    from dlightrag.engine.answer.memory import render_auto_recall

    record = MemoryRecord(
        owner_id="owner",
        memory_id="m1",
        kind="fact",
        body="prefers short answers",
        provenance=MemoryProvenance(origin_kind="answer_run", origin_id="origin", run_id="origin"),
    )

    class _Memory:
        async def recall(self, **_kwargs: Any) -> RecallResult:
            return RecallResult(facts=(record,))

    executor, session, sent = await _drive_fast_execute(tmp_path=tmp_path, memory=_Memory())
    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))

    (request,) = sent
    # Fast composes no tools, so its one answer call offers the model none.
    assert "tools" not in request
    # The recalled block rides last, after the request it must not outrank.
    assert request["messages"][-1] == {
        "role": "user",
        "content": render_auto_recall(RecallResult(facts=(record,))),
    }


@pytest.mark.asyncio
async def test_fast_recall_is_suppressed_when_memory_capability_is_disabled(
    tmp_path: Path,
) -> None:
    class _Memory:
        async def recall(self, **_kwargs: Any) -> Any:
            raise AssertionError("disabled capability must not recall")

    async def disabled(**_kwargs: Any) -> bool:
        return False

    executor, session, sent = await _drive_fast_execute(
        tmp_path=tmp_path,
        memory=_Memory(),
        memory_capability_current=disabled,
    )
    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))

    (request,) = sent
    assert not any("Remembered about this owner" in str(m["content"]) for m in request["messages"])


@pytest.mark.asyncio
async def test_a_fast_answer_resolves_its_citations_to_the_documents_it_was_shown(
    tmp_path: Path,
) -> None:
    """The stored answer resolves each marker against the numbers its request printed.

    Retrieval numbers documents by how many chunks cite them, so here the first excerpt
    is retrieval's document 2. Fast once labelled no excerpt when that order differed
    from the request's, and resolved the answer's markers against retrieval's numbers.
    """

    def chunk(chunk_id: str, reference_id: str, name: str, content: str, page: int) -> dict:
        return {
            "chunk_id": chunk_id,
            "reference_id": reference_id,
            "file_path": f"/docs/{name}",
            "content": content,
            "page_number": page,
            "_workspace": "default",
            "metadata": {
                "source_uri": f"local://default/{name}",
                "source_download_locator": f"/docs/{name}",
            },
        }

    executor, session, sent = await _drive_fast_execute(
        tmp_path=tmp_path,
        contexts={
            "chunks": [
                chunk("a1", "2", "report.pdf", "Revenue grew.", 3),
                chunk("b1", "1", "other.pdf", "Other one.", 1),
                chunk("b2", "1", "other.pdf", "Other two.", 2),
            ],
            "entities": [],
            "relationships": [],
        },
        answer="Revenue grew [1-1], and the other report says two [2-2].",
    )

    await executor.execute(cast(RunSession, session))

    (request,) = sent
    shown = [
        block["text"] for block in request["messages"][-1]["content"] if block["type"] == "text"
    ]
    assert "[1-1] report.pdf, Page 3\nRevenue grew." in shown
    assert "[2-2] other.pdf, Page 2\nOther two." in shown
    stored = session.execution.progress_store.settle_terminal.await_args.kwargs["result"]
    assert stored["answer"] == "Revenue grew [1-1], and the other report says two [2-2]."
    assert [
        (source["id"], source["title"], source["cited_chunk_ids"]) for source in stored["sources"]
    ] == [("1", "report.pdf", ["a1"]), ("2", "other.pdf", ["b2"])]


@pytest.mark.parametrize(
    ("original", "filename"),
    [
        ("[download](artifact:data[9].txt)", "data[9].txt"),
        ("[download][9]\n\n[9]: artifact:data.txt", "data.txt"),
    ],
)
async def test_citation_preparation_preserves_resource_links_before_staging(
    tmp_path: Path, original: str, filename: str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    report = root / "report.md"
    report.write_text(original)
    (root / filename).write_text("data")
    accepted = await _publication_plan(
        root,
        answer="Report generated. [Report](artifact:report.md)",
        attachments=(prepare_artifact_attachment(root, path="report.md"),),
        contexts={},
        limits=PublicationLimits(),
    )
    publications, descriptors, _ = _stage_publications(
        plan=accepted, answer=accepted.answer, session_id="session"
    )
    assert accepted.outcome["status"] == "complete"
    assert [item.relative_path for item in accepted.artifacts] == ["report.md", filename]
    assert publications[0].content == original.encode()
    assert list(accepted.artifacts[0].artifact_bindings.values()) == [publications[1].resource_id]
    assert descriptors[0]["digest"] == artifact_digest(publications[0].content)
    assert report.read_text() == original


def _public_artifact_context() -> dict[str, Any]:
    return {
        "chunks": [
            {
                "chunk_id": "public-fact",
                "reference_id": "1",
                "file_path": "Source",
                "content": "Fact.",
                "_workspace": "__web_search__",
                "metadata": {
                    "source_uri": "https://example.com/source",
                    "source_download_locator": "https://example.com/source",
                },
            }
        ]
    }


async def test_citation_preparation_precedes_binding_and_staging_uses_validated_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    original = "Fact [1]. [Data](artifact:data.txt)"
    (root / "report.md").write_text(original)
    (root / "data.txt").write_text("data")
    attachment = prepare_artifact_attachment(root, path="report.md")

    plan = await _publication_plan(
        root,
        answer="[Report](artifact:report.md)",
        attachments=(attachment,),
        contexts=_public_artifact_context(),
        limits=PublicationLimits(),
    )
    publications, descriptors, sources = _stage_publications(
        plan=plan, answer=plan.answer, session_id="session"
    )

    assert plan.outcome["status"] == "complete"
    report = plan.artifacts[0]
    assert report.source_digest == attachment.content_digest == artifact_digest(original.encode())
    assert report.content == publications[0].content
    assert b"[1](<https://example.com/source>" in report.content
    assert b"[Data](artifact:data.txt)" in report.content
    assert report.artifact_bindings == {"artifact:data.txt": publications[1].resource_id}
    assert descriptors[0]["byte_size"] == len(report.content)
    assert descriptors[0]["digest"] == artifact_digest(report.content)
    assert [source.id for source in sources[report.resource_id]] == ["1"]


@pytest.mark.parametrize(
    ("limits", "issue"),
    [
        (PublicationLimits(max_file_bytes=10), "file_too_large"),
        (PublicationLimits(max_total_bytes=10), "answer_too_large"),
    ],
)
async def test_publication_budgets_include_projected_citation_bytes(
    tmp_path: Path, limits: PublicationLimits, issue: str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("Fact [1].")

    plan = await _publication_plan(
        root,
        answer="Report generated. [Report](artifact:report.md)",
        attachments=(prepare_artifact_attachment(root, path="report.md"),),
        contexts=_public_artifact_context(),
        limits=limits,
    )
    publications, descriptors, sources = _stage_publications(
        plan=plan, answer=plan.answer, session_id="session"
    )

    assert plan.outcome["status"] == "failed"
    assert plan.issues[0].kind == issue
    assert publications == []
    assert descriptors[0]["status"] == "unavailable"
    assert sources == {}


async def test_publication_plan_validates_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dlightrag.engine.answer.execution import executor

    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.md").write_text("Fact.")
    validate = executor.validate_publication
    threads: list[threading.Thread] = []

    def probe(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread())
        return validate(*args, **kwargs)

    monkeypatch.setattr(executor, "validate_publication", probe)

    plan = await _publication_plan(
        root,
        answer="Report generated. [Report](artifact:report.md)",
        attachments=(prepare_artifact_attachment(root, path="report.md"),),
        contexts={},
        limits=PublicationLimits(),
    )

    assert [item.relative_path for item in plan.artifacts] == ["report.md"]
    assert threads and threads[0] is not threading.current_thread()
    assert threads[0].name.startswith("artifact-check")


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [("notes.txt", "text/plain"), ("photo.png", "image/png")],
    ids=["lazy-attachment", "eager-image"],
)
async def test_answer_run_attachments_must_match_their_accepted_digest(
    filename: str, mime_type: str
) -> None:
    import hashlib

    from dlightrag.engine.answer.execution.input import AnswerRunInput

    accepted = b"accepted bytes"
    stored = {"bytes": b"tampered bytes"}

    def stream(*, owner_id: str, digest: str) -> Any:
        del owner_id, digest

        async def pieces() -> Any:
            # The digest covers every piece the store streams, not only the first.
            yield stored["bytes"][:5]
            yield stored["bytes"][5:]

        return pieces()

    executor = _executor()
    executor._blob_store = SimpleNamespace(stream=stream)  # type: ignore[assignment]
    request = AnswerRunInput(
        query="q",
        pinned_models=(),
        context_policy_revision="policy",
        model_catalog_revision="catalog",
        idempotency_fingerprint="fingerprint",
        resource_identity=new_resource_identity(),
        attachments=(
            AttachmentReference(
                digest=hashlib.sha256(accepted).hexdigest(),
                filename=filename,
                mime_type=mime_type,
                ordinal=0,
            ),
        ),
    )

    async def load_attachment() -> bytes:
        resources = await executor._answer_run_resources(request, owner_id="owner")
        assert resources is not None
        (resource,) = resources
        if resource.loader is None:
            assert resource.content is not None
            return resource.content
        return await resource.loader()

    with pytest.raises(RunExecutionError, match="do not match their accepted digest"):
        await load_attachment()

    stored["bytes"] = accepted
    assert await load_attachment() == accepted
