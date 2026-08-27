# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.application.answer_runs.capabilities import AnswerCapabilities
from dlightrag.application.answer_runs.capability import AnswerImageCapability
from dlightrag.application.answer_runs.errors import CurrentDocumentParseError
from dlightrag.application.answer_runs.execution import (
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.execution import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
)
from dlightrag.engine.answer.execution.executor import (
    _close_execution_resources,
    _memory_recall_allowed,
)
from dlightrag.engine.answer.fast import ensure_session_lane
from dlightrag.engine.answer.highlights import SemanticHighlightSettings
from dlightrag.engine.runtime import RunExecutionError, RunSession, artifact_digest


@pytest.mark.asyncio
async def test_missing_fork_session_is_a_typed_run_conflict() -> None:
    store = MemoryAgentSessionRepository[None]()

    with pytest.raises(RunExecutionError) as raised:
        await ensure_session_lane(
            transactions=store,
            load=store.load,
            fencing_epoch=1,
            session_id=SessionId.new(),
            lane_id=LaneId.new(),
            source_lane_id=LaneId.main(),
        )

    assert raised.value.kind == "agent_session_conflict"


def _fingerprint(role: str) -> ModelFingerprint:
    return ModelFingerprint("openai", f"test-{role}", None)


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
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
    )


def test_acceptance_research_tools_include_every_configured_non_resource_surface() -> None:
    from pydantic import BaseModel

    from dlightrag.engine.agent.tools import AgentTool, ToolResult

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
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
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
    from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
    from dlightrag.engine.agent.skills import SkillCatalog
    from dlightrag.engine.answer.evidence import EvidenceLedger
    from dlightrag.engine.answer.tools.composition import compose_research_tools
    from dlightrag.engine.answer.tools.subagents import SubagentHost

    executor = AnswerExecutor(
        store=MagicMock(),
        pool=MagicMock(),
        retrieve=AsyncMock(),
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

    from dlightrag.engine.agent.tools import AgentTool, ToolResult
    from dlightrag.engine.answer.execution import IncompatibleActiveRunError

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


def test_execution_rejects_changed_context_or_model_pins() -> None:
    from dlightrag.engine.answer.execution import IncompatibleActiveRunError

    pins = tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=_fingerprint(role),
            profile=ModelProfile(context_window_tokens=10_000),
        )
        for role in ("extract", "keyword", "query", "vlm")
    )
    executor = _executor()
    request = MagicMock(
        pinned_models=pins,
        context_policy_revision=CONTEXT_POLICY_REVISION,
    )
    executor.validate_pinned_model_profiles(request)

    request.context_policy_revision = "stale-policy"
    with pytest.raises(IncompatibleActiveRunError, match="context policy"):
        executor.validate_pinned_model_profiles(request)

    request.context_policy_revision = CONTEXT_POLICY_REVISION
    mismatched = _executor()
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
