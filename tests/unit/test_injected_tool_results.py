# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Injected tool results cross the ordinary Research/Child and resource read seams."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.session.ids import EntryId, IntentId, SessionId
from dlightrag.engine.agent.tools import (
    AgentTool,
    EvidenceSourceFact,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.workspace import RunWorkspace
from tests.unit.conftest import answer_model_profile


class Args(BaseModel):
    pass


def prepared_tools(tmp_path, result, *, child=False, workspace=True):
    execute = AsyncMock(return_value=result)
    injected = AgentTool("external", "External fixture", Args, execute, guidance="Exact guidance")
    profile = answer_model_profile()
    orchestrator = AnswerOrchestrator(
        synthesizer=MagicMock(),
        retrieve_knowledge_base=AsyncMock(),
        model_func=AsyncMock(),
        injected_tools=[injected],
        model_profile=profile,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
    )
    if workspace:
        orchestrator.bind_workspace(
            RunWorkspace(
                epoch=1,
                workspace=tmp_path,
                spill_dir=tmp_path / "spill",
                environment=LocalExecutionEnvironment(tmp_path),
            )
        )
    if child:
        from dlightrag.engine.answer.tools.subagents import ChildContextSnapshot, ChildRequest

        prepared = orchestrator.prepare_child_session(
            ChildRequest(objective="Anything", tools=("external", "read")),
            context_snapshot=ChildContextSnapshot(
                parent_session_id=SessionId.new(),
                parent_entry_id=EntryId.new(),
                depth=0,
                messages_json="[]",
            ),
            child_session_id=str(uuid.uuid7()),
        )
    else:
        prepared = orchestrator.prepare_run("question")
    return injected, {tool.name: tool for tool in prepared.tools}, execute


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_injected_tool_small_result_unchanged_and_large_spilled_readable(tmp_path, child):
    original = ToolResult.text("small", is_error=True, details={"keep": True})
    injected, tools, execute = prepared_tools(tmp_path, original, child=child)
    runtime = ToolRuntime("c", "external", IntentId.new(), str(uuid.uuid7()), AsyncMock(), 1)
    assert await tools["external"].execute(Args(), runtime) is original
    assert tools["external"].definition == injected.definition
    assert tools["external"].replay_policy == injected.replay_policy
    assert tools["external"].guidance == injected.guidance
    evidence = EvidenceSourceFact("r", "fixture", "fixture:r", "Keep")
    full = "payload line\n" * 6000
    execute.return_value = ToolResult.text(full, effects=ToolEffects(evidence_sources=(evidence,)))
    result = await tools["external"].execute(Args(), runtime)
    assert result.effects.evidence_sources == (evidence,)
    assert len(result.effects.committed_outputs) == 1
    receipt = result.effects.committed_outputs[0]
    assert receipt.resource_id in result.protected_text
    assert (tmp_path / "spill" / (receipt.resource_id + ".txt")).read_text() == full
    read = tools["read"]
    page = await read.execute(
        read.input_model.model_validate({"resource_id": receipt.resource_id}), runtime
    )
    assert not page.is_error and "payload line" in page.text_content
    _, other_tools, _ = prepared_tools(tmp_path / "other-owner-run", ToolResult.text("other"))
    other_read = other_tools["read"]
    missing = await other_read.execute(
        other_read.input_model.model_validate({"resource_id": receipt.resource_id}), runtime
    )
    assert missing.is_error  # no cross-owner/run workspace fallback


@pytest.mark.asyncio
async def test_oversize_without_spill_is_explicit_failure_not_reexecution(tmp_path):
    _, tools, execute = prepared_tools(tmp_path, ToolResult.text("x" * 60000), workspace=False)
    runtime = ToolRuntime("c", "external", IntentId.new(), str(uuid.uuid7()), AsyncMock(), 1)
    result = await tools["external"].execute(Args(), runtime)
    assert result.is_error and "Do not retry" in result.text_content
    execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("backing", ["cursor", "spill"])
async def test_injected_tool_existing_continuation_and_effects_are_not_respilled(tmp_path, backing):
    from dlightrag.engine.agent.tools.contracts import CommittedOutput

    effects = (
        ToolEffects(committed_outputs=(CommittedOutput("existing", "a" * 64, 100000),))
        if backing == "spill"
        else ToolEffects()
    )
    result = ToolResult.text(
        "x" * 60000,
        protected_text="read existing cursor" if backing == "cursor" else "",
        effects=effects,
    )
    _, tools, _ = prepared_tools(tmp_path, result)
    runtime = ToolRuntime("c", "external", IntentId.new(), str(uuid.uuid7()), AsyncMock(), 1)
    assert await tools["external"].execute(Args(), runtime) is result
    assert not (tmp_path / "spill").exists()
