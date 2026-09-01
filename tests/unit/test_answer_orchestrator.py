# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer Host coordination around the deep AgentSessionRuntime."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.skills import SkillsBundle
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from tests.unit.conftest import answer_model_profile


def _orchestrator(*, mode: str, model=None, retrieve=None, synthesizer=None):

    profile = answer_model_profile()

    async def default_retrieve(_query: str):
        return MagicMock(contexts={"chunks": [], "entities": [], "relationships": []}, trace={})

    return AnswerOrchestrator(
        synthesizer=synthesizer or MagicMock(spec=AnswerSynthesizer),
        retrieve_knowledge_base=retrieve or default_retrieve,
        model_func=model,
        stream_model_func=None,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        resolved_mode=mode,  # type: ignore[arg-type]
    )


def test_requested_skill_contribution_precedes_skill_metadata(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    (global_root / "review").mkdir(parents=True)
    (global_root / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )

    bundle = SkillsBundle(global_root=global_root, requested_skill="review")
    contributions = bundle.context_contributions()

    assert [item.source for item in contributions] == ["agent.skills.requested", "agent.skills"]
    assert contributions[0].authority == "user"
    assert contributions[1].authority == "reference"
    assert "load_skill(name='review')" in str(contributions[0].messages[0]["content"])


def test_context_contributions_without_requested_skill_keep_metadata_only(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    (global_root / "review").mkdir(parents=True)
    (global_root / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )

    bundle = SkillsBundle(global_root=global_root)
    contributions = bundle.context_contributions()

    assert [item.source for item in contributions] == ["agent.skills"]
    assert contributions[0].authority == "reference"


def test_skills_bundle_tool_membership_differs_between_parent_and_child(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    bundle = SkillsBundle(owner_root=owner_root)

    parent = {tool.name for tool in bundle.tools(child=False)}
    child = {tool.name for tool in bundle.tools(child=True)}

    assert {"load_skill", "publish_skill", "delete_skill"} <= parent
    assert "load_skill" in child
    assert "publish_skill" not in child
    assert "delete_skill" not in child


@pytest.mark.asyncio
async def test_fast_path_retrieves_then_streams_one_synthesis() -> None:
    retrieval = MagicMock(
        contexts={"chunks": [{"content": "grounded"}], "entities": [], "relationships": []},
        trace={"retrieval": True},
    )

    retrieve_calls = 0

    async def retrieve(query: str):
        nonlocal retrieve_calls
        retrieve_calls += 1
        assert query == "question"
        return retrieval

    class Stream:
        def __aiter__(self):
            async def chunks():
                yield "answer"

            return chunks()

    synthesizer = MagicMock(spec=AnswerSynthesizer)
    synthesizer.generate_stream = AsyncMock(return_value=(retrieval.contexts, Stream()))
    orchestrator = _orchestrator(mode="fast", retrieve=retrieve, synthesizer=synthesizer)
    query_images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    contexts, stream = await orchestrator.answer_stream(
        "question",
        query_images=query_images,
    )
    assert contexts == retrieval.contexts
    assert stream is not None
    assert [chunk async for chunk in stream] == ["answer"]
    assert retrieve_calls == 1
    synthesizer.generate_stream.assert_awaited_once()
    assert synthesizer.generate_stream.await_args.kwargs["current_images"] == query_images


@pytest.mark.asyncio
async def test_research_cannot_bypass_agent_session_runtime() -> None:
    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _orchestrator(mode="research", model=model)
    with pytest.raises(RuntimeError, match="AgentSessionRuntime"):
        await orchestrator.answer_stream("question")


def test_research_preparation_composes_closed_host_tools() -> None:
    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    prepared = _orchestrator(mode="research", model=model).prepare_run("question")
    names = {tool.name for tool in prepared.tools}
    assert "search_knowledge_base" in names
    assert "subagent_status" not in names


def test_child_preparation_excludes_every_parent_subagent_control() -> None:
    from dlightrag.engine.agent.session.ids import EntryId, SessionId
    from dlightrag.engine.answer.tools.subagents import (
        ChildContextSnapshot,
        ChildRequest,
        SubagentHost,
    )

    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _orchestrator(mode="research", model=model)
    orchestrator._subagent_host = SubagentHost()  # same Host composition owner
    child = orchestrator.prepare_child_session(
        ChildRequest(objective="investigate"),
        context_snapshot=ChildContextSnapshot.from_values(
            parent_session_id=SessionId.new(),
            parent_entry_id=EntryId.new(),
            depth=0,
            messages=[],
        ),
    )
    names = {tool.name for tool in child.tools}
    assert names.isdisjoint({"spawn_agent", "subagent_status", "wait_subagent", "cancel_subagent"})
