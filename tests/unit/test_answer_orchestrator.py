# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer Host coordination around the deep AgentSessionRuntime."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.ai.messages import AssistantTurn
from dlightrag.ai.telemetry import NOOP_TELEMETRY
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.resources.models import TextWindowBudget
from dlightrag.answer.synthesizer import AnswerSynthesizer
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


@pytest.mark.asyncio
async def test_fast_path_retrieves_then_streams_one_synthesis() -> None:
    retrieval = MagicMock(
        contexts={"chunks": [{"content": "grounded"}], "entities": [], "relationships": []},
        trace={"retrieval": True},
    )

    async def retrieve(query: str):
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
    contexts, stream = await orchestrator.answer_stream("question")
    assert contexts == retrieval.contexts
    assert stream is not None
    assert [chunk async for chunk in stream] == ["answer"]


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
    from dlightrag.agent.session.ids import EntryId, SessionId
    from dlightrag.answer.tools.subagents import (
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
