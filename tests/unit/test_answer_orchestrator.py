# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the capability-driven answer orchestrator."""

import asyncio
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.core.agent.orchestrator import (
    INITIAL_SCOPE_OUTPUT,
    AgentProtocolError,
    AnswerOrchestrator,
    InitialScopeDecision,
    SearchInput,
)
from dlightrag.core.agent.tool_loop import AgentTool, ToolResult
from dlightrag.core.answer.errors import AnswerInputOverflowError
from dlightrag.core.answer.synthesizer import NO_CONTEXT_DISCLAIMER, AnswerSynthesizer
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchHit, WebSearchResult
from dlightrag.models.tool_turn import AssistantTurn, ToolCall


class ScriptedAgent:
    def __init__(
        self,
        *turns: AssistantTurn,
        include_web: bool = True,
        final_text: str = "Final answer generation.",
    ) -> None:
        self._turns = list(turns)
        self.include_web = include_web
        self._final_text = final_text
        self.turn_calls: list[dict[str, Any]] = []
        self.scope_calls: list[dict[str, Any]] = []
        self.final_calls: list[list[dict[str, Any]]] = []

    async def turn(self, **kwargs: Any) -> AssistantTurn:
        self.turn_calls.append(kwargs)
        return self._turns.pop(0)

    async def select_scope(self, **kwargs: Any) -> InitialScopeDecision:
        self.scope_calls.append(kwargs)
        return InitialScopeDecision(include_web=self.include_web)

    async def final(self, *, messages: list[dict[str, Any]]) -> str:
        """Tools-disabled final text call the orchestrator must route through."""
        self.final_calls.append(messages)
        return self._final_text


def _tool(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(text="", tool_calls=tuple(calls), stop_reason="tool_use")


def _call(*, query: str, source: str, call_id: str = "search") -> ToolCall:
    return ToolCall(id=call_id, name=f"search_{source}", arguments={"query": query})


def _answer(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=(), stop_reason="stop")


def _corpus_result(text: str = "corpus fact") -> RetrievalResult:
    row = {
        "chunk_id": "corpus-1",
        "reference_id": "upstream-doc",
        "full_doc_id": "doc-1",
        "file_path": "report.pdf",
        "content": text,
        "_workspace": "alpha",
        "metadata": {
            "source_type": "file",
            "source_uri": "file:///alpha/report.pdf",
            "source_download_locator": "file:///alpha/report.pdf",
        },
    }
    return RetrievalResult(contexts={"chunks": [row], "entities": [], "relationships": []})


def _web_result(text: str = "web fact", *, url: str = "https://example.com/a") -> WebSearchResult:
    return WebSearchResult(
        hits=(WebSearchHit(url=url, title="Web page", text=text),),
        cost_dollars=0.007,
    )


def _research_synthesizer() -> AnswerSynthesizer:
    """Real synthesizer that owns research finalization via injected callables.

    Its own ``model_func`` stays ``None``: the research path must generate the
    final answer through the injected tools-disabled callables, never through
    the synthesizer's fast-path ``generate`` model function.
    """
    return AnswerSynthesizer(image_max_pixels=40_000_000, model_func=None)


def _fast_synthesizer(answer_text: str = "Fast answer [1-1].") -> AnswerSynthesizer:
    async def model_func(*, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return answer_text

    return AnswerSynthesizer(
        image_max_pixels=40_000_000,
        model_func=model_func,
        effective_max_images=0,
    )


def _research(
    agent: ScriptedAgent,
    retrieve: Any,
    search: Any,
    *,
    stream_model_func: Any = None,
    context_window_tokens: int = 260_000,
    resource_tools: list[AgentTool] | None = None,
    has_resources: bool = False,
) -> AnswerOrchestrator:
    return AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=search,
        model_func=agent.turn,
        scope_model_func=agent.select_scope,
        stream_model_func=stream_model_func,
        final_text_func=agent.final,
        resource_tools=resource_tools,
        has_resources=has_resources,
        context_window_tokens=context_window_tokens,
    )


class _ReadResourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1)


def _fake_read_tool(
    content: str = "attachment evidence", *, calls: list[str] | None = None
) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = (
            raw if isinstance(raw, _ReadResourceInput) else _ReadResourceInput.model_validate(raw)
        )
        if calls is not None:
            calls.append(args.resource_id)
        return ToolResult(content=content, details={"resource_id": args.resource_id})

    return AgentTool("read_resource", "Read a registered attachment.", _ReadResourceInput, execute)


# ---------------------------------------------------------------------------
# Fast path: no resources and no web means one synthesis, no control turn.
# ---------------------------------------------------------------------------


async def test_pure_knowledge_base_takes_fast_path_with_no_control_turn() -> None:
    retrieved: list[str] = []

    async def retrieve(query: str) -> RetrievalResult:
        retrieved.append(query)
        return _corpus_result()

    orchestrator = AnswerOrchestrator(
        synthesizer=_fast_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=None,
    )
    assert orchestrator.uses_research_path is False

    result = await orchestrator.answer("what is X?", retrieval_query="standalone X")

    # Fast path: one fixed KB retrieval, one synthesis, and no control turn.
    assert result.answer is not None
    assert result.answer.startswith("Fast answer")
    assert retrieved == ["standalone X"]


async def test_fast_path_streams_one_synthesis() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    orchestrator = AnswerOrchestrator(
        synthesizer=_fast_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=None,
    )

    async def model_func(*, messages: list[dict[str, Any]], stream: bool = False, **_kw: Any):
        async def tokens():
            yield "Fast "
            yield "answer [1-1]."

        return tokens()

    orchestrator._synthesizer.model_func = model_func  # type: ignore[attr-defined]
    contexts, stream = await orchestrator.answer_stream("q", retrieval_query="q")

    assert stream is not None
    assert [token async for token in stream] == ["Fast ", "answer [1-1]."]
    assert len(contexts["chunks"]) == 1


# ---------------------------------------------------------------------------
# Research path selection.
# ---------------------------------------------------------------------------


async def test_resources_without_web_still_research_and_read_attachments() -> None:
    read_calls: list[str] = []

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    agent = ScriptedAgent(
        _tool(ToolCall(id="r", name="read_resource", arguments={"resource_id": "att-1"})),
        _answer("draft that is not the final answer"),
        final_text="From the attachment [2-1].",
    )
    orchestrator = AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=None,
        model_func=agent.turn,
        final_text_func=agent.final,
        resource_tools=[_fake_read_tool(calls=read_calls)],
        has_resources=True,
    )
    assert orchestrator.uses_research_path is True

    result = await orchestrator.answer("Summarize the attachment")

    # No web scope decision is made when Exa is absent.
    assert agent.scope_calls == []
    assert read_calls == ["att-1"]
    # search_web is never offered; read_resource is a peer tool.
    tool_names = {tool.name for tool in agent.turn_calls[0]["tools"]}
    assert "search_web" not in tool_names
    assert tool_names == {"search_knowledge_base", "read_resource"}
    # The final answer comes from one distinct tools-disabled synthesis call.
    assert len(agent.final_calls) == 1
    assert result.answer == "From the attachment [2-1]."


async def test_initial_decision_runs_fixed_corpus_and_web_wave_in_parallel() -> None:
    corpus_started = asyncio.Event()
    web_started = asyncio.Event()
    release = asyncio.Event()

    async def retrieve(_query: str) -> RetrievalResult:
        corpus_started.set()
        await release.wait()
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        web_started.set()
        await release.wait()
        return _web_result()

    agent = ScriptedAgent(_answer("draft"), final_text="Both agree [1-1][2-1].")
    task = asyncio.create_task(
        _research(agent, retrieve, search).answer(
            "What about it?",
            retrieval_query="standalone subject",
        )
    )
    await asyncio.wait_for(corpus_started.wait(), timeout=1)
    await asyncio.wait_for(web_started.wait(), timeout=1)
    release.set()
    result = await task

    assert result.answer == "Both agree [1-1][2-1]."
    assert [source.id for source in result.sources] == ["1", "2"]
    assert agent.scope_calls[0]["structured_output"] is INITIAL_SCOPE_OUTPUT
    assert agent.scope_calls[0]["messages"][-1]["content"] == "What about it?"
    assert [tool.name for tool in agent.turn_calls[0]["tools"]] == [
        "search_knowledge_base",
        "search_web",
    ]
    payload = str(agent.turn_calls[0]["messages"][-1]["content"])
    assert "Knowledge-base evidence" in payload
    assert "Open-web evidence" in payload


async def test_explicit_knowledge_base_decision_never_calls_web() -> None:
    web_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal web_calls
        web_calls += 1
        return _web_result()

    agent = ScriptedAgent(_answer("draft"), include_web=False, final_text="Corpus only [1-1].")
    result = await _research(agent, retrieve, search).answer("Use only my knowledge base")

    assert result.answer == "Corpus only [1-1]."
    assert web_calls == 0
    assert [tool.name for tool in agent.turn_calls[0]["tools"]] == ["search_knowledge_base"]


def test_source_decisions_are_closed_and_do_not_absorb_future_tools() -> None:
    initial = InitialScopeDecision.model_json_schema()["properties"]
    followup = SearchInput.model_json_schema()["properties"]

    assert set(initial) == {"include_web"}
    assert set(followup) == {"query"}
    assert "all" not in str(followup)
    assert "read_page" not in str(followup)


async def test_followup_search_uses_one_explicit_source_and_can_deepen() -> None:
    web_queries: list[str] = []

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        web_queries.append(query)
        return _web_result(
            "survey" if len(web_queries) == 1 else "deep",
            url=f"https://example.com/{len(web_queries)}",
        )

    agent = ScriptedAgent(
        _tool(_call(query="deeper angle", source="web")),
        _answer("draft"),
        final_text="Deep answer [3-1].",
    )
    result = await _research(agent, retrieve, search).answer("Research this")

    assert web_queries == ["Research this", "deeper angle"]
    assert result.answer == "Deep answer [3-1]."
    assert [source.id for source in result.sources] == ["3"]
    # scope + one search turn + one model-stop turn: no fixed round count.
    assert result.trace["agent_turns"] == 3


async def test_two_followup_sources_execute_as_parallel_tool_calls() -> None:
    kb_started = asyncio.Event()
    web_started = asyncio.Event()
    release = asyncio.Event()

    async def retrieve(query: str) -> RetrievalResult:
        if query == "gap":
            kb_started.set()
            await release.wait()
        return _corpus_result(query)

    async def search(query: str) -> WebSearchResult:
        if query == "gap":
            web_started.set()
            await release.wait()
        return _web_result(query, url=f"https://example.com/{query}")

    agent = ScriptedAgent(
        _tool(
            _call(query="gap", source="knowledge_base", call_id="kb"),
            _call(query="gap", source="web", call_id="web"),
        ),
        _answer("Combined answer."),
    )
    task = asyncio.create_task(_research(agent, retrieve, search).answer("Question"))
    await asyncio.wait_for(kb_started.wait(), timeout=1)
    await asyncio.wait_for(web_started.wait(), timeout=1)
    release.set()

    await task


async def test_no_new_evidence_ends_loop_and_triggers_final_synthesis() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    # The model repeats the initial web query, adds no new evidence, and the
    # loop ends into one distinct tools-disabled final answer generation -- there is no
    # second forced tools-none control turn.
    agent = ScriptedAgent(
        _tool(_call(query="Question", source="web")),
        final_text="Use available evidence [1-1][2-1].",
    )
    result = await _research(agent, retrieve, search).answer("Question")

    assert len(agent.turn_calls) == 1
    assert len(agent.final_calls) == 1
    assert result.answer == "Use available evidence [1-1][2-1]."
    assert result.trace["agent_stop_reason"] == "no_new_evidence"


async def test_equivalent_search_shares_work_and_reports_no_new_evidence() -> None:
    web_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal web_calls
        web_calls += 1
        return _web_result()

    agent = ScriptedAgent(
        _tool(_call(query="Question", source="web")),
        final_text="Answer [1-1][2-1].",
    )
    await _research(agent, retrieve, search).answer("Question")

    assert web_calls == 1
    # The equivalent-call notice lands in the transcript the final generation reads.
    assert "already executed" in str(agent.final_calls[0])
    assert "added 1" not in str(agent.final_calls[0])


async def test_failed_tool_call_is_evicted_and_can_be_retried() -> None:
    attempts = 0

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient web failure")
        return _web_result()

    # The initial wave scope requests web, its search fails, then the model
    # retries the same web query and it succeeds -- a failed cache entry is
    # evicted rather than pinned.
    agent = ScriptedAgent(
        _tool(_call(query="Question", source="web")),
        _answer("draft"),
        final_text="Recovered [1-1][2-1].",
    )
    result = await _research(agent, retrieve, search).answer("Question")

    assert attempts == 2
    assert result.answer == "Recovered [1-1][2-1]."


async def test_no_tool_control_turn_stops_research_before_final_synthesis() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_answer("control draft"), final_text="Done [1-1][2-1].")
    result = await _research(agent, retrieve, search).answer("Question")

    assert result.trace["agent_stop_reason"] == "model_stop"
    assert len(agent.turn_calls) == 1
    assert agent.turn_calls[0]["tool_choice"] == "auto"
    control_instruction = str(agent.turn_calls[0]["messages"][-1]["content"])
    assert "Do not draft the answer" in control_instruction
    assert "brief readiness acknowledgement" in control_instruction
    assert len(agent.final_calls) == 1
    assert result.answer == "Done [1-1][2-1]."


async def test_only_latest_exchange_is_replayed_with_canonical_evidence() -> None:
    async def retrieve(query: str) -> RetrievalResult:
        return _corpus_result(query)

    async def search(query: str) -> WebSearchResult:
        return _web_result(query, url=f"https://example.com/{query}")

    agent = ScriptedAgent(
        _tool(_call(query="second", source="web", call_id="wave-two")),
        _tool(_call(query="third", source="web", call_id="wave-three")),
        _answer("Answer."),
    )
    await _research(agent, retrieve, search).answer("Question")

    serialized = str(agent.turn_calls[2]["messages"])
    assert "wave-three" in serialized
    assert "wave-two" not in serialized
    assert "Open-web evidence" in serialized


async def test_preloaded_composer_evidence_joins_fixed_wave_one() -> None:
    attachment = {
        "chunk_id": "attachment-1",
        "reference_id": "attachment-upstream",
        "full_doc_id": "attachment-doc",
        "file_path": "uploaded.pdf",
        "content": "uploaded evidence",
        "_workspace": "__web_attachment__",
        "metadata": {
            "source_type": "web_attachment",
            "source_uri": "web-attachment://1",
            "source_download_locator": "web-attachment://1",
        },
    }

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_answer("draft"), final_text="Combined [1-1][2-1][3-1].")
    result = await _research(agent, retrieve, search).answer(
        "Question",
        initial_contexts={"chunks": [attachment], "entities": [], "relationships": []},
    )

    assert [source.id for source in result.sources] == ["1", "2", "3"]
    payload = str(agent.turn_calls[0]["messages"][-1]["content"])
    assert "User-attached documents" in payload


async def test_input_over_envelope_stops_before_scope_or_retrieval() -> None:
    retrieval_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        nonlocal retrieval_calls
        retrieval_calls += 1
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent()
    # An input budget of one token cannot fit the fixed system prompt/query.
    orchestrator = _research(agent, retrieve, search, context_window_tokens=32_769)

    with pytest.raises(AnswerInputOverflowError):
        await orchestrator.answer("Question")

    assert agent.scope_calls == []
    assert agent.turn_calls == []
    assert retrieval_calls == 0


async def test_invalid_scope_result_stops_before_retrieval() -> None:
    retrieval_calls = 0

    class WrongDecision(BaseModel):
        value: str

    async def wrong_scope(**_kwargs: Any) -> BaseModel:
        return WrongDecision(value="wrong")

    async def retrieve(_query: str) -> RetrievalResult:
        nonlocal retrieval_calls
        retrieval_calls += 1
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent()
    orchestrator = AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=search,
        model_func=agent.turn,
        scope_model_func=wrong_scope,
        final_text_func=agent.final,
    )

    with pytest.raises(AgentProtocolError, match="InitialScopeDecision"):
        await orchestrator.answer("Question")

    assert retrieval_calls == 0


async def test_research_answer_can_be_cancelled_mid_flight() -> None:
    started = asyncio.Event()

    async def retrieve(_query: str) -> RetrievalResult:
        started.set()
        await asyncio.sleep(3600)
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_answer("never reached"), include_web=False)
    task = asyncio.create_task(_research(agent, retrieve, search).answer("Question"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_streaming_no_tool_turn_starts_distinct_native_final_stream() -> None:
    streamed_messages: list[list[dict[str, Any]]] = []

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    async def tokens():
        yield "Final "
        yield "answer [1-1][2-1]."

    def stream_text(*, messages: list[dict[str, Any]]):
        streamed_messages.append(messages)
        return tokens()

    agent = ScriptedAgent(_answer("control draft"))
    contexts, stream = await _research(
        agent,
        retrieve,
        search,
        stream_model_func=stream_text,
    ).answer_stream("Question")

    assert stream is not None
    assert len(contexts["chunks"]) == 2
    assert {tool.name for tool in agent.turn_calls[0]["tools"]} == {
        "search_knowledge_base",
        "search_web",
    }
    assert agent.turn_calls[0]["tool_choice"] == "auto"
    assert "Answer the original request now" in str(streamed_messages[0][-1]["content"])
    assert [token async for token in stream] == ["Final ", "answer [1-1][2-1]."]
    assert cast(Any, stream).answer == "Final answer [1-1][2-1]."
    assert cast(Any, stream).trace["agent_stop_reason"] == "model_stop"


# ---------------------------------------------------------------------------
# The research path routes its final answer through the AnswerSynthesizer.
# ---------------------------------------------------------------------------


async def test_research_final_answer_is_a_distinct_tools_disabled_synthesis() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(
        _answer("DRAFT control-turn text that must never be the final answer"),
        final_text="Synthesized final [1-1][2-1].",
    )
    result = await _research(agent, retrieve, search).answer("Question")

    # The control turn only signals a stop; a distinct tools-disabled synthesis
    # produces the answer and the synthesizer owns finalization + answer media.
    assert len(agent.final_calls) == 1
    assert result.answer == "Synthesized final [1-1][2-1]."
    assert "DRAFT control-turn text" not in (result.answer or "")
    assert [source.id for source in result.sources] == ["1", "2"]
    assert result.references
    assert result.answer_blocks
    # The final call carries the reasoning-bearing tool transcript, no live tools.
    final_messages = agent.final_calls[0]
    assert any(msg.get("role") == "assistant" for msg in final_messages)


async def test_research_stream_final_flows_through_synthesizer_no_context_and_warnings() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return RetrievalResult(contexts={"chunks": [], "entities": [], "relationships": []})

    async def search(_query: str) -> WebSearchResult:
        return WebSearchResult(hits=(), cost_dollars=0.0)

    async def tokens():
        yield "Best-effort answer."

    def stream_text(*, messages: list[dict[str, Any]]):
        return tokens()

    agent = ScriptedAgent(_answer("control draft"))
    contexts, stream = await _research(
        agent,
        retrieve,
        search,
        stream_model_func=stream_text,
    ).answer_stream("Question")

    assert stream is not None
    # The synthesizer owns the no-context disclaimer and the warnings list for
    # the streaming research branch, not the orchestrator.
    assert isinstance(cast(Any, stream).warnings, list)
    emitted = [token async for token in stream]
    assert emitted[0].startswith(NO_CONTEXT_DISCLAIMER)
    assert "Best-effort answer." in "".join(emitted)
    assert not contexts["chunks"]
