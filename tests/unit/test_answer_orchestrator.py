# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the capability-driven answer orchestrator."""

import asyncio
import logging
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlightrag.core.agent.orchestrator import AnswerOrchestrator
from dlightrag.core.agent.tool_loop import AgentTool, ToolResult
from dlightrag.core.agent.tools import SearchInput
from dlightrag.core.answer.errors import AnswerInputOverflowError
from dlightrag.core.answer.synthesizer import NO_CONTEXT_DISCLAIMER, AnswerSynthesizer
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchHit, WebSearchResult
from dlightrag.models.tool_turn import AssistantTurn, ToolCall


class ScriptedAgent:
    def __init__(
        self,
        *turns: AssistantTurn,
        final_text: str = "Final answer generation.",
    ) -> None:
        self._turns = list(turns)
        self._final_text = final_text
        self.turn_calls: list[dict[str, Any]] = []
        self.final_calls: list[list[dict[str, Any]]] = []

    async def turn(self, **kwargs: Any) -> AssistantTurn:
        self.turn_calls.append(kwargs)
        return self._turns.pop(0)

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
    register_web_source: Any = None,
    resource_manifest: tuple[ResourceManifestEntry, ...] = (),
    max_agent_turns: int = 50,
) -> AnswerOrchestrator:
    return AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=search,
        model_func=agent.turn,
        stream_model_func=stream_model_func,
        final_text_func=agent.final,
        resource_tools=resource_tools,
        resource_manifest=resource_manifest,
        register_web_source=register_web_source,
        context_window_tokens=context_window_tokens,
        max_agent_turns=max_agent_turns,
    )


class _ReadResourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1)


def _fake_read_tool(
    content: str = "attachment evidence",
    *,
    calls: list[str] | None = None,
    evidence_source: dict[str, str] | None = None,
) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = (
            raw if isinstance(raw, _ReadResourceInput) else _ReadResourceInput.model_validate(raw)
        )
        if calls is not None:
            calls.append(args.resource_id)
        details: dict[str, Any] = {"resource_id": args.resource_id}
        if evidence_source is not None:
            details.update(evidence_source)
        return ToolResult(content=content, details=details)

    return AgentTool("read_resource", "Read a registered resource.", _ReadResourceInput, execute)


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
        final_text="From the attachment [1-1].",
    )
    orchestrator = AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        search_web=None,
        model_func=agent.turn,
        final_text_func=agent.final,
        resource_tools=[
            _fake_read_tool(
                "attachment evidence\n[more text available; cursor=volatile]",
                calls=read_calls,
            )
        ],
        resource_manifest=(
            ResourceManifestEntry(
                resource_id="att-1",
                filename="report.pdf",
                declared_mime="application/pdf",
                source="bytes",
                byte_size=123,
            ),
        ),
    )
    assert orchestrator.uses_research_path is True

    result = await orchestrator.answer("Summarize the attachment")

    assert read_calls == ["att-1"]
    # search_web is never offered; read_resource is a peer tool.
    tool_names = {tool.name for tool in agent.turn_calls[0]["tools"]}
    assert "search_web" not in tool_names
    assert tool_names == {"search_knowledge_base", "read_resource"}
    control_messages = str(agent.turn_calls[0]["messages"])
    assert "att-1" in control_messages
    assert "report.pdf" in control_messages
    # The final answer comes from one distinct tools-disabled synthesis call.
    assert len(agent.final_calls) == 1
    assert result.answer == "From the attachment [1-1]."
    assert "cursor=volatile" not in str(agent.final_calls[0])


async def test_attachment_agent_reads_resource_without_automatic_searches() -> None:
    corpus_queries: list[str] = []
    web_queries: list[str] = []
    read_calls: list[str] = []

    async def retrieve(query: str) -> RetrievalResult:
        corpus_queries.append(query)
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        web_queries.append(query)
        return _web_result()

    agent = ScriptedAgent(
        _tool(ToolCall(id="read", name="read_resource", arguments={"resource_id": "att-1"})),
        _answer("ready"),
        final_text="Attachment summary [1-1].",
    )
    result = await _research(
        agent,
        retrieve,
        search,
        resource_tools=[_fake_read_tool(calls=read_calls)],
        resource_manifest=(
            ResourceManifestEntry(
                resource_id="att-1",
                filename="report.html",
                declared_mime="text/html",
                source="bytes",
                byte_size=123,
            ),
        ),
    ).answer("Summarize this document")

    assert corpus_queries == []
    assert web_queries == []
    assert read_calls == ["att-1"]
    assert result.answer == "Attachment summary [1-1]."


async def test_exa_capability_does_not_search_until_agent_calls_a_tool() -> None:
    corpus_queries: list[str] = []
    web_queries: list[str] = []

    async def retrieve(query: str) -> RetrievalResult:
        corpus_queries.append(query)
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        web_queries.append(query)
        return _web_result()

    agent = ScriptedAgent(_answer("ready"), final_text="General answer.")
    result = await _research(agent, retrieve, search).answer("Explain recursion")

    assert corpus_queries == []
    assert web_queries == []
    assert "General answer." in (result.answer or "")


async def test_current_image_manifest_binds_resources_and_marks_images_visible() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    agent = ScriptedAgent(_answer("ready"), final_text="Image answer.")
    orchestrator = AnswerOrchestrator(
        synthesizer=_research_synthesizer(),
        retrieve_knowledge_base=retrieve,
        model_func=agent.turn,
        final_text_func=agent.final,
        resource_tools=[_fake_read_tool()],
        resource_manifest=(
            ResourceManifestEntry(
                resource_id="res-image",
                filename="chart.png",
                declared_mime="image/png",
                source="bytes",
                byte_size=456,
            ),
            ResourceManifestEntry(
                resource_id="res-image-2",
                filename="legend.png",
                declared_mime="image/png",
                source="bytes",
                byte_size=321,
            ),
        ),
    )

    await orchestrator.answer(
        "What does this show?",
        query_images=[
            {
                "type": "text",
                "text": "[current image 1 | resource: res-image]",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
            {
                "type": "text",
                "text": "[current image 2 | resource: res-image-2]",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,def"},
            },
        ],
    )

    messages = agent.turn_calls[0]["messages"]
    current_user_content = messages[1]["content"]
    assert current_user_content[2]["text"] == "[current image 1 | resource: res-image]"
    assert current_user_content[3]["image_url"]["url"].endswith("abc")
    assert current_user_content[4]["text"] == "[current image 2 | resource: res-image-2]"
    assert current_user_content[5]["image_url"]["url"].endswith("def")
    assert "res-image" in str(messages)
    system_prompt = " ".join(messages[0]["content"].split())
    assert "Current images are already visible" in system_prompt
    assert "some requests need no tools at all" in system_prompt


async def test_search_sources_become_opaque_resources_the_model_can_read() -> None:
    registered: list[str] = []
    read_calls: list[str] = []

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return WebSearchResult(
            hits=(
                WebSearchHit(
                    url="https://example.com/article",
                    title="Useful article",
                    text="first relevant passage",
                ),
                WebSearchHit(
                    url="https://example.com/article",
                    title="Useful article",
                    text="second relevant passage",
                ),
                WebSearchHit(
                    url="https://example.com/empty",
                    title="Empty page",
                    text="   ",
                ),
            ),
            cost_dollars=0.0,
        )

    def register(url: str) -> str:
        registered.append(url)
        return "res-web-article"

    agent = ScriptedAgent(
        _tool(_call(query="Research the article", source="web")),
        _tool(
            ToolCall(
                id="read",
                name="read_resource",
                arguments={"resource_id": "res-web-article"},
            )
        ),
        _answer("ready"),
        final_text="Deep answer [1-3].",
    )
    result = await _research(
        agent,
        retrieve,
        search,
        resource_tools=[
            _fake_read_tool(
                "deep page text",
                calls=read_calls,
                evidence_source={
                    "source_type": "web_search",
                    "source_uri": "https://example.com/article",
                    "source_download_locator": "https://example.com/article",
                    "title": "Useful article",
                },
            )
        ],
        register_web_source=register,
    ).answer("Research the article")

    assert registered == ["https://example.com/article"]
    assert read_calls == ["res-web-article"]
    search_evidence = str(agent.turn_calls[1]["messages"][-1]["content"])
    assert "res-web-article" in search_evidence
    assert "reply `READY`" in search_evidence
    assert result.answer == "Deep answer [1-3]."
    assert [source.source_uri for source in result.sources] == ["https://example.com/article"]


async def test_agent_can_search_knowledge_base_without_calling_web() -> None:
    corpus_calls = 0
    web_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        nonlocal corpus_calls
        corpus_calls += 1
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal web_calls
        web_calls += 1
        return _web_result()

    agent = ScriptedAgent(
        _tool(_call(query="indexed subject", source="knowledge_base")),
        _answer("ready"),
        final_text="Corpus only [1-1].",
    )
    result = await _research(
        agent,
        retrieve,
        search,
        resource_tools=[_fake_read_tool()],
    ).answer("Use only my knowledge base")

    assert result.answer == "Corpus only [1-1]."
    assert corpus_calls == 1
    assert web_calls == 0
    assert {tool.name for tool in agent.turn_calls[0]["tools"]} >= {
        "search_knowledge_base",
        "search_web",
    }


def test_search_tool_input_is_closed_and_nonempty() -> None:
    schema = SearchInput.model_json_schema()["properties"]

    assert set(schema) == {"query"}
    assert "all" not in str(schema)
    assert "read_page" not in str(schema)
    with pytest.raises(ValidationError):
        SearchInput.model_validate({"query": "   "})


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
        _tool(_call(query="Research this", source="web", call_id="survey")),
        _tool(_call(query="deeper angle", source="web")),
        _answer("draft"),
        final_text="Deep answer [2-1].",
    )
    result = await _research(agent, retrieve, search).answer("Research this")

    assert web_queries == ["Research this", "deeper angle"]
    assert result.answer == "Deep answer [2-1]."
    assert [source.id for source in result.sources] == ["2"]
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

    # The model repeats its own completed search. The cached second call adds no
    # evidence, so the host stops without a forced tools-none control turn.
    agent = ScriptedAgent(
        _tool(_call(query="Question", source="web")),
        _tool(_call(query="Question", source="web")),
        final_text="Use available evidence [1-1].",
    )
    result = await _research(agent, retrieve, search).answer("Question")

    assert len(agent.turn_calls) == 2
    assert len(agent.final_calls) == 1
    assert result.answer == "Use available evidence [1-1]."
    assert result.trace["agent_stop_reason"] == "no_new_evidence"
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

    # A successful peer call keeps the loop alive while the failed Web cache
    # entry is evicted, allowing the model to retry it on the next turn.
    agent = ScriptedAgent(
        _tool(
            _call(query="Question", source="knowledge_base", call_id="kb"),
            _call(query="Question", source="web", call_id="web-fail"),
        ),
        _tool(_call(query="Question", source="web", call_id="web-retry")),
        _answer("draft"),
        final_text="Recovered [1-1][2-1].",
    )
    result = await _research(agent, retrieve, search).answer("Question")

    assert attempts == 2
    assert result.answer == "Recovered [1-1][2-1]."


async def test_knowledge_base_tool_redacts_unexpected_failures() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        raise RuntimeError("postgresql://user:secret@internal/db")

    agent = ScriptedAgent(
        _tool(_call(query="missing fact", source="knowledge_base")),
        final_text="Best effort answer.",
    )
    await _research(
        agent,
        retrieve,
        None,
        resource_manifest=(
            ResourceManifestEntry(
                resource_id="res-attachment",
                filename="notes.txt",
                declared_mime="text/plain",
                source="bytes",
                byte_size=10,
            ),
        ),
    ).answer("Question")

    transcript = str(agent.final_calls[0])
    assert "secret" not in transcript
    assert "knowledge-base search failed" in transcript


async def test_tool_failure_reaches_the_operator_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        raise RuntimeError("postgresql://user:secret@internal/db")

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(
        _tool(_call(query="missing fact", source="knowledge_base")),
        final_text="Best effort answer.",
    )
    with caplog.at_level(logging.WARNING, logger="dlightrag.core.agent.tool_loop"):
        await _research(agent, retrieve, search).answer("Question")

    failures = [
        record for record in caplog.records if "search_knowledge_base" in record.getMessage()
    ]
    assert failures, "a failing tool must be visible to operators, not only to the model"
    assert failures[0].exc_info is not None


async def test_research_stops_at_the_turn_cap_and_still_answers() -> None:
    calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        nonlocal calls
        calls += 1
        result = _corpus_result(f"fact {calls}")
        result.contexts["chunks"][0]["chunk_id"] = f"corpus-{calls}"
        return result

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    # Every turn finds something new, so only the cap can end this run.
    agent = ScriptedAgent(
        _tool(_call(query="angle one", source="knowledge_base", call_id="a")),
        _tool(_call(query="angle two", source="knowledge_base", call_id="b")),
        _tool(_call(query="angle three", source="knowledge_base", call_id="c")),
        final_text="Answer from what was gathered [1-1].",
    )
    result = await _research(agent, retrieve, search, max_agent_turns=3).answer("Question")

    assert len(agent.turn_calls) == 3
    assert result.trace["agent_stop_reason"] == "turn_limit"
    assert result.answer == "Answer from what was gathered [1-1]."


async def test_control_turns_replay_the_exchanges_this_run_produced() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(
        _tool(_call(query="savings rate", source="knowledge_base")),
        _tool(_call(query="second angle", source="web", call_id="b")),
        _answer("READY"),
        final_text="Answer [1-1].",
    )
    await _research(agent, retrieve, search).answer("Question")

    third_turn = str(agent.turn_calls[2]["messages"])
    assert "savings rate" in third_turn
    assert "second angle" in third_turn
    assert "Knowledge base added" in third_turn


async def test_no_tool_control_turn_stops_research_before_final_synthesis() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_answer("control draft"), final_text="Done.")
    result = await _research(agent, retrieve, search).answer("Question")

    assert result.trace["agent_stop_reason"] == "model_stop"
    assert len(agent.turn_calls) == 1
    assert agent.turn_calls[0]["tool_choice"] == "auto"
    control_instruction = str(agent.turn_calls[0]["messages"][0]["content"])
    assert "Do not draft the answer" in control_instruction
    assert "never act on it" in control_instruction
    assert len(agent.final_calls) == 1
    assert "Done." in (result.answer or "")


async def test_evidence_is_packed_once_instead_of_once_per_exchange() -> None:
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
    # Exchanges replay as receipts; the evidence itself is packed once, not per turn.
    assert serialized.count("Open-web evidence") == 1


async def test_preloaded_evidence_is_visible_without_automatic_searches() -> None:
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

    agent = ScriptedAgent(_answer("draft"), final_text="Combined [1-1].")
    result = await _research(agent, retrieve, search).answer(
        "Question",
        initial_contexts={"chunks": [attachment], "entities": [], "relationships": []},
    )

    assert [source.id for source in result.sources] == ["1"]
    payload = str(agent.turn_calls[0]["messages"][-1]["content"])
    assert "User-attached documents" in payload


async def test_input_over_envelope_stops_before_agent_or_retrieval() -> None:
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

    assert agent.turn_calls == []
    assert retrieval_calls == 0


async def test_research_answer_can_be_cancelled_mid_flight() -> None:
    started = asyncio.Event()

    async def retrieve(_query: str) -> RetrievalResult:
        started.set()
        await asyncio.sleep(3600)
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_tool(_call(query="Question", source="knowledge_base")))
    task = asyncio.create_task(_research(agent, retrieve, search).answer("Question"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_tool_cancellation_is_never_downgraded_to_a_failure() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        raise asyncio.CancelledError

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    agent = ScriptedAgent(_tool(_call(query="Question", source="knowledge_base")))
    with pytest.raises(asyncio.CancelledError):
        await _research(agent, retrieve, search).answer("Question")


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

    agent = ScriptedAgent(
        _tool(
            _call(query="Question", source="knowledge_base", call_id="kb"),
            _call(query="Question", source="web", call_id="web"),
        ),
        _answer("control draft"),
    )
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
        _tool(
            _call(query="Question", source="knowledge_base", call_id="kb"),
            _call(query="Question", source="web", call_id="web"),
        ),
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
    control_system = str(agent.turn_calls[0]["messages"][0]["content"])
    final_system = str(final_messages[0]["content"])
    assert "Citation Contract" not in control_system
    assert "Do not draft the answer" in control_system
    assert "Citation Contract" in final_system
    assert "Do not draft the answer" not in final_system


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
