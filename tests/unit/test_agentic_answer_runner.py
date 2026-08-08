# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the Exa-enabled answer runner."""

import asyncio
from typing import Any

from dlightrag.core.agent.runner import AgenticAnswerRunner, AgentInputOverflowError
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchHit, WebSearchResult
from dlightrag.models.tool_turn import AssistantTurn, ToolCall


class ScriptedModel:
    def __init__(self, *turns: AssistantTurn) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> AssistantTurn:
        self.calls.append(kwargs)
        return self._turns.pop(0)


def _call(name: str, arguments: dict[str, Any], *, call_id: str = "call") -> AssistantTurn:
    return AssistantTurn(
        text="",
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        stop_reason="tool_use",
    )


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


async def test_wave_one_all_runs_corpus_and_web_in_parallel_then_finalizes_sources() -> None:
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

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}),
        _answer("The corpus and current page agree [1-1][2-1]."),
    )
    runner = AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    )
    task = asyncio.create_task(runner.run("What is true now?"))
    await asyncio.wait_for(corpus_started.wait(), timeout=1)
    await asyncio.wait_for(web_started.wait(), timeout=1)
    release.set()

    result = await task

    assert result.answer == "The corpus and current page agree [1-1][2-1]."
    assert [source.id for source in result.sources] == ["1", "2"]
    assert [tool.name for tool in model.calls[0]["tools"]] == ["retrieve_evidence"]
    assert model.calls[0]["tool_choice"] == "required"
    assert {tool.name for tool in model.calls[1]["tools"]} == {
        "search_knowledge_base",
        "search_web",
    }
    second_payload = str(model.calls[1]["messages"][-1]["content"])
    assert "Knowledge-base evidence" in second_payload
    assert "Open-web evidence" in second_payload


async def test_explicit_knowledge_base_scope_never_calls_web() -> None:
    web_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal web_calls
        web_calls += 1
        return _web_result()

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "knowledge_base"}),
        _answer("Corpus only [1-1]."),
    )
    result = await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("Use only the knowledge base")

    assert result.answer == "Corpus only [1-1]."
    assert web_calls == 0


async def test_a_followup_search_can_add_a_new_source_before_answering() -> None:
    web_queries: list[str] = []

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        web_queries.append(query)
        if len(web_queries) == 1:
            return _web_result("survey fact", url="https://example.com/survey")
        return _web_result("deeper fact", url="https://example.com/deep")

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}, call_id="first"),
        _call("search_web", {"query": "deeper angle"}, call_id="second"),
        _answer("Deep answer [3-1]."),
    )
    result = await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("Research this")

    assert web_queries == ["Research this", "deeper angle"]
    assert result.answer == "Deep answer [3-1]."
    assert [source.id for source in result.sources] == ["3"]
    assert result.trace["agent_turns"] == 3


async def test_no_new_evidence_removes_tools_from_the_next_turn() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}, call_id="first"),
        _call("search_web", {"query": "same page"}, call_id="second"),
        _answer("Use what is available [1-1][2-1]."),
    )
    result = await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("Question")

    assert result.answer == "Use what is available [1-1][2-1]."
    assert model.calls[2]["tools"] == []
    assert model.calls[2]["tool_choice"] == "none"
    assert result.trace["agent_stop_reason"] == "no_new_evidence"


async def test_first_turn_cannot_skip_required_retrieval() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    model = ScriptedModel(_answer("Ungrounded shortcut"))
    runner = AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    )

    try:
        await runner.run("Question")
    except RuntimeError as exc:
        assert "required retrieval" in str(exc)
    else:
        raise AssertionError("runner accepted an answer before wave-one retrieval")


async def test_user_intent_sees_original_words_while_tools_use_standalone_query() -> None:
    retrieval_queries: list[str] = []

    async def retrieve(query: str) -> RetrievalResult:
        retrieval_queries.append(query)
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        retrieval_queries.append(query)
        return _web_result()

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}),
        _answer("Answer [1-1][2-1]."),
    )
    await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("What about it?", retrieval_query="standalone subject query")

    assert model.calls[0]["messages"][-1]["content"] == "What about it?"
    assert retrieval_queries == ["standalone subject query", "standalone subject query"]


async def test_preloaded_composer_evidence_joins_but_does_not_replace_wave_one() -> None:
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

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}),
        _answer("Combined [1-1][2-1][3-1]."),
    )
    result = await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run(
        "Question",
        initial_contexts={"chunks": [attachment], "entities": [], "relationships": []},
    )

    assert [source.id for source in result.sources] == ["1", "2", "3"]
    payload = str(model.calls[1]["messages"][-1]["content"])
    assert "User-attached documents" in payload
    assert "Knowledge-base evidence" in payload
    assert "Open-web evidence" in payload


async def test_input_over_envelope_is_rejected_before_a_model_call() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        return _web_result()

    model = ScriptedModel(_call("retrieve_evidence", {"scope": "all"}))
    runner = AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
        input_token_envelope=1,
    )

    try:
        await runner.run("Question")
    except AgentInputOverflowError:
        pass
    else:
        raise AssertionError("runner called a model above its input envelope")

    assert model.calls == []


async def test_each_retrieval_lane_uses_the_existing_answer_context_cap() -> None:
    corpus = _corpus_result()
    corpus.contexts["chunks"].append(
        {**corpus.contexts["chunks"][0], "chunk_id": "corpus-2", "content": "second corpus"}
    )

    async def retrieve(_query: str) -> RetrievalResult:
        return corpus

    async def search(_query: str) -> WebSearchResult:
        return WebSearchResult(
            hits=(
                WebSearchHit(url="https://example.com/a", title="A", text="web a"),
                WebSearchHit(url="https://example.com/b", title="B", text="web b"),
            ),
            cost_dollars=0.007,
        )

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}),
        _answer("Answer [1-1][2-1]."),
    )
    result = await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
        context_top_k=1,
    ).run("Question")

    assert len(result.contexts["chunks"]) == 2
    assert {row["metadata"]["source_type"] for row in result.contexts["chunks"]} == {
        "file",
        "web_search",
    }


async def test_only_the_latest_tool_exchange_is_replayed_with_canonical_evidence() -> None:
    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(query: str) -> WebSearchResult:
        return _web_result(query, url=f"https://example.com/{query}")

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}, call_id="wave-one"),
        _call("search_web", {"query": "second"}, call_id="wave-two"),
        _answer("Answer [1-1][2-1][3-1]."),
    )
    await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("Question")

    third_messages = model.calls[2]["messages"]
    serialized = str(third_messages)
    assert "wave-two" in serialized
    assert "wave-one" not in serialized
    assert "Knowledge-base evidence" in serialized
    assert "Open-web evidence" in serialized


async def test_equivalent_tool_calls_share_work_but_report_the_repeat_truthfully() -> None:
    web_calls = 0

    async def retrieve(_query: str) -> RetrievalResult:
        return _corpus_result()

    async def search(_query: str) -> WebSearchResult:
        nonlocal web_calls
        web_calls += 1
        return _web_result()

    model = ScriptedModel(
        _call("retrieve_evidence", {"scope": "all"}, call_id="wave-one"),
        _call("search_web", {"query": "Question"}, call_id="repeat"),
        _answer("Answer [1-1][2-1]."),
    )
    await AgenticAnswerRunner(
        model_func=model,
        retrieve_knowledge_base=retrieve,
        search_web=search,
    ).run("Question")

    assert web_calls == 1
    repeat_result = model.calls[2]["messages"][-2]
    assert repeat_result["role"] == "tool"
    assert "already executed" in repeat_result["content"]
    assert "added 1" not in repeat_result["content"]
