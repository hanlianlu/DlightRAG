# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Optional evidence-gathering answer runner."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.citations.finalization import finalize_answer
from dlightrag.core.agent.evidence import EvidenceSession
from dlightrag.core.agent.tool_loop import AgentTool, ToolResult, ToolTurnExecutor
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchResult, web_context_rows
from dlightrag.models.tool_turn import AssistantTurn
from dlightrag.prompts.agent import agentic_answer_prompt

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[str], Awaitable[WebSearchResult]]
ToolModel = Callable[..., Awaitable[AssistantTurn]]


class RetrieveEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["all", "knowledge_base"] = "all"


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class AgentProtocolError(RuntimeError):
    """The model violated a required agent turn contract."""


class AgenticAnswerRunner:
    """Gather request-local evidence and produce one citation-checked answer."""

    def __init__(
        self,
        *,
        model_func: ToolModel,
        retrieve_knowledge_base: KnowledgeRetrieval,
        search_web: WebSearch,
    ) -> None:
        self._executor = ToolTurnExecutor(model_func)
        self._retrieve_knowledge_base = retrieve_knowledge_base
        self._search_web = search_web

    async def run(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        query_images: list[dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        session = EvidenceSession()
        cache = _ToolCallCache()
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
        }

        async def retrieve_initial(raw: BaseModel) -> ToolResult:
            args = _as(raw, RetrieveEvidenceInput)
            return await cache.run(
                f"retrieve_evidence:{args.scope}",
                lambda: self._retrieve_initial(query, args.scope, session, trace),
            )

        async def search_corpus(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("search_knowledge_base", args.query),
                lambda: self._search_corpus(args.query, session),
            )

        async def search_web(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("search_web", args.query),
                lambda: self._search_open_web(args.query, session, trace),
            )

        first_tool = AgentTool(
            "retrieve_evidence",
            (
                "Retrieve the first evidence wave. Scope defaults to all, which searches the "
                "knowledge base and open web in parallel. Use knowledge_base only when the "
                "user explicitly asks to exclude outside sources."
            ),
            RetrieveEvidenceInput,
            retrieve_initial,
        )
        followup_tools = [
            AgentTool(
                "search_knowledge_base",
                "Search the indexed knowledge base for one concrete unresolved fact.",
                SearchInput,
                search_corpus,
            ),
            AgentTool(
                "search_web",
                "Search the open web for one concrete unresolved or current fact.",
                SearchInput,
                search_web,
            ),
        ]

        transcript = _initial_messages(
            query,
            conversation_history=conversation_history,
            query_images=query_images,
        )
        evidence_message: dict[str, Any] | None = None
        first_turn = True
        force_answer = False
        stop_reason = "model_stop"

        while True:
            tools = [first_tool] if first_turn else ([] if force_answer else followup_tools)
            tool_choice = "required" if first_turn else ("none" if force_answer else "auto")
            call_messages = [*transcript]
            if evidence_message is not None:
                call_messages.append(evidence_message)
            previous_evidence_count = _evidence_count(session)
            executed = await self._executor.run_turn(
                call_messages,
                tools,
                tool_choice=tool_choice,
            )
            trace["agent_turns"] += 1
            transcript.extend(executed.messages[len(call_messages) :])

            if not executed.assistant.tool_calls:
                if first_turn:
                    raise AgentProtocolError("The model skipped the required retrieval turn.")
                _, indexer = session.render_blocks()
                finalized = finalize_answer(
                    executed.assistant.text,
                    session.contexts,
                    indexer=indexer,
                )
                trace["agent_stop_reason"] = stop_reason
                return RetrievalResult(
                    answer=finalized.answer,
                    contexts=session.contexts,
                    sources=finalized.sources,
                    trace=trace,
                )

            if force_answer:
                raise AgentProtocolError("The model called a tool after tools were withdrawn.")

            first_turn = False
            evidence_message = _evidence_message(session)
            if _evidence_count(session) == previous_evidence_count:
                force_answer = True
                stop_reason = "no_new_evidence"

    async def _retrieve_initial(
        self,
        query: str,
        scope: Literal["all", "knowledge_base"],
        session: EvidenceSession,
        trace: dict[str, Any],
    ) -> ToolResult:
        corpus_task = asyncio.ensure_future(self._retrieve_knowledge_base(query))
        web_task = asyncio.ensure_future(self._search_web(query)) if scope == "all" else None
        tasks: list[asyncio.Future[Any]] = [corpus_task]
        if web_task is not None:
            tasks.append(web_task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        messages: list[str] = []
        successes = 0
        corpus = results[0]
        if isinstance(corpus, RetrievalResult):
            delta = session.add_contexts(corpus.contexts)
            messages.append(f"Knowledge base added {delta.new_chunks} passages.")
            successes += 1
        else:
            messages.append(f"Knowledge-base retrieval failed: {corpus}")
        if web_task is not None:
            web = results[1]
            if isinstance(web, WebSearchResult):
                delta = session.add_rows(web_context_rows(web.hits))
                trace["web_search_cost_dollars"] += web.cost_dollars
                messages.append(f"Open web added {delta.new_chunks} passages.")
                successes += 1
            else:
                messages.append(f"Open-web retrieval failed: {web}")
        if successes == 0:
            raise RuntimeError("; ".join(messages))
        return ToolResult(content=" ".join(messages))

    async def _search_corpus(self, query: str, session: EvidenceSession) -> ToolResult:
        result = await self._retrieve_knowledge_base(query)
        delta = session.add_contexts(result.contexts)
        return ToolResult(content=f"Knowledge base added {delta.new_chunks} new passages.")

    async def _search_open_web(
        self,
        query: str,
        session: EvidenceSession,
        trace: dict[str, Any],
    ) -> ToolResult:
        result = await self._search_web(query)
        delta = session.add_rows(web_context_rows(result.hits))
        trace["web_search_cost_dollars"] += result.cost_dollars
        return ToolResult(content=f"Open web added {delta.new_chunks} new passages.")


class _ToolCallCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Future[ToolResult]] = {}

    async def run(
        self,
        key: str,
        operation: Callable[[], Awaitable[ToolResult]],
    ) -> ToolResult:
        async with self._lock:
            task = self._tasks.get(key)
            if task is None:
                task = asyncio.ensure_future(operation())
                self._tasks[key] = task
        return await task


def _initial_messages(
    query: str,
    *,
    conversation_history: list[dict[str, Any]] | None,
    query_images: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": agentic_answer_prompt()},
        *(conversation_history or []),
    ]
    if query_images:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *query_images,
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": query})
    return messages


def _evidence_message(session: EvidenceSession) -> dict[str, Any]:
    blocks, _ = session.render_blocks()
    return {
        "role": "user",
        "content": [
            *blocks,
            {
                "type": "text",
                "text": (
                    "Use the current evidence above. Answer now if it is sufficient; otherwise "
                    "call one or more tools for a concrete missing fact."
                ),
            },
        ],
    }


def _evidence_count(session: EvidenceSession) -> int:
    return sum(len(rows) for rows in session.contexts.values())


def _call_key(name: str, query: str) -> str:
    return f"{name}:{json.dumps(query.strip(), ensure_ascii=False)}"


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = ["AgentProtocolError", "AgenticAnswerRunner"]
