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
from dlightrag.core.answer.engine import ANSWER_INPUT_TOKEN_ENVELOPE
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchResult, web_context_rows
from dlightrag.models.tool_turn import AssistantTurn
from dlightrag.prompts.agent import agentic_answer_prompt
from dlightrag.utils.tokens import estimate_messages_tokens, truncate_conversation_history

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


class AgentInputOverflowError(RuntimeError):
    """The current agent turn exceeds the query model's input envelope."""


class AgenticAnswerRunner:
    """Gather request-local evidence and produce one citation-checked answer."""

    def __init__(
        self,
        *,
        model_func: ToolModel,
        retrieve_knowledge_base: KnowledgeRetrieval,
        search_web: WebSearch,
        context_top_k: int | None = None,
        input_token_envelope: int = ANSWER_INPUT_TOKEN_ENVELOPE,
        history_token_ceiling: int = 81_920,
        composer_image_budget: AnswerImageBudget | None = None,
        rag_image_budget: AnswerImageBudget | None = None,
    ) -> None:
        self._executor = ToolTurnExecutor(model_func)
        self._retrieve_knowledge_base = retrieve_knowledge_base
        self._search_web = search_web
        self._context_top_k = context_top_k if context_top_k and context_top_k > 0 else None
        self._input_token_envelope = max(1, input_token_envelope)
        self._history_token_ceiling = max(0, history_token_ceiling)
        self._composer_image_budget = composer_image_budget
        self._rag_image_budget = rag_image_budget

    async def run(
        self,
        query: str,
        *,
        retrieval_query: str | None = None,
        initial_contexts: dict[str, list[dict[str, Any]]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        query_images: list[dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        session = EvidenceSession(
            composer_image_budget=self._composer_image_budget,
            rag_image_budget=self._rag_image_budget,
        )
        if initial_contexts:
            session.add_contexts(initial_contexts)
        effective_retrieval_query = retrieval_query or query
        cache = _ToolCallCache()
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
        }

        async def retrieve_initial(raw: BaseModel) -> ToolResult:
            args = _as(raw, RetrieveEvidenceInput)
            return await self._retrieve_initial(
                effective_retrieval_query,
                args.scope,
                session,
                trace,
                cache,
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

        base_messages = _initial_messages(
            query,
            conversation_history=_bounded_history(
                conversation_history,
                token_ceiling=self._history_token_ceiling,
            ),
            query_images=query_images,
        )
        evidence_message: dict[str, Any] | None = None
        last_exchange: list[dict[str, Any]] = []
        first_turn = True
        force_answer = False
        stop_reason = "model_stop"

        while True:
            tools = [first_tool] if first_turn else ([] if force_answer else followup_tools)
            tool_choice = "required" if first_turn else ("none" if force_answer else "auto")
            call_messages = [*base_messages, *last_exchange]
            if evidence_message is not None:
                call_messages.append(evidence_message)
            input_tokens = estimate_messages_tokens(call_messages)
            if input_tokens > self._input_token_envelope:
                raise AgentInputOverflowError(
                    "Agent input exceeds the answer envelope: "
                    f"{input_tokens} > {self._input_token_envelope} estimated tokens"
                )
            previous_evidence_count = _evidence_count(session)
            executed = await self._executor.run_turn(
                call_messages,
                tools,
                tool_choice=tool_choice,
            )
            trace["agent_turns"] += 1
            last_exchange = executed.messages[len(call_messages) :]

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
        cache: _ToolCallCache,
    ) -> ToolResult:
        tasks: list[Awaitable[ToolResult]] = [
            cache.run(
                _call_key("search_knowledge_base", query),
                lambda: self._search_corpus(query, session),
            )
        ]
        if scope == "all":
            tasks.append(
                cache.run(
                    _call_key("search_web", query),
                    lambda: self._search_open_web(query, session, trace),
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        messages: list[str] = []
        successes = 0
        corpus = results[0]
        if isinstance(corpus, ToolResult):
            messages.append(corpus.content)
            successes += 1
        else:
            messages.append(f"Knowledge-base retrieval failed: {corpus}")
        if scope == "all":
            web = results[1]
            if isinstance(web, ToolResult):
                messages.append(web.content)
                successes += 1
            else:
                messages.append(f"Open-web retrieval failed: {web}")
        if successes == 0:
            raise RuntimeError("; ".join(messages))
        return ToolResult(content=" ".join(messages))

    async def _search_corpus(self, query: str, session: EvidenceSession) -> ToolResult:
        result = await self._retrieve_knowledge_base(query)
        delta = session.add_contexts(_limit_contexts(result.contexts, self._context_top_k))
        return ToolResult(content=f"Knowledge base added {delta.new_chunks} new passages.")

    async def _search_open_web(
        self,
        query: str,
        session: EvidenceSession,
        trace: dict[str, Any],
    ) -> ToolResult:
        result = await self._search_web(query)
        rows = web_context_rows(result.hits)
        delta = session.add_rows(
            rows if self._context_top_k is None else rows[: self._context_top_k]
        )
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
            repeated = task is not None
            if task is None:
                task = asyncio.ensure_future(operation())
                self._tasks[key] = task
        result = await task
        if repeated:
            return ToolResult(
                content="Equivalent tool call already executed; no new evidence was added.",
                details=result.details,
            )
        return result


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


def _bounded_history(
    history: list[dict[str, Any]] | None,
    *,
    token_ceiling: int,
) -> list[dict[str, Any]]:
    if not history or token_ceiling <= 0:
        return []
    return truncate_conversation_history(
        history,
        max_messages=len(history),
        max_tokens=token_ceiling,
    )


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


def _limit_contexts(
    contexts: dict[str, list[dict[str, Any]]],
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [dict(row) for row in (rows if key != "chunks" or limit is None else rows[:limit])]
        for key, rows in contexts.items()
    }


def _call_key(name: str, query: str) -> str:
    return f"{name}:{json.dumps(query.strip(), ensure_ascii=False)}"


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = ["AgentInputOverflowError", "AgentProtocolError", "AgenticAnswerRunner"]
