# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. A request with no registered resources and no
open-web capability takes the standard-RAG fast path: fixed knowledge-base
retrieval and one final synthesis, with no control turn. A request with
attachments/resources or a web-search capability enters the research loop:
fixed initial retrieval, an optional strict web-scope decision when Exa exists,
peer tools, evidence-growth convergence, and one tools-disabled final answer.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.citations.finalization import finalize_answer
from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.agent.evidence import EvidenceLedger
from dlightrag.core.agent.tool_loop import (
    AgentTool,
    ExecutedTurn,
    ToolResult,
    ToolTurnExecutor,
)
from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE, AnswerCapacity
from dlightrag.core.answer.errors import AnswerInputOverflowError
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.retrieval.web_search import WebSearchResult, web_context_rows
from dlightrag.models.structured import StructuredOutput
from dlightrag.models.tool_turn import AssistantTurn
from dlightrag.prompts.agent import agentic_answer_prompt
from dlightrag.utils.tokens import estimate_messages_tokens

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[str], Awaitable[WebSearchResult]]
ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]
ScopeModel = Callable[..., Awaitable[BaseModel]]

_SEARCH_TOOL_NAMES = frozenset(
    {"search_knowledge_base", "search_web", "read_resource", "inspect_resource"}
)


class InitialScopeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_web: bool = Field(
        description=("Use true unless the user explicitly requires knowledge-base-only evidence.")
    )


INITIAL_SCOPE_OUTPUT = StructuredOutput(
    name="initial_evidence_scope",
    schema=InitialScopeDecision,
)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class FinishResearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentProtocolError(RuntimeError):
    """The model violated a required agent turn contract."""


class ResearchAnswerStream(AnswerStream):
    """Citation-validating final stream with request-level research trace."""

    def __init__(
        self,
        raw_iterator: AsyncIterator[str],
        *,
        indexer: Any,
        trace: dict[str, Any],
    ) -> None:
        super().__init__(raw_iterator, indexer=indexer)
        self.trace = trace
        self.image_descriptions: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class PreparedResearchStream:
    contexts: RetrievalContexts
    stream: ResearchAnswerStream


@dataclass(slots=True)
class _RunState:
    session: EvidenceLedger
    cache: _ToolCallCache
    trace: dict[str, Any]
    search_tools: list[AgentTool]
    finish_tool: AgentTool
    base_messages: list[dict[str, Any]]
    initial_query: str
    evidence_message: dict[str, Any] | None = None
    last_exchange: list[dict[str, Any]] = field(default_factory=list)
    force_answer: bool = False
    stop_reason: str = "model_stop"

    @property
    def research_tools(self) -> list[AgentTool]:
        return [*self.search_tools, self.finish_tool]


class AnswerOrchestrator:
    """Route every answer through one fast or research path and one final answer."""

    def __init__(
        self,
        *,
        synthesizer: AnswerSynthesizer,
        retrieve_knowledge_base: KnowledgeRetrieval,
        search_web: WebSearch | None = None,
        model_func: ToolModel | None = None,
        scope_model_func: ScopeModel | None = None,
        stream_model_func: StreamModel | None = None,
        resource_tools: list[AgentTool] | None = None,
        has_resources: bool = False,
        context_window_tokens: int = 260_000,
    ) -> None:
        self._synthesizer = synthesizer
        self._retrieve_knowledge_base = retrieve_knowledge_base
        self._search_web = search_web
        self._model_func = model_func
        self._scope_model_func = scope_model_func
        self._stream_model_func = stream_model_func
        self._resource_tools = list(resource_tools or [])
        self._has_resources = has_resources
        self._capacity = AnswerCapacity(max(1, context_window_tokens))
        self._input_budget = max(
            1, self._capacity.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
        )

    @property
    def uses_research_path(self) -> bool:
        """A request researches when it has resources or a web-search capability."""
        return self._has_resources or self._search_web is not None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def answer(
        self,
        query: str,
        *,
        retrieval_query: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        query_images: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> RetrievalResult:
        if not self.uses_research_path:
            return await self._fast_answer(
                query,
                retrieval_query=retrieval_query,
                conversation_history=conversation_history,
                query_images=query_images,
                history_images=history_images,
                initial_contexts=initial_contexts,
            )
        return await self._run_research(
            query,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )

    async def answer_stream(
        self,
        query: str,
        *,
        retrieval_query: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        query_images: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        if not self.uses_research_path:
            return await self._fast_answer_stream(
                query,
                retrieval_query=retrieval_query,
                conversation_history=conversation_history,
                query_images=query_images,
                history_images=history_images,
                initial_contexts=initial_contexts,
            )
        prepared = await self._prepare_research_stream(
            query,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )
        return prepared.contexts, prepared.stream

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------

    async def _fast_answer(
        self,
        query: str,
        *,
        retrieval_query: str | None,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        history_images: list[dict[str, Any]] | None = None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> RetrievalResult:
        retrieval = await self._retrieve_knowledge_base(retrieval_query or query)
        contexts = _merge_initial_contexts(initial_contexts, retrieval.contexts)
        generate_kwargs: dict[str, Any] = {
            "query_images": query_images,
            "conversation_history": conversation_history,
        }
        if history_images:
            generate_kwargs["history_images"] = history_images
        result = await self._synthesizer.generate(query, contexts, **generate_kwargs)
        result.trace.update(retrieval.trace)
        return result

    async def _fast_answer_stream(
        self,
        query: str,
        *,
        retrieval_query: str | None,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        history_images: list[dict[str, Any]] | None = None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        retrieval = await self._retrieve_knowledge_base(retrieval_query or query)
        contexts = _merge_initial_contexts(initial_contexts, retrieval.contexts)
        generate_kwargs: dict[str, Any] = {
            "query_images": query_images,
            "conversation_history": conversation_history,
        }
        if history_images:
            generate_kwargs["history_images"] = history_images
        contexts, stream = await self._synthesizer.generate_stream(
            query, contexts, **generate_kwargs
        )
        if stream is not None:
            existing = getattr(stream, "trace", None)
            merged = (
                {**retrieval.trace, **existing} if isinstance(existing, dict) else retrieval.trace
            )
            stream.trace = merged  # type: ignore[attr-defined]
        return contexts, stream

    # ------------------------------------------------------------------
    # Research path
    # ------------------------------------------------------------------

    async def _run_research(
        self,
        query: str,
        *,
        retrieval_query: str | None,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> RetrievalResult:
        if self._model_func is None:
            raise RuntimeError("Research answer requires a tool model")
        state = self._new_state(
            query,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )
        await self._run_initial_wave(state)

        while True:
            tools = [] if state.force_answer else state.research_tools
            tool_choice: Literal["auto", "none"] = "none" if state.force_answer else "auto"
            executed, changed = await self._execute_control_turn(
                state,
                tools,
                tool_choice=tool_choice,
            )

            if not executed.assistant.tool_calls:
                _, indexer = self._render_evidence(state, final=True)
                finalized = finalize_answer(
                    executed.assistant.text,
                    state.session.contexts,
                    indexer=indexer,
                )
                state.trace["agent_stop_reason"] = state.stop_reason
                return RetrievalResult(
                    answer=finalized.answer,
                    contexts=state.session.contexts,
                    sources=finalized.sources,
                    trace=state.trace,
                )

            if state.force_answer:
                raise AgentProtocolError("The model called a tool after tools were withdrawn.")

            names = {result.call.name for result in executed.results if not result.is_error}
            searched = bool(names & _SEARCH_TOOL_NAMES)
            if "finish_research" in names and not searched:
                state.force_answer = True
                state.stop_reason = "finish_research"
            elif not changed:
                state.force_answer = True
                state.stop_reason = "no_new_evidence"

    async def _prepare_research_stream(
        self,
        query: str,
        *,
        retrieval_query: str | None,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> PreparedResearchStream:
        if self._stream_model_func is None or self._model_func is None:
            raise RuntimeError("Streaming research answer requires a final text stream")
        state = self._new_state(
            query,
            retrieval_query=retrieval_query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )
        await self._run_initial_wave(state)

        while True:
            executed, changed = await self._execute_control_turn(
                state,
                state.research_tools,
                tool_choice="required",
                finish_control=True,
            )
            if not executed.assistant.tool_calls:
                raise AgentProtocolError("The model skipped a required research action.")

            names = {result.call.name for result in executed.results if not result.is_error}
            searched = bool(names & _SEARCH_TOOL_NAMES)
            if "finish_research" in names and not searched:
                state.stop_reason = "finish_research"
                break
            if not changed:
                state.stop_reason = "no_new_evidence"
                break

        blocks, indexer = self._render_evidence(state, final=True)
        final_messages = [
            *state.base_messages,
            *state.last_exchange,
            {"role": "user", "content": blocks},
        ]
        self._check_envelope(final_messages)
        state.trace["agent_stop_reason"] = state.stop_reason
        stream = ResearchAnswerStream(
            self._stream_model_func(messages=final_messages),
            indexer=indexer,
            trace=state.trace,
        )
        return PreparedResearchStream(contexts=state.session.contexts, stream=stream)

    # ------------------------------------------------------------------
    # Research helpers
    # ------------------------------------------------------------------

    def _new_state(
        self,
        query: str,
        *,
        retrieval_query: str | None,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> _RunState:
        session = EvidenceLedger()
        if initial_contexts:
            session.add_contexts(initial_contexts)
        cache = _ToolCallCache()
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
        }
        effective_retrieval_query = retrieval_query or query

        async def search_knowledge_base(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("knowledge_base", args.query),
                lambda: self._search_corpus(args.query, session),
            )

        search_tools = [
            AgentTool(
                "search_knowledge_base",
                "Search the indexed knowledge base for one concrete unresolved fact.",
                SearchInput,
                search_knowledge_base,
            ),
        ]
        if self._search_web is not None:
            search_tools.append(
                AgentTool(
                    "search_web",
                    "Search the open web for one concrete unresolved or current fact.",
                    SearchInput,
                    self._make_web_tool(session, cache, trace),
                )
            )
        for tool in self._resource_tools:
            search_tools.append(self._wrap_resource_tool(tool, session, cache))

        finish_tool = AgentTool(
            "finish_research",
            "Finish research when the current evidence is sufficient to answer.",
            FinishResearchInput,
            _finish_research,
        )
        return _RunState(
            session=session,
            cache=cache,
            trace=trace,
            search_tools=search_tools,
            finish_tool=finish_tool,
            base_messages=_initial_messages(
                query,
                conversation_history=conversation_history,
                query_images=query_images,
            ),
            initial_query=effective_retrieval_query,
        )

    def _make_web_tool(
        self,
        session: EvidenceLedger,
        cache: _ToolCallCache,
        trace: dict[str, Any],
    ) -> Callable[[BaseModel], Awaitable[ToolResult]]:
        async def search_web(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("web", args.query),
                lambda: self._search_open_web(args.query, session, trace),
            )

        return search_web

    def _wrap_resource_tool(
        self,
        tool: AgentTool,
        session: EvidenceLedger,
        cache: _ToolCallCache,
    ) -> AgentTool:
        """Cache equivalent resource calls and land each observation in the ledger."""

        async def execute(raw: BaseModel) -> ToolResult:
            key = _resource_call_key(tool.name, raw)

            async def run_once() -> ToolResult:
                result = await tool.execute(raw)
                row = _resource_row(tool.name, result)
                if row is not None:
                    session.add_rows([row])
                return result

            return await cache.run(key, run_once)

        return AgentTool(tool.name, tool.description, tool.input_model, execute)

    async def _execute_control_turn(
        self,
        state: _RunState,
        tools: list[AgentTool],
        *,
        tool_choice: Literal["auto", "required", "none"],
        finish_control: bool = False,
    ) -> tuple[ExecutedTurn, bool]:
        executor = ToolTurnExecutor(cast(ToolModel, self._model_func))
        call_messages = [*state.base_messages, *state.last_exchange]
        if state.evidence_message is not None:
            call_messages.append(state.evidence_message)
        self._check_envelope(call_messages)
        previous_evidence_count = _evidence_count(state.session)
        executed = await executor.run_turn(
            call_messages,
            tools,
            tool_choice=tool_choice,
        )
        state.trace["agent_turns"] += 1
        state.last_exchange = executed.messages[len(call_messages) :]
        if executed.assistant.tool_calls:
            blocks, _ = self._render_evidence(state, finish_control=finish_control)
            state.evidence_message = {"role": "user", "content": blocks}
        return executed, _evidence_count(state.session) != previous_evidence_count

    def _render_evidence(
        self,
        state: _RunState,
        *,
        final: bool = False,
        finish_control: bool = False,
    ) -> tuple[list[dict[str, Any]], Any]:
        fixed = estimate_messages_tokens([*state.base_messages, *state.last_exchange])
        blocks, indexer = state.session.transform(self._capacity, fixed_input_tokens=fixed)
        if final:
            instruction = "Answer the original request now from the current evidence above."
        elif finish_control:
            instruction = (
                "Use the current evidence above. Call finish_research if it is sufficient; "
                "otherwise call one or more tools for a concrete missing fact."
            )
        else:
            instruction = (
                "Use the current evidence above. Answer now if it is sufficient; otherwise "
                "call one or more tools for a concrete missing fact."
            )
        return [*blocks, {"type": "text", "text": instruction}], indexer

    def _check_envelope(self, messages: list[dict[str, Any]]) -> None:
        input_tokens = estimate_messages_tokens(messages)
        if input_tokens > self._input_budget:
            raise AnswerInputOverflowError(
                "Research input does not fit beside the generation reserve: "
                f"{input_tokens} > {self._input_budget} estimated input tokens"
            )

    async def _run_initial_wave(self, state: _RunState) -> None:
        self._check_envelope(state.base_messages)
        include_web = False
        if self._search_web is not None:
            decision = await self._require_scope(state)
            state.trace["agent_turns"] += 1
            include_web = decision.include_web
            if not include_web:
                state.search_tools = [
                    tool for tool in state.search_tools if tool.name != "search_web"
                ]
        sources: tuple[Literal["knowledge_base", "web"], ...] = (
            ("knowledge_base", "web") if include_web else ("knowledge_base",)
        )
        await self._search_sources(state.initial_query, sources, state)
        blocks, _ = self._render_evidence(state)
        state.evidence_message = {"role": "user", "content": blocks}

    async def _require_scope(self, state: _RunState) -> InitialScopeDecision:
        if self._scope_model_func is None:
            raise RuntimeError("Web scope decision requires a scope model")
        decision = await self._scope_model_func(
            messages=state.base_messages,
            structured_output=INITIAL_SCOPE_OUTPUT,
        )
        if not isinstance(decision, InitialScopeDecision):
            raise AgentProtocolError(
                f"Expected InitialScopeDecision, got {type(decision).__name__}"
            )
        return decision

    async def _search_sources(
        self,
        query: str,
        sources: tuple[Literal["knowledge_base", "web"], ...],
        state: _RunState,
    ) -> None:
        operations: list[tuple[str, Awaitable[ToolResult]]] = []
        if "knowledge_base" in sources:
            operations.append(
                (
                    "Knowledge-base",
                    state.cache.run(
                        _call_key("knowledge_base", query),
                        lambda: self._search_corpus(query, state.session),
                    ),
                )
            )
        if "web" in sources and self._search_web is not None:
            operations.append(
                (
                    "Open-web",
                    state.cache.run(
                        _call_key("web", query),
                        lambda: self._search_open_web(query, state.session, state.trace),
                    ),
                )
            )
        results = await asyncio.gather(
            *(operation for _, operation in operations),
            return_exceptions=True,
        )
        messages: list[str] = []
        successes = 0
        for (label, _), result in zip(operations, results, strict=True):
            if isinstance(result, ToolResult):
                messages.append(result.content)
                successes += 1
            else:
                messages.append(f"{label} retrieval failed: {result}")
        if successes == 0:
            raise RuntimeError("; ".join(messages))

    async def _search_corpus(self, query: str, session: EvidenceLedger) -> ToolResult:
        result = await self._retrieve_knowledge_base(query)
        delta = session.add_contexts(result.contexts)
        return ToolResult(content=f"Knowledge base added {delta.new_chunks} new passages.")

    async def _search_open_web(
        self,
        query: str,
        session: EvidenceLedger,
        trace: dict[str, Any],
    ) -> ToolResult:
        search_web = cast(WebSearch, self._search_web)
        result = await search_web(query)
        rows = web_context_rows(result.hits)
        delta = session.add_rows(rows)
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
        try:
            result = await task
        except BaseException:
            async with self._lock:
                if self._tasks.get(key) is task:
                    self._tasks.pop(key, None)
            raise
        if repeated:
            return ToolResult(
                content="Equivalent tool call already executed; no new evidence was added.",
                details=result.details,
            )
        return result


async def _finish_research(_raw: BaseModel) -> ToolResult:
    return ToolResult(content="Research is complete; proceed to the final answer.")


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


def _resource_row(tool_name: str, result: ToolResult) -> dict[str, Any] | None:
    """Project a resource observation into a citable, re-readable evidence row."""
    details = result.details or {}
    resource_id = str(details.get("resource_id") or "")
    if not resource_id or not result.content.strip():
        return None
    identity = hashlib.sha256(result.content.encode("utf-8")).hexdigest()[:16]
    return {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": resource_id,
        "content": result.content,
        "page_number": None,
        "_workspace": "__attachment__",
        "metadata": {
            "source_type": "web_attachment",
            "source_uri": resource_id,
            "source_download_locator": resource_id,
            "derived_by_vlm": bool(details.get("derived_by_vlm")),
        },
    }


def _evidence_count(session: EvidenceLedger) -> int:
    return sum(len(rows) for rows in session.contexts.values())


def _merge_initial_contexts(
    initial: RetrievalContexts | None,
    retrieved: RetrievalContexts,
) -> RetrievalContexts:
    """Place server-prepared evidence ahead of retrieved rows without reranking."""
    if not initial:
        return retrieved
    merged: RetrievalContexts = {
        "chunks": [*initial.get("chunks", []), *retrieved.get("chunks", [])],
        "entities": [*retrieved.get("entities", []), *initial.get("entities", [])],
        "relationships": [
            *retrieved.get("relationships", []),
            *initial.get("relationships", []),
        ],
    }
    for key, rows in retrieved.items():
        if key not in merged:
            merged[key] = list(rows)
    return merged


def _call_key(name: str, query: str) -> str:
    return f"{name}:{json.dumps(query.strip(), ensure_ascii=False)}"


def _resource_call_key(name: str, raw: BaseModel) -> str:
    payload = json.dumps(raw.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{payload}"


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = [
    "INITIAL_SCOPE_OUTPUT",
    "AgentProtocolError",
    "AnswerOrchestrator",
    "InitialScopeDecision",
    "PreparedResearchStream",
    "ResearchAnswerStream",
    "SearchInput",
]
