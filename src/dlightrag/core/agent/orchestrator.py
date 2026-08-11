# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. A request with no registered resources and no
open-web capability takes the standard-RAG fast path: fixed knowledge-base
retrieval and one final answer generation, with no control turn. A request with
attachments/resources or a web-search capability enters the research loop:
the model selects from the available peer tools, evidence-growth convergence,
and one additional tools-disabled final answer generation.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.core.agent.episode import RunEpisode
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
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.retrieval.web_search import (
    WebSearchResult,
    WebSearchUnavailable,
    web_context_rows,
)
from dlightrag.models.tool_turn import AssistantTurn
from dlightrag.prompts import (
    CONTROL_TURN_INSTRUCTION,
    FINAL_TURN_INSTRUCTION,
    agent_control_prompt,
    answer_core,
)
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.utils.tokens import estimate_messages_tokens

logger = logging.getLogger(__name__)

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[str], Awaitable[WebSearchResult]]
ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]
FinalText = Callable[..., Awaitable[str]]


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)


@dataclass(slots=True)
class _RunState:
    # Memory: what this run has found, and what it has done.
    evidence: EvidenceLedger
    episode: RunEpisode
    base_messages: list[dict[str, Any]]
    tools: list[AgentTool]
    cache: _ToolCallCache
    trace: dict[str, Any]
    control_message: dict[str, Any] | None = None
    stop_reason: str = "model_stop"


class AnswerOrchestrator:
    """Route every answer through one fast or research path and one final answer."""

    def __init__(
        self,
        *,
        synthesizer: AnswerSynthesizer,
        retrieve_knowledge_base: KnowledgeRetrieval,
        search_web: WebSearch | None = None,
        model_func: ToolModel | None = None,
        stream_model_func: StreamModel | None = None,
        final_text_func: FinalText | None = None,
        resource_tools: list[AgentTool] | None = None,
        resource_manifest: tuple[ResourceManifestEntry, ...] = (),
        register_web_source: Callable[[str], str | None] | None = None,
        context_window_tokens: int = 260_000,
        max_agent_turns: int = 50,
    ) -> None:
        self._synthesizer = synthesizer
        self._retrieve_knowledge_base = retrieve_knowledge_base
        self._search_web = search_web
        self._model_func = model_func
        self._stream_model_func = stream_model_func
        self._final_text_func = final_text_func
        self._resource_tools = list(resource_tools or [])
        self._resource_manifest = tuple(resource_manifest)
        self._register_web_source = register_web_source
        self._capacity = AnswerCapacity(max(1, context_window_tokens))
        self._input_budget = max(
            1, self._capacity.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
        )
        self._max_agent_turns = max(1, max_agent_turns)

    @property
    def uses_research_path(self) -> bool:
        """A request researches when it has resources or a web-search capability."""
        return bool(self._resource_manifest) or self._search_web is not None

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
        return await self._run_research_stream(
            query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )

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
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> RetrievalResult:
        if self._model_func is None:
            raise RuntimeError("Research answer requires a tool model")
        state = self._new_state(
            query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )
        self._prepare_research(state)
        await self._research_until_stopped(state)

        if self._final_text_func is None:
            raise RuntimeError("Research answer requires a tools-disabled final model")
        final_messages, indexer = self._finalize_transcript(state)
        state.trace["agent_stop_reason"] = state.stop_reason
        return await self._synthesizer.synthesize_research(
            final_messages,
            state.evidence.contexts,
            complete=self._final_text_func,
            indexer=indexer,
            trace=state.trace,
        )

    async def _run_research_stream(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        if self._stream_model_func is None or self._model_func is None:
            raise RuntimeError("Streaming research answer requires a final text stream")
        state = self._new_state(
            query,
            conversation_history=conversation_history,
            query_images=query_images,
            initial_contexts=initial_contexts,
        )
        self._prepare_research(state)
        await self._research_until_stopped(state)

        final_messages, indexer = self._finalize_transcript(state)
        state.trace["agent_stop_reason"] = state.stop_reason
        return await self._synthesizer.synthesize_research_stream(
            final_messages,
            state.evidence.contexts,
            stream=self._stream_model_func,
            indexer=indexer,
            trace=state.trace,
        )

    async def _research_until_stopped(self, state: _RunState) -> None:
        """Run evidence turns until the model stops, adds nothing, or hits the cap."""
        for _ in range(self._max_agent_turns):
            executed, changed = await self._execute_control_turn(state)
            if not executed.assistant.tool_calls:
                state.stop_reason = "model_stop"
                return
            if not changed:
                state.stop_reason = "no_new_evidence"
                return
        state.stop_reason = "turn_limit"
        logger.warning(
            "Research stopped at the %d-turn cap; answering from the evidence gathered so far",
            self._max_agent_turns,
        )

    def _finalize_transcript(self, state: _RunState) -> tuple[list[dict[str, Any]], Any]:
        """Pack the ledger's final citable evidence beside the tool transcript.

        Returns the tools-disabled final message list and the matching citation
        indexer so the synthesizer finalizes over the same stable identities the
        transcript's evidence blocks were numbered from.
        """
        blocks, indexer = self._render_turn_blocks(state, final=True)
        final_messages = [
            {"role": "system", "content": answer_core()},
            *state.base_messages[1:],
            *state.episode.last_exchange,
            {"role": "user", "content": blocks},
        ]
        self._check_envelope(final_messages)
        return final_messages, indexer

    # ------------------------------------------------------------------
    # Research helpers
    # ------------------------------------------------------------------

    def _new_state(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> _RunState:
        evidence = EvidenceLedger()
        if initial_contexts:
            evidence.add_contexts(initial_contexts)
        cache = _ToolCallCache()
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
        }

        async def search_knowledge_base(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("knowledge_base", args.query),
                lambda: self._search_corpus(args.query, evidence),
            )

        tools = [
            AgentTool(
                "search_knowledge_base",
                "Search the indexed knowledge base for one concrete unresolved fact.",
                SearchInput,
                search_knowledge_base,
            ),
        ]
        if self._search_web is not None:
            tools.append(
                AgentTool(
                    "search_web",
                    "Search the open web for one concrete unresolved or current fact.",
                    SearchInput,
                    self._make_web_tool(evidence, cache, trace),
                )
            )
        for tool in self._resource_tools:
            tools.append(self._wrap_resource_tool(tool, evidence, cache))

        return _RunState(
            evidence=evidence,
            episode=RunEpisode(),
            base_messages=_initial_messages(
                query,
                conversation_history=conversation_history,
                query_images=query_images,
                resource_manifest=self._resource_manifest,
            ),
            tools=tools,
            cache=cache,
            trace=trace,
        )

    def _make_web_tool(
        self,
        evidence: EvidenceLedger,
        cache: _ToolCallCache,
        trace: dict[str, Any],
    ) -> Callable[[BaseModel], Awaitable[ToolResult]]:
        async def search_web(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("web", args.query),
                lambda: self._search_open_web(args.query, evidence, trace),
            )

        return search_web

    def _wrap_resource_tool(
        self,
        tool: AgentTool,
        evidence: EvidenceLedger,
        cache: _ToolCallCache,
    ) -> AgentTool:
        """Cache equivalent resource calls and land each observation in the ledger."""

        async def execute(raw: BaseModel) -> ToolResult:
            key = _resource_call_key(tool.name, raw)

            async def run_once() -> ToolResult:
                result = await tool.execute(raw)
                row = _resource_row(tool.name, result)
                if row is not None:
                    evidence.add_rows([row])
                return result

            return await cache.run(key, run_once)

        return AgentTool(tool.name, tool.description, tool.input_model, execute)

    async def _execute_control_turn(
        self,
        state: _RunState,
    ) -> tuple[ExecutedTurn, bool]:
        executor = ToolTurnExecutor(cast(ToolModel, self._model_func))
        call_messages = [*state.base_messages, *state.episode.messages()]
        if state.control_message is not None:
            call_messages.append(state.control_message)
        self._check_envelope(call_messages)
        previous_evidence_count = _evidence_count(state.evidence)
        executed = await executor.run_turn(
            call_messages,
            state.tools,
            tool_choice="auto",
        )
        state.trace["agent_turns"] += 1
        state.episode.record(executed.messages[len(call_messages) :])
        if executed.assistant.tool_calls:
            blocks, _ = self._render_turn_blocks(state)
            state.control_message = {"role": "user", "content": blocks}
        return executed, _evidence_count(state.evidence) != previous_evidence_count

    def _render_turn_blocks(
        self,
        state: _RunState,
        *,
        final: bool = False,
    ) -> tuple[list[dict[str, Any]], Any]:
        # The answer turn carries one exchange instead of the episode, so it packs more evidence.
        carried = state.episode.last_exchange if final else state.episode.messages()
        fixed = estimate_messages_tokens([*state.base_messages, *carried])
        blocks, indexer = state.evidence.transform(self._capacity, fixed_input_tokens=fixed)
        instruction = FINAL_TURN_INSTRUCTION if final else CONTROL_TURN_INSTRUCTION
        return [*blocks, {"type": "text", "text": instruction}], indexer

    def _check_envelope(self, messages: list[dict[str, Any]]) -> None:
        input_tokens = estimate_messages_tokens(messages)
        if input_tokens > self._input_budget:
            raise AnswerInputOverflowError(
                "Research input does not fit beside the generation reserve: "
                f"{input_tokens} > {self._input_budget} estimated input tokens"
            )

    def _prepare_research(self, state: _RunState) -> None:
        self._check_envelope(state.base_messages)
        if _evidence_count(state.evidence):
            blocks, _ = self._render_turn_blocks(state)
            state.control_message = {"role": "user", "content": blocks}

    async def _search_corpus(self, query: str, evidence: EvidenceLedger) -> ToolResult:
        try:
            result = await self._retrieve_knowledge_base(query)
        except Exception as exc:
            raise RuntimeError("knowledge-base search failed") from exc
        delta = evidence.add_contexts(result.contexts)
        return ToolResult(content=f"Knowledge base added {delta.new_chunks} new passages.")

    async def _search_open_web(
        self,
        query: str,
        evidence: EvidenceLedger,
        trace: dict[str, Any],
    ) -> ToolResult:
        search_web = cast(WebSearch, self._search_web)
        try:
            result = await search_web(query)
        except WebSearchUnavailable:
            raise
        except Exception as exc:
            raise RuntimeError("open-web search failed") from exc
        rows = web_context_rows(result.hits)
        readable_sources: dict[str, str] = {}
        if self._register_web_source is not None:
            resources_by_url: dict[str, str | None] = {}
            for row in rows:
                metadata = row.get("metadata") or {}
                url = str(metadata.get("source_uri") or "")
                if url not in resources_by_url:
                    resources_by_url[url] = self._register_web_source(url)
                resource_id = resources_by_url[url]
                if resource_id is not None:
                    metadata["resource_id"] = resource_id
                    readable_sources.setdefault(resource_id, str(metadata.get("title") or "Source"))
        delta = evidence.add_rows(rows)
        trace["web_search_cost_dollars"] += result.cost_dollars
        content = f"Open web added {delta.new_chunks} new passages."
        if delta.new_chunks and readable_sources:
            content += "\nResource handles:\n" + "\n".join(
                f"- {title} [resource: {resource_id}]"
                for resource_id, title in readable_sources.items()
            )
        return ToolResult(content=content)


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


def _initial_messages(
    query: str,
    *,
    conversation_history: list[dict[str, Any]] | None,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": agent_control_prompt()},
        *(conversation_history or []),
    ]
    resource_context = _resource_manifest_context(resource_manifest)
    if query_images or resource_context:
        content: list[dict[str, Any]] = [{"type": "text", "text": query}]
        if resource_context:
            content.append({"type": "text", "text": resource_context})
        content.extend(query_images or [])
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
    else:
        messages.append({"role": "user", "content": query})
    return messages


def _resource_manifest_context(manifest: tuple[ResourceManifestEntry, ...]) -> str:
    if not manifest:
        return ""
    lines = ["## Registered request-local resources"]
    for entry in manifest:
        filename = safe_source_filename(entry.filename or "resource")
        kind = "image" if (entry.declared_mime or "").lower().startswith("image/") else "resource"
        lines.append(f"- [resource: {entry.resource_id}] {filename} ({kind})")
    lines.append("Use only these opaque resource ids with read_resource or inspect_resource.")
    return "\n".join(lines)


def _resource_row(tool_name: str, result: ToolResult) -> dict[str, Any] | None:
    """Project a resource observation into a citable, re-readable evidence row."""
    details = result.details or {}
    resource_id = str(details.get("resource_id") or "")
    if not resource_id or not result.content.strip():
        return None
    source_type = str(details.get("source_type") or "web_attachment")
    source_uri = str(details.get("source_uri") or resource_id)
    metadata = {
        "source_type": source_type,
        "source_uri": source_uri,
        "source_download_locator": str(details.get("source_download_locator") or source_uri),
        "title": str(details.get("title") or resource_id),
    }
    evidence_key = result.content
    if tool_name == "read_resource":
        content, marker, cursor = evidence_key.rpartition("\n[more text available; cursor=")
        if marker and cursor.endswith("]"):
            evidence_key = content
    identity = hashlib.sha256(f"{tool_name}\0{evidence_key}".encode()).hexdigest()[:16]
    return {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": str(metadata.get("title") or resource_id),
        "content": evidence_key if tool_name == "read_resource" else result.content,
        "page_number": None,
        "_workspace": "__web_search__" if source_type == "web_search" else "__attachment__",
        "_evidence_key": f"{tool_name}:{identity}",
        "metadata": metadata,
    }


def _evidence_count(evidence: EvidenceLedger) -> int:
    return sum(len(rows) for rows in evidence.contexts.values())


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
    "AnswerOrchestrator",
    "SearchInput",
]
