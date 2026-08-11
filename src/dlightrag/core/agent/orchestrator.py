# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. A request with no registered resources and no
open-web capability takes the standard-RAG fast path: fixed knowledge-base
retrieval and one final answer generation, with no control turn. A request with
attachments/resources or a web-search capability enters the research loop:
the model selects from the available peer tools, evidence-growth convergence,
and one additional tools-disabled final answer generation.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from dlightrag.core.agent.context import ContextAssembler
from dlightrag.core.agent.tool_loop import AgentTool, ExecutedTurn, ToolTurnExecutor
from dlightrag.core.agent.tools import (
    KnowledgeRetrieval,
    WebSearch,
    build_run_tools,
)
from dlightrag.core.answer.capacity import AnswerCapacity
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.models.tool_turn import AssistantTurn

logger = logging.getLogger(__name__)

ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]
FinalText = Callable[..., Awaitable[str]]


@dataclass(slots=True)
class _RunState:
    # Memory: what this run has found, and what it has done.
    evidence: EvidenceLedger
    episode: RunEpisode
    context: ContextAssembler
    tools: list[AgentTool]
    trace: dict[str, Any]
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
        image_budget: AnswerImageBudget | None = None,
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
        self._image_budget = image_budget
        self._capacity = AnswerCapacity(max(1, context_window_tokens))
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
        conversation_history: PriorTurns | None = None,
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
        conversation_history: PriorTurns | None = None,
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
        conversation_history: PriorTurns | None,
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
        conversation_history: PriorTurns | None,
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
        conversation_history: PriorTurns | None,
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
        await self._research_until_stopped(state)

        if self._final_text_func is None:
            raise RuntimeError("Research answer requires a tools-disabled final model")
        final_messages, indexer = state.context.answer_turn(
            evidence=state.evidence, episode=state.episode
        )
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
        conversation_history: PriorTurns | None,
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
        await self._research_until_stopped(state)

        final_messages, indexer = state.context.answer_turn(
            evidence=state.evidence, episode=state.episode
        )
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

    # ------------------------------------------------------------------
    # Research helpers
    # ------------------------------------------------------------------

    def _new_state(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None,
        query_images: list[dict[str, Any]] | None,
        initial_contexts: RetrievalContexts | None = None,
    ) -> _RunState:
        evidence = EvidenceLedger(image_budget=self._image_budget)
        if initial_contexts:
            evidence.add_contexts(initial_contexts)
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
        }
        return _RunState(
            evidence=evidence,
            episode=RunEpisode(),
            context=ContextAssembler(
                self._capacity,
                query=query,
                history=conversation_history or PriorTurns(),
                query_images=query_images,
                resource_manifest=self._resource_manifest,
            ),
            tools=build_run_tools(
                evidence=evidence,
                trace=trace,
                retrieve_knowledge_base=self._retrieve_knowledge_base,
                search_web=self._search_web,
                resource_tools=self._resource_tools,
                register_web_source=self._register_web_source,
            ),
            trace=trace,
        )

    async def _execute_control_turn(
        self,
        state: _RunState,
    ) -> tuple[ExecutedTurn, bool]:
        executor = ToolTurnExecutor(cast(ToolModel, self._model_func))
        call_messages = state.context.control_turn(evidence=state.evidence, episode=state.episode)
        previous_rows = state.evidence.row_count
        executed = await executor.run_turn(
            call_messages,
            state.tools,
            tool_choice="auto",
        )
        state.trace["agent_turns"] += 1
        state.episode.record(executed.messages[len(call_messages) :])
        return executed, state.evidence.row_count != previous_rows


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


__all__ = ["AnswerOrchestrator"]
