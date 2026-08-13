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
from typing import Any, Protocol, cast

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
from dlightrag.core.answer_runs.models import AgentRunState
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.core.retrieval.protocols import RetrievalContexts
from dlightrag.models.tool_turn import AssistantTurn

logger = logging.getLogger(__name__)

ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]


class RunBoundaries(Protocol):
    """Durable boundaries a run may observe between agent steps."""

    async def enter_phase(self, phase: str) -> None: ...

    async def turn_completed(self, state: AgentRunState) -> None: ...

    async def check_cancelled(self) -> None: ...


class _NoBoundaries:
    """An in-process answer observes no durable boundary."""

    async def enter_phase(self, phase: str) -> None:
        return None

    async def turn_completed(self, state: AgentRunState) -> None:
        return None

    async def check_cancelled(self) -> None:
        return None


@dataclass(slots=True)
class PreparedRun:
    """One run's restorable memory plus the wiring that executes it here."""

    state: AgentRunState
    context: ContextAssembler
    tools: list[AgentTool]


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

    async def answer_stream(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None = None,
        query_images: list[dict[str, Any]] | None = None,
        run: PreparedRun | None = None,
        boundaries: RunBoundaries | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        limits = boundaries or _NoBoundaries()
        if not self.uses_research_path:
            if query_images:
                raise RuntimeError("Current images require request resources")
            return await self._fast_answer_stream(
                query,
                conversation_history=conversation_history,
                boundaries=limits,
            )
        return await self._run_research_stream(
            run
            or self.prepare_run(
                query,
                conversation_history=conversation_history,
                query_images=query_images,
            ),
            boundaries=limits,
        )

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------

    async def _fast_answer_stream(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None,
        boundaries: RunBoundaries,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        await boundaries.enter_phase("searching")
        retrieval = await self._retrieve_knowledge_base(query)
        await boundaries.check_cancelled()
        await boundaries.enter_phase("generating")
        contexts, stream = await self._synthesizer.generate_stream(
            query,
            retrieval.contexts,
            conversation_history=conversation_history,
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

    async def _run_research_stream(
        self,
        run: PreparedRun,
        *,
        boundaries: RunBoundaries,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        if self._stream_model_func is None or self._model_func is None:
            raise RuntimeError("Streaming research answer requires a final text stream")
        state = run.state
        try:
            await self._research_until_stopped(run, boundaries=boundaries)

            await boundaries.enter_phase("generating")
            final_messages, indexer = await run.context.answer_turn(
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
        finally:
            await state.tool_cache.aclose()

    async def _research_until_stopped(self, run: PreparedRun, *, boundaries: RunBoundaries) -> None:
        """Run evidence turns until the model stops, adds nothing, or hits the cap.

        A tool error is not convergence: an invalid, unavailable, or failed result
        is replayed so the model can correct it, and only ``max_agent_turns``
        bounds that correction. The cap spans the whole run, not one process
        lifetime, so a resumed run continues its recorded turn count.
        """
        state = run.state
        while state.completed_turns < self._max_agent_turns:
            await boundaries.check_cancelled()
            await boundaries.enter_phase("researching")
            executed, changed = await self._execute_control_turn(run)
            state.completed_turns += 1
            state.trace["agent_turns"] = state.completed_turns
            await boundaries.turn_completed(state)
            if not executed.assistant.tool_calls:
                state.stop_reason = "model_stop"
                return
            if not changed and not any(result.is_error for result in executed.results):
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

    def prepare_run(
        self,
        query: str,
        *,
        conversation_history: PriorTurns | None = None,
        query_images: list[dict[str, Any]] | None = None,
        registry: ResourceRegistry | None = None,
    ) -> PreparedRun:
        """Build one run's memory and the tools bound to it, before any restore."""
        evidence = EvidenceLedger(image_budget=self._image_budget)
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
            "tool_observations": [],
        }
        tools, tool_cache = build_run_tools(
            evidence=evidence,
            trace=trace,
            retrieve_knowledge_base=self._retrieve_knowledge_base,
            search_web=self._search_web,
            resource_tools=self._resource_tools,
            register_web_source=self._register_web_source,
        )
        return PreparedRun(
            state=AgentRunState(
                evidence=evidence,
                episode=RunEpisode(),
                tool_cache=tool_cache,
                registry=registry,
                trace=trace,
            ),
            context=ContextAssembler(
                self._capacity,
                query=query,
                history=conversation_history or PriorTurns(),
                query_images=query_images,
                resource_manifest=self._resource_manifest,
            ),
            tools=tools,
        )

    async def _execute_control_turn(
        self,
        run: PreparedRun,
    ) -> tuple[ExecutedTurn, bool]:
        state = run.state
        executor = ToolTurnExecutor(cast(ToolModel, self._model_func))
        call_messages = await run.context.control_turn(
            evidence=state.evidence, episode=state.episode
        )
        previous_rows = state.evidence.row_count
        executed = await executor.run_turn(
            call_messages,
            run.tools,
            tool_choice="auto",
        )
        state.trace["tool_observations"].extend(
            execution.observation.as_dict() for execution in executed.results
        )
        state.episode.record(executed.messages[len(call_messages) :])
        return executed, state.evidence.row_count != previous_rows


__all__ = ["AnswerOrchestrator", "PreparedRun", "RunBoundaries"]
