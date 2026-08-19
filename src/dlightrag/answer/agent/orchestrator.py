# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. A request with no registered resources and no
open-web capability takes the standard-RAG fast path: fixed knowledge-base
retrieval and one final answer generation, with no control turn. A request with
attachments/resources or a web-search capability enters the research loop:
the model selects from the available peer tools, evidence-growth convergence,
and one additional tools-disabled final answer generation.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast
from uuid import uuid4

from dlightrag_agent.session.fold import PriorTurns, SessionEpisode, fold_entries
from dlightrag_agent.tools import (
    AgentTool,
    ExecutedTurn,
    ToolResultCapacityError,
    ToolTurnExecutor,
)
from dlightrag_ai.capacity import (
    CONTEXT_POLICY,
    ContextPolicy,
    ModelProfile,
)
from dlightrag_ai.messages import AssistantTurn
from dlightrag_ai.telemetry import Telemetry
from dlightrag_ai.tokens import estimate_tokens
from dlightrag_rag.retrieval import RetrievalContexts

from dlightrag.answer.agent.context import ContextAssembler
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.answer.resources.models import ResourceManifestEntry, TextWindowBudget
from dlightrag.answer.resources.registry import ResourceRegistry
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools import KnowledgeRetrieval, WebSearch, compose_research_tools

logger = logging.getLogger(__name__)

ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]


class RunBoundaries(Protocol):
    """Durable boundaries a run may observe between agent steps.

    ``commit_turn`` journals the complete assistant response and its ordered
    intents, settles every intent in source order, and advances durable
    progress; the live executor executes tool calls before it is invoked.
    """

    async def enter_phase(self, phase: str) -> None: ...

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None: ...

    async def check_cancelled(self) -> None: ...


class _NoBoundaries:
    """An in-process answer observes no durable boundary."""

    async def enter_phase(self, phase: str) -> None:
        return None

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None:
        return None

    async def check_cancelled(self) -> None:
        return None


@dataclass(slots=True)
class PreparedRun:
    """One run's live memory plus the wiring that executes it here.

    The episode and evidence are request-local materializers rebuilt from the
    durable journal on recovery; they carry no export/restore interface.
    """

    context: ContextAssembler
    tools: list[AgentTool]
    evidence: EvidenceLedger
    episode: SessionEpisode
    registry: ResourceRegistry | None
    trace: dict[str, Any]
    agent_turn_count: int = 0
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
        resource_tools: list[AgentTool] | None = None,
        resource_manifest: tuple[ResourceManifestEntry, ...] = (),
        register_web_source: Callable[[str], str | None] | None = None,
        image_budget: AnswerImageBudget | None = None,
        text_window_budget: TextWindowBudget,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        max_agent_turns: int = 50,
        telemetry: Telemetry,
        environment: object | None = None,
        resource_reader: object | None = None,
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
        self._text_window_budget = text_window_budget
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._max_agent_turns = max(1, max_agent_turns)
        self._telemetry = telemetry
        self._environment = environment
        self._resource_reader = resource_reader
        self._workspace: Any = None

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
        await self._research_until_stopped(run, boundaries=boundaries)

        await boundaries.enter_phase("generating")
        final_messages, indexer = await run.context.answer_turn(
            evidence=run.evidence, episode=run.episode
        )
        run.trace["agent_stop_reason"] = run.stop_reason
        return await self._synthesizer.synthesize_research_stream(
            final_messages,
            run.evidence.contexts,
            stream=self._stream_model_func,
            indexer=indexer,
            trace=run.trace,
        )

    async def _research_until_stopped(self, run: PreparedRun, *, boundaries: RunBoundaries) -> None:
        """Run evidence turns until the model stops, adds nothing, or hits the cap.

        A tool error is not convergence: an invalid, unavailable, or failed result
        is replayed so the model can correct it, and only ``max_agent_turns``
        bounds that correction. The cap spans the whole run, not one process
        lifetime, so a resumed run continues its recorded turn count.
        """
        while run.agent_turn_count < self._max_agent_turns:
            await boundaries.check_cancelled()
            await boundaries.enter_phase("researching")
            executed, changed = await self._execute_control_turn(run)
            run.agent_turn_count += 1
            run.trace["agent_turns"] = run.agent_turn_count
            await boundaries.commit_turn(executed, turn_number=run.agent_turn_count)
            if not executed.assistant.tool_calls:
                run.stop_reason = "model_stop"
                return
            if not changed and not any(result.is_error for result in executed.results):
                run.stop_reason = "no_new_evidence"
                return
        run.stop_reason = "turn_limit"
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
        agent_turn_count: int = 0,
    ) -> PreparedRun:
        """Build one run's memory and the tools bound to it, before any restore."""
        evidence = EvidenceLedger(image_budget=self._image_budget)
        retained_tail_tokens = self._context_policy.retained_tail_target(self._model_profile)
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
            "tool_observations": [],
        }
        tools = self._compose_tools(evidence, trace)
        return PreparedRun(
            context=ContextAssembler(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                query=query,
                history=conversation_history or PriorTurns(),
                query_images=query_images,
                resource_manifest=self._resource_manifest,
            ),
            tools=tools,
            evidence=evidence,
            episode=SessionEpisode(retained_tail_tokens=retained_tail_tokens),
            registry=registry,
            trace=trace,
            agent_turn_count=agent_turn_count,
        )

    def adopt_agent_turn_count(self, run: PreparedRun, turns: int) -> None:
        """Continue a resumed run's recorded turn count from the journal."""
        run.agent_turn_count = int(turns)
        run.trace["agent_turns"] = run.agent_turn_count

    async def recover_from_fold(self, run: PreparedRun, snapshot: Any) -> None:
        """Rebuild the live episode from the folded journal exchanges."""
        messages = fold_entries(snapshot.entries)
        exchanges: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls") and current:
                exchanges.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            exchanges.append(current)
        for exchange in exchanges:
            run.episode.record(exchange)
        self.adopt_agent_turn_count(
            run,
            sum(
                1
                for entry in snapshot.entries
                if entry.__class__.__name__ == "AssistantMessageEntry"
            ),
        )

    def _compose_tools(
        self,
        evidence: EvidenceLedger,
        trace: dict[str, Any],
    ) -> list[AgentTool]:
        return compose_research_tools(
            evidence=evidence,
            trace=trace,
            retrieve_knowledge_base=self._retrieve_knowledge_base,
            search_web=self._search_web,
            resource_tools=self._resource_tools,
            register_web_source=self._register_web_source,
            resource_reader=self._resource_reader,
            environment=self._environment,  # type: ignore[arg-type]
            spill=self._spill_writer() if self._workspace is not None else None,
        )

    def _spill_writer(self) -> Any:
        from dlightrag.answer.workspace import spill_receipt, write_spill_file

        workspace = self._workspace

        async def write(text: str) -> dict[str, object]:
            resource_id = f"spill_{uuid4().hex}"
            write_spill_file(workspace.spill_dir, resource_id, text)
            return spill_receipt(resource_id, text)

        return write

    async def _execute_control_turn(
        self,
        run: PreparedRun,
    ) -> tuple[ExecutedTurn, bool]:
        executor = ToolTurnExecutor(
            cast(ToolModel, self._model_func),
            telemetry=self._telemetry,
        )
        tool_schema_tokens = _tool_schema_tokens(run.tools)
        call_messages = await run.context.control_turn(
            evidence=run.evidence,
            episode=run.episode,
            tool_schema_tokens=tool_schema_tokens,
        )
        max_output_tokens = run.context.control_output_allowance(
            call_messages,
            tool_schema_tokens=tool_schema_tokens,
        )

        def observation_budget(transcript: list[dict[str, Any]]) -> int:
            residual = run.context.observation_residual(
                transcript,
                tool_schema_tokens=tool_schema_tokens,
            )
            if residual < 1:
                raise AnswerInputOverflowError(
                    "Research tool calls exhausted the resolved model input residual"
                )
            call_count = max(1, len(transcript[-1].get("tool_calls") or ()))
            self._text_window_budget.update(max(1, residual // call_count))
            return residual

        previous_rows = run.evidence.row_count
        try:
            executed = await executor.run_turn(
                call_messages,
                run.tools,
                tool_choice="auto",
                observation_budget=observation_budget,
                max_tokens=max_output_tokens,
            )
        except ToolResultCapacityError as exc:
            raise AnswerInputOverflowError(str(exc)) from exc
        run.trace["tool_observations"].extend(
            execution.observation.as_dict() for execution in executed.results
        )
        run.episode.record(executed.messages[len(call_messages) :])
        return executed, run.evidence.row_count != previous_rows


def _tool_schema_tokens(tools: list[AgentTool]) -> int:
    return estimate_tokens(
        json.dumps(
            [asdict(tool.definition) for tool in tools],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def research_history_input_measure(
    *,
    model_profile: ModelProfile,
    context_policy: ContextPolicy,
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
    image_budget: AnswerImageBudget | None,
    tools: list[AgentTool],
    retained_tail_tokens: int,
) -> Callable[[list[dict[str, Any]]], int]:
    """Return the exact zero-evidence Research seed serializer used at acceptance."""
    tool_schema_tokens = _tool_schema_tokens(tools)

    def measure(history: list[dict[str, Any]]) -> int:
        context = ContextAssembler(
            model_profile=model_profile,
            context_policy=context_policy,
            query=query,
            history=PriorTurns(history),
            query_images=query_images,
            resource_manifest=resource_manifest,
        )
        return (
            context.measure_control_input(
                evidence=EvidenceLedger(image_budget=image_budget),
                episode=SessionEpisode(retained_tail_tokens=retained_tail_tokens),
            )
            + tool_schema_tokens
        )

    return measure


__all__ = [
    "AnswerOrchestrator",
    "PreparedRun",
    "RunBoundaries",
    "research_history_input_measure",
]
