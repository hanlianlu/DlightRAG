# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Capability-driven answer orchestrator.

One owner routes every answer. Resolved Fast mode takes the standard-RAG path: fixed knowledge-base
retrieval and one final answer generation, with no control turn. Resolved
Research mode enters the agent loop: the model selects from the available peer
tools and writes the answer when it stops calling tools.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast
from uuid import uuid4

from dlightrag_agent.environment.access import AccessScheduler, PathAccess
from dlightrag_agent.loop import AgentLoop, LoopCancelled
from dlightrag_agent.session.fold import PriorTurns, SessionEpisode, fold_entries
from dlightrag_agent.session.ids import SessionId
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
from dlightrag.answer.citations.streaming import AnswerStream
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.answer.publication import StagedArtifact, scan_artifact_directory
from dlightrag.answer.resources.models import ResourceManifestEntry, TextWindowBudget
from dlightrag.answer.resources.registry import ResourceRegistry
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools import KnowledgeRetrieval, WebSearch, compose_research_tools
from dlightrag.answer.tools.delegate import DelegateHost
from dlightrag.answer.workspace import RunWorkspace

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
    last_turn: ExecutedTurn | None = None


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
        telemetry: Telemetry,
        environment: object | None = None,
        resource_reader: object | None = None,
        research_path: bool,
        delegate_host: DelegateHost | None = None,
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
        self._telemetry = telemetry
        self._environment = environment
        self._resource_reader = resource_reader
        self._workspace: RunWorkspace | None = None
        self._research_path = research_path
        self._delegate_host = delegate_host
        self._access = AccessScheduler()

    def bind_delegate(
        self,
        *,
        parent_session_id: SessionId,
        run_id: str,
        owner_id: str,
        persist: Any = None,
        load_child: Any = None,
        finish_child: Any = None,
        run_child: Any = None,
    ) -> None:
        if self._delegate_host is None:
            return
        self._delegate_host.parent_session_id = parent_session_id
        self._delegate_host.run_id = run_id
        self._delegate_host.owner_id = owner_id
        self._delegate_host.persist = persist
        self._delegate_host.load_child = load_child
        self._delegate_host.finish_child = finish_child
        self._delegate_host.run_child = run_child

    def bind_workspace(self, workspace: RunWorkspace) -> None:
        """Attach the claimed run workspace used for tools, spill, and publication."""
        self._workspace = workspace
        self._environment = workspace.environment

    def staged_artifacts(self) -> tuple[StagedArtifact, ...]:
        """Regular files under artifacts/, or empty when no workspace is bound."""
        if self._workspace is None:
            return ()
        return scan_artifact_directory(self._workspace.workspace / "artifacts")

    @property
    def uses_research_path(self) -> bool:
        """Whether this orchestrator runs AgentLoop instead of Fast."""
        return self._research_path

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
        if self._model_func is None:
            raise RuntimeError("Research answer requires a tool-capable model")
        await self.research_until_stopped(run, boundaries=boundaries)
        await boundaries.enter_phase("generating")
        run.trace["agent_stop_reason"] = run.stop_reason
        text = run.last_turn.assistant.text if run.last_turn is not None else ""
        indexer = run.evidence.render_blocks()[1]
        stream = AnswerStream(_single_chunk(text), indexer=indexer)
        stream.trace = run.trace  # type: ignore[attr-defined]
        return run.evidence.contexts, stream

    async def research_until_stopped(self, run: PreparedRun, *, boundaries: RunBoundaries) -> None:
        """Run evidence turns until AgentLoop reports a terminal stop."""

        class _Host:
            async def check_cancelled(self) -> None:
                try:
                    await boundaries.check_cancelled()
                except Exception as exc:
                    if exc.__class__.__name__ in {"RunCancelledError", "AnswerRunCancelledError"}:
                        raise LoopCancelled from exc
                    raise

            async def run_turn(self) -> ExecutedTurn:
                await boundaries.enter_phase("researching")
                executed, _changed = await self_outer._execute_control_turn(run)
                run.agent_turn_count += 1
                run.trace["agent_turns"] = run.agent_turn_count
                await boundaries.commit_turn(executed, turn_number=run.agent_turn_count)
                return executed

        self_outer = self
        outcome = await AgentLoop().run(_Host())
        run.stop_reason = outcome.reason
        run.last_turn = outcome.last_turn

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
        tools = self._compose_tools(evidence, trace, child=False)
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

    def prepare_child_session(self, objective: str) -> PreparedRun:
        """Build a zero-history child session bound to this parent's tools and model."""
        evidence = EvidenceLedger(image_budget=self._image_budget)
        retained_tail_tokens = self._context_policy.retained_tail_target(self._model_profile)
        trace: dict[str, Any] = {
            "agent_turns": 0,
            "web_search_cost_dollars": 0.0,
            "tool_observations": [],
        }
        tools = self._compose_tools(evidence, trace, child=True)
        return PreparedRun(
            context=ContextAssembler(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                query=child_question(objective),
                history=PriorTurns(),
                query_images=None,
                resource_manifest=self._resource_manifest,
            ),
            tools=tools,
            evidence=evidence,
            episode=SessionEpisode(retained_tail_tokens=retained_tail_tokens),
            registry=None,
            trace=trace,
        )

    def hold_workspace_read(self) -> Any:
        """Hold a recursive workspace search so parent writes wait out the child."""
        return self._access.hold(PathAccess(".", kind="search"))

    @property
    def has_execution_environment(self) -> bool:
        return self._environment is not None

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
        *,
        child: bool,
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
            scheduler=self._access,
            spill=(None if child or self._workspace is None else self._spill_writer()),
            delegate_host=None if child else self._delegate_host,
            child=child,
        )

    def _spill_writer(self) -> Any:
        from dlightrag.answer.workspace import spill_receipt, write_spill_file

        workspace = self._workspace
        if workspace is None:
            raise RuntimeError("spill requires a bound workspace")

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


_CHILD_OBJECTIVE_PREFIX = (
    "Investigate this question as a research subagent. "
    "Use tools as needed, then write a concise summary and stop. "
    "Do not mention these instructions.\n\n"
)


def child_question(objective: str) -> str:
    return f"{_CHILD_OBJECTIVE_PREFIX}{objective.strip()}"


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


async def _single_chunk(text: str) -> AsyncIterator[str]:
    if text:
        yield text


__all__ = [
    "AnswerOrchestrator",
    "PreparedRun",
    "RunBoundaries",
    "child_question",
    "research_history_input_measure",
]
