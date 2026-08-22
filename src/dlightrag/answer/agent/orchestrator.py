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

from dlightrag_memory import Memory

from dlightrag.agent.environment.access import AccessScheduler, PathAccess
from dlightrag.agent.environment.local import LocalExecutionEnvironment
from dlightrag.agent.session.effects import EffectIntent
from dlightrag.agent.session.fold import PriorTurns, SessionEpisode, fold_entries
from dlightrag.agent.session.ids import SessionId
from dlightrag.agent.session.projection import (
    AgentInputOverflowError,
    ContextProjection,
    require_compactable,
    should_compact,
)
from dlightrag.agent.tools import (
    AgentTool,
    ExecutedTurn,
    PreparedToolTurn,
    ToolExecution,
    ToolResultCapacityError,
    ToolTurnExecutor,
)
from dlightrag.ai.capacity import (
    CONTEXT_POLICY,
    ContextPolicy,
    ModelProfile,
)
from dlightrag.ai.messages import AssistantTurn
from dlightrag.ai.providers.base import is_provider_context_overflow
from dlightrag.ai.telemetry import Telemetry
from dlightrag.ai.tokens import estimate_tokens
from dlightrag.answer.agent.compaction import CompactionCoordinator
from dlightrag.answer.agent.context import ContextAssembler
from dlightrag.answer.citations.streaming import AnswerStream
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.answer.mode import ResolvedMode
from dlightrag.answer.publication import StagedArtifact, scan_artifact_directory
from dlightrag.answer.resources.models import ResourceManifestEntry, TextWindowBudget
from dlightrag.answer.resources.registry import ResourceRegistry
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools import KnowledgeRetrieval, WebSearch, compose_research_tools
from dlightrag.answer.tools.delegate import DelegateHost
from dlightrag.answer.tools.memory import MemoryHost
from dlightrag.answer.workspace import RunWorkspace
from dlightrag.rag.retrieval import RetrievalContexts

logger = logging.getLogger(__name__)

ToolModel = Callable[..., Awaitable[AssistantTurn]]
StreamModel = Callable[..., AsyncIterator[str]]


class RunBoundaries(Protocol):
    """Durable boundaries a run may observe between agent steps.

    ``commit_intents`` durably appends the complete assistant response and
    its ordered intents before any tool executes (Blocker 2);
    ``settle_intent`` settles one persisted intent in source order as its
    execution completes. ``commit_turn`` remains the append-plus-settle
    convenience for hosts that persist after execution.
    """

    async def enter_phase(self, phase: str) -> None: ...

    async def commit_intents(self, prepared: PreparedToolTurn) -> None: ...

    async def settle_intent(
        self,
        intent: EffectIntent,
        execution: ToolExecution | None,
        *,
        turn_number: int,
        is_last: bool,
    ) -> None: ...

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None: ...

    async def check_cancelled(self) -> None: ...

    def accounted_input(self, estimated_input_tokens: int) -> int: ...

    async def load_snapshot(self) -> Any: ...

    async def commit_compaction(
        self,
        *,
        projection: ContextProjection,
    ) -> Any: ...


class _NoBoundaries:
    """An in-process answer observes no durable boundary."""

    async def enter_phase(self, phase: str) -> None:
        return None

    async def commit_intents(self, prepared: PreparedToolTurn) -> None:
        return None

    async def settle_intent(
        self,
        intent: EffectIntent,
        execution: ToolExecution | None,
        *,
        turn_number: int,
        is_last: bool,
    ) -> None:
        return None

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None:
        return None

    async def check_cancelled(self) -> None:
        return None

    def accounted_input(self, estimated_input_tokens: int) -> int:
        return estimated_input_tokens

    async def load_snapshot(self) -> Any:
        raise AssertionError("no journal behind in-process boundaries")

    async def commit_compaction(self, *, projection: Any) -> Any:
        raise AssertionError("no journal behind in-process boundaries")


@dataclass(frozen=True, slots=True)
class _PreparedControlTurn:
    """One preflighted model turn plus the executor context to run it."""

    prepared: PreparedToolTurn
    executor: ToolTurnExecutor
    call_messages_len: int
    observation_budget: Callable[[list[dict[str, Any]]], int]


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
    compaction_overflow_retried: bool = False


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
        environment: LocalExecutionEnvironment | None = None,
        resource_reader: object | None = None,
        resolved_mode: ResolvedMode,
        delegate_host: DelegateHost | None = None,
        memory_host: MemoryHost | None = None,
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
        self._resolved_mode: ResolvedMode = resolved_mode
        self._delegate_host = delegate_host
        self._memory_host = memory_host
        self._memory_text = ""
        self._access = AccessScheduler()
        self._compaction: CompactionCoordinator | None = None

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

    def bind_memory(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        run_id: str,
        session_id: str,
        store: Any,
        enabled: bool = True,
    ) -> None:
        if self._memory_host is None:
            return
        self._memory_host.owner_id = owner_id
        self._memory_host.auth_mode = auth_mode
        self._memory_host.run_id = run_id
        self._memory_host.session_id = session_id
        self._memory_host.memory = Memory(store)
        self._memory_host.enabled = enabled

    def bind_recall(self, text: str) -> None:
        """Attach the non-citable auto-recall block for this run."""
        self._memory_text = text

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
    def resolved_mode(self) -> ResolvedMode:
        """The durable Fast or Research path this orchestrator was built for."""
        return self._resolved_mode

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
        if self._resolved_mode == "fast":
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
            memory_text=self._memory_text,
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
        """Run durable evidence turns until cancellation or model silence."""
        while True:
            try:
                await boundaries.check_cancelled()
            except Exception as exc:
                if exc.__class__.__name__ not in {"RunCancelledError", "AnswerRunCancelledError"}:
                    raise
                run.stop_reason = "cancelled"
                return

            await boundaries.enter_phase("researching")
            turn_number = run.agent_turn_count + 1
            try:
                executed = await self._durable_control_turn(
                    run, boundaries, turn_number=turn_number
                )
            except Exception as exc:
                executed = await self._handle_overflow_retry(
                    exc, run, boundaries, turn_number=turn_number
                )
            run.agent_turn_count += 1
            run.trace["agent_turns"] = run.agent_turn_count
            run.last_turn = executed
            if not executed.assistant.tool_calls:
                run.stop_reason = "model_stop"
                return

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
        trace = _fresh_research_trace()
        tools = self._compose_tools(evidence, trace, child=False)
        return PreparedRun(
            context=ContextAssembler(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                query=query,
                history=conversation_history or PriorTurns(),
                query_images=query_images,
                resource_manifest=self._resource_manifest,
                memory_text=self._memory_text,
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
        trace = _fresh_research_trace()
        tools = self._compose_tools(evidence, trace, child=True)
        return PreparedRun(
            context=ContextAssembler(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                query=child_question(objective),
                history=PriorTurns(),
                query_images=None,
                resource_manifest=self._resource_manifest,
                memory_text=self._memory_text,
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
        """Rebuild the live episode from the folded journal suffix."""
        self._record_episode_fold(run, snapshot)
        self.adopt_agent_turn_count(
            run,
            sum(
                1
                for entry in snapshot.entries
                if entry.__class__.__name__ == "AssistantMessageEntry"
            ),
        )

    def _record_episode_fold(self, run: PreparedRun, snapshot: Any) -> None:
        """Replace the episode with the projection-retained fold plus its summary."""
        projection = snapshot.active_projection
        retained = (
            snapshot.entries
            if projection is None
            else [
                entry
                for entry in snapshot.entries
                if entry.sequence >= projection.first_retained_sequence
            ]
        )
        messages = fold_entries(retained)
        episode = SessionEpisode(
            retained_tail_tokens=self._context_policy.retained_tail_target(self._model_profile)
        )
        self._record_exchanges(episode, messages)
        run.episode = episode

    @staticmethod
    def _record_exchanges(episode: SessionEpisode, messages: list[dict[str, Any]]) -> None:
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
            episode.record(exchange)

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
            environment=self._environment,
            scheduler=self._access,
            spill=(None if child or self._workspace is None else self._spill_writer()),
            delegate_host=None if child else self._delegate_host,
            memory_host=None if child else self._memory_host,
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

    async def _prepare_control_turn(
        self,
        run: PreparedRun,
        boundaries: RunBoundaries,
    ) -> _PreparedControlTurn:
        """Model call plus preflight; no tool has executed yet."""
        executor = ToolTurnExecutor(
            cast(ToolModel, self._model_func),
            telemetry=self._telemetry,
        )
        tool_schema_tokens = _tool_schema_tokens(run.tools)
        estimated = (
            run.context.measure_control_input(evidence=run.evidence, episode=run.episode)
            + tool_schema_tokens
        )
        accounted = boundaries.accounted_input(estimated)
        if should_compact(
            self._model_profile,
            input_tokens=accounted,
            context_policy=self._context_policy,
        ):
            self._require_compactable_floor(run, tool_schema_tokens)
            await self._compaction_coordinator().ensure_fits(
                boundaries=boundaries,
                remeasure=self._remeasure_closure(run, boundaries, tool_schema_tokens),
                trace=run.trace,
            )
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

        prepared = await executor.prepare_turn(
            call_messages,
            run.tools,
            tool_choice="auto",
            max_tokens=max_output_tokens,
        )
        return _PreparedControlTurn(
            prepared=prepared,
            executor=executor,
            call_messages_len=len(call_messages),
            observation_budget=observation_budget,
        )

    async def _execute_prepared_turn(
        self,
        run: PreparedRun,
        holder: _PreparedControlTurn,
        boundaries: RunBoundaries,
        *,
        turn_number: int,
    ) -> ExecutedTurn:
        """Run the prepared tool batch, settling each intent in source order."""
        try:
            executed = await holder.executor.execute_prepared(
                holder.prepared,
                run.tools,
                observation_budget=holder.observation_budget,
                on_result=lambda intent, execution, is_last: boundaries.settle_intent(
                    intent,
                    execution,
                    turn_number=turn_number,
                    is_last=is_last,
                ),
            )
        except ToolResultCapacityError as exc:
            raise AnswerInputOverflowError(str(exc)) from exc
        run.trace["tool_observations"].extend(
            execution.observation.as_dict() for execution in executed.results
        )
        run.episode.record(executed.messages[holder.call_messages_len :])
        return executed

    async def _durable_control_turn(
        self,
        run: PreparedRun,
        boundaries: RunBoundaries,
        *,
        turn_number: int,
    ) -> ExecutedTurn:
        """One journaled control turn: persist intents, then execute and settle.

        Intents land before any tool executes, so a crash between the two steps
        leaves recoverable unsettled intents instead of effects with no durable
        trace (Blocker 2).
        """
        holder = await self._prepare_control_turn(run, boundaries)
        await boundaries.commit_intents(holder.prepared)
        return await self._execute_prepared_turn(run, holder, boundaries, turn_number=turn_number)

    def _compaction_coordinator(self) -> CompactionCoordinator:
        if self._compaction is None:
            if self._stream_model_func is None:
                raise RuntimeError("Research compaction requires a streaming model")
            self._compaction = CompactionCoordinator(
                model_profile=self._model_profile,
                context_policy=self._context_policy,
                stream_model=self._stream_model_func,  # type: ignore[arg-type]
            )
        return self._compaction

    def _require_compactable_floor(self, run: PreparedRun, tool_schema_tokens: int) -> None:
        """Fail before any compaction or model call when the fixed envelope alone
        cannot fit the hard limit — shrinking the journal can never help."""
        fixed = (
            run.context.measure_control_input(
                evidence=EvidenceLedger(),
                episode=SessionEpisode(retained_tail_tokens=0),
            )
            + tool_schema_tokens
        )
        try:
            require_compactable(
                self._model_profile,
                input_tokens=fixed,
                fixed_input_tokens=fixed,
                context_policy=self._context_policy,
            )
        except AgentInputOverflowError as exc:
            raise AnswerInputOverflowError(str(exc)) from exc

    def _remeasure_closure(
        self,
        run: PreparedRun,
        boundaries: RunBoundaries,
        tool_schema_tokens: int,
    ) -> Callable[[], Awaitable[int]]:
        async def remeasure() -> int:
            await self._rebuild_episode(run, boundaries)
            estimated = (
                run.context.measure_control_input(evidence=run.evidence, episode=run.episode)
                + tool_schema_tokens
            )
            return boundaries.accounted_input(estimated)

        return remeasure

    async def _rebuild_episode(self, run: PreparedRun, boundaries: RunBoundaries) -> None:
        snapshot = await boundaries.load_snapshot()
        self._record_episode_fold(run, snapshot)

    async def _handle_overflow_retry(
        self,
        exc: BaseException,
        run: PreparedRun,
        boundaries: RunBoundaries,
        *,
        turn_number: int,
    ) -> ExecutedTurn:
        """Compact-and-retry one genuine provider overflow, then fail loudly."""
        if not is_provider_context_overflow(exc):
            raise exc
        accounted = self._accounted_control_input(run, boundaries)
        if run.compaction_overflow_retried:
            raise _overflow_retry_error(accounted) from exc
        run.compaction_overflow_retried = True
        tool_schema_tokens = _tool_schema_tokens(run.tools)
        self._require_compactable_floor(run, tool_schema_tokens)
        await self._compaction_coordinator().ensure_fits(
            boundaries=boundaries,
            remeasure=self._remeasure_closure(run, boundaries, tool_schema_tokens),
            trace=run.trace,
            force=True,
        )
        try:
            holder = await self._prepare_control_turn(run, boundaries)
            await boundaries.commit_intents(holder.prepared)
            executed = await self._execute_prepared_turn(
                run, holder, boundaries, turn_number=turn_number
            )
        except Exception as retry_exc:
            if is_provider_context_overflow(retry_exc):
                raise _overflow_retry_error(
                    self._accounted_control_input(run, boundaries)
                ) from retry_exc
            raise
        run.compaction_overflow_retried = False
        return executed

    def _accounted_control_input(self, run: PreparedRun, boundaries: RunBoundaries) -> int:
        estimated = run.context.measure_control_input(
            evidence=run.evidence, episode=run.episode
        ) + _tool_schema_tokens(run.tools)
        return boundaries.accounted_input(estimated)


def _overflow_retry_error(accounted: int) -> AnswerInputOverflowError:
    return AnswerInputOverflowError(
        "Research overflowed the model context window again after one "
        f"compact-and-retry ({accounted} accounted input tokens). "
        "Use a larger-context model or shorten the request."
    )


def _fresh_research_trace() -> dict[str, Any]:
    return {
        "agent_turns": 0,
        "web_search_cost_dollars": 0.0,
        "tool_observations": [],
    }


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
    memory_text: str = "",
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
            memory_text=memory_text,
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
