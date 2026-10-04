# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution over durable Runtime sessions and RAG workspaces."""

import asyncio
import datetime
import hashlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from dlightrag_memory import Memory, MemoryStore

from dlightrag.engine.agent.environment import (
    ExecutionMode,
    SearchToolchain,
    confinement_state,
    resolve_execution_adapter,
)
from dlightrag.engine.agent.environment.confinement import ConfinementPolicy
from dlightrag.engine.agent.session.effects import (
    canonical_json,
)
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    SessionEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.fold import (
    PriorTurns,
    conversation_messages,
    host_turn_starts,
    project_session_messages,
)
from dlightrag.engine.agent.session.ids import (
    EntryId,
    LaneId,
    OperationId,
    SessionId,
)
from dlightrag.engine.agent.session.operation import (
    OperationCompleted,
    OperationFailed,
)
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.projection import ContextProjection
from dlightrag.engine.agent.session.registers import (
    ContextProjectionRegister,
    DeleteRegister,
    HostTurnReservation,
    LaneHead,
    LaneState,
    RegisterRef,
    SetRegister,
)
from dlightrag.engine.agent.session.repository import (
    AgentSessionSnapshot,
    validate_snapshot_refresh,
)
from dlightrag.engine.agent.session.runtime import (
    AgentSessionRuntime,
    AgentSessionSnapshotSeed,
    FollowUpCommand,
    OperationConflictError,
    SessionLeaseLostError,
    close_operation,
)
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionLeaseLost,
)
from dlightrag.engine.agent.skills import SkillsBundle, SkillsBundleFactory
from dlightrag.engine.agent.tools import (
    AgentTool,
)
from dlightrag.engine.agent.tools.contracts import ToolDeclaration
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.reasoning import (
    ReasoningConfigurationError,
    ReasoningLevel,
    resolve_reasoning,
)
from dlightrag.engine.ai.scheduler import model_call_scope
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ChatModelSelector
from dlightrag.engine.ai.telemetry import (
    Observation,
    Telemetry,
    bounded_telemetry_text,
    safe_log_text,
)
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserBinding,
    BrowserHolder,
    RunAgentAccounts,
    RunAgentBrowser,
)
from dlightrag.engine.answer.attachment_replay import AttachmentReplaySelection
from dlightrag.engine.answer.capabilities import AnswerCapabilityCoordinator
from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.sources import project_contexts_for_client
from dlightrag.engine.answer.citations.streaming import aclose_answer_stream
from dlightrag.engine.answer.client_contracts import AnswerEffort
from dlightrag.engine.answer.compaction import CompactionCoordinator
from dlightrag.engine.answer.continuation_handles import compose_session_notes
from dlightrag.engine.answer.errors import (
    ROUTING_FAILED,
    AnswerInputError,
    InvalidToolConfigurationError,
    classify_answer_error,
    reasoning_control_rejection_message,
)
from dlightrag.engine.answer.execution.acceptance import (
    AnswerHistoryBudget,
    RetrievalPlanning,
    reserved_memory_text,
    routing_resources,
)
from dlightrag.engine.answer.execution.connection_binding import (
    ResearchConnectionToolResolver,
    ResearchToolClaim,
)
from dlightrag.engine.answer.execution.input import (
    AnswerRunInput,
    PinnedModelProfile,
    build_current_answer_resources,
    child_model_guidance,
    model_reasoning_settings,
    pinned_model_selectors,
    validate_active_answer_input,
)
from dlightrag.engine.answer.execution.lineage import LineageResourceStore, RetainedResourceLoader
from dlightrag.engine.answer.execution.resources import AnswerResourceResolver
from dlightrag.engine.answer.fast import (
    FastRunBoundaries,
    FastSessionHost,
    ensure_session_lane,
    projection_from_compaction_at,
)
from dlightrag.engine.answer.highlights import SemanticHighlightSettings, enrich_semantic_highlights
from dlightrag.engine.answer.history import HistoryProjectionTarget
from dlightrag.engine.answer.links.cards import collect_link_cards
from dlightrag.engine.answer.media import evidence_images_from_sources
from dlightrag.engine.answer.memory import render_auto_recall
from dlightrag.engine.answer.mode import ResolvedMode
from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.publication import (
    ArtifactAttachment,
    PublicationLimits,
    PublicationPlan,
    StagedArtifact,
    is_empty_answer,
    run_artifact_check,
    validate_publication,
)
from dlightrag.engine.answer.research.persistence import ResearchRunStore
from dlightrag.engine.answer.research.runtime import (
    AnswerRuntimeControls,
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _accepted_operation_keys,
    _answer_runtime_event_sink,
    _bound_child_dispatch_preparer,
    _bound_child_runner,
    _buffered_fetched_bytes_sink,
    _check_child_write,
    _drive_answer_operation,
    _durable_child_usage,
    _fenced_control_ack,
    _fenced_control_reader,
    _held_operation,
    _oldest_pending_input,
    _restore_durable_evidence,
    _usage_from_snapshot_entries,
)
from dlightrag.engine.answer.resource_settlement import attached_resource_update
from dlightrag.engine.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.engine.answer.resources.converters import MAX_CONVERSION_SECONDS, is_convertible
from dlightrag.engine.answer.resources.lineage import LINEAGE_ADOPTION_KIND, LineageResourceLoader
from dlightrag.engine.answer.resources.models import (
    ResourceRegistryError,
)
from dlightrag.engine.answer.resources.registry import (
    BROWSER_ACQUISITIONS,
    BROWSER_RENDER,
    FetchedBytesSink,
    ResourceStateMismatchError,
)
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.results import store_answer_result
from dlightrag.engine.answer.router import AnswerModeRouter, RoutingFailedError
from dlightrag.engine.answer.runs.routing import AnswerRoutingStore, decide_resolved_mode
from dlightrag.engine.answer.session_notes import (
    SESSION_NOTES_DEGRADED_KEY,
    SessionNotesBinding,
    SessionNotesPlane,
    read_working_copy_or_reason,
)
from dlightrag.engine.answer.tools.browser import BrowserToolHost
from dlightrag.engine.answer.tools.memory import MemoryHost
from dlightrag.engine.answer.tools.resources import (
    make_admitted_bytes_reader,
    make_resource_reader,
    make_resource_viewer,
)
from dlightrag.engine.answer.tools.subagents import (
    ChildContextSnapshot,
    SubagentHost,
)
from dlightrag.engine.answer.workspace import (
    RunWorkspace,
    WorkspaceIntegrityError,
    bind_run_workspace,
)
from dlightrag.engine.dependencies import (
    DependencyComponent,
    classify_transient_dependency,
    dependency_component_from_checkpoint,
    next_dependency_retry,
)
from dlightrag.engine.rag.retrieval import (
    MetadataFilter,
    RetrievalContexts,
    RetrievalOptions,
    RetrievalResult,
)
from dlightrag.engine.rag.workspace.lifecycle import defer_cancellation
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.runtime.coordinator import (
    LeaseLostError,
    RunCancellationObserved,
    RunSession,
)
from dlightrag.engine.runtime.errors import (
    IncompatibleActiveRunError,
    RunExecutionError,
)
from dlightrag.engine.runtime.records import (
    AlreadyCommittedTerminal,
    Deferred,
    PendingPublication,
    RunExecutionOutcome,
    RunFetchedResource,
    RunRecord,
    Succeeded,
    WaitingForRepair,
)
from dlightrag.engine.runtime.settlements import (
    ArtifactAttachmentUpdate,
    EffectHostUpdate,
)
from dlightrag.engine.runtime.workspace import (
    DEFAULT_SESSION_NOTES_LIMITS,
    SessionNoteRecord,
    SessionNotesLimits,
    WorkspaceStore,
)

logger = logging.getLogger(__name__)
_FAST_COMPACTION_ATTEMPT_LIMIT = 3
_DEPENDENCY_DEFER_BASE_SECONDS = 5
_DEPENDENCY_DEFER_MAX_SECONDS = 60

type DependencyStateCallback = Callable[[DependencyComponent], None]


def _incomplete_operation_error(operation: Any, *, message: str) -> RunExecutionError:
    """Classify a terminal Agent Operation failure into a public Run failure.

    A refused payload describes the *value*, not the stored Session, so it gets
    its own kind: the Run failed, but the conversation survives and can be used
    again. Generic messages stay as they were for every other terminal state.
    """
    if isinstance(operation.state, OperationFailed) and (
        operation.state.kind == "payload_unrepresentable"
    ):
        return RunExecutionError(
            "payload_unrepresentable",
            "The research context could not be stored: it contained a character the "
            "durable store cannot keep.",
        )
    return RunExecutionError("run_execution_failed", message)


def _require_current_child_lifecycle(plan: AgentRunPlan | None) -> None:
    """Refuse a Research plan whose pinned spawn contract predates asynchronous Children."""
    if plan is None:
        raise IncompatibleActiveRunError("Research answer run is missing its accepted Agent Plan")
    spawn = next((tool for tool in plan.tools if tool.name == "spawn_agent"), None)
    if spawn is not None and spawn.contract_version != 5:
        raise IncompatibleActiveRunError("Research answer run uses an unsupported child lifecycle")


class RunBlobReader(Protocol):
    """Stream one owner-scoped opaque Run blob by digest."""

    def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...


class ArtifactReader(Protocol):
    """Read Answer-owned artifact metadata without owning blob bytes."""

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]: ...
    async def list_artifact_attachments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[ArtifactAttachmentUpdate, ...]: ...
    async def list_fetched_resources(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunFetchedResource, ...]: ...


class AnswerExecutionStore(
    ArtifactReader, AnswerRoutingStore, ResearchRunStore, LineageResourceStore, Protocol
):
    """Answer execution persistence, including required Research child operations."""

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None: ...

    async def load_child_attachment_occurrences(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_session_id: str,
        context_snapshot: dict[str, Any],
    ) -> tuple[RunFetchedResource, ...]: ...

    async def retain_attachment_occurrences(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        selection: AttachmentReplaySelection,
    ) -> tuple[RunFetchedResource, ...]: ...


type WorkspaceWarmer = Callable[[Sequence[str]], None]


class RawRetrieval(Protocol):
    async def __call__(
        self,
        query: str,
        *,
        workspaces: Sequence[str],
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        retrieval: RetrievalOptions = RetrievalOptions(),
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: Sequence[Mapping[str, Any]] = (),
        image_descriptions: Sequence[str] = (),
        preserve_query: bool | None = None,
        model_profile: ModelProfile | None = None,
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class AnswerExecutorSettings:
    default_top_k: int
    default_chunk_top_k: int
    semantic_highlights: SemanticHighlightSettings
    publication: PublicationLimits = PublicationLimits()
    child_guidance_timeout_seconds: int = 300
    lineage_adoption: bool = True


@dataclass
class OrchestratorRun:
    """One durable request resolved into an orchestrator and its exact inputs."""

    orchestrator: AnswerOrchestrator
    image_descriptions: list[str]
    query_images: list[dict[str, Any]] | None
    history: PriorTurns
    fast_history_targets: tuple[HistoryProjectionTarget, ...]
    current_image_count: int
    workspaces: list[str]
    registry: ResourceRegistry | None


async def _memory_recall_allowed(
    checker: Callable[..., Awaitable[bool]] | None, *, owner_id: str
) -> bool:
    """Whether answer injection may use this owner's memory.

    A missing checker means the composition has no settings store and memory
    stays enabled — the historical default.
    """
    return checker is None or await checker(owner_id=owner_id)


class AnswerExecutor:
    """Execute durable Answer runs without composition or storage dependencies."""

    def __init__(
        self,
        *,
        store: AnswerExecutionStore,
        blob_store: RunBlobReader,
        pool: WorkspacePool,
        warm: WorkspaceWarmer,
        retrieve: RawRetrieval,
        planning: RetrievalPlanning,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
        resources: AnswerResourceResolver,
        settings: AnswerExecutorSettings,
        telemetry: Telemetry,
        model_invocation_fingerprint_for_role: Callable[
            [ChatModelSelector], ModelInvocationFingerprint
        ],
        execution_environment: ExecutionMode = "trust",
        shell_confinement: ConfinementPolicy,
        workspace_root: str | None = None,
        session_notes_limits: SessionNotesLimits | None = None,
        search_toolchain: SearchToolchain,
        working_dir: str = "./dlightrag_storage",
        memory_store: MemoryStore | None = None,
        memory_recall_enabled: Callable[..., Awaitable[bool]] | None = None,
        memory_capability_current: Callable[..., Awaitable[bool]] | None = None,
        connection_tool_resolver: ResearchConnectionToolResolver | None = None,
        skills_bundle_factory: SkillsBundleFactory | None = None,
        browser: AgentBrowserBinding | None = None,
        now: Callable[[], datetime.datetime] | None = None,
        on_dependency_unavailable: DependencyStateCallback | None = None,
        on_dependency_recovered: DependencyStateCallback | None = None,
    ) -> None:
        self._store = store
        self._blob_store = blob_store
        self._pool = pool
        self._warm = warm
        #: The settled Session view each running Run already drove, held only until
        #: its own exit hook records the Fork Point and then dropped.
        self._settled_views: dict[str, AgentSessionSnapshot] = {}
        self._retrieve_result = retrieve
        self._planning = planning
        self._models = models
        self._capabilities = capabilities
        self._resources = resources
        self._settings = settings
        self._telemetry = telemetry
        self._model_invocation_fingerprint_for_role = model_invocation_fingerprint_for_role
        self._execution_environment: ExecutionMode = execution_environment
        self._workspace_root_setting = workspace_root
        self._session_notes_limits = session_notes_limits or DEFAULT_SESSION_NOTES_LIMITS
        self._search_toolchain = search_toolchain
        self._working_dir = working_dir
        self._memory_store = memory_store
        self._memory = Memory(memory_store) if memory_store is not None else None
        self._memory_recall_enabled = memory_recall_enabled
        self._memory_capability_current = memory_capability_current
        self._connection_tool_resolver = connection_tool_resolver
        self._skills_bundle_factory = skills_bundle_factory
        self._browser = browser
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))
        self._on_dependency_unavailable = on_dependency_unavailable
        self._on_dependency_recovered = on_dependency_recovered
        self._execution_adapter = resolve_execution_adapter(
            execution_environment, confinement=shell_confinement
        )

    async def aclose(self) -> None:
        """Finish adapter-owned process cleanup after the coordinator stops claims."""
        try:
            if self._execution_adapter is not None:
                await self._execution_adapter.aclose()
        finally:
            if self._browser is not None:
                await self._browser.provider.aclose()

    def validate_active_prepared_input(self, prepared: Mapping[str, Any]) -> None:
        """Validate active durable Answer input using the executor's model bindings."""
        validate_active_answer_input(
            prepared,
            model_invocation_fingerprint_for_role=self._model_invocation_fingerprint_for_role,
            model_settings_for_role=self._models.model_settings,
        )

    def research_tool_declarations(
        self,
        *,
        web_search: bool,
        memory: bool,
        model_guidance: str,
        injected: Sequence[ToolDeclaration],
    ) -> tuple[ToolDeclaration, ...]:
        """Project configured capabilities without constructing execution dependencies."""
        from dlightrag.engine.answer.tools.composition import research_tool_declarations
        from dlightrag.engine.answer.tools.subagents import subagent_declarations

        accounts = None if self._browser is None else self._browser.accounts
        return research_tool_declarations(
            web_search=web_search,
            resource_read=True,
            agent_browser=self._browser is not None,
            agent_accounts=accounts is not None,
            agent_mailbox=accounts is not None and accounts.mailbox is not None,
            resource_view=True,
            environment=self._execution_adapter is not None,
            artifact_publication=self._execution_adapter is not None,
            subagents=subagent_declarations(model_guidance=model_guidance),
            memory=memory and self._memory_store is not None,
            skills=(
                self._skills_bundle_factory.declarations(child=False)
                if self._skills_bundle_factory is not None
                else ()
            ),
            injected=injected,
        )

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        """Execute one claimed run; this is the trace that owns everything the run does."""
        prepared = session.prepared_input if isinstance(session.prepared_input, Mapping) else {}
        query = str(prepared.get("query") or "")
        with model_call_scope((session.owner_id, session.run_id)):
            with self._telemetry.trace(
                session_id=str(prepared.get("agent_session_id") or "") or None,
                user_id=session.owner_id,
            ):
                async with self._telemetry.observe(
                    "run-answer",
                    input={"query": bounded_telemetry_text(query, max_length=2000)},
                    metadata={
                        "run_id": session.run_id,
                        "parent_run_id": prepared.get("parent_run_id"),
                        "workspaces": tuple(prepared.get("workspaces") or ()),
                    },
                ) as run_trace:
                    try:
                        outcome = await self._execute_run(session, run_trace)
                    except RunCancellationObserved, LeaseLostError:
                        # A cancelled or fenced Run is a terminal outcome a user
                        # asked for, not a failure: name it instead of letting
                        # the generic exception mapping mark the trace ERROR.
                        run_trace.update(
                            level="DEFAULT",
                            output={
                                "outcome": "cancelled" if session.cancel_requested else "lease_lost"
                            },
                        )
                        await self._record_fork_point(
                            session,
                            run_trace,
                            snapshot=self._settled_views.pop(session.run_id, None),
                        )
                        raise
                    except asyncio.CancelledError:
                        # Shutdown cancels the task and *requeues* the Run, so this exit
                        # is not a settlement: a point recorded here would outlive the
                        # attempt that never ended, and the requeue clears it anyway.
                        self._settled_views.pop(session.run_id, None)
                        raise
                    except BaseException:
                        # A failure the coordinator will terminalize is still a state
                        # a Fork may branch from: record before it is written.
                        await self._record_fork_point(
                            session,
                            run_trace,
                            snapshot=self._settled_views.pop(session.run_id, None),
                        )
                        raise
                    if isinstance(outcome, Deferred | WaitingForRepair):
                        # The Run keeps its claim and will run again, so its settled
                        # state does not exist yet. Recording the head it happens to
                        # be at would leave a stale branch point behind if the attempt
                        # that does settle could not write its own.
                        self._settled_views.pop(session.run_id, None)
                        return outcome
                    if isinstance(outcome, AlreadyCommittedTerminal):
                        # Fast already recorded the point before committing its own
                        # terminal row, and the fence would refuse a second write.
                        self._settled_views.pop(session.run_id, None)
                        return outcome
                    await self._record_fork_point(
                        session, run_trace, snapshot=self._settled_views.pop(session.run_id, None)
                    )
                    return outcome

    async def _record_fork_point(
        self,
        session: RunSession,
        run_trace: Observation,
        *,
        snapshot: AgentSessionSnapshot | None = None,
    ) -> None:
        """Record the state this Run ended at, for a Fork to branch from.

        Every terminal outcome is a state worth branching from — a Run that died
        established whatever it established — so this runs on the way out of both.
        It is deliberately best effort: the fact is one a Fork refuses to guess
        without, so a lost lease or a store error leaves the Run without a Fork
        Point rather than turning a settled Run into a failed one.

        The Lane comes from the durable routing row, which is the same record that
        fixed it at acceptance, rather than from the prepared input: the two ids
        are routing facts, and decoding the whole pinned contract here would let an
        unrelated pin problem cost the Run its Fork Point.
        """
        try:
            routing = await self._store.load_routing(
                owner_id=session.owner_id, run_id=session.run_id
            )
            if routing is None:
                raise LookupError("answer run has no routing row")
            lane_id = LaneId(routing.agent_lane_id)
            settled = snapshot
            if settled is None:
                # Only a path that never held the settled snapshot pays for it — a Fast
                # run that failed, say, whose Session is a single turn. A long research
                # run passes the view it just drove.
                settled = await session.execution.session_repository.load(
                    SessionId(routing.agent_session_id)
                )
            head = settled.tree.lane(lane_id).head_entry_id
            projection = _lane_projection(settled, lane_id)
            written = await self._store.record_fork_point(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
                entry_id=head.value if head is not None else None,
                projection_id=(projection.projection_id.value if projection is not None else None),
            )
        except LeaseLostError:
            raise
        except Exception as exc:
            # Deliberately no exception text and no traceback: a store failure's
            # message may carry a connection string, and this hook is a convenience
            # rather than an operator diagnostic. The kind is enough to find it.
            logger.warning(
                "answer run could not record its fork point (%s)",
                type(exc).__name__,
                extra={"run_id": session.run_id},
            )
            return
        if not written:
            # The claim moved on: this worker's state is not the Run's settled one.
            run_trace.update(metadata={"fork_point": "unwritten"})

    async def _reclaim_lane(
        self,
        session: RunSession,
        snapshot: AgentSessionSnapshot,
        lane_id: LaneId,
    ) -> AgentSessionSnapshot:
        """Take this Run's Lane back from a holder that is no longer live.

        A Lane is held by the Run that accepted its active Operation or reserved its
        Fast turn. A live holder keeps it, and acceptance refuses as it always has. A
        Run that has ended can never release it, so this Run closes what that one
        left: its Operation through the Runtime's typed close, or its reservation,
        whose unanswered question stays on the Lane.
        """
        try:
            state = snapshot.tree.lane(lane_id).state.value
        except KeyError:
            return snapshot
        if not isinstance(state, LaneState):
            raise TypeError("Lane State register has the wrong value type")
        ref = RegisterRef("host_turn_reservation", lane_id.value)
        record = next((item for item in snapshot.registers if item.ref == ref), None)
        reservation = record.value if record is not None else None
        if reservation is not None and not isinstance(reservation, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        holder = state.holder or (reservation.reservation_id if reservation is not None else None)
        if holder is None or holder == session.run_id:
            return snapshot
        run = await self._store.get_run(owner_id=session.owner_id, run_id=holder)
        if run is not None and not run.terminal:
            return snapshot
        repository = session.execution.session_repository
        if state.active_operation_id is not None:
            await close_operation(
                repository,
                fencing_epoch=session.execution.fencing_epoch,
                holder=session.run_id,
                session_id=snapshot.session_id,
                operation_id=OperationId(state.active_operation_id),
            )
        elif record is not None:
            outcome = await repository.transact(
                session_id=snapshot.session_id,
                fencing_epoch=session.execution.fencing_epoch,
                transaction=SessionTransaction.from_parts(
                    register_writes=[DeleteRegister(ref)],
                    expectations=[RegisterExpectation(ref, record.sequence)],
                ),
            )
            if isinstance(outcome, TransactionLeaseLost):
                raise LeaseLostError
            if isinstance(outcome, RegisterConflict):
                raise RunExecutionError(
                    "agent_session_conflict",
                    "The Agent Lane changed before it could be reclaimed.",
                )
        refreshed = await repository.refresh(snapshot.session_id, previous=snapshot)
        validate_snapshot_refresh(snapshot.session_id, previous=snapshot, snapshot=refreshed)
        return refreshed

    async def _resolve_fork_seed(
        self,
        session: RunSession,
        request: AnswerRunInput,
        snapshot: AgentSessionSnapshot,
    ) -> tuple[EntryId, ContextProjection | None]:
        """Return the recorded Fork Point a new Lane must open at.

        Missing, foreign, or unrecorded parents refuse with a remedy rather than
        falling back to the source Lane's current head.
        """
        if not request.parent_run_id:
            raise RunExecutionError(
                "fork_point_missing",
                "A Fork needs a parent Run. Fork from a Run that has one.",
            )
        parent = await self._store.load_routing(
            owner_id=session.owner_id, run_id=request.parent_run_id
        )
        if parent is None:
            raise RunExecutionError(
                "fork_point_missing",
                "The parent Run is missing. Fork from a Run that still exists.",
            )
        if parent.agent_session_id != request.agent_session_id:
            raise RunExecutionError(
                "fork_point_missing",
                "The parent Run belongs to another Session. Fork from a Run in this conversation.",
            )
        if parent.fork_point_entry_id is None:
            raise RunExecutionError(
                "fork_point_missing",
                "This Run has no recorded Fork Point. Fork from a Run that has one.",
            )
        try:
            head = EntryId(parent.fork_point_entry_id)
            reconstructed = projection_from_compaction_at(snapshot, head)
        except (KeyError, ValueError) as exc:
            raise RunExecutionError(
                "fork_point_stale",
                "The recorded Fork Point is no longer on this Session. "
                "Fork from a Run whose head is still present.",
            ) from exc
        if not snapshot.tree.is_stable_checkpoint(head):
            # A terminally failed Research run can settle with a Tool Call whose
            # Result never arrived; branching there would start inside a batch.
            raise RunExecutionError(
                "fork_point_stale",
                "The recorded Fork Point is not a settled turn. Fork from a Run that ended on one.",
            )
        if request.source_lane_id:
            try:
                source_ids = {
                    entry.entry_id
                    for entry in snapshot.tree.ancestry(LaneId(request.source_lane_id))
                }
            except KeyError as exc:
                raise RunExecutionError(
                    "fork_point_stale",
                    "The recorded Fork Point is no longer on this Session. "
                    "Fork from a Run whose head is still present.",
                ) from exc
            if head not in source_ids:
                raise RunExecutionError(
                    "fork_point_stale",
                    "The recorded Fork Point is no longer on this Session. "
                    "Fork from a Run whose head is still present.",
                )
        if parent.fork_point_projection_id is None:
            return head, None
        if (
            reconstructed is None
            or reconstructed.projection_id.value != parent.fork_point_projection_id
        ):
            raise RunExecutionError(
                "fork_point_stale",
                "The recorded Fork Point is no longer on this Session. "
                "Fork from a Run whose head is still present.",
            )
        return head, reconstructed

    async def _claim_run_workspace(
        self,
        *,
        session: RunSession,
        workspace_store: WorkspaceStore | None,
        session_id: str | None,
    ) -> tuple[RunWorkspace | None, SessionNotesBinding, SessionNotesPlane | None]:
        """Bind this Run's Agent Workspace epoch and materialize the Session's notes.

        Fast composes no tools, so its workspace is inert: it exists so a later
        Research turn of the same Session can read what Fast's memory holds, and so
        Fast's own compaction can name it. A disabled execution environment has no
        root, so this returns nothing and there is no memory to materialize. Memory
        degrades rather than failing the Run: an unreadable plane costs this Run the
        notes, not the answer. A workspace volume that does not answer defers the Run
        as a dependency outage; content recovery refuses fails it.
        """
        from dlightrag.engine.answer.execution_settings import validate_agent_execution

        root = validate_agent_execution(
            execution_environment=self._execution_environment,
            workspace_root=self._workspace_root_setting,
            working_dir=self._working_dir,
        )
        if root is None:
            return None, SessionNotesBinding(), None
        binding = SessionNotesBinding()
        notes: tuple[SessionNoteRecord, ...] = ()
        plane: SessionNotesPlane | None = None
        if workspace_store is not None and session_id:
            plane = SessionNotesPlane(
                store=workspace_store,
                session_id=session_id,
                limits=self._session_notes_limits,
            )
            # A fresh epoch materializes the Session's notes; a recovered one already
            # holds what this Run had, and must not be materialized over.
            if session.workspace_epoch is None:
                binding = await plane.load()
                notes = binding.records
        try:
            bound = await bind_run_workspace(
                workspace_root=root,
                owner_id=session.owner_id,
                run_id=session.run_id,
                fencing_epoch=session.execution.fencing_epoch,
                recorded_epoch=session.workspace_epoch,
                store=workspace_store,
                execution_adapter=self._execution_adapter,
                notes=(notes if session.workspace_epoch is None else ()),
            )
        except WorkspaceIntegrityError as exc:
            # Reading again reads the same content, so the Run fails. What was refused
            # stays in the server log; the public error names no path or internal text.
            logger.warning("Agent Workspace of run %s refused: %s", session.run_id, exc)
            raise RunExecutionError(
                "workspace_integrity_error",
                "This Run's Agent Workspace failed its integrity check, so the Run cannot "
                "resume. Fork from it or start a new Run.",
            ) from exc
        if plane is not None:
            # The baseline is the working copy itself, on both paths: a recovered
            # attempt states what its own epoch holds (so its first request names
            # notes it can actually open), and any change this Run already made is
            # promoted at its next settlement rather than silently reverted.
            records, reason = read_working_copy_or_reason(
                bound.workspace, self._session_notes_limits
            )
            binding = SessionNotesBinding(
                records=records,
                degraded_reason=(binding.degraded_reason or reason or bound.notes_degraded),
            )
            plane.rebind(workspace=bound.workspace, records=binding.records)
        return bound, binding, plane

    async def _execute_run(
        self,
        session: RunSession,
        run_trace: Observation,
    ) -> RunExecutionOutcome:
        """Run one attempt and report its product on the run's own trace.

        Only this layer and the terminal site in :meth:`_execute` write the root
        output, and never both: the pipeline reports the Answer it produced, the
        deferral path reports why the attempt ended without one.
        """
        try:
            if session.prepared_input is not None:
                self.validate_active_prepared_input(session.prepared_input)
            outcome = await self._execute(session, run_trace)
        except IncompatibleActiveRunError as exc:
            raise RunExecutionError(
                "incompatible_answer_run",
                "This Answer Run uses an incompatible model or execution contract. Start a new Run.",
            ) from exc
        except (
            asyncio.CancelledError,
            RunCancellationObserved,
            LeaseLostError,
            RunExecutionError,
        ):
            raise
        except Exception as exc:
            component = classify_transient_dependency(exc, component_hint="providers")
            if component is not None:
                await session.check_cancelled()
                # Clear any durable optimistic draft before releasing the
                # Query permit. The same Run and Agent Session resume from
                # their fenced durable state after the bounded delay.
                await session.reset_output()
                checkpoint, delay = next_dependency_retry(
                    session.checkpoint,
                    component,
                    base_seconds=_DEPENDENCY_DEFER_BASE_SECONDS,
                    max_seconds=_DEPENDENCY_DEFER_MAX_SECONDS,
                )
                self._notify_dependency(self._on_dependency_unavailable, component)
                run_trace.update(output={"outcome": "deferred", "component": component})
                return Deferred(
                    checkpoint=checkpoint,
                    next_attempt_at=self._now() + datetime.timedelta(seconds=delay),
                )
            logger.warning(
                "Answer run %s execution failed",
                session.run_id,
                extra={"error_type": type(exc).__name__},
            )
            message = (
                exc.public_message
                if isinstance(exc, AnswerInputError | InvalidToolConfigurationError)
                and exc.public_message
                else reasoning_control_rejection_message(exc) or "Answer run failed."
            )
            raise RunExecutionError(classify_answer_error(exc), message) from exc
        recovered = dependency_component_from_checkpoint(session.checkpoint)
        if recovered is not None:
            self._notify_dependency(self._on_dependency_recovered, recovered)
        return outcome

    @staticmethod
    def _notify_dependency(
        callback: DependencyStateCallback | None,
        component: DependencyComponent,
    ) -> None:
        if callback is not None:
            callback(component)

    async def _ensure_resolved_mode(
        self,
        session: RunSession,
        request: AnswerRunInput,
        *,
        history: Sequence[Mapping[str, Any]],
    ) -> ResolvedMode:
        record = await self._store.load_routing(owner_id=session.owner_id, run_id=session.run_id)
        if record is None:
            raise RunExecutionError(ROUTING_FAILED, "Routing record is missing.")
        if record.resolved_mode:
            return _require_resolved_mode(record.resolved_mode)
        try:
            decided = decide_resolved_mode(
                requested_mode=record.requested_mode,
                valid_modes=frozenset(record.valid_modes),
            )
        except ValueError as exc:
            raise RunExecutionError(ROUTING_FAILED, "Answer mode routing failed.") from exc
        if decided is None:
            decided = _require_resolved_mode(
                await self._route_with_model(
                    request,
                    history=history,
                    valid_modes=record.valid_modes,
                )
            )
        written = await self._store.resolve(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            resolved_mode=decided,
        )
        return _require_resolved_mode(written or decided)

    async def _route_with_model(
        self,
        request: AnswerRunInput,
        *,
        history: Sequence[Mapping[str, Any]],
        valid_modes: tuple[str, ...],
    ) -> str:
        model, _telemetry = self._models.new_highlight_model()
        router = AnswerModeRouter(model)
        web_sources = self._models.web_sources()
        try:
            return await router.choose(
                query=request.query,
                history=history,
                resources=routing_resources(request.attachments, request.history_attachments),
                web_search=web_sources is not None and web_sources.search_enabled,
                valid_modes=valid_modes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # An invalid mode is the model's answer, not a fault: its message is enough.
            logger.warning(
                "auto router failed; defaulting to research: %s",
                exc,
                extra={"valid_modes": list(valid_modes)},
                exc_info=not isinstance(exc, RoutingFailedError),
            )
            if "research" in valid_modes:
                return "research"
            raise RunExecutionError(ROUTING_FAILED, "Answer mode routing failed.") from exc

    async def _compact_fast_history_if_needed(
        self,
        *,
        host: FastSessionHost,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        accepted_user_entry_id: EntryId,
        targets: Sequence[HistoryProjectionTarget],
        compaction_model_profile: ModelProfile,
        workspace_store: WorkspaceStore | None = None,
    ) -> tuple[PriorTurns, dict[str, Any], bool]:
        """Commit one canonical projection satisfying every reachable Fast serializer."""
        snapshot = await host.snapshot(session_id, selected_lane_id=lane_id)
        coordinator = CompactionCoordinator(
            model_profile=compaction_model_profile,
            context_policy=CONTEXT_POLICY,
            stream_model=self._models.query_tool_model().stream_text,
            exchange_starts_func=host_turn_starts,
        )
        failures: list[dict[str, Any]] = []
        refreshed_after_commit_error = False
        tail_reductions = 0
        for attempt in range(1, _FAST_COMPACTION_ATTEMPT_LIMIT + 1):
            history = _project_fast_history_before_current_user(
                snapshot,
                lane_id=lane_id,
                projection=snapshot.active_projection,
                accepted_user_entry_id=accepted_user_entry_id,
            )
            before = _measure_fast_history_targets(history, targets)
            if _fast_history_targets_fit(before):
                recovered_projection = (
                    refreshed_after_commit_error and snapshot.active_projection is not None
                )
                trace = _durable_fast_compaction_trace(snapshot) if recovered_projection else {}
                return history, trace, recovered_projection

            attempt_trace: dict[str, Any] = {}
            tail = CONTEXT_POLICY.retained_tail_target(compaction_model_profile) // (
                2**tail_reductions
            )
            run_notes = (
                compose_session_notes(await workspace_store.load_inventory())
                if workspace_store is not None
                else []
            )
            try:
                projection, _outcome = await coordinator.prepare(
                    snapshot,
                    tail_target_tokens=tail,
                    accounted_before=max(item["input_tokens"] for item in before.values()),
                    # Fast has no EvidenceLedger at compaction — that happens
                    # before retrieval — and inventing citation ordinals the model
                    # never saw is worse than naming none.
                    run_notes=run_notes,
                    trace=attempt_trace,
                )
                candidate = _project_fast_history_before_current_user(
                    snapshot,
                    lane_id=lane_id,
                    projection=projection,
                    accepted_user_entry_id=accepted_user_entry_id,
                )
                after = _measure_fast_history_targets(candidate, targets)
                if not _fast_history_targets_fit(after):
                    overflowing = ", ".join(
                        name
                        for name, item in after.items()
                        if item["input_tokens"] > item["input_limit_tokens"]
                    )
                    raise ValueError(
                        f"prepared Fast projection still exceeds targets: {overflowing}"
                    )
            except asyncio.CancelledError, SessionLeaseLostError:
                raise
            except Exception as exc:
                failures.append(_fast_compaction_failure(attempt, "prepare", exc))
                tail_reductions += 1
                continue

            try:
                await host.commit_compaction(
                    snapshot=snapshot,
                    session_id=session_id,
                    lane_id=lane_id,
                    reservation_id=reservation_id,
                    projection=projection,
                )
            except asyncio.CancelledError, SessionLeaseLostError:
                raise
            except Exception as exc:
                failures.append(_fast_compaction_failure(attempt, "commit", exc))
                authoritative = await host.snapshot(
                    session_id,
                    selected_lane_id=lane_id,
                    force_reload=True,
                )
                if _active_fast_compaction(authoritative, projection) is not None:
                    recovered = _project_fast_history_before_current_user(
                        authoritative,
                        lane_id=lane_id,
                        projection=authoritative.active_projection,
                        accepted_user_entry_id=accepted_user_entry_id,
                    )
                    return recovered, _durable_fast_compaction_trace(authoritative), True
                _require_fast_turn_reservation(
                    authoritative,
                    lane_id=lane_id,
                    reservation_id=reservation_id,
                    accepted_user_entry_id=accepted_user_entry_id,
                )
                snapshot = authoritative
                refreshed_after_commit_error = True
                continue

            trace = dict(attempt_trace)
            trace["fast_compaction_attempt"] = attempt
            if failures:
                trace["fast_compaction_retries"] = list(failures)
            trace["fast_compaction_targets"] = {
                name: {
                    "input_tokens_before": before[name]["input_tokens"],
                    "input_tokens_after": after[name]["input_tokens"],
                    "input_limit_tokens": after[name]["input_limit_tokens"],
                }
                for name in before
            }
            return candidate, trace, True

        failure_trace = {"compaction_failed": {"attempts": failures}}
        logger.warning(
            "Fast Session compaction failed",
            extra={"trace": failure_trace, "session_id": session_id.value},
        )
        raise RunExecutionError(
            "compaction_failed",
            "Fast Answer could not compact the conversation within model capacity.",
        )

    async def _execute(self, session: RunSession, run_trace: Observation) -> RunExecutionOutcome:
        request = AnswerRunInput.from_prepared_input(session.prepared_input)
        model_profiles = self.validate_pinned_model_profiles(request)
        agent_session_id = SessionId(request.agent_session_id)
        agent_lane_id = LaneId(request.agent_lane_id)
        repository = session.execution.session_repository
        progress_store = session.execution.progress_store
        workspace_store = session.execution.workspace_store
        if repository is None or progress_store is None:
            raise RunExecutionError(
                "run_execution_failed",
                "Answer execution state is unavailable.",
            )
        loaded_snapshot = await repository.load(agent_session_id)
        canonical_snapshot = await _reserve_agent_session_boundary(
            repository,
            session_id=agent_session_id,
            fencing_epoch=session.execution.fencing_epoch,
            previous=loaded_snapshot,
        )
        canonical_snapshot = await self._reclaim_lane(session, canonical_snapshot, agent_lane_id)
        lane_ids = {lane.lane_id for lane in canonical_snapshot.tree.lanes}
        source_lane_id = LaneId(request.source_lane_id) if request.source_lane_id else None
        fork_head: EntryId | None = None
        fork_projection: ContextProjection | None = None
        if request.continuation_kind == "fork":
            # Every Fork resolves its parent's recorded point, whether or not its own
            # Lane already exists: a re-claimed Fork reuses the Lane its first attempt
            # seeded, and skipping the parent checks for an existing Lane would be a
            # way to branch from the tip without ever naming a Fork Point.
            fork_head, fork_projection = await self._resolve_fork_seed(
                session, request, canonical_snapshot
            )
        if fork_head is not None and agent_lane_id not in lane_ids:
            # A Fork's first request is bounded by the recorded Fork Point, not by
            # whatever the source Lane has since grown to.
            selected_snapshot = replace(
                canonical_snapshot,
                selected_lane_id=source_lane_id or LaneId.main(),
            )
            authoritative_messages = project_session_messages(
                canonical_snapshot.tree.graph.ancestry(fork_head),
                fork_projection,
                included_incomplete_host_user_entry_ids=_continued_turns(
                    canonical_snapshot.tree.graph.ancestry(fork_head)
                ),
                # Pre-routing: this fold serves the router. Research folds its own
                # history, and Fast folds its own once its turn is reserved.
                re_readable_handles=False,
            )
        else:
            history_lane_id = (
                agent_lane_id
                if agent_lane_id in lane_ids
                else source_lane_id
                if source_lane_id in lane_ids
                else LaneId.main()
            )
            selected_snapshot = replace(
                canonical_snapshot,
                selected_lane_id=history_lane_id,
            )
            authoritative_messages = project_session_messages(
                canonical_snapshot.tree.ancestry(history_lane_id),
                selected_snapshot.active_projection,
                included_incomplete_host_user_entry_ids=_continued_turns(
                    canonical_snapshot.tree.ancestry(history_lane_id)
                ),
                # Pre-routing: this fold serves the router. Research folds its own
                # history, and Fast folds its own once its turn is reserved.
                re_readable_handles=False,
            )
        has_agent_history = bool(authoritative_messages)
        if has_agent_history:
            routing_history = PriorTurns(authoritative_messages)
        else:
            routing_history = PriorTurns(
                [dict(message) for message in request.history],
                episodic_summary=request.episodic_summary,
            )
        await session.enter_phase("routing")
        resolved_mode = await self._ensure_resolved_mode(
            session,
            request,
            history=routing_history.messages,
        )
        await session.enter_phase("planning")
        # A Session's history is folded where it is sent: Research folds its own, and
        # Fast folds the one its model receives once its turn is reserved, below.
        projected_history = PriorTurns() if has_agent_history else routing_history

        fast_boundaries: FastRunBoundaries | None = None
        fast_session_host: FastSessionHost | None = None
        fast_compaction_trace: dict[str, Any] = {}
        fast_reservation_active = False
        agent_runtime: AgentSessionRuntime[EffectHostUpdate] | None = None
        research_operation_id: OperationId | None = None
        research_plan: AgentRunPlan | None = None
        agent_operations: list[dict[str, Any]] = []

        fetched_buffer = FetchedResourceBuffer()
        if resolved_mode == "research":
            _require_current_child_lifecycle(request.agent_run_plan)

        connection_tools: tuple[AgentTool, ...] = ()
        if resolved_mode == "research" and request.run_connection_bindings:
            if self._connection_tool_resolver is None:
                raise IncompatibleActiveRunError("Research Connection resolver unavailable")
            if any(
                binding.owner_id != session.owner_id for binding in request.run_connection_bindings
            ):
                raise IncompatibleActiveRunError("Research Connection owner mismatch")
            connection_tools = await self._connection_tool_resolver(
                bindings=request.run_connection_bindings,
                claim=ResearchToolClaim(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    worker_id=session.worker_id,
                    fencing_epoch=session.fencing_epoch,
                    check_cancelled=session.check_cancelled,
                ),
            )

        agent_browser = self._run_agent_browser(session, resolved_mode)
        agent_accounts = self._run_agent_accounts(session, agent_browser)
        try:
            run = await self.prepare_orchestrated_run(
                query=request.query,
                agent_effort=request.effort,
                worst_case_memory=_worst_case_recall_block(session.prepared_input),
                workspaces=list(request.workspaces),
                retrieval=request.retrieval,
                filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
                resources=await self._answer_run_resources(request, owner_id=session.owner_id),
                fetched_bytes_sink=_buffered_fetched_bytes_sink(fetched_buffer),
                resolved_mode=resolved_mode,
                resource_identity=request.resource_identity,
                pinned_image_descriptions=request.image_descriptions,
                projected_history=projected_history,
                model_profiles=model_profiles,
                pinned_models=request.pinned_models,
                connection_tools=connection_tools,
                lineage_loader=self._lineage_loader(session, agent_session_id),
                skills=(
                    self._skills_bundle_factory(session.owner_id, request.requested_skill)
                    if self._skills_bundle_factory is not None
                    else None
                ),
                agent_browser=agent_browser,
                agent_accounts=agent_accounts,
            )
        except BaseException:
            if agent_browser is not None:
                await agent_browser.aclose()
            raise
        attachment_snapshots: dict[str, bytes] = {}
        if run.registry is not None:
            attachment_snapshots = await self._restore_registry_fetches(
                run.registry,
                owner_id=session.owner_id,
                run_id=str(session.run_id),
            )
        retained_snapshots = await self._restore_selected_attachments(session, selected_snapshot)
        for resource_id, content in retained_snapshots.items():
            if resource_id in attachment_snapshots and attachment_snapshots[resource_id] != content:
                raise ValueError("retained attachment conflicts with current Run snapshot")
            attachment_snapshots[resource_id] = content
        attachment_admissions = run.orchestrator.admit_durable_attachments(
            authoritative_messages,
            retained_snapshots,
        )
        prepared_input = session.prepared_input or {}
        recall_allowed = bool(prepared_input.get("profile_memory_enabled", True))
        memory_epoch = int(prepared_input.get("profile_memory_epoch") or 0)
        memory_recall_record_count = 0
        memory_recall_chars = 0
        if self._memory is None:
            recall_allowed = False
        elif recall_allowed and self._memory_capability_current is not None:
            recall_allowed = await self._memory_capability_current(
                owner_id=session.owner_id, epoch=memory_epoch
            )
        elif recall_allowed:
            recall_allowed = await _memory_recall_allowed(
                self._memory_recall_enabled, owner_id=session.owner_id
            )
        stream: AsyncIterator[str] | None = None
        subagent_host = run.orchestrator.subagent_host
        cancel_children_on_exit = False
        try:
            try:
                await ensure_session_lane(
                    repository=repository,
                    snapshot=canonical_snapshot,
                    fencing_epoch=session.execution.fencing_epoch,
                    session_id=agent_session_id,
                    lane_id=agent_lane_id,
                    source_lane_id=(
                        LaneId(request.source_lane_id)
                        if request.source_lane_id is not None
                        else None
                    ),
                    head_entry_id=fork_head,
                    projection=fork_projection,
                )
            except RunExecutionError as exc:
                if fork_head is None or exc.kind != "agent_session_conflict":
                    raise
                # A Fork Point that could not open a branch is a stale point, not a
                # Session race: the caller asked for a state that is no longer there.
                raise RunExecutionError("fork_point_stale", exc.public_message) from exc
            bound, notes_binding, notes_plane = await self._claim_run_workspace(
                session=session,
                workspace_store=workspace_store,
                session_id=agent_session_id.value,
            )
            if notes_binding.degraded_reason is not None:
                # Memory is not worth failing a Run over, so the degradation is stated
                # where it can be seen instead: the Run's own trace.
                logger.warning(
                    "Session notes degraded for run %s: %s",
                    session.run_id,
                    notes_binding.degraded_reason,
                )
                run_trace.update(
                    metadata={SESSION_NOTES_DEGRADED_KEY: notes_binding.degraded_reason}
                )
            prepared_early: Any = None
            if resolved_mode == "research":
                if bound is not None:
                    run.orchestrator.bind_workspace(
                        bound,
                        workspace_store,
                        session_notes=notes_binding.records,
                        memory_degradation=notes_binding.degraded_reason,
                    )
                session_id = agent_session_id
                store = self._store
                run.orchestrator.bind_memory(
                    owner_id=session.owner_id,
                    run_id=session.run_id,
                    session_id=session_id.value,
                    store=self._memory_store,
                    enabled=recall_allowed,
                    epoch=memory_epoch,
                    capability_current=self._memory_capability_current,
                )
                # Keep partial's fence keywords explicit: expanding a dict here
                # makes Pyright erase the remaining callback parameters.
                persist_child_runtime = _check_child_write(
                    partial(
                        store.upsert_child_session,
                        worker_id=session.worker_id,
                        fencing_epoch=session.fencing_epoch,
                    )
                )
                claim_child = _check_child_write(
                    partial(
                        store.claim_child_session,
                        worker_id=session.worker_id,
                        fencing_epoch=session.fencing_epoch,
                    )
                )
                renew_child = _check_child_write(
                    partial(
                        store.heartbeat_child_session,
                        worker_id=session.worker_id,
                        fencing_epoch=session.fencing_epoch,
                    )
                )
                run.orchestrator.bind_subagents(
                    parent_session_id=session_id,
                    run_id=session.run_id,
                    owner_id=session.owner_id,
                    persist=persist_child_runtime,
                    load_child=store.load_child_session,
                    list_children=store.list_child_sessions,
                    finish_child=_check_child_write(
                        partial(
                            store.finish_child_session,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        ),
                        false_is_lease_loss=False,
                    ),
                    request_cancel=_check_child_write(
                        partial(
                            store.request_child_cancellation,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    release_children=_check_child_write(
                        partial(
                            store.release_child_sessions,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    steer_child=_check_child_write(
                        partial(
                            store.enqueue_child_control,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    continue_child=_check_child_write(
                        partial(
                            store.continue_child_session,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    reply_guidance=_check_child_write(
                        partial(
                            store.reply_child_guidance,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    create_guidance=_check_child_write(
                        partial(
                            store.create_child_guidance,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        )
                    ),
                    load_guidance=store.load_child_guidance,
                    wait_guidance=store.wait_for_child_guidance,
                    expire_guidance=_check_child_write(
                        partial(
                            store.expire_child_guidance,
                            worker_id=session.worker_id,
                            fencing_epoch=session.fencing_epoch,
                        ),
                        false_is_lease_loss=False,
                    ),
                    list_guidance=store.list_pending_child_guidance,
                    prepare_dispatch=_bound_child_dispatch_preparer(run.orchestrator),
                    run_child=_bound_child_runner(
                        close_agent_page=(
                            agent_browser.close_page if agent_browser is not None else None
                        ),
                        orchestrator=run.orchestrator,
                        telemetry=self._telemetry,
                        repository=repository,
                        session=session,
                        fetched_buffer=fetched_buffer,
                        parent_session_id=session_id,
                        session_notes=notes_plane,
                        restore_child_attachments=lambda child_id, context: (
                            self._restore_child_attachments(session, child_id, context)
                        ),
                        persist_child_runtime=persist_child_runtime,
                        claim_child=claim_child,
                        renew_child=renew_child,
                        load_child=store.load_child_session,
                        control_reader=_fenced_control_reader(store, session),
                        control_ack=_fenced_control_ack(store, session),
                        is_detaching=(
                            (lambda: subagent_host.detaching) if subagent_host is not None else None
                        ),
                    ),
                    check_cancelled=session.check_cancelled,
                )
                # Resolve and compare every accepted execution pin before the
                # first Session mutation. The later post-recall comparison is
                # a second guard immediately before provider/tool effects.
                pin_probe = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, pin_probe.tools)
                snapshot = await repository.refresh(
                    session_id,
                    previous=canonical_snapshot,
                )
                validate_snapshot_refresh(
                    session_id,
                    previous=canonical_snapshot,
                    snapshot=snapshot,
                )
                is_new_session = snapshot.commit_sequence == 0
                memory_text = ""
                if self._memory is not None and recall_allowed:
                    recalled = await self._memory.recall(
                        owner_id=session.owner_id,
                        query=request.query,
                    )
                    memory_text = render_auto_recall(recalled)
                    memory_recall_record_count = len(recalled.records)
                    memory_recall_chars = recalled.content_chars
                run.orchestrator.bind_recall(memory_text)
                prepared_early = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                    attachment_snapshots=attachment_snapshots,
                    attachment_admissions=attachment_admissions,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, prepared_early.tools)
                if not is_new_session:
                    await _restore_durable_evidence(prepared_early, repository, session_id)
                    if run.registry is not None:
                        run.registry.restore_discovered_resources(prepared_early.evidence.contexts)
                plan = request.agent_run_plan
                if plan is None:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "Research answer run is missing its accepted Agent Plan",
                    )
                research_plan = plan

                def validate_research_pins() -> None:
                    self.validate_pinned_model_profiles(request)
                    self.validate_pinned_agent_run_plan(request, prepared_early.tools)

                effects = ResearchRuntimeEffects(
                    orchestrator=run.orchestrator,
                    telemetry=self._telemetry,
                    prepared=prepared_early,
                    session=session,
                    session_id=session_id,
                    fetched_buffer=fetched_buffer,
                    persist_child_intent=persist_child_runtime,
                    validate_pins=validate_research_pins,
                    publish_provider_text=True,
                    session_notes=notes_plane,
                )
                control_reader = _fenced_control_reader(store, session)
                control_ack = _fenced_control_ack(store, session)
                controls = AnswerRuntimeControls(
                    reader=control_reader, acknowledge=control_ack, run_id=session.run_id
                )
                agent_runtime = AgentSessionRuntime(
                    repository=repository,
                    effects=effects,
                    tools=prepared_early.tools,
                    fencing_epoch=session.execution.fencing_epoch,
                    holder=session.run_id,
                    provider_attempt_limit=plan.provider_attempt_limit,
                    event_sink=_answer_runtime_event_sink(session),
                    controls=controls,
                    initial_snapshot=AgentSessionSnapshotSeed(
                        repository=repository,
                        session_id=session_id,
                        snapshot=snapshot,
                    ),
                )
                if subagent_host is not None:
                    await subagent_host.restore_pending()
                accepted = await agent_runtime.accept(
                    session_id=session_id,
                    lane_id=agent_lane_id,
                    idempotency_key=f"answer-run:{session.run_id}",
                    content=request.query,
                    plan=plan,
                )
                accepted_purpose = "research"
                research_operation_id = accepted.operation_id
                await session.enter_phase("researching")
                while True:
                    usage_floor = accepted.cursor.last_entry_sequence
                    # A previous operation's settled view is not this operation's
                    # state: if this one fails, nothing here has settled to record.
                    self._settled_views.pop(session.run_id, None)
                    operation = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=session_id,
                        operation_id=accepted.operation_id,
                    )
                    if not isinstance(operation.state, OperationCompleted):
                        raise _incomplete_operation_error(
                            operation,
                            message=(
                                f"Research Agent operation ended as {operation.state.state_type}."
                            ),
                        )
                    snapshot = operation.context.snapshot
                    # The run's settled state is already in hand here, so the exit
                    # hook records from it rather than loading the whole Session
                    # again at the one moment a long research run is finishing.
                    self._settled_views[session.run_id] = snapshot
                    operation_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry for entry in snapshot.entries if entry.sequence > usage_floor
                            )
                        )
                        or {}
                    )
                    agent_operations.append(
                        {
                            "operation_id": accepted.operation_id.value,
                            "purpose": accepted_purpose,
                            "status": "completed",
                            "usage": operation_usage,
                        }
                    )
                    held = _held_operation(snapshot, agent_lane_id, holder=session.run_id)
                    if held is not None:
                        # An earlier attempt accepted the Operation that still holds
                        # the Lane. It is this Run's program counter and its input may
                        # be gone, so it resumes before any new input is taken.
                        accepted = replace(accepted, operation_id=held, cursor=snapshot.cursor)
                        accepted_purpose = "resumed"
                        research_operation_id = held
                        continue
                    accepted_keys = _accepted_operation_keys(snapshot)
                    next_input = _oldest_pending_input(snapshot, agent_lane_id)
                    next_purpose = "follow_up"
                    command_ids: tuple[str, ...] = ()
                    if next_input is None and controls is not None:
                        commands = await controls.poll(operation.context)
                        if commands:
                            command = commands[0]
                            command_ids = (command.command_id,)
                            if isinstance(command, FollowUpCommand):
                                next_input = (
                                    command.idempotency_key,
                                    command.content,
                                )
                            else:
                                next_input = (command.command_id, command.content)
                    while next_input is None and subagent_host is not None:
                        notifications = await subagent_host.completed_dispatch_notifications(
                            seen=accepted_keys
                        )
                        if notifications:
                            next_input = notifications[0]
                            next_purpose = "child_result"
                            break
                        if not await subagent_host.has_running_children():
                            # A Child can settle between the notification scan
                            # and the running-row check. Re-scan before breaking
                            # so that durable completion cannot be lost in that
                            # subscribe/park window.
                            notifications = await subagent_host.completed_dispatch_notifications(
                                seen=accepted_keys
                            )
                            if notifications:
                                next_input = notifications[0]
                                next_purpose = "child_result"
                            break
                        # Parent completion parks on child lifecycle activity,
                        # not on an all-child gather. A future durable question
                        # can wake this same seam while its Child remains parked.
                        await subagent_host.wait_for_activity()
                    if next_input is None:
                        break
                    validate_research_pins()
                    accepted = await agent_runtime.accept(
                        session_id=session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=next_input[0],
                        content=next_input[1],
                        plan=plan,
                    )
                    accepted_purpose = next_purpose
                    research_operation_id = accepted.operation_id
                    if subagent_host is not None and next_purpose == "child_result":
                        subagent_host.note_notification_accepted(next_input[0])
                    if command_ids and controls is not None:
                        if not await controls.acknowledge(command_ids):
                            raise LeaseLostError
                run.orchestrator.restore_runtime_snapshot(prepared_early, snapshot)
            else:
                memory_text = ""
                if self._memory is not None and recall_allowed:
                    recalled = await self._memory.recall(
                        owner_id=session.owner_id,
                        query=request.query,
                    )
                    memory_text = render_auto_recall(recalled)
                    memory_recall_record_count = len(recalled.records)
                    memory_recall_chars = recalled.content_chars
                run.orchestrator.bind_recall(memory_text)
                fast_boundaries = FastRunBoundaries(
                    session=session,
                    progress=progress_store,
                    run_id=session.run_id,
                    initial_progress_version=session.durable_progress_version,
                    plan={
                        "query": request.query,
                        "workspaces": list(request.workspaces),
                        "top_k": request.retrieval.top_k,
                        "chunk_top_k": request.retrieval.chunk_top_k,
                    },
                )
                fast_session_host = FastSessionHost(
                    repository=repository,
                    initial_snapshot=canonical_snapshot,
                    load_settled_result=fast_boundaries.load_settled_result,
                    fencing_epoch=session.execution.fencing_epoch,
                )
                fast_turn = await fast_session_host.accept(
                    session_id=agent_session_id,
                    lane_id=agent_lane_id,
                    reservation_id=session.run_id,
                    idempotency_key=request.idempotency_fingerprint,
                    content=request.query,
                )
                if fast_turn.progress_advanced:
                    fast_boundaries.observe_session_progress()
                if fast_turn.settled_payload is not None:
                    stored = dict(fast_turn.settled_payload)
                    # Fast commits its own terminal row, so the Fork Point has to be
                    # recorded while the claim still reads as running, from the view
                    # the Host already refreshed rather than a fresh Session load.
                    await self._record_fork_point(
                        session,
                        run_trace,
                        snapshot=await fast_session_host.snapshot(
                            agent_session_id, selected_lane_id=agent_lane_id
                        ),
                    )
                    terminal = await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                    return AlreadyCommittedTerminal(terminal)
                fast_reservation_active = True
                if not fast_turn.created:
                    replay_snapshot = await fast_session_host.snapshot(
                        agent_session_id,
                        selected_lane_id=agent_lane_id,
                    )
                    run.history = _project_fast_history_before_current_user(
                        replay_snapshot,
                        lane_id=agent_lane_id,
                        projection=replay_snapshot.active_projection,
                        accepted_user_entry_id=fast_turn.user_entry_id,
                    )
                    if replay_snapshot.active_projection is not None:
                        fast_compaction_trace.update(
                            _durable_fast_compaction_trace(replay_snapshot)
                        )
                if has_agent_history:
                    (
                        compacted_history,
                        compaction_trace,
                        compacted,
                    ) = await self._compact_fast_history_if_needed(
                        host=fast_session_host,
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                        accepted_user_entry_id=fast_turn.user_entry_id,
                        targets=run.fast_history_targets,
                        compaction_model_profile=model_profiles["query"],
                        workspace_store=workspace_store,
                    )
                    run.history = compacted_history
                    if "fast_compaction_attempt" in compaction_trace:
                        fast_compaction_trace.clear()
                    fast_compaction_trace.update(compaction_trace)
                    if compacted:
                        fast_boundaries.observe_session_progress()
                    # Fast projection rebuilds history from durable Session entries,
                    # so restore the already-budgeted transport pixels on that copy.
                    run.orchestrator.hydrate_admitted_attachments(
                        run.history.messages,
                        retained_snapshots,
                        attachment_admissions,
                    )
                await fast_boundaries.settle_planner()

            async with self._telemetry.observe(
                "generate-answer",
                metadata={
                    "resolved_mode": resolved_mode,
                    "history_turns": len(run.history or []),
                    "query_image_count": run.current_image_count,
                    "semantic_highlights": request.semantic_highlights,
                },
            ) as generation_trace:
                prepared = prepared_early
                stream: AsyncIterator[str] | None = None
                if resolved_mode == "research":
                    if prepared is None:
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Research Runtime lost its prepared Host state.",
                        )
                    contexts, answer_text, already_streamed = run.orchestrator.runtime_answer(
                        prepared
                    )
                    if not already_streamed:
                        await session.enter_phase("generating")
                        await session.emit_token(answer_text)
                        prepared.streamed_terminal_text = answer_text
                else:
                    contexts, stream = await run.orchestrator.answer_stream(
                        request.query,
                        conversation_history=run.history,
                        query_images=run.query_images,
                        boundaries=fast_boundaries,
                    )
                    answer_parts: list[str] = []
                    if stream is not None:
                        async for chunk in stream:
                            answer_parts.append(chunk)
                            await session.emit_token(chunk)
                    answer_text = getattr(stream, "answer", "") or "".join(answer_parts)
                emitted_answer_text = answer_text
                await session.flush_tokens()
                finalized = finalize_answer(answer_text, contexts)
                artifact_root = run.orchestrator.artifact_root()
                artifact_attachments = (
                    _publication_attachments(
                        await self._store.list_artifact_attachments(
                            owner_id=session.owner_id,
                            run_id=session.run_id,
                        )
                    )
                    if artifact_root is not None
                    else ()
                )
                publication = await _publication_plan(
                    artifact_root,
                    answer=finalized.answer,
                    attachments=artifact_attachments,
                    limits=self._settings.publication,
                    contexts=contexts,
                )
                finalized.answer = publication.answer
                if (
                    publication.issues
                    and agent_runtime is not None
                    and research_plan is not None
                    and prepared_early is not None
                ):
                    correction = await agent_runtime.accept(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=f"publication-correction:{session.run_id}",
                        content=publication.correction_feedback(),
                        plan=research_plan,
                    )
                    correction_usage_floor = correction.cursor.last_entry_sequence
                    corrected = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=agent_session_id,
                        operation_id=correction.operation_id,
                    )
                    if not isinstance(corrected.state, OperationCompleted):
                        raise _incomplete_operation_error(
                            corrected,
                            message="Publication correction Agent operation did not complete.",
                        )
                    corrected_snapshot = corrected.context.snapshot
                    correction_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry
                                for entry in corrected_snapshot.entries
                                if entry.sequence > correction_usage_floor
                            )
                        )
                        or {}
                    )
                    correction_record = {
                        "operation_id": correction.operation_id.value,
                        "purpose": "publication_correction",
                        "status": "completed",
                        "usage": correction_usage,
                    }
                    agent_operations.append(correction_record)
                    research_operation_id = correction.operation_id
                    run.orchestrator.restore_runtime_snapshot(
                        prepared_early,
                        corrected_snapshot,
                    )
                    contexts, answer_text, already_streamed = run.orchestrator.runtime_answer(
                        prepared_early
                    )
                    if not already_streamed:
                        await session.enter_phase("generating")
                        await session.emit_token(answer_text)
                        prepared_early.streamed_terminal_text = answer_text
                    emitted_answer_text = answer_text
                    await session.flush_tokens()
                    finalized = finalize_answer(answer_text, contexts)
                    artifact_root = run.orchestrator.artifact_root()
                    artifact_attachments = (
                        _publication_attachments(
                            await self._store.list_artifact_attachments(
                                owner_id=session.owner_id,
                                run_id=session.run_id,
                            )
                        )
                        if artifact_root is not None
                        else ()
                    )
                    publication = await _publication_plan(
                        artifact_root,
                        answer=finalized.answer,
                        attachments=artifact_attachments,
                        limits=self._settings.publication,
                        contexts=contexts,
                    )
                    finalized.answer = publication.answer
                    correction_record["publication_outcome"] = publication.outcome
                if resolved_mode == "research" and finalized.answer != emitted_answer_text:
                    await session.reset_output()
                    if prepared_early is not None:
                        prepared_early.streamed_terminal_text = None
                    await session.enter_phase("generating")
                    await session.emit_token(finalized.answer)
                    await session.flush_tokens()
                    emitted_answer_text = finalized.answer
                if request.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        settings=self._settings.semantic_highlights,
                        model_factory=self._models.new_highlight_model,
                    )
                trace = dict(
                    prepared.trace
                    if resolved_mode == "research" and prepared is not None
                    else getattr(stream, "trace", None) or {}
                )
                trace["agent_shell_confinement"] = confinement_state(self._execution_environment)
                trace["agent_effort"] = _agent_effort_trace(
                    request.effort,
                    resolved_mode,
                    None if prepared is None else prepared.model_profile,
                    self._models.model_settings("query").raw_agentic_reasoning_keys,
                )
                if fast_compaction_trace:
                    trace.update(fast_compaction_trace)
                if agent_runtime is not None and research_operation_id is not None:
                    root_usage: dict[str, int] = {}
                    for item in agent_operations:
                        for key, value in item["usage"].items():
                            root_usage[key] = root_usage.get(key, 0) + int(value)
                    child_usage = await _durable_child_usage(
                        self._store.list_child_sessions,
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                    )
                    inclusive = dict(root_usage)
                    for key, value in child_usage.items():
                        inclusive[key] = inclusive.get(key, 0) + value
                    trace["usage"] = {
                        "usage_details": root_usage,
                        "child_usage_details": child_usage,
                        "inclusive_usage_details": inclusive,
                    }
                    trace["agent_operations"] = list(agent_operations)
                trace.setdefault("agent_operations", list(agent_operations))
                trace["query_image_description_count"] = len(run.image_descriptions)
                trace["memory_recall_record_count"] = memory_recall_record_count
                trace["memory_recall_chars"] = memory_recall_chars
                images = evidence_images_from_sources(finalized.sources, contexts=contexts)
                answer_output = answer_trace_output(
                    finalized.answer,
                    finalized.sources,
                    contexts,
                    capture_sensitive_data=self._telemetry.capture_sensitive_data,
                )
                generation_trace.update(output=answer_output)
                # The root drives the trace table, so the answer a reviewer sees
                # first is written where the answer exists, not derived later
                # from an outcome that may not carry it (Fast commits terminal).
                run_trace.update(output=answer_output)
                publications, artifact_descriptors, artifact_sources = _stage_publications(
                    plan=publication,
                    answer=finalized.answer,
                    session_id=agent_session_id.value,
                    # Only an adopting Run reads a stored view, so with adoption off no
                    # product spends a conversion on one.
                    views=await _publication_views(
                        publication.artifacts,
                        registry=run.registry if self._settings.lineage_adoption else None,
                    ),
                )
                # Fast terminal settlement has no publication channel; Research
                # leaves publication ownership with the coordinator.
                session.pending_publications = publications if fast_boundaries is None else []
                stored = store_answer_result(
                    answer=finalized.answer,
                    link_cards=[
                        card.as_dict() for card in await collect_link_cards(finalized.answer)
                    ],
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    evidence_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                    artifacts=artifact_descriptors,
                    artifact_outcome=publication.outcome,
                    artifact_bindings=publication.artifact_bindings,
                    artifact_sources=artifact_sources,
                )
                if fast_boundaries is not None:
                    await fast_boundaries.settle_retrieval(contexts)
                    await fast_boundaries.stage_result(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                if fast_session_host is not None:
                    fast_commit = await fast_session_host.complete(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                        content=finalized.answer,
                        usage=(
                            trace.get("usage") if isinstance(trace.get("usage"), Mapping) else None
                        ),
                    )
                    fast_reservation_active = False
                    if fast_boundaries is not None and fast_commit is not None:
                        fast_boundaries.observe_session_progress()
                if fast_boundaries is not None and fast_session_host is not None:
                    await self._record_fork_point(
                        session,
                        run_trace,
                        snapshot=await fast_session_host.snapshot(
                            agent_session_id, selected_lane_id=agent_lane_id
                        ),
                    )
                    terminal = await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                    return AlreadyCommittedTerminal(terminal)
                return Succeeded(stored)
        except BaseException as exc:
            cancel_children_on_exit = not isinstance(exc, LeaseLostError) and not (
                isinstance(exc, asyncio.CancelledError) and not session.cancel_requested
            )
            if fast_session_host is not None and fast_reservation_active:
                try:
                    await fast_session_host.fail(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                    )
                except Exception:
                    logger.exception("Failed to clear Fast Host turn reservation")
            raise
        finally:
            if subagent_host is not None:
                try:
                    await subagent_host.stop(cancel=cancel_children_on_exit)
                except Exception:
                    logger.exception("Failed to settle local Child Session tasks")
            await _close_execution_resources(stream, run.registry, agent_browser)

    def _run_agent_browser(
        self, session: RunSession, resolved_mode: ResolvedMode
    ) -> RunAgentBrowser | None:
        """The browser this Research Run leases from the pool on first need, if there is one.

        Fast has no tools and gains no hidden rendering (ADR 0020), so it gets none.
        """
        if resolved_mode != "research" or self._browser is None:
            return None
        return RunAgentBrowser(
            self._browser.provider,
            BrowserHolder(
                owner_id=session.owner_id,
                run_id=session.run_id,
                worker_id=session.worker_id,
                fencing_epoch=session.fencing_epoch,
            ),
            self._browser.settings,
        )

    def _run_agent_accounts(
        self, session: RunSession, agent_browser: RunAgentBrowser | None
    ) -> RunAgentAccounts | None:
        """The Agent Accounts this Research Run registers and logs in with, if it has a browser
        and the deployment allows them. Their Child-scoped accounts live in this object, so
        they go with the Run and there is nothing to close."""
        if agent_browser is None or self._browser is None or self._browser.accounts is None:
            return None
        return RunAgentAccounts(owner_id=session.owner_id, binding=self._browser.accounts)

    async def prepare_orchestrated_run(
        self,
        *,
        query: str,
        workspaces: list[str],
        retrieval: RetrievalOptions,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        pinned_image_descriptions: tuple[str, ...],
        #: A worst case, not the injected text: the capability check that can disable
        #: recall runs later, and reserving a block recall does not use is the safe
        #: direction, while under-reserving spends the difference on chunks.
        worst_case_memory: str = "",
        projected_history: PriorTurns,
        model_profiles: Mapping[ChatModelSelector, ModelProfile],
        resolved_mode: ResolvedMode,
        resource_identity: str,
        skills: SkillsBundle | None = None,
        pinned_models: tuple[PinnedModelProfile, ...],
        agent_effort: ReasoningLevel | None = None,
        connection_tools: tuple[AgentTool, ...] = (),
        lineage_loader: LineageResourceLoader | None = None,
        agent_browser: RunAgentBrowser | None = None,
        agent_accounts: RunAgentAccounts | None = None,
    ) -> OrchestratorRun:
        pinned_model_selectors(pinned_models)
        child_pins = {pin.role: pin for pin in pinned_models}
        history = projected_history
        models = self._capabilities.request_model_context(model_profiles)
        query_profile = models.query
        if not workspaces:
            raise ValueError("an Answer run requires at least one workspace")
        self._warm(workspaces)
        resolved = await self._resources.resolve(
            resources,
            models=models,
            confirm_image_context=self._capabilities.pinned_answer_context,
            fetched_bytes_sink=fetched_bytes_sink,
            resolved_mode=resolved_mode,
            resource_identity=resource_identity,
            page_renderer=agent_browser.render if agent_browser is not None else None,
        )
        try:
            models = resolved.models
            query_profile = models.query
            image_descriptions = list(pinned_image_descriptions)
            fast_history_targets: tuple[HistoryProjectionTarget, ...] = ()
            if resolved_mode == "fast":
                # The calls acceptance fitted, measured again over the durable
                # Session history this Run actually continues.
                fast_history_targets = AnswerHistoryBudget(
                    query=query,
                    profiles=model_profiles,
                    planner_for=self._planning.planner_for,
                    schema=await self._planning.schema_for(workspaces),
                    answer_image_policy=self._capabilities.answer_image_policy,
                    image_descriptions=image_descriptions,
                    current_images=resolved.current_images,
                    memory_text=worst_case_memory,
                ).fast_targets()
            orchestrated_run: OrchestratorRun | None = None

            async def retrieve_knowledge_base(search_query: str) -> RetrievalResult:
                active_history = (
                    orchestrated_run.history if orchestrated_run is not None else history
                )
                return await self._retrieve_result(
                    search_query,
                    workspaces=workspaces,
                    conversation_history=active_history.messages,
                    retrieval=retrieval,
                    filters=filters,
                    query_images=resolved.current_images,
                    image_descriptions=image_descriptions,
                    preserve_query=True if resolved_mode == "research" else None,
                    model_profile=models.extract,
                )

            model_func: Callable[..., Any] | None = None
            stream_model_func: Callable[..., AsyncIterator[str]] | None = None
            if resolved_mode == "research":
                tool_model = self._models.query_tool_model(agentic_reasoning=agent_effort)
                model_func = tool_model
                stream_model_func = tool_model.stream_text

            def resolve_child_model(
                role: str,
            ) -> tuple[Callable[..., Any], Callable[..., AsyncIterator[str]], ModelProfile]:
                if role not in CHAT_MODEL_SELECTORS:
                    raise ValueError(f"unknown child model role: {role}")
                selected_role = cast(ChatModelSelector, role)
                pin = child_pins[role]
                profile = pin.profile
                selected = self._models.tool_model(selected_role)
                if (
                    selected.fingerprint != pin.fingerprint
                    or model_reasoning_settings(selected.settings) != pin.reasoning_settings
                ):
                    raise IncompatibleActiveRunError("child model binding changed after acceptance")
                return selected, selected.stream_text, profile

            resource_reader = None
            admitted_bytes_reader = None
            browser = None
            if resolved.registry is not None:
                resource_reader = make_resource_reader(
                    resolved.registry,
                    CONTEXT_POLICY.read_window_tokens(query_profile),
                    lineage=lineage_loader,
                )
                admitted_bytes_reader = make_admitted_bytes_reader(
                    resolved.registry, lineage=lineage_loader
                )
                if agent_browser is not None:
                    browser = BrowserToolHost(
                        agent_browser, resolved.registry, resource_reader, agent_accounts
                    )
            orchestrator = AnswerOrchestrator(
                synthesizer=self._models.answer_synthesizer(query_profile),
                retrieve_knowledge_base=retrieve_knowledge_base,
                search_web=(
                    resolved.web_sources.search
                    if resolved.web_sources is not None and resolved.web_sources.search_enabled
                    else None
                ),
                model_func=model_func,
                stream_model_func=stream_model_func,
                injected_tools=list(connection_tools) if resolved_mode == "research" else [],
                resource_manifest=resolved.resource_manifest,
                register_web_source=(
                    resolved.registry.register_discovered_link
                    if resolved.registry is not None and resolved.web_sources is not None
                    else None
                ),
                image_budget=resolved.image_budget,
                model_profile=query_profile,
                context_policy=CONTEXT_POLICY,
                publication_limits=self._settings.publication,
                telemetry=self._telemetry,
                search_toolchain=self._search_toolchain,
                resolved_mode=resolved_mode,
                subagent_host=(
                    SubagentHost(
                        guidance_timeout_seconds=self._settings.child_guidance_timeout_seconds,
                        model_guidance=child_model_guidance(pinned_models),
                    )
                    if resolved_mode == "research"
                    else None
                ),
                memory_host=(
                    MemoryHost()
                    if resolved_mode == "research" and self._memory_store is not None
                    else None
                ),
                resource_viewer=(
                    make_resource_viewer(resolved.registry, lineage=lineage_loader)
                    if resolved.registry
                    else None
                ),
                resource_reader=resource_reader,
                admitted_bytes_reader=admitted_bytes_reader,
                browser=browser,
                child_model_resolver=resolve_child_model,
                child_model_identities={
                    pin.role: {
                        **pin.as_json()["fingerprint"],
                        "reasoning_settings": pin.reasoning_settings,
                    }
                    for pin in pinned_models
                },
                skills=skills,
            )
            orchestrated_run = OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=image_descriptions,
                query_images=resolved.query_images,
                history=history,
                fast_history_targets=fast_history_targets,
                current_image_count=resolved.current_image_count,
                workspaces=workspaces,
                registry=resolved.registry,
            )
            return orchestrated_run
        except BaseException:
            if resolved.registry is not None:
                await resolved.registry.aclose()
            raise

    async def _restore_child_attachments(
        self, session: RunSession, child_id: SessionId, context: ChildContextSnapshot
    ) -> dict[str, bytes]:
        references = await self._store.load_child_attachment_occurrences(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            child_session_id=child_id.value,
            context_snapshot=context.canonical_payload(),
        )
        if len(references) != len(context.attachment_occurrences):
            raise ValueError("Child attachment hydration returned incomplete references")
        return await self._load_attachment_blobs(session, references)

    async def _restore_selected_attachments(
        self, session: RunSession, snapshot: Any
    ) -> dict[str, bytes]:
        selection = AttachmentReplaySelection.from_snapshot(snapshot)
        if not selection.occurrences:
            return {}
        references = await self._store.retain_attachment_occurrences(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            selection=selection,
        )
        if len(references) != len(selection.occurrences):
            raise ValueError("selected attachment retention returned incomplete references")
        return await self._load_attachment_blobs(session, references)

    async def _load_attachment_blobs(
        self, session: RunSession, references: Sequence[RunFetchedResource]
    ) -> dict[str, bytes]:
        content_by_resource: dict[str, bytes] = {}
        for reference in references:
            content = b"".join(
                [
                    chunk
                    async for chunk in self._blob_store.stream(
                        owner_id=session.owner_id, digest=reference.digest
                    )
                ]
            )
            if not content or hashlib.sha256(content).hexdigest() != reference.digest:
                raise ValueError("retained attachment Blob is missing or has a mismatched digest")
            if (
                reference.resource_id in content_by_resource
                and content_by_resource[reference.resource_id] != content
            ):
                raise ValueError("selected attachment resource identity conflict")
            content_by_resource[reference.resource_id] = content
        return content_by_resource

    async def _restore_registry_fetches(
        self,
        registry: ResourceRegistry,
        *,
        owner_id: str,
        run_id: str,
    ) -> dict[str, bytes]:
        attachment_snapshots: dict[str, bytes] = {}
        conversions: list[tuple[str, bytes]] = []
        renderings: list[tuple[RunFetchedResource, bytes]] = []
        for resource in await self._store.list_fetched_resources(
            owner_id=owner_id,
            run_id=run_id,
        ):
            pieces: list[bytes] = []
            async for piece in self._blob_store.stream(
                owner_id=owner_id,
                digest=resource.digest,
            ):
                pieces.append(piece)
            if not pieces:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource representation no longer exists.",
                )
            content = b"".join(pieces)
            if hashlib.sha256(content).hexdigest() != resource.digest:
                raise ValueError("durable resource blob digest mismatch")
            capabilities = resource.capabilities
            attachment_snapshots[resource.resource_id] = content
            kind = capabilities.get("resource_kind")
            if kind == "conversion_snapshot":
                conversions.append((resource.source_locator.decode(), content))
                continue
            if kind in {"tool_attachment", "conversion_asset"}:
                continue
            if kind == "web_render":
                # A rendering is restored beside the Web Resource it renders, which a row
                # of the same ordinal may come after.
                renderings.append((resource, content))
                continue
            raw_aliases = _resource_aliases(capabilities)
            if kind == LINEAGE_ADOPTION_KIND:
                # An adopted Resource is durable Run state: it keeps the handle it was
                # recorded under and every earlier handle, is charged but never
                # refused, and its text still reads only through the view restored
                # with it.
                registry.restore_adopted(
                    resource.resource_id,
                    filename=resource.filename,
                    mime_type=resource.mime_type,
                    content=content,
                    aliases=raw_aliases,
                )
                continue
            origin = str(capabilities.get("admission_origin") or "")
            if origin not in {"caller", "search", "agent"}:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                )
            acquisition = str(capabilities.get("acquisition") or "")
            if acquisition in BROWSER_ACQUISITIONS:
                # A capture or a download holds its own bytes, and only the Agent admits one.
                if origin != "agent":
                    raise RunExecutionError(
                        "run_execution_failed",
                        "A durable Web resource catalog entry is invalid.",
                    )
                try:
                    registry.restore_browser_resource(
                        resource_id=resource.resource_id,
                        ordinal=resource.ordinal,
                        filename=resource.filename,
                        mime_type=resource.mime_type,
                        locator=resource.source_locator.decode("utf-8"),
                        content=content,
                        acquisition=acquisition,
                    )
                except (
                    UnicodeError,
                    ValueError,
                    ResourceRegistryError,
                    ResourceStateMismatchError,
                ) as exc:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "A durable Web resource catalog entry is invalid.",
                    ) from exc
                continue
            if acquisition not in {"direct_http", "exa_extract", "tavily_extract"}:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                )
            try:
                registry.restore_fetched_resource(
                    resource_id=resource.resource_id,
                    ordinal=resource.ordinal,
                    filename=resource.filename,
                    mime_type=resource.mime_type,
                    url=resource.source_locator.decode("utf-8"),
                    content=content,
                    admission_origin=origin,  # type: ignore[arg-type]
                    acquisition=acquisition,
                    aliases=tuple(raw_aliases),
                )
            except (UnicodeError, ValueError, ResourceRegistryError) as exc:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable Web resource catalog entry is invalid.",
                ) from exc
        for rendering, content in renderings:
            capabilities = rendering.capabilities
            origin = str(capabilities.get("admission_origin") or "")
            if origin not in {"caller", "search", "agent"} or (
                capabilities.get("acquisition") != BROWSER_RENDER
            ):
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable rendered representation is invalid.",
                )
            try:
                registry.restore_rendered(
                    rendering.source_locator.decode("utf-8"),
                    url=str(capabilities.get("url") or ""),
                    admission_origin=origin,  # type: ignore[arg-type]
                    final_url=str(capabilities.get("final_url") or ""),
                    content=content,
                )
            except (UnicodeError, ValueError, ResourceRegistryError) as exc:
                raise RunExecutionError(
                    "run_execution_failed",
                    "A durable rendered representation is invalid.",
                ) from exc
        for parent_id, encoded in conversions:
            snapshot = ConversionSnapshot.restore(encoded, attachment_snapshots)
            if snapshot.resource_id != parent_id:
                raise ValueError("conversion snapshot parent mismatch")
            # A rendering is held already, with the bytes this view must match. Any other
            # recovery verifies durable source bytes, including lazy inputs; registry
            # adoption separately guards any already-materialized source.
            if not registry.holds_rendered(parent_id):
                original = await registry.materialize(parent_id)
                if hashlib.sha256(original).hexdigest() != snapshot.input_digest:
                    raise ValueError("conversion snapshot input digest mismatch")
            registry.adopt_conversion_snapshot(snapshot)
        return attachment_snapshots

    def _lineage_loader(
        self, session: RunSession, agent_session_id: SessionId
    ) -> LineageResourceLoader | None:
        """Adopt an earlier Run's Resources when this deployment allows it.

        The loader carries this Run's owner and Agent Session, so a row from another
        Session or owner cannot be reached through it even by a forged handle, and
        this Run's claim, so an adoption is recorded only while this worker holds it.
        """
        if not self._settings.lineage_adoption:
            return None
        return RetainedResourceLoader(
            store=self._store,
            blobs=self._blob_store,
            owner_id=session.owner_id,
            session_id=agent_session_id.value,
            run_id=str(session.run_id),
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )

    async def _answer_run_resources(
        self,
        request: AnswerRunInput,
        *,
        owner_id: str,
    ) -> list[ResourceInput] | None:
        if not request.links and not request.attachments and not request.history_attachments:
            return None

        async def load(digest: str) -> bytes:
            pieces: list[bytes] = []
            # Hashed as it streams, so the loop never hashes a whole blob at once.
            streamed = hashlib.sha256()
            async for piece in self._blob_store.stream(owner_id=owner_id, digest=digest):
                streamed.update(piece)
                pieces.append(piece)
            if not pieces:
                raise RunExecutionError(
                    "run_execution_failed",
                    "Answer run attachment bytes no longer exist.",
                )
            # The accepted reference names its bytes by content address; every
            # other restoration path already refuses bytes that do not match.
            if streamed.hexdigest() != digest:
                raise RunExecutionError(
                    "run_execution_failed",
                    "Answer run attachment bytes do not match their accepted digest.",
                )
            return b"".join(pieces)

        def loader(digest: str) -> Callable[[], Awaitable[bytes]]:
            async def read() -> bytes:
                return await load(digest)

            return read

        resources = await build_current_answer_resources(
            links=request.links,
            attachments=request.attachments,
            attachment_loaders=[loader(attachment.digest) for attachment in request.attachments],
        )
        resources.extend(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                loader=loader(attachment.digest),
            )
            for attachment in request.history_attachments
        )
        return resources

    @staticmethod
    def validate_pinned_agent_run_plan(
        request: AnswerRunInput,
        tools: Sequence[AgentTool],
    ) -> None:
        """Reject execution when runtime tools differ from acceptance."""
        pinned = request.agent_run_plan
        if pinned is None:
            raise IncompatibleActiveRunError(
                "Research answer run is missing its accepted Agent Plan"
            )
        actual = AgentRunPlan.from_tools(
            tools,
            model_role="query",
            context_policy_revision=request.context_policy_revision,
        )
        if (
            actual.model_role != pinned.model_role
            or actual.context_policy_revision != pinned.context_policy_revision
            or actual.tools != pinned.tools
        ):
            raise IncompatibleActiveRunError(
                "answer run Agent Plan differs from its accepted tool contracts"
            )

    def validate_pinned_model_profiles(
        self,
        request: AnswerRunInput,
    ) -> dict[ChatModelSelector, ModelProfile]:
        # Capacity is recalculated from the pinned model facts for each segment.
        # A global arithmetic revision is not a reason to strand an otherwise
        # replayable run.
        pinned = {item.role: item for item in request.pinned_models}
        selectors = pinned_model_selectors(request.pinned_models)
        if any(
            pinned[role].reasoning_settings
            != model_reasoning_settings(self._models.model_settings(role))
            for role in selectors
        ):
            raise IncompatibleActiveRunError(
                "answer run targets another model reasoning configuration"
            )
        if request.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise IncompatibleActiveRunError("answer run uses another context policy revision")
        if request.model_catalog_revision != current_model_catalog_revision():
            raise IncompatibleActiveRunError("answer run uses another model catalog revision")
        if any(
            pinned[role].fingerprint != self._model_invocation_fingerprint_for_role(role)
            for role in selectors
        ):
            raise IncompatibleActiveRunError(
                "answer run targets another model invocation configuration"
            )
        return {role: pinned[role].profile for role in selectors}


def _measure_fast_history_targets(
    history: PriorTurns,
    targets: Sequence[HistoryProjectionTarget],
) -> dict[str, dict[str, int]]:
    """Measure the authoritative history with every exact Fast serializer."""
    measured: dict[str, dict[str, int]] = {}
    for target in targets:
        if target.name in measured:
            raise ValueError(f"duplicate Fast history target: {target.name}")
        measured[target.name] = {
            "input_tokens": target.measure_input(
                history.messages,
                history.episodic_summary,
            ),
            "input_limit_tokens": target.acceptance_limit(CONTEXT_POLICY),
        }
    return measured


def _fast_history_targets_fit(measured: Mapping[str, Mapping[str, int]]) -> bool:
    return all(item["input_tokens"] <= item["input_limit_tokens"] for item in measured.values())


def _fast_compaction_failure(
    attempt: int,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "stage": stage,
        "error_type": type(error).__name__,
        "detail": safe_log_text(str(error)),
    }


def _active_fast_compaction(
    snapshot: Any,
    projection: ContextProjection,
) -> CompactionEntry | None:
    """Return the active checkpoint only when it exactly materializes a projection."""
    if snapshot.active_projection != projection:
        return None
    ancestry = snapshot.graph.ancestry()
    latest = ancestry[-1] if ancestry else None
    if not isinstance(latest, CompactionEntry):
        return None
    if (
        latest.projection_id != projection.projection_id
        or latest.summary != projection.summary
        or latest.covered_through_sequence != projection.covered_through_sequence
        or latest.first_retained_sequence != projection.first_retained_sequence
        or latest.covered_through_entry_id != projection.covered_through_entry_id
        or latest.first_retained_entry_id != projection.first_retained_entry_id
        or latest.source_digest != projection.source_digest
    ):
        return None
    return latest


def _durable_fast_compaction_trace(snapshot: Any) -> dict[str, Any]:
    """Reconstruct only coverage facts that the durable checkpoint actually stores."""
    projection = snapshot.active_projection
    if projection is None:
        return {}
    entry = _active_fast_compaction(snapshot, projection)
    if entry is None:
        return {}
    return {
        "fast_compaction_recovered": True,
        "fast_compaction_coverage": {
            "projection_id": projection.projection_id.value,
            "covered_through_sequence": projection.covered_through_sequence,
            "first_retained_sequence": projection.first_retained_sequence,
        },
    }


def _resource_aliases(capabilities: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the earlier durable handles one restored Resource carries."""
    raw = capabilities.get("resource_aliases", [])
    if (
        not isinstance(raw, list)
        or len(raw) > 64
        or not all(isinstance(alias, str) for alias in raw)
    ):
        raise RunExecutionError(
            "run_execution_failed",
            "A durable Web resource catalog entry is invalid.",
        )
    return tuple(raw)


async def _reserve_agent_session_boundary(
    repository: Any,
    *,
    session_id: SessionId,
    fencing_epoch: int,
    previous: Any,
) -> Any:
    """Claim one Host-neutral Session boundary before history-sensitive routing.

    The durable repository binds Session writes to the active Answer run. An
    empty transaction establishes that lease without choosing Fast or Research,
    then the refresh supplies the exact authoritative history boundary that no
    earlier run can advance while this run routes and accepts its operation.
    """
    snapshot = previous
    while True:
        lanes = tuple(snapshot.tree.lanes)
        if lanes:
            lane = next(
                (item for item in lanes if item.lane_id == snapshot.selected_lane_id),
                lanes[0],
            )
            lane_state = lane.state
            if not isinstance(lane_state.value, LaneState):
                raise TypeError("Lane State register has the wrong value type")
            transaction = SessionTransaction.from_parts(
                register_writes=[SetRegister(lane_state.value)],
                expectations=[RegisterExpectation(lane_state.ref, lane_state.sequence)],
            )
        else:
            lane_id = LaneId.main()
            head = LaneHead(lane_id, None)
            state = LaneState(lane_id)
            transaction = SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            )
        outcome = await repository.transact(
            session_id=session_id,
            fencing_epoch=fencing_epoch,
            transaction=transaction,
        )
        if isinstance(outcome, TransactionLeaseLost):
            raise SessionLeaseLostError(session_id.value)
        refreshed = await repository.refresh(session_id, previous=snapshot)
        validate_snapshot_refresh(
            session_id,
            previous=snapshot,
            snapshot=refreshed,
        )
        if isinstance(outcome, RegisterConflict):
            snapshot = refreshed
            continue
        return refreshed


def _require_fast_turn_reservation(
    snapshot: Any,
    *,
    lane_id: LaneId,
    reservation_id: str,
    accepted_user_entry_id: EntryId,
) -> None:
    record = next(
        (
            item
            for item in snapshot.registers
            if item.ref == RegisterRef("host_turn_reservation", lane_id.value)
        ),
        None,
    )
    if record is None or not isinstance(record.value, HostTurnReservation):
        raise OperationConflictError("Fast Host turn reservation is not active")
    reservation = record.value
    if (
        reservation.reservation_id != reservation_id
        or reservation.user_entry_id != accepted_user_entry_id
    ):
        raise OperationConflictError("Fast Host turn reservation identity changed")


def _project_fast_history_before_current_user(
    snapshot: Any,
    *,
    lane_id: LaneId,
    projection: ContextProjection | None,
    accepted_user_entry_id: EntryId,
) -> PriorTurns:
    """Fold a prepared projection while leaving the separately serialized query out."""
    selected = replace(snapshot, selected_lane_id=lane_id)
    ancestry = selected.graph.ancestry()
    semantic_entries = [entry for entry in ancestry if not isinstance(entry, CompactionEntry)]
    latest = semantic_entries[-1] if semantic_entries else None
    if not isinstance(latest, UserMessageEntry) or latest.entry_id != accepted_user_entry_id:
        raise ValueError("Fast compaction lost the current accepted User Entry")
    messages = project_session_messages(
        ancestry,
        projection,
        # A failed or cancelled turn this one continues stays: it is the question
        # being continued. The current query is removed again below.
        included_incomplete_host_user_entry_ids=(
            *_continued_turns(semantic_entries[:-1]),
            accepted_user_entry_id,
        ),
        # Fast composes no tools, so its history must not name re-read calls.
        re_readable_handles=False,
    )
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Fast compaction projection did not retain the current User query")
    return PriorTurns(conversation_messages(messages[:-1]))


async def _close_execution_resources(
    stream: AsyncIterator[str] | None,
    registry: ResourceRegistry | None,
    agent_browser: RunAgentBrowser | None,
) -> None:
    cancellation: asyncio.CancelledError | None = None
    try:
        await aclose_answer_stream(stream)
    except asyncio.CancelledError as exc:
        cancellation = defer_cancellation(cancellation, exc)
    except Exception:
        logger.warning("Failed to close Answer stream", exc_info=True)
    if registry is not None:
        try:
            await registry.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close Answer resource registry", exc_info=True)
    # The registry's pending renders end first, then the browser they used is given back,
    # before the coordinator's terminal write.
    if agent_browser is not None:
        try:
            await agent_browser.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close the Run's Agent Browser", exc_info=True)
    if cancellation is not None:
        raise cancellation


def _require_resolved_mode(value: str | None) -> ResolvedMode:
    if value == "fast" or value == "research":
        return value
    raise RunExecutionError(ROUTING_FAILED, "Answer mode routing failed.")


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _publication_attachments(
    records: Sequence[ArtifactAttachmentUpdate],
) -> tuple[ArtifactAttachment, ...]:
    return tuple(
        ArtifactAttachment(
            relative_path=record.relative_path,
            label=record.label,
            content_digest=record.content_digest,
            size_bytes=record.size_bytes,
            presentation=record.presentation,  # type: ignore[arg-type]
        )
        for record in records
    )


async def _publication_plan(
    root: Path | None,
    *,
    answer: str,
    attachments: Sequence[ArtifactAttachment],
    limits: PublicationLimits,
    contexts: RetrievalContexts,
) -> PublicationPlan:
    if not isinstance(root, Path):
        return PublicationPlan(answer=answer)
    # Validation scans the Agent Workspace and decodes every candidate file;
    # the writer's event loop keeps serving HTTP, SSE and Run leases meanwhile.
    return await run_artifact_check(
        validate_publication,
        root,
        answer=answer,
        attachments=attachments,
        limits=limits,
        contexts=contexts,
    )


async def _publication_views(
    artifacts: Sequence[StagedArtifact], *, registry: ResourceRegistry | None
) -> dict[str, ConversionSnapshot]:
    """Convert each product a later turn reads through a view, as a read of it would.

    An adopting Run never converts what it adopts, so a convertible product is
    readable later only through a view stored with it. The products share one
    conversion's budget: once it is spent no further conversion starts or is
    adopted. A product whose conversion fails, is refused, or finds the budget
    spent is published without one, and a later read of it refuses as it does
    for any convertible document it holds no view for. Without a registry, as when
    lineage adoption is off, no view is built.
    """
    if registry is None:
        return {}
    deadline = time.monotonic() + MAX_CONVERSION_SECONDS
    views: dict[str, ConversionSnapshot] = {}
    for artifact in artifacts:
        if not is_convertible(artifact.filename, artifact.media_type):
            continue
        try:
            views[artifact.resource_id] = await registry.conversion_view(
                artifact.resource_id,
                artifact.content,
                filename=artifact.filename,
                declared_mime=artifact.media_type,
                deadline=deadline,
            )
        except Exception:  # noqa: BLE001 - a product without a view still publishes
            logger.warning(
                "A published product keeps no conversion view",
                extra={"resource_id": artifact.resource_id},
                exc_info=True,
            )
    return views


def _stage_publications(
    *,
    plan: PublicationPlan,
    answer: str,
    session_id: str,
    views: Mapping[str, ConversionSnapshot] | None = None,
) -> tuple[list[PendingPublication], list[dict[str, Any]], dict[str, list[Any]]]:
    """Stage one accepted answer's publications, or reject an answer-less run."""
    if is_empty_answer(answer=answer, has_artifacts=bool(plan.artifacts)):
        raise RunExecutionError("empty_answer", "The run produced no answer.")
    publications: list[PendingPublication] = []
    artifact_sources = {
        resource_id: list(sources) for resource_id, sources in plan.artifact_sources.items()
    }
    descriptors = [dict(item) for item in plan.descriptors]
    labels = {
        str(descriptor.get("resource_id")): str(descriptor.get("label") or "")
        for descriptor in descriptors
    }
    for item in plan.artifacts:
        view = (views or {}).get(item.resource_id)
        publications.append(
            PendingPublication(
                resource_id=item.resource_id,
                reference_kind="published_artifact",
                filename=item.filename,
                mime_type=item.media_type,
                content=item.content,
                session_id=session_id,
                relative_path=item.relative_path,
                presentation=item.presentation,
                label=labels.get(item.resource_id) or item.filename,
                view=(
                    tuple(
                        attached_resource_update(effect, session_id=session_id, intent_id=None)
                        for effect in view.effects()
                    )
                    if view is not None
                    else ()
                ),
            )
        )
    return publications, descriptors, artifact_sources


def answer_trace_output(
    answer: str | None,
    sources: Sequence[Any] | None,
    contexts: RetrievalContexts,
    *,
    capture_sensitive_data: bool = False,
) -> dict[str, Any]:
    """Shape what a pipeline observation reports for one Answer result."""
    output: dict[str, Any] = {
        "answer_len": len(answer or ""),
        "source_count": len(sources or []),
        "context_chunk_count": _context_count(contexts, "chunks"),
    }
    if capture_sensitive_data:
        output["answer"] = answer or ""
    return output


def _worst_case_recall_block(prepared_input: Mapping[str, Any] | None) -> str:
    """Return the standing memory block acceptance reserved for this Run.

    The reservation is acceptance's, read back from what it recorded: an owner
    whose memory was disabled reserved nothing.
    """
    prepared = prepared_input if isinstance(prepared_input, Mapping) else {}
    return reserved_memory_text(enabled=bool(prepared.get("profile_memory_enabled", True)))


def _continued_turns(entries: Sequence[SessionEntry]) -> tuple[EntryId, ...]:
    """Return the host turn a branch ends on while it still has no answer.

    A continuation of a failed or cancelled Fast turn continues that question, and
    its Fold would otherwise drop it: `fold_entries` omits a Host user entry whose
    Assistant never committed, which is right for history and wrong for the one turn
    a continuation is continuing. A Research question carries no acceptance id and is
    already folded, so this names nothing for it.
    """
    if not entries or not isinstance(entries[-1], UserMessageEntry):
        return ()
    last = entries[-1]
    if last.acceptance_id is None:
        return ()
    if any(
        isinstance(entry, AssistantMessageEntry) and entry.acceptance_id == last.acceptance_id
        for entry in entries
    ):
        return ()
    return (last.entry_id,)


def _lane_projection(
    snapshot: AgentSessionSnapshot | None, lane_id: LaneId
) -> ContextProjection | None:
    """Return one Lane's active projection from an exact register snapshot.

    Read by Lane rather than through the snapshot's selected Lane: this records a
    settled fact about one specific Lane, and a selection that drifted for any
    reason would otherwise record another branch's projection at this Run's head.
    A caller with no snapshot simply has none to read.
    """
    if snapshot is None:
        return None
    for record in snapshot.registers:
        value = record.value
        if isinstance(value, ContextProjectionRegister) and value.lane_id == lane_id:
            return value.projection
    return None


__all__ = [
    "AnswerExecutionStore",
    "AnswerExecutor",
    "AnswerExecutorSettings",
    "OrchestratorRun",
    "answer_trace_output",
]


def _agent_effort_trace(
    effort: AnswerEffort | None,
    resolved_mode: ResolvedMode,
    profile: ModelProfile | None,
    raw_reasoning_keys: tuple[str, ...] = (),
) -> dict[str, str | None] | None:
    """Return the chosen agent effort and what the answering model did with it.

    A chosen effort never fails a run: it is applied, clamped to the nearest level the
    model's ladder has, or ignored. Every ending is stated, because the stored effort
    alone misreports a clamped run and says nothing about a choice that never applied —
    a Fast answer enters no agent loop, a role that owns reasoning through raw model
    kwargs takes no typed level, and a model that names no level has nothing to clamp to.
    """
    if effort is None:
        return None
    if resolved_mode != "research":
        return {"requested": effort, "effective": None, "ignored": "fast"}
    if raw_reasoning_keys:
        return {"requested": effort, "effective": None, "ignored": "raw_kwargs"}
    if profile is None:
        return {"requested": effort, "effective": None}
    try:
        resolved = resolve_reasoning(profile.reasoning, effort)
    except ReasoningConfigurationError:
        return {"requested": effort, "effective": None, "ignored": "no_level"}
    return {
        "requested": effort,
        "effective": None if resolved is None else resolved.effective,
    }
