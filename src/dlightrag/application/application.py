# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application lifecycle, capability accessors, and startup ordering."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.errors import ApplicationClosedError
from dlightrag.application.skills import SkillProtection

if TYPE_CHECKING:
    from dlightrag.application.access.control import AccessControl
    from dlightrag.application.agent_accounts import AgentAccountMaintenance, AgentAccounts
    from dlightrag.application.answer_runs.service import AnswerService
    from dlightrag.application.connections.service import Connections
    from dlightrag.application.corpus_admin.mutations import CorpusMutationService
    from dlightrag.application.corpus_admin.service import CorpusAdmin
    from dlightrag.application.health import ApplicationHealth, HealthComponentName
    from dlightrag.application.live_browser.service import LiveBrowserService
    from dlightrag.application.memory.service import MemoryService
    from dlightrag.application.model_catalogue.service import ModelCatalogueAdmin
    from dlightrag.application.policies.service import (
        RuntimePolicies,
    )
    from dlightrag.application.privacy import Privacy
    from dlightrag.application.retrieval.service import RetrievalService
    from dlightrag.application.runs import RunService
    from dlightrag.application.schedules.service import ScheduleService
    from dlightrag.application.web_conversations.service import WebConversationService
    from dlightrag.engine.answer.goal_service import AnswerGoalService
    from dlightrag.engine.dependencies import DependencyComponent


logger = logging.getLogger(__name__)

#: How long startup waits for the cancellation listener before it reports the
#: process not ready and leaves the Run coordinator to start once the listener is.
_LISTENER_STARTUP_WAIT_SECONDS = 30.0
_LATE_COORDINATOR_START = "Run coordinator start after a late cancellation listener"
#: How often the recovery process re-runs the goal scan. A settlement observer's task can be
#: lost with the process that owned it, so startup is not the only scan: the same durable-state
#: pass runs on this cadence, and CAS plus the pending-round row fence duplicate attempts.
_GOAL_RECOVERY_INTERVAL_SECONDS = 60.0
_GOAL_RECOVERY_TASK = "Goal recovery sweep"
_SCHEDULE_LOOK_TASK = "Schedule look"


def _noop_initialize_process(_config: DlightragConfig) -> None:
    return None


async def _noop_close_process() -> None:
    return None


def _noop_start_process() -> None:
    return None


class RunStoreLifecycle(Protocol):
    """The startup-only operational-store capability owned by Application."""

    async def initialize(self, *, validate_only: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class _ApplicationComponents:
    """Every injected collaborator one Application owns before startup."""

    health: ApplicationHealth
    capabilities: Any
    pool: Any
    models: Any
    run_store: RunStoreLifecycle
    web_store: Any
    coordinator: Any
    cancellation_listener: Any
    validate_active_runs: Callable[[], Awaitable[None]]
    corpora: CorpusAdmin
    access_control: AccessControl
    policies: RuntimePolicies
    privacy: Privacy
    retrieval: RetrievalService
    runs: RunService
    answers: AnswerService
    memory: MemoryService
    memory_store: Any
    memory_embedder: Any
    web_conversations: WebConversationService
    agent_accounts: AgentAccounts
    #: The Session goal's durable service; the harness projects it to clients without reading
    #: round Run rows. Every components value must inject it: goal continuation is product state,
    #: not an optional integration.
    goals: AnswerGoalService
    #: The live preview of a Run's leased browser; None only where none was composed.
    live_browser: LiveBrowserService | None = None
    connections: Connections | None = None
    agent_account_maintenance: AgentAccountMaintenance | None = None
    search_toolchain: Any | None = None
    model_catalogue: ModelCatalogueAdmin | None = None
    corpus_mutations: CorpusMutationService | None = None
    initialize_process: Callable[[DlightragConfig], None] = _noop_initialize_process
    close_agent_execution: Callable[[], Awaitable[None]] = _noop_close_process
    start_session_memory_passes: Callable[[], None] = _noop_start_process
    #: Take up the active goals the database holds when this process starts (ADR 0077). Composed
    #: by :mod:`dlightrag._compose`; a components value built without one starts no round.
    recover_goal_rounds: Callable[[], Awaitable[None]] = _noop_close_process
    #: One look for due Schedules (ADR 0103), repeated by the process that hosts Web conversations
    #: while ``deployment.schedules`` is on; a components value built without one starts no look.
    look_schedules: Callable[[], Awaitable[None]] = _noop_close_process
    #: The owner-facing Schedule service; None only where none was composed.
    schedules: ScheduleService | None = None
    close_session_memory_passes: Callable[[], Awaitable[None]] = _noop_close_process
    close_process: Callable[[], Awaitable[None]] = _noop_close_process
    #: The reader content gate, composed by :mod:`dlightrag._compose`. A components value
    #: built without one — a unit test's — must supply its own; the property refuses rather
    #: than falling back to an unprotected gate, because a deployment that forgot to compose
    #: one would otherwise hand readers protected text and say nothing.
    skill_protection: SkillProtection | None = None


class Application:
    """This process's services, their startup order, and their shutdown order."""

    def __init__(
        self,
        config: DlightragConfig,
        components: _ApplicationComponents,
        *,
        web_enabled: bool = False,
    ) -> None:
        self._config = config
        self._components = components
        self._web_enabled = web_enabled
        self._runs_ready = True
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._memory_janitor: asyncio.Task[None] | None = None
        self._goal_recovery_task: asyncio.Task[None] | None = None
        self._schedule_look_task: asyncio.Task[None] | None = None
        self._late_coordinator_start: asyncio.Task[None] | None = None

    @property
    def config(self) -> DlightragConfig:
        """The one configuration this process was composed from."""
        return self._config

    @property
    def health(self) -> ApplicationHealth:
        """Process health, readable after close so shutdown stays diagnosable."""
        return self._components.health

    @property
    def answers(self) -> AnswerService:
        return self._open().answers

    @property
    def skill_protection(self) -> SkillProtection:
        """The process's content fingerprints of the protected Skill documents.

        Every reader-facing surface asks this one object whether a passage is a Skill
        author's own text; the model's own transcript never does (ADR 0061).
        """
        protection = self._open().skill_protection
        if protection is None:
            raise RuntimeError("Reader content protection is not composed")
        return protection

    @property
    def runs(self) -> RunService:
        """The common owner-scoped durable lifecycle service."""
        return self._open().runs

    @property
    def live_browser(self) -> LiveBrowserService:
        """The live preview of a Run's browser, wherever the deployment composes one."""
        service = self._open().live_browser
        if service is None:
            raise RuntimeError("The live browser preview is unavailable")
        return service

    @property
    def connections(self) -> Connections:
        service = self._open().connections
        if service is None:
            raise RuntimeError("Personal Connections are unavailable")
        return service

    @property
    def memory(self) -> MemoryService:
        return self._open().memory

    @property
    def agent_accounts(self) -> AgentAccounts:
        """An owner's Agent Accounts in Settings: their list, the switch for new sign-ups, and
        removal."""
        return self._open().agent_accounts

    @property
    def retrieval(self) -> RetrievalService:
        return self._open().retrieval

    @property
    def model_catalogue(self) -> ModelCatalogueAdmin:
        from dlightrag.application.model_catalogue.service import ModelCatalogueUnavailableError

        catalogue = self._open().model_catalogue
        if catalogue is None:
            raise ModelCatalogueUnavailableError("runtime model catalogue is unavailable")
        return catalogue

    @property
    def corpora(self) -> CorpusAdmin:
        return self._open().corpora

    @property
    def access_control(self) -> AccessControl:
        """Who may do what to which workspace, over this deployment's catalog."""
        return self._open().access_control

    @property
    def policies(self) -> RuntimePolicies:
        """The settings an administrator changes while the deployment runs, read per request."""
        return self._open().policies

    @property
    def privacy(self) -> Privacy:
        """An owner's Conversation Contribution in Settings: whether their conversations may be
        used to evaluate and train models, and since when."""
        return self._open().privacy

    @property
    def corpus_mutations(self) -> CorpusMutationService:
        """The sole product acceptance surface for corpus writes."""
        service = self._open().corpus_mutations
        if service is None:
            raise RuntimeError("Corpus Mutation acceptance is unavailable")
        return service

    @property
    def web_conversations(self) -> WebConversationService:
        """The browser conversation service, guarded by Application lifetime."""
        return self._open().web_conversations

    @property
    def goals(self) -> AnswerGoalService:
        """The Session goal service: the durable phase a round continues under."""
        return self._open().goals

    @property
    def schedules(self) -> ScheduleService:
        """An owner's Schedules: the records of work the clock starts."""
        service = self._open().schedules
        if service is None:
            raise RuntimeError("Schedules are unavailable")
        return service

    def _open(self) -> _ApplicationComponents:
        if self._closed:
            raise ApplicationClosedError()
        return self._components

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def astart(self) -> None:
        """Run the startup sequence in dependency order.

        A schema or accepted-run incompatibility needs operator action, so it
        closes what startup already began and propagates. A transient registry,
        store, recovery, or default-workspace fault only degrades the process.
        """
        components = self._components
        from dlightrag.engine.answer.execution_settings import validate_agent_execution

        validate_agent_execution(
            execution_environment=self._config.answer.agent.execution_environment,
            workspace_root=self._config.answer.agent.workspace_root,
            working_dir=self._config.deployment.working_dir,
        )
        components.initialize_process(self._config)
        try:
            await self._initialize_search_toolchain()
            catalogue_ready = await self._initialize_model_catalogue()
            components.capabilities.resolve_profiles()
            await self._initialize_run_stores()
            if components.connections is not None:
                await components.connections.start(validate_only=self._config.is_reader)
            if components.agent_account_maintenance is not None:
                components.agent_account_maintenance.start()
            await self._validate_active_runs()
            corpora_ready = await self._initialize_corpora()
            # Bind the retrieval-planner LLM; this does not make a model call.
            components.retrieval.planner_for()
            # Vision probes run once at startup, not from health endpoints. A
            # provider interruption degrades capability health without taking
            # down durable Run admission. A workspace is built with the rerank
            # verdict, so that probe settles first; the others are not read while
            # one is built, and overlap its warm-up.
            capabilities = components.capabilities
            rerank_probed = await self._probe_providers(capabilities.probe_rerank)
            chat_probes = asyncio.create_task(self._probe_providers(capabilities.probe_chat))
            try:
                degraded = await self._warm_default_workspace()
            except BaseException:
                chat_probes.cancel()
                await asyncio.gather(chat_probes, return_exceptions=True)
                raise
            chat_probed = await chat_probes
            if rerank_probed and chat_probed:
                components.health.mark_component_healthy("providers")
            self._start_promotion_worker()
            await self._start_run_coordinator()
            await self._initialize_web_conversations()
            await self._recover_goal_rounds()
            self._start_goal_recovery()
            self._start_schedule_look()
            await self._start_memory_janitor()
            # Last, once the stores it reads are initialized: the sweep for Sessions whose pass a
            # restart or a failed model call lost. A store that is not ready only costs it a look.
            components.start_session_memory_passes()
        except BaseException:
            try:
                await self.aclose()
            except BaseException:
                logger.warning("Application cleanup failed during startup", exc_info=True)
            raise
        if degraded == "corpus_storage" or not corpora_ready:
            components.health.mark_component_degraded("corpus_storage")
        else:
            components.health.mark_component_healthy("corpus_storage")
        if degraded is not None:
            components.health.mark_component_degraded(degraded)
            logger.error(
                "DlightRAG started with the default workspace unavailable: %s is unavailable",
                degraded,
            )
        if not catalogue_ready:
            components.health.mark_component_degraded("providers")
        if self._runs_ready:
            components.health.mark_ready()
        else:
            components.health.mark_not_ready()

    async def _initialize_search_toolchain(self) -> None:
        """Fail startup before admission when required path executables are unusable."""
        toolchain = self._components.search_toolchain
        if toolchain is None or self._config.answer.agent.execution_environment == "disabled":
            return
        await toolchain.ensure()
        self._components.health.set_search_toolchain(toolchain.provenance)

    async def _initialize_model_catalogue(self) -> bool:
        """Synchronize the runtime overlay before resolving any model profile."""
        catalogue = self._components.model_catalogue
        if catalogue is None:
            return True
        from dlightrag.application.model_catalogue.service import (
            ModelCatalogueSchemaError,
            ModelCatalogueValidationError,
        )

        try:
            await catalogue.start(validate_only=self._config.is_reader)
        except ModelCatalogueSchemaError, ModelCatalogueValidationError:
            raise
        except Exception as exc:
            self._components.health.mark_component_degraded("providers")
            logger.warning(
                "Runtime model catalogue initialization failed",
                extra={"error_type": type(exc).__name__},
            )
            return False
        return True

    async def _probe_providers(self, probe: Callable[[], Awaitable[None]]) -> bool:
        """Run one startup capability probe; a failure degrades providers, never startup."""
        try:
            await probe()
        except Exception as exc:
            self._components.health.mark_component_degraded("providers")
            logger.warning(
                "Startup model capability probe failed",
                extra={"error_type": type(exc).__name__},
            )
            return False
        return True

    async def _initialize_run_stores(self) -> None:
        """Migrate the durable operational schema, or validate it on a reader.

        Durable runs are startup state, not first-request state: a process whose
        run schema is absent must fail before readiness rather than accept runs
        it cannot durably record. The Web conversation link table is part of the
        same schema because run retention cascades turns through it, so every
        process that owns runs also establishes that table.
        """
        from dlightrag.engine.runtime.errors import RunSchemaError

        from .web_conversations.models import WebConversationSchemaError

        components = self._components
        validate_only = self._config.is_reader
        try:
            await components.run_store.initialize(validate_only=validate_only)
            await components.web_store.initialize(validate_only=validate_only)
            if validate_only:
                await components.memory_store.verify()
            else:
                await components.memory_store.initialize()
        except RunSchemaError, WebConversationSchemaError:
            raise
        except Exception as exc:
            self._runs_ready = False
            components.health.mark_component_degraded("operational_state")
            logger.warning(
                "Run store initialization failed",
                extra={"error_type": type(exc).__name__},
            )
        else:
            components.health.mark_component_healthy("operational_state")

    async def _validate_active_runs(self) -> None:
        """Run the composed operation-owner compatibility checks in startup order."""
        if self._runs_ready:
            await self._components.validate_active_runs()

    async def _initialize_corpora(self) -> bool:
        from .errors import StorageSchemaError

        try:
            await self._components.corpora.initialize()
        except StorageSchemaError:
            raise
        except Exception as exc:
            self._components.health.mark_component_degraded("corpus_storage")
            logger.warning(
                "Workspace registry initialization failed",
                extra={"error_type": type(exc).__name__},
            )
            return False
        return True

    async def _warm_default_workspace(self) -> DependencyComponent | None:
        """Warm the default workspace; return the dependency whose outage degrades startup.

        Building a workspace reaches corpus storage and, when it probes image
        embedding, the model provider: a transient outage of either is no reason
        to refuse to start, while any other failure is.
        """
        from dlightrag.engine.dependencies import classify_transient_dependency
        from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

        from .errors import StorageSchemaError

        workspace = self._config.deployment.workspace_id
        try:
            await self._components.pool.acquire(workspace)
        except CorpusSchemaError as exc:
            raise StorageSchemaError(str(exc)) from exc
        except Exception as exc:
            component = classify_transient_dependency(exc)
            if component is None:
                raise
            logger.warning(
                "Failed to warm the default workspace",
                extra={
                    "workspace": workspace,
                    "error_type": type(exc).__name__,
                    "component": component,
                },
            )
            return component
        self._components.health.mark_component_healthy("corpus_storage")
        logger.info("Warmed up default workspace service '%s'", workspace)
        return None

    def _start_promotion_worker(self) -> None:
        """Start the background hot-workspace promotion worker (writers only)."""
        try:
            self._components.corpora.start_promotion_worker()
        except Exception as exc:
            self._components.health.mark_component_degraded("corpus_storage")
            logger.warning(
                "Promotion worker failed to start",
                extra={"error_type": type(exc).__name__},
            )

    async def _start_run_coordinator(self) -> None:
        """Begin executing accepted runs once startup validated their schema.

        The cancellation listener's initial LISTEN and locally leased rescan
        must succeed before the coordinator claims work. A listener still
        retrying after the startup wait keeps readiness false and never permits
        heartbeat-only claiming; the coordinator starts, and readiness returns,
        in the background once the listener is ready.
        """
        if not self._runs_ready:
            return
        listener = self._components.cancellation_listener
        try:
            await listener.start()
        except Exception as exc:
            self._degrade_runs("cancellation_listener")
            logger.warning(
                "Run cancellation listener failed to start",
                extra={"error_type": type(exc).__name__},
            )
            return
        try:
            await asyncio.wait_for(listener.ready.wait(), timeout=_LISTENER_STARTUP_WAIT_SECONDS)
        except TimeoutError:
            self._degrade_runs("cancellation_listener")
            logger.warning(
                "Run cancellation listener is not ready; the Run coordinator starts once it is"
            )
            self._late_coordinator_start = asyncio.create_task(
                self._start_run_coordinator_once_listener_ready(),
                name=_LATE_COORDINATOR_START,
            )
            return
        await self._start_coordinator()

    async def _start_run_coordinator_once_listener_ready(self) -> None:
        """Finish the Run startup a slow cancellation listener outlasted."""
        await self._components.cancellation_listener.ready.wait()
        if self._closed or not await self._start_coordinator():
            return
        self._runs_ready = True
        await self._initialize_web_conversations()
        await self._recover_goal_rounds()
        self._start_goal_recovery()
        self._start_schedule_look()
        self._components.health.mark_ready()
        logger.info("Run cancellation listener is ready; the Run coordinator started")

    async def _start_coordinator(self) -> bool:
        """Start claiming work behind a ready cancellation listener."""
        self._components.health.mark_component_healthy("cancellation_listener")
        try:
            await self._components.coordinator.start()
        except Exception as exc:
            self._degrade_runs("run_coordinator")
            logger.warning(
                "Run coordinator failed to start",
                extra={"error_type": type(exc).__name__},
            )
            return False
        self._components.health.mark_component_healthy("run_coordinator")
        return True

    def _degrade_runs(self, component: HealthComponentName) -> None:
        self._runs_ready = False
        self._components.health.mark_component_degraded(component)
        self._components.health.mark_component_degraded("operational_state")

    async def _stop_late_coordinator_start(self) -> None:
        task = self._late_coordinator_start
        self._late_coordinator_start = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _recover_goal_rounds(self) -> None:
        """Take up the goals a previous process left active (ADR 0077).

        Runs after the coordinator exists, so a round the driver starts has something to admit it,
        and after the Web conversations are initialized, so the round can become their next turn.
        Only a process whose deployment setting asks for it runs the startup scan; the durable row
        and its compare-and-set fence the rest, so a second process cannot duplicate a round. A
        failure here costs the continuation, never the boot: the goal stays readable and the next
        settlement or human turn continues it.
        """
        if not self._config.deployment.goal_recovery:
            return
        if not self._runs_ready:
            return
        try:
            await self._components.recover_goal_rounds()
        except Exception:
            logger.warning("Goal rounds could not be recovered at startup", exc_info=True)

    def _start_goal_recovery(self) -> None:
        """Start the periodic goal scan in the one process that owns startup recovery.

        Only a process whose ``deployment.goal_recovery`` is true scans; the scan itself reuses
        the startup pass and CAS + the pending-round row fence a second process's attempts. A
        components value built without a recovery callback (a unit test's) has nothing to repeat,
        so no task starts there.
        """
        if not self._config.deployment.goal_recovery or not self._runs_ready or self._closed:
            return
        if self._components.recover_goal_rounds is _noop_close_process:
            return
        if self._goal_recovery_task is None:
            self._goal_recovery_task = asyncio.create_task(
                self._goal_recovery_forever(), name=_GOAL_RECOVERY_TASK
            )

    async def _goal_recovery_forever(self) -> None:
        """Re-run the goal recovery scan on its cadence; failures cost one pass, never the loop."""
        while True:
            await asyncio.sleep(_GOAL_RECOVERY_INTERVAL_SECONDS)
            try:
                await self._components.recover_goal_rounds()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Periodic goal recovery failed", exc_info=True)

    async def _stop_goal_recovery(self) -> None:
        task = self._goal_recovery_task
        self._goal_recovery_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    def _start_schedule_look(self) -> None:
        """Start looking for due Schedules, where a fire can become a conversation turn.

        Only a process that hosts Web conversations looks, and only while ``deployment.schedules``
        is on and its runs are ready. Any number of processes may: the database fences each slot,
        so the MCP-only service turns the setting off to spare itself the queries, not for safety.
        """
        if not self._config.deployment.schedules or not self._web_enabled:
            return
        if not self._runs_ready or self._closed:
            return
        if self._components.look_schedules is _noop_close_process:
            return
        if self._schedule_look_task is None:
            self._schedule_look_task = asyncio.create_task(
                self._schedule_look_forever(), name=_SCHEDULE_LOOK_TASK
            )

    async def _schedule_look_forever(self) -> None:
        """Look on the configured cadence; a failed pass costs that pass, never the loop."""
        while True:
            try:
                await self._components.look_schedules()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Schedule look failed", exc_info=True)
            await asyncio.sleep(self._config.schedules.look_interval_seconds)

    async def _stop_schedule_look(self) -> None:
        task = self._schedule_look_task
        self._schedule_look_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _initialize_web_conversations(self) -> None:
        if not self._web_enabled:
            return
        if not self._runs_ready:
            return
        await self._components.web_conversations.start_retention()

    async def _start_memory_janitor(self) -> None:
        try:
            await self._components.memory.purge_expired()
        except Exception:
            logger.warning("Memory retention failed", exc_info=True)
        if self._memory_janitor is None:
            self._memory_janitor = asyncio.create_task(self._purge_memory_forever())

    async def _purge_memory_forever(self) -> None:
        from dlightrag.engine.runtime.coordinator import MAINTENANCE_SECONDS

        while True:
            await asyncio.sleep(MAINTENANCE_SECONDS)
            try:
                await self._components.memory.purge_expired()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Memory retention failed", exc_info=True)

    async def _stop_memory_janitor(self) -> None:
        task = self._memory_janitor
        self._memory_janitor = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close every owned collaborator in dependency order. Idempotent.

        An ordinary close failure is logged so later cleanup still runs, while
        cancellation is deferred and re-raised once nothing is left to close.
        """
        from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup

        close_task = self._close_task
        if close_task is None:
            self._closed = True
            self._components.health.mark_closed()
            close_task = asyncio.create_task(self._close_components())
            self._close_task = close_task
        await await_shared_cleanup(close_task)

    async def _close_components(self) -> None:
        """Run the one shared shutdown sequence every close caller joins."""
        from dlightrag.engine.rag.workspace.lifecycle import defer_cancellation

        components = self._components
        cancellation: asyncio.CancelledError | None = None
        for label, close in (
            (
                "Connection refresh",
                components.connections.stop_refresh
                if components.connections is not None
                else _noop_close_process,
            ),
            (
                "live browser sessions",
                components.live_browser.aclose
                if components.live_browser is not None
                else _noop_close_process,
            ),
            (
                "Agent Account maintenance",
                components.agent_account_maintenance.aclose
                if components.agent_account_maintenance is not None
                else _noop_close_process,
            ),
            ("memory janitor", self._stop_memory_janitor),
            ("goal recovery sweep", self._stop_goal_recovery),
            ("schedule look", self._stop_schedule_look),
            ("Session Memory Passes", components.close_session_memory_passes),
            ("the late Run coordinator start", self._stop_late_coordinator_start),
            ("corpus admin promotion worker", components.corpora.aclose),
            ("the durable run coordinator", components.coordinator.aclose),
            ("Agent execution", components.close_agent_execution),
            (
                "Connections",
                components.connections.aclose
                if components.connections is not None
                else _noop_close_process,
            ),
            ("the cancellation listener", components.cancellation_listener.aclose),
            ("Web conversation retention", components.web_conversations.aclose),
            ("the Retrieval service", components.retrieval.aclose),
            (
                "the runtime model catalogue",
                components.model_catalogue.aclose
                if components.model_catalogue is not None
                else _noop_close_process,
            ),
            ("the Answer model runtime", components.models.aclose),
            ("the workspace pool", components.pool.aclose),
            ("the memory embedder", components.memory_embedder.aclose),
            ("process resources", components.close_process),
        ):
            try:
                await close()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close %s", label, exc_info=True)
        if cancellation is not None:
            raise cancellation


__all__ = [
    "Application",
    "ApplicationClosedError",
]
