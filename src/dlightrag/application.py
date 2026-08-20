# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one local composition root: build this process, start it, and close it.

Composition is eager. ``Application.acreate`` resolves configuration once,
constructs every collaborator, and then runs the startup sequence in dependency
order, so no request ever races a half-built store or an unstarted coordinator.
Shutdown reverses that order and is part of the interface: an owner is always
closed before the resources it borrows.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dlightrag.config import DlightragConfig, get_config

if TYPE_CHECKING:
    from dlightrag_ai.fingerprints import ModelFingerprint
    from dlightrag_ai.settings import ModelRole
    from dlightrag_rag.pool import WorkspacePool

    from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
    from dlightrag.adapters.postgres.memory import PGAnswerMemoryStore
    from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore
    from dlightrag.answer.capabilities import AnswerCapabilityCoordinator
    from dlightrag.answer.model_runtime import AnswerModelRuntime
    from dlightrag.health import ApplicationHealth
    from dlightrag.runtime import RunCoordinator
    from dlightrag.runtime.cancellation import RunCancellationListener
    from dlightrag.services.answers import AnswerService
    from dlightrag.services.corpora import CorpusAdmin
    from dlightrag.services.memory import MemoryService
    from dlightrag.services.retrieval import RetrievalService
    from dlightrag.web.conversations import WebConversationService

logger = logging.getLogger(__name__)


class ApplicationClosedError(RuntimeError):
    """Raised when a closed Application is asked for one of its services."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "Application is shutting down"
        super().__init__(self.detail)


@dataclass(frozen=True, slots=True)
class _ApplicationComponents:
    """Every collaborator one Application owns, composed before startup."""

    health: ApplicationHealth
    capabilities: AnswerCapabilityCoordinator
    pool: WorkspacePool
    models: AnswerModelRuntime
    run_store: PGAnswerRunStore
    web_store: PGWebConversationStore
    coordinator: RunCoordinator
    cancellation_listener: RunCancellationListener
    corpora: CorpusAdmin
    retrieval: RetrievalService
    answers: AnswerService
    memory: MemoryService
    memory_store: PGAnswerMemoryStore
    web_conversations: WebConversationService


def _actionable_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    text = str(exc).lower()
    if "connection" in text and ("refused" in text or "reset" in text):
        return f"{msg}. Check DLIGHTRAG_POSTGRES_* or model server settings."
    if "asyncpg" in type(exc).__module__:
        return f"{msg}. Check DLIGHTRAG_POSTGRES_HOST/PORT/USER/PASSWORD."
    if "timeout" in text or "timed out" in text:
        return f"{msg}. Service may be overloaded or unreachable."
    if "authentication" in text or "password" in text or "denied" in text:
        return f"{msg}. Check API keys or database credentials."
    return msg


def _compose(config: DlightragConfig) -> _ApplicationComponents:
    """Construct this process's collaborators from one resolved configuration."""
    from dlightrag_ai.fingerprints import model_fingerprint
    from dlightrag_ai.media import MAX_DECODE_IMAGE_PIXELS
    from dlightrag_ai.scheduler import ModelScheduler
    from dlightrag_ai.telemetry import safe_log_text
    from dlightrag_ai.vision import ModelImageCapabilities
    from dlightrag_rag.ingestion.jobs import IngestJobCoordinator
    from dlightrag_rag.pool import WorkspacePool
    from dlightrag_rag.ports import CorpusSchemaError, WorkspaceCorpusBackend
    from dlightrag_rag.settings import RagSettings
    from dlightrag_rag.source_download import SourceDownloadService
    from dlightrag_rag.workspace_rag import WorkspaceRag
    from dlightrag_rag.workspaces import normalize_workspace
    from PIL import Image

    from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
    from dlightrag.adapters.postgres.corpus import PGCorpusBackendFactory, PGReadinessProbe
    from dlightrag.adapters.postgres.file_panel import PGFilePanelStore
    from dlightrag.adapters.postgres.memory import PGAnswerMemoryStore
    from dlightrag.adapters.postgres.pg_metadata_index import PGMetadataIndex
    from dlightrag.adapters.postgres.retrieval import PGWorkspaceSchemaLookup
    from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore
    from dlightrag.adapters.retrieval import (
        AnswerQueryImagePreparer,
        AnswerRetrievalProjector,
    )
    from dlightrag.answer.capabilities import (
        AnswerCapabilityCoordinator,
        AnswerCapabilityView,
    )
    from dlightrag.answer.executor import AnswerExecutor, AnswerResourceResolver
    from dlightrag.answer.model_runtime import AnswerModelRuntime
    from dlightrag.health import ApplicationHealth
    from dlightrag.model_settings import (
        answer_capability_settings,
        answer_executor_settings,
        answer_model_runtime_settings,
        answer_resource_settings,
        corpus_admin_settings,
        model_profile_for_role,
        model_settings_for_role,
        rag_settings,
        rerank_scoring_model_settings,
        retrieval_settings,
    )
    from dlightrag.observability import LangfuseTelemetry
    from dlightrag.runtime import RunCoordinator
    from dlightrag.services.answers import AnswerService
    from dlightrag.services.corpora import CorpusAdmin
    from dlightrag.services.memory import MemoryService
    from dlightrag.services.retrieval import RetrievalPlannerRuntime, RetrievalService
    from dlightrag.web.conversations import WebConversationService

    # Large document scans are DlightRAG product policy, not an AI package import side effect.
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS
    health = ApplicationHealth(readiness_probe=PGReadinessProbe(config))
    scheduler = ModelScheduler(max_concurrency=config.max_async)
    telemetry = LangfuseTelemetry()
    corpus_backend = PGCorpusBackendFactory(config).create()

    # Image capability is role-specific but cached per resolved model config,
    # so roles that share one model share one probe.
    capabilities = AnswerCapabilityCoordinator(
        settings=answer_capability_settings(config),
        profile_for_role=lambda role: model_profile_for_role(config, role),
        model_settings_for_role=lambda role: model_settings_for_role(config, role),
        rerank_model_settings=lambda: rerank_scoring_model_settings(config),
        image_capabilities=ModelImageCapabilities(scheduler=scheduler, telemetry=telemetry),
        on_answer_capability=health.set_answer_image_capability,
    )

    def workspace_config(workspace_id: str) -> DlightragConfig:
        return config.model_copy(update={"workspace": workspace_id})

    async def build_workspace(
        workspace_id: str,
        settings: RagSettings,
        backend: WorkspaceCorpusBackend,
    ) -> WorkspaceRag:
        try:
            runtime = await WorkspaceRag.acreate(
                workspace_id=workspace_id,
                settings=settings,
                backend=backend,
                scheduler=scheduler,
                telemetry=LangfuseTelemetry(),
                rerank_supports_vision=capabilities.rerank_supports_vision,
            )
        except CorpusSchemaError:
            raise
        except Exception as exc:
            raise RuntimeError(_actionable_error(exc)) from exc
        logger.info("Created WorkspaceRag for workspace '%s'", safe_log_text(workspace_id))
        return runtime

    pool = WorkspacePool(
        settings_for=lambda workspace: rag_settings(workspace_config(workspace)),
        backend_for=lambda workspace: PGCorpusBackendFactory(workspace_config(workspace)).create(),
        build=build_workspace,
    )

    source_download_settings = rag_settings(config)
    corpora = CorpusAdmin(
        settings=corpus_admin_settings(config),
        pool=pool,
        maintenance=corpus_backend.maintenance,
        ingest_jobs=IngestJobCoordinator(
            lambda workspace: pool.acquire(workspace),
            input_root=config.input_dir_path,
            store=corpus_backend.ingest_jobs,
        ),
        file_panel=PGFilePanelStore(),
        source_download_for=lambda workspace: SourceDownloadService(
            settings=source_download_settings,
            metadata_index=PGMetadataIndex(workspace=workspace),
            workspace_id=workspace,
        ),
    )

    models = AnswerModelRuntime(
        settings=answer_model_runtime_settings(config),
        scheduler=scheduler,
        telemetry=telemetry,
        answer_image_policy=capabilities.answer_image_policy,
        vlm_image_policy=capabilities.vlm_image_policy,
        vlm_profile=lambda: capabilities.model_profile("vlm"),
    )
    resources = AnswerResourceResolver(
        settings=answer_resource_settings(config),
        models=models,
        capabilities=capabilities,
    )
    retrieval = RetrievalService(
        pool=pool,
        planners=RetrievalPlannerRuntime(
            model_settings=model_settings_for_role(config, "extract"),
            default_profile=lambda: capabilities.model_profile("extract"),
            scheduler=scheduler,
            telemetry=telemetry,
        ),
        schema_lookup=PGWorkspaceSchemaLookup(
            default_workspace=normalize_workspace(config.workspace)
        ),
        image_preparer=AnswerQueryImagePreparer(capabilities=capabilities, models=models),
        projector=AnswerRetrievalProjector(),
        settings=retrieval_settings(config),
        telemetry=telemetry,
    )

    run_store = PGAnswerRunStore()
    memory_store = PGAnswerMemoryStore()
    coordinator = RunCoordinator(
        store=run_store,
        executor=AnswerExecutor(
            store=run_store,
            pool=pool,
            retrieve=retrieval.retrieve_result,
            models=models,
            capabilities=capabilities,
            resources=resources,
            settings=answer_executor_settings(config),
            telemetry=telemetry,
            execution_environment=config.agent.execution_environment,
            workspace_root=config.agent.workspace_root,
            working_dir=config.working_dir,
            memory_store=memory_store,
        ),
        answer_worker_concurrency=config.runtime.answer_worker_concurrency,
    )

    async def _cancel_local(owner: str, run_id: str) -> None:
        coordinator.cancel_local(owner, run_id)

    cancellation_listener = run_store.build_cancellation_listener(
        worker_id=coordinator.worker_id,
        on_cancel=_cancel_local,
    )

    answers = AnswerService(
        store=run_store,
        coordinator=coordinator,
        retrieval=retrieval,
        capabilities=capabilities,
        capability_view=AnswerCapabilityView(capabilities),
        models=models,
        resources=resources,
        model_fingerprint_for_role=lambda role: model_fingerprint(
            model_settings_for_role(config, role)
        ),
    )
    memory = MemoryService(memory_store)
    web_store = PGWebConversationStore(run_store=run_store)
    return _ApplicationComponents(
        health=health,
        capabilities=capabilities,
        pool=pool,
        models=models,
        run_store=run_store,
        web_store=web_store,
        coordinator=coordinator,
        cancellation_listener=cancellation_listener,
        corpora=corpora,
        retrieval=retrieval,
        answers=answers,
        memory=memory,
        memory_store=memory_store,
        web_conversations=WebConversationService(
            store=web_store,
            answers=answers,
            max_turns=config.web_conversations.max_turns,
            ttl_days=config.web_conversations.ttl_days,
            max_attachments=config.answer.max_attachments,
        ),
    )


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

    @classmethod
    async def acreate(
        cls,
        config: DlightragConfig | None = None,
        *,
        web_enabled: bool = False,
    ) -> Application:
        """Resolve configuration once, compose every service, and start them."""
        resolved = config or get_config()
        application = cls(resolved, _compose(resolved), web_enabled=web_enabled)
        await application.astart()
        return application

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
    def memory(self) -> MemoryService:
        return self._open().memory

    @property
    def retrieval(self) -> RetrievalService:
        return self._open().retrieval

    @property
    def corpora(self) -> CorpusAdmin:
        return self._open().corpora

    @property
    def web_conversations(self) -> WebConversationService:
        """The browser conversation service, guarded by Application lifetime."""
        return self._open().web_conversations

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
        from dlightrag.adapters.postgres._pool import pg_pool
        from dlightrag.observability import init_tracing

        components = self._components
        components.capabilities.resolve_profiles()
        components.capabilities.validate_startup()
        from dlightrag.answer.execution_settings import validate_agent_execution

        self._workspace_root = validate_agent_execution(
            execution_environment=self._config.agent.execution_environment,
            workspace_root=self._config.agent.workspace_root,
            working_dir=self._config.working_dir,
        )
        init_tracing(self._config)
        # Bind the process-wide domain pool to this config so the endpoint and
        # role cannot silently diverge from a caller-supplied SDK config that
        # never called set_config().
        pg_pool.bind(self._config)
        try:
            await self._initialize_run_stores()
            await self._validate_active_runs()
            corpora_ready = await self._initialize_corpora()
            # Bind the retrieval-planner LLM; this does not make a model call.
            components.retrieval.planner_for()
            # Vision probes run once at startup, not per workspace.
            await components.capabilities.probe_all()
            degraded = await self._warm_default_workspace()
            recovery_ready = await self._start_ingest_recovery()
            await self._start_run_coordinator()
            await self._initialize_web_conversations()
            await self._start_memory_janitor()
        except BaseException:
            try:
                await self.aclose()
            except BaseException:
                logger.warning("Application cleanup failed during startup", exc_info=True)
            raise
        if degraded is not None:
            warning = f"Default workspace init failed: {degraded}"
            components.health.add_warning(warning)
            logger.error("DlightRAG started in degraded mode: %s", degraded)
        if not self._runs_ready:
            components.health.add_warning("Answer runtime unavailable")
        if self._runs_ready and corpora_ready and recovery_ready and degraded is None:
            components.health.mark_ready()
        else:
            components.health.mark_degraded()

    async def _initialize_run_stores(self) -> None:
        """Migrate the durable operational schema, or validate it on a reader.

        Answer runs are startup state, not first-request state: a process whose
        run schema is absent must fail before readiness rather than accept runs
        it cannot durably record. The Web conversation link table is part of the
        same schema because run retention exempts conversation-linked runs, so
        every process that owns runs also establishes that table.
        """
        from dlightrag.runtime import RunSchemaError
        from dlightrag.web.conversation_models import WebConversationSchemaError

        components = self._components
        validate_only = self._config.is_reader
        try:
            await components.run_store.initialize(validate_only=validate_only)
            await components.web_store.initialize(validate_only=validate_only)
        except RunSchemaError, WebConversationSchemaError:
            raise
        except Exception as exc:
            self._runs_ready = False
            components.health.add_warning("Answer run store unavailable")
            logger.warning("Answer run store initialization failed: %s", exc)

    async def _validate_active_runs(self) -> None:
        """Reject a rolling deployment that cannot execute already accepted inputs."""
        from dlightrag_ai.fingerprints import model_fingerprint
        from dlightrag_ai.settings import MODEL_ROLE_NAMES

        from dlightrag.model_settings import model_settings_for_role

        if not self._runs_ready:
            return
        current_fingerprints: dict[ModelRole, ModelFingerprint] = {
            role: model_fingerprint(model_settings_for_role(self._config, role))
            for role in MODEL_ROLE_NAMES
        }
        for requirement in await self._components.run_store.list_active_run_requirements():
            _require_compatible_run(requirement, current_fingerprints)

    async def _initialize_corpora(self) -> bool:
        from dlightrag.services.errors import StorageSchemaError

        try:
            await self._components.corpora.initialize()
        except StorageSchemaError:
            raise
        except Exception as exc:
            self._components.health.add_warning("Workspace registry unavailable")
            logger.warning("Workspace registry initialization failed: %s", exc)
            return False
        return True

    async def _warm_default_workspace(self) -> str | None:
        """Warm the default workspace; return the detail that degrades startup."""
        from dlightrag_rag.pool import WorkspaceUnavailableError
        from dlightrag_rag.ports import CorpusSchemaError
        from dlightrag_rag.workspaces import normalize_workspace

        from dlightrag.services.errors import CorpusUnavailableError, StorageSchemaError

        workspace = normalize_workspace(self._config.workspace)
        try:
            await self._components.pool.acquire(workspace)
        except CorpusSchemaError as exc:
            raise StorageSchemaError(str(exc)) from exc
        except WorkspaceUnavailableError as exc:
            raise CorpusUnavailableError(str(exc)) from exc
        except Exception as exc:
            logger.warning("Failed to warm up default workspace '%s'", workspace, exc_info=True)
            return str(getattr(exc, "detail", None) or exc) or "unknown"
        logger.info("Warmed up default workspace service '%s'", workspace)
        return None

    async def _start_ingest_recovery(self) -> bool:
        try:
            await self._components.corpora.start_recovery()
        except Exception:
            self._components.health.add_warning("Ingest job recovery unavailable")
            logger.warning("Ingest job recovery initialization failed", exc_info=True)
            return False
        return True

    async def _start_run_coordinator(self) -> None:
        """Begin executing accepted runs once startup validated their schema.

        The cancellation listener's initial LISTEN and locally leased rescan
        must succeed before the coordinator claims work (M3-D41); connection
        failure keeps readiness false while the listener retries and never
        permits heartbeat-only claiming.
        """
        if not self._runs_ready:
            return
        try:
            await self._components.cancellation_listener.start()
            await asyncio.wait_for(
                self._components.cancellation_listener.ready.wait(), timeout=30.0
            )
        except Exception as exc:
            self._runs_ready = False
            self._components.health.add_warning("Answer runtime unavailable")
            logger.warning("Answer cancellation listener failed to start: %s", exc)
            return
        try:
            await self._components.coordinator.start()
        except Exception as exc:
            self._runs_ready = False
            self._components.health.add_warning("Answer runtime unavailable")
            logger.warning("Answer runtime failed to start: %s", exc)

    async def _initialize_web_conversations(self) -> None:
        if not self._web_enabled:
            return
        if not self._runs_ready:
            self._components.health.add_warning("Web conversations unavailable")
            return
        await self._components.web_conversations.start_retention()

    async def _start_memory_janitor(self) -> None:
        purge = getattr(self._components.memory, "purge_expired", None)
        if purge is None:
            return
        try:
            await purge()
        except Exception:
            logger.warning("Memory retention failed", exc_info=True)
        if self._memory_janitor is None:
            self._memory_janitor = asyncio.create_task(self._purge_memory_forever())

    async def _purge_memory_forever(self) -> None:
        from dlightrag.runtime.coordinator import MAINTENANCE_SECONDS

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
        from dlightrag_rag.lifecycle import await_shared_cleanup

        close_task = self._close_task
        if close_task is None:
            self._closed = True
            self._components.health.mark_closed()
            close_task = asyncio.create_task(self._close_components())
            self._close_task = close_task
        await await_shared_cleanup(close_task)

    async def _close_components(self) -> None:
        """Run the one shared shutdown sequence every close caller joins."""
        from dlightrag_rag.lifecycle import defer_cancellation

        from dlightrag.adapters.postgres._pool import pg_pool
        from dlightrag.observability import shutdown_tracing

        components = self._components
        cancellation: asyncio.CancelledError | None = None
        for label, close in (
            ("memory janitor", self._stop_memory_janitor),
            ("ingest jobs", components.corpora.aclose),
            ("the durable answer coordinator", components.coordinator.aclose),
            ("the cancellation listener", components.cancellation_listener.aclose),
            ("Web conversation retention", components.web_conversations.aclose),
            ("the Retrieval service", components.retrieval.aclose),
            ("the Answer model runtime", components.models.aclose),
            ("the workspace pool", components.pool.aclose),
            ("the operational PostgreSQL pool", pg_pool.close),
        ):
            try:
                await close()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close %s", label, exc_info=True)
        shutdown_tracing()
        if cancellation is not None:
            raise cancellation


def _require_compatible_run(
    requirement: Mapping[str, Any],
    current_fingerprints: Mapping[ModelRole, ModelFingerprint],
) -> None:
    """Fail startup when one accepted run cannot execute under this binary."""
    from dlightrag_ai.capacity import CONTEXT_POLICY_REVISION
    from dlightrag_ai.settings import MODEL_ROLE_NAMES

    from dlightrag.answer.executor import IncompatibleActiveRunError
    from dlightrag.answer.runs.execution import PinnedModelProfile

    try:
        policy_revision = str(requirement.get("context_policy_revision") or "")
        raw_pins = requirement.get("pinned_models")
        if not isinstance(raw_pins, list):
            raise ValueError("pinned_models must be an array")
        pinned_models = tuple(PinnedModelProfile.from_json(item) for item in raw_pins)
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompatibleActiveRunError(
            "active answer runs use an incompatible durable input schema; "
            "drain or owner-cancel them before deployment"
        ) from exc
    if policy_revision != CONTEXT_POLICY_REVISION:
        raise IncompatibleActiveRunError(
            "active answer runs use another context policy revision; "
            "drain or owner-cancel them before deployment"
        )
    pinned = {item.role: item for item in pinned_models}
    if len(pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(MODEL_ROLE_NAMES):
        raise IncompatibleActiveRunError(
            "active answer runs do not contain the complete model role set; "
            "drain or owner-cancel them before deployment"
        )
    if any(pinned[role].fingerprint != current_fingerprints[role] for role in MODEL_ROLE_NAMES):
        raise IncompatibleActiveRunError(
            "active answer runs target another model endpoint configuration; "
            "drain or owner-cancel them before deployment"
        )


__all__ = [
    "Application",
    "ApplicationClosedError",
]
