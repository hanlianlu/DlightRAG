# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Temporary composition coordinator pending final application-service extraction."""

import asyncio
import logging
from collections.abc import (
    AsyncGenerator,
    Mapping,
    Sequence,
)
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore
    from dlightrag.answer.resources import ResourceInput
    from dlightrag.config import DlightragConfig

from dlightrag_ai.capacity import (
    CONTEXT_POLICY,
    CONTEXT_POLICY_REVISION,
    ModelProfile,
)
from dlightrag_ai.catalog import MODEL_CATALOG_REVISION
from dlightrag_ai.fingerprints import model_fingerprint
from dlightrag_ai.media import MAX_DECODE_IMAGE_PIXELS
from dlightrag_ai.scheduler import ModelScheduler
from dlightrag_ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag_ai.telemetry import safe_log_text
from dlightrag_ai.vision import ModelImageCapabilities
from dlightrag_rag.ingestion.jobs import IngestJobCoordinator
from dlightrag_rag.lifecycle import defer_cancellation
from dlightrag_rag.pool import WorkspacePool
from dlightrag_rag.ports import (
    CorpusSchemaError,
    WorkspaceCorpusBackend,
)
from dlightrag_rag.retrieval import (
    MetadataFilter,
    RetrievalResult,
)
from dlightrag_rag.settings import RagSettings
from dlightrag_rag.source_download import SourceDownloadService
from dlightrag_rag.sourcing.source_contract import safe_source_filename
from dlightrag_rag.workspace_rag import WorkspaceRag
from dlightrag_rag.workspaces import (
    normalize_workspace,
    normalize_workspace_ids,
)
from PIL import Image

from dlightrag.access import (
    DEPLOYMENT_OWNER_ID,
    resolve_query_workspaces,
    validate_query_workspace_selection,
)
from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.adapters.postgres.corpus import PGCorpusBackendFactory, PGReadinessProbe
from dlightrag.adapters.postgres.file_panel import PGFilePanelStore
from dlightrag.adapters.postgres.pg_metadata_index import PGMetadataIndex
from dlightrag.adapters.retrieval import (
    AnswerQueryImagePreparer,
    AnswerRetrievalProjector,
    PGWorkspaceSchemaLookup,
)
from dlightrag.answer.agent.orchestrator import (
    research_history_input_measure,
)
from dlightrag.answer.capabilities import (
    AnswerCapabilityCoordinator,
    AnswerCapabilityView,
)
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.executor import (
    AnswerExecutor,
    AnswerResourceResolver,
    IncompatibleAnswerRunError,
)
from dlightrag.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.answer.model_runtime import AnswerModelRuntime
from dlightrag.answer.resources.images import prepare_query_images
from dlightrag.answer.resources.models import (
    TextWindowBudget,
)
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.answer.runs.results import (
    AnswerResult,
    restore_answer_result,
)
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.memory.conversation import PriorTurns
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
from dlightrag.runtime import (
    AnswerRunCancelledError,
    AnswerRunEvent,
    AnswerRunFailedError,
    AnswerRunRecord,
    CancellationOutcome,
    PendingArtifact,
    PendingArtifactReference,
    RunCoordinator,
    RunCreation,
    RunSchemaError,
    answer_run_request_fingerprint,
    artifact_digest,
)
from dlightrag.services.corpora import CorpusAdmin
from dlightrag.services.retrieval import RetrievalPlannerRuntime, RetrievalService
from dlightrag.web.conversation_models import WebConversationSchemaError

logger = logging.getLogger(__name__)


def _attachment_bytes(resources: list[ResourceInput] | None) -> list[bytes]:
    """Return the inline bytes an accepted run must persist with its input."""
    return [resource.content for resource in resources or () if resource.content is not None]


@dataclass(frozen=True, slots=True)
class _AcceptanceProjection:
    history: PriorTurns
    image_descriptions: tuple[str, ...]
    pinned_models: tuple[PinnedModelProfile, ...]


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class RAGServiceUnavailableError(Exception):
    """Raised when a temporary manager-owned service is unavailable."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


class RAGServiceManager:
    """Multi-workspace RAG coordinator.

    Manages a pool of WorkspaceRag instances (one per workspace).
    Routes read operations to single workspace or federation.
    """

    def __init__(self, config: DlightragConfig | None = None) -> None:
        from dlightrag.config import get_config

        # Large document scans are DlightRAG product policy, not an AI package import side effect.
        Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS
        self._config = config or get_config()
        corpus_backend = PGCorpusBackendFactory(self._config).create()

        self._health = ApplicationHealth(
            readiness_probe=PGReadinessProbe(self._config),
        )

        self._model_scheduler = ModelScheduler(max_concurrency=self._config.max_async)
        self._telemetry = LangfuseTelemetry()
        self._workspace_pool = WorkspacePool(
            settings_for=self._workspace_settings,
            backend_for=self._workspace_backend,
            build=self._build_workspace,
        )

        ingest_jobs = IngestJobCoordinator(
            lambda workspace: self._workspace_pool.acquire(workspace),
            input_root=self._config.input_dir_path,
            store=corpus_backend.ingest_jobs,
        )
        source_download_settings = rag_settings(self._config)
        self.corpora = CorpusAdmin(
            settings=corpus_admin_settings(self._config),
            pool=self._workspace_pool,
            maintenance=corpus_backend.maintenance,
            ingest_jobs=ingest_jobs,
            file_panel=PGFilePanelStore(),
            source_download_for=lambda workspace: SourceDownloadService(
                settings=source_download_settings,
                metadata_index=PGMetadataIndex(workspace=workspace),
                workspace_id=workspace,
            ),
        )
        # Image capability is role-specific but cached per resolved model config,
        # so roles that share one model share one probe.
        self._capabilities = AnswerCapabilityCoordinator(
            settings=answer_capability_settings(self._config),
            profile_for_role=lambda role: model_profile_for_role(self._config, role),
            model_settings_for_role=lambda role: model_settings_for_role(self._config, role),
            rerank_model_settings=lambda: rerank_scoring_model_settings(self._config),
            image_capabilities=ModelImageCapabilities(
                scheduler=self._model_scheduler,
                telemetry=self._telemetry,
            ),
            on_answer_capability=self._health.set_answer_image_capability,
        )
        self._answer_models = AnswerModelRuntime(
            settings=answer_model_runtime_settings(self._config),
            scheduler=self._model_scheduler,
            telemetry=self._telemetry,
            answer_image_policy=self._capabilities.answer_image_policy,
            vlm_image_policy=self._capabilities.vlm_image_policy,
            vlm_profile=lambda: self._capabilities.model_profile("vlm"),
        )
        self._answer_resources = AnswerResourceResolver(
            settings=answer_resource_settings(self._config),
            models=self._answer_models,
            capabilities=self._capabilities,
        )
        self.retrieval = RetrievalService(
            pool=self._workspace_pool,
            planners=RetrievalPlannerRuntime(
                model_settings=model_settings_for_role(self._config, "extract"),
                default_profile=lambda: self._capabilities.model_profile("extract"),
                scheduler=self._model_scheduler,
                telemetry=self._telemetry,
            ),
            schema_lookup=PGWorkspaceSchemaLookup(
                default_workspace=normalize_workspace(self._config.workspace)
            ),
            image_preparer=AnswerQueryImagePreparer(
                capabilities=self._capabilities,
                models=self._answer_models,
            ),
            projector=AnswerRetrievalProjector(),
            settings=retrieval_settings(self._config),
            telemetry=self._telemetry,
        )
        self.answer_capabilities = AnswerCapabilityView(self._capabilities)
        self._answer_run_store: PGAnswerRunStore | None = None
        self._web_conversation_store: PGWebConversationStore | None = None
        self._answer_coordinator: RunCoordinator | None = None
        self._answer_executor: AnswerExecutor | None = None
        # Separate locks: starting the runtime needs the store, so one lock for
        # both would deadlock on itself.
        self._answer_store_lock = asyncio.Lock()
        self._answer_runtime_lock = asyncio.Lock()

    @property
    def config(self) -> DlightragConfig:
        """Read-only access to the manager configuration for UI/API adapters."""
        return self._config

    @property
    def health(self) -> ApplicationHealth:
        return self._health

    @classmethod
    async def acreate(cls, config: DlightragConfig | None = None) -> RAGServiceManager:
        """Async factory — creates manager and warms the default workspace."""
        from dlightrag.observability import init_tracing

        manager = cls(config=config)
        manager._capabilities.resolve_profiles()
        manager._capabilities.validate_startup()
        init_tracing(manager._config)

        # Bind the process-wide domain pool to this service config so the
        # endpoint and role cannot silently diverge from a caller-supplied SDK
        # config that never called set_config().
        from dlightrag.adapters.postgres._pool import pg_pool

        pg_pool.bind(manager._config)

        default_ws = normalize_workspace(manager._config.workspace)
        default_err: Exception | None = None
        default_ready = False
        try:
            await manager._initialize_answer_run_store()
            await manager._validate_active_answer_run_compatibility()
            try:
                await manager.corpora.initialize()
            except CorpusSchemaError:
                raise
            except Exception as exc:
                manager._health.add_warning("Workspace registry unavailable")
                logger.warning("Workspace registry initialization failed: %s", exc)

            # Bind the retrieval-planner LLM during startup; this does not make a model call.
            manager.retrieval.planner_for()

            # ── Vision probes (once at startup, not per workspace) ─────────
            await manager._capabilities.probe_all()

            try:
                await manager._workspace_pool.acquire(default_ws)
                default_ready = True
                logger.info("Warmed up default workspace service '%s'", default_ws)
            except CorpusSchemaError:
                raise
            except Exception as exc:
                default_err = exc
                logger.warning(
                    "Failed to warm up default workspace '%s'", default_ws, exc_info=True
                )
        except (
            CorpusSchemaError,
            IncompatibleAnswerRunError,
            RunSchemaError,
            WebConversationSchemaError,
        ):
            # Schema/run incompatibility needs operator action; never degrade into it.
            await manager.aclose()
            raise
        except Exception:
            await manager.aclose()
            raise

        try:
            await manager.corpora.start_recovery()
        except Exception:
            manager._health.add_warning("Ingest job recovery unavailable")
            logger.warning("Ingest job recovery initialization failed", exc_info=True)
        await manager._start_answer_runtime()
        if default_ready:
            manager._health.mark_ready()
        else:
            detail = getattr(default_err, "detail", str(default_err)) if default_err else "unknown"
            manager._health.mark_degraded(f"Default workspace init failed: {detail}")
            logger.error("RAG service started in degraded mode: %s", detail)
        return manager

    async def _initialize_answer_run_store(self) -> None:
        """Migrate the durable operational schema, or validate it on a reader.

        Answer runs are startup state, not first-request state: a process whose
        run schema is absent must fail before readiness rather than accept runs
        it cannot durably record. The Web conversation link table is part of the
        same schema because run retention exempts conversation-linked runs, so
        every process that owns runs also establishes that table.
        """
        from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore

        try:
            run_store = await self._get_answer_run_store()
            store = PGWebConversationStore(run_store=run_store)
            await store.initialize(validate_only=self._config.is_reader)
            self._web_conversation_store = store
        except RunSchemaError, WebConversationSchemaError:
            raise
        except Exception as exc:
            self._health.add_warning("Answer run store unavailable")
            logger.warning("Answer run store initialization failed: %s", exc)

    def create_web_conversation_service(self):
        """Compose the Web owner with the PostgreSQL stores validated at startup."""
        from dlightrag.web.conversations import WebConversationService

        store = self._web_conversation_store
        run_store = self._answer_run_store
        if store is None or run_store is None:
            raise RuntimeError("Web conversation stores are not initialized")
        return WebConversationService(
            store=store,
            run_store=run_store,
            prepare_run_input=self.aprepare_answer_run_input,
            max_turns=self._config.web_conversations.max_turns,
            ttl_days=self._config.web_conversations.ttl_days,
            max_attachments=self._config.answer.max_attachments,
            validate_schema_only=self._config.is_reader,
        )

    async def _validate_active_answer_run_compatibility(self) -> None:
        """Reject a rolling deployment that cannot execute already accepted inputs."""
        store = self._answer_run_store
        if store is None:
            return
        current_fingerprints = {
            role: model_fingerprint(model_settings_for_role(self._config, role))
            for role in MODEL_ROLE_NAMES
        }
        for requirement in await store.list_active_run_requirements():
            try:
                policy_revision = str(requirement.get("context_policy_revision") or "")
                raw_pins = requirement.get("pinned_models")
                if not isinstance(raw_pins, list):
                    raise ValueError("pinned_models must be an array")
                pinned_models = tuple(PinnedModelProfile.from_json(item) for item in raw_pins)
            except (KeyError, TypeError, ValueError) as exc:
                raise IncompatibleAnswerRunError(
                    "active answer runs use an incompatible durable input schema; "
                    "drain or owner-cancel them before deployment"
                ) from exc
            if policy_revision != CONTEXT_POLICY_REVISION:
                raise IncompatibleAnswerRunError(
                    "active answer runs use another context policy revision; "
                    "drain or owner-cancel them before deployment"
                )
            pinned = {item.role: item for item in pinned_models}
            if len(pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(MODEL_ROLE_NAMES):
                raise IncompatibleAnswerRunError(
                    "active answer runs do not contain the complete model role set; "
                    "drain or owner-cancel them before deployment"
                )
            if any(
                pinned[role].fingerprint != current_fingerprints[role] for role in MODEL_ROLE_NAMES
            ):
                raise IncompatibleAnswerRunError(
                    "active answer runs target another model endpoint configuration; "
                    "drain or owner-cancel them before deployment"
                )

    async def _start_answer_runtime(self) -> None:
        """Begin executing accepted runs once startup validated their schema.

        A store that never initialized already reported its startup warning, so
        this neither retries it nor turns a transient fault into a hard failure.
        """
        if self._answer_run_store is None:
            return
        try:
            await self.astart_answer_runtime()
        except Exception as exc:
            self._health.add_warning("Answer runtime unavailable")
            logger.warning("Answer runtime failed to start: %s", exc)

    @staticmethod
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

    def _workspace_config(self, workspace_id: str) -> DlightragConfig:
        return self._config.model_copy(update={"workspace": workspace_id})

    def _workspace_settings(self, workspace_id: str) -> RagSettings:
        return rag_settings(self._workspace_config(workspace_id))

    def _workspace_backend(self, workspace_id: str) -> WorkspaceCorpusBackend:
        return PGCorpusBackendFactory(self._workspace_config(workspace_id)).create()

    async def _build_workspace(
        self,
        workspace_id: str,
        settings: RagSettings,
        backend: WorkspaceCorpusBackend,
    ) -> WorkspaceRag:
        try:
            runtime = await WorkspaceRag.acreate(
                workspace_id=workspace_id,
                settings=settings,
                backend=backend,
                scheduler=self._model_scheduler,
                telemetry=LangfuseTelemetry(),
                rerank_supports_vision=self._capabilities.rerank_supports_vision,
            )
        except CorpusSchemaError:
            raise
        except Exception as exc:
            raise RuntimeError(self._actionable_error(exc)) from exc
        logger.info("Created WorkspaceRag for workspace '%s'", safe_log_text(workspace_id))
        return runtime

    # --- Durable answer runs ---

    async def _get_answer_run_store(self) -> PGAnswerRunStore:
        if self._answer_run_store is not None:
            return self._answer_run_store
        if self._health.is_closed:
            raise RAGServiceUnavailableError("Answer runtime is shutting down")
        async with self._answer_store_lock:
            if self._answer_run_store is None:
                from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore as _Store

                store = _Store()
                await store.initialize(validate_only=self._config.is_reader)
                self._answer_run_store = store
            return self._answer_run_store

    async def astart_answer_runtime(self) -> None:
        """Create the durable run schema and begin executing accepted runs.

        Idempotent and safe to call concurrently: every caller ends up sharing
        the one store and the one coordinator this process owns, so no orphaned
        worker identity can claim runs that ``aclose`` would never join.
        """
        if self._answer_coordinator is not None:
            return
        if self._health.is_closed:
            raise RAGServiceUnavailableError("Answer runtime is shutting down")
        async with self._answer_runtime_lock:
            if self._answer_coordinator is not None:
                return
            if self._health.is_closed:
                raise RAGServiceUnavailableError("Answer runtime is shutting down")
            store = await self._get_answer_run_store()
            executor = self._answer_executor
            if executor is None:
                executor = AnswerExecutor(
                    store=store,
                    pool=self._workspace_pool,
                    retrieve=self.retrieval.retrieve_result,
                    models=self._answer_models,
                    capabilities=self._capabilities,
                    resources=self._answer_resources,
                    settings=answer_executor_settings(self._config),
                    telemetry=self._telemetry,
                )
                self._answer_executor = executor
            coordinator = RunCoordinator(
                store=store,
                executor=executor,
                answer_worker_concurrency=self._config.runtime.answer_worker_concurrency,
            )
            await coordinator.start()
            if self._health.is_closed:
                # The manager closed while the store was initializing; publishing
                # now would leave a claiming worker that ``aclose`` never joins.
                await coordinator.aclose()
                raise RAGServiceUnavailableError("Answer runtime is shutting down")
            self._answer_coordinator = coordinator

    async def astart_answer_run(
        self,
        *,
        owner_id: str,
        request: AnswerRunInput,
        idempotency_key: str | None = None,
        attachment_bytes: Sequence[bytes] = (),
    ) -> RunCreation:
        """Accept one run, its input blobs, and its references in one transaction.

        The runtime is started first, so an accepted run executes even when no
        caller ever subscribes to its events.
        """
        await self.astart_answer_runtime()
        store = await self._get_answer_run_store()
        artifacts = [PendingArtifact(content=content) for content in attachment_bytes]
        references = [
            PendingArtifactReference(
                resource_id=attachment.resource_id,
                reference_kind="current_attachment",
                ordinal=attachment.ordinal,
                digest=attachment.digest,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
            )
            for attachment in request.attachments
        ]
        references.extend(
            PendingArtifactReference(
                resource_id=attachment.history_resource_id,
                reference_kind="history_attachment",
                ordinal=attachment.ordinal,
                digest=attachment.digest,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
            )
            for attachment in request.history_attachments
        )
        creation = await store.create_run(
            owner_id=owner_id,
            request=request.as_request(),
            idempotency_fingerprint=request.idempotency_fingerprint,
            idempotency_key=idempotency_key,
            artifacts=artifacts,
            references=references,
        )
        if self._answer_coordinator is not None:
            self._answer_coordinator.wake()
        return creation

    async def areplay_answer_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
    ) -> AnswerRunRecord | None:
        """Return an already accepted keyed run without rebuilding resolved input."""
        store = await self._get_answer_run_store()
        replay = await store.replay_run(
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return replay.run if replay is not None else None

    async def aget_answer_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None:
        """Read one owned run; unknown and foreign identifiers both return ``None``."""
        store = await self._get_answer_run_store()
        return await store.get_run(owner_id=owner_id, run_id=run_id)

    async def acancel_answer_run(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        """Request cancellation; only this mutates a run on a caller's behalf."""
        store = await self._get_answer_run_store()
        return await store.request_cancellation(owner_id=owner_id, run_id=run_id)

    async def asubscribe_answer_run(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[AnswerRunEvent]:
        """Follow one run's durable events; detaching never cancels the run."""
        await self.astart_answer_runtime()
        coordinator = self._answer_coordinator
        if coordinator is None:  # pragma: no cover - started above
            raise RAGServiceUnavailableError("Answer runtime is unavailable")
        return coordinator.subscribe(
            owner_id=owner_id, run_id=run_id, after_sequence=after_sequence
        )

    async def aanswer(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        history: list[dict[str, Any]] | None = None,
        semantic_highlights: bool = False,
        resources: list[ResourceInput] | None = None,
        idempotency_key: str | None = None,
        owner_id: str | None = None,
    ) -> AnswerResult:
        """Create one durable answer run and wait for its canonical result.

        Current-turn images and documents are supplied through ``resources``:
        inline bytes become owner artifacts and HTTPS links become inert link
        descriptors, so a resumed run reads exactly what was accepted. A caller
        that cancels this wait only detaches; use ``acancel_answer_run`` to stop
        the run itself.

        ``history`` is caller-supplied prior turns (``role``/``content`` dicts).
        It is stateless -- the caller owns persistence and passes it per request.
        """
        creation = await self.acreate_answer_run(
            query,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            history=history,
            semantic_highlights=semantic_highlights,
            resources=resources,
            idempotency_key=idempotency_key,
            owner_id=owner_id,
        )
        run = creation.run
        async with aclosing(
            await self.asubscribe_answer_run(owner_id=run.owner_id, run_id=run.run_id)
        ) as events:
            async for _event in events:
                pass
        final = await self.aget_answer_run(owner_id=run.owner_id, run_id=run.run_id)
        if final is None:
            raise RAGServiceUnavailableError("Answer run disappeared before it finished")
        if final.status == "succeeded":
            return restore_answer_result(final.result or {})
        if final.status == "cancelled":
            raise AnswerRunCancelledError(final.run_id)
        raise AnswerRunFailedError(
            final.error_kind or "answer_stream_failed",
            final.error_message or "Answer run failed.",
        )

    async def acreate_answer_run(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        history: list[dict[str, Any]] | None = None,
        semantic_highlights: bool = False,
        resources: list[ResourceInput] | None = None,
        idempotency_key: str | None = None,
        owner_id: str | None = None,
    ) -> RunCreation:
        """Accept one durable answer run and return it without waiting.

        The descriptor-only entry point every transport shares: the run outlives
        the call that created it, and its state is read back through
        ``aget_answer_run``, ``asubscribe_answer_run``, and
        ``acancel_answer_run``.
        """
        owner = owner_id or DEPLOYMENT_OWNER_ID
        request = await self._normalized_answer_run_request(
            query,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            history=history,
            semantic_highlights=semantic_highlights,
            resources=resources,
        )
        fingerprint = answer_run_request_fingerprint(request.as_request())
        if idempotency_key is not None:
            replay = await self.areplay_answer_run(
                owner_id=owner,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
            )
            if replay is not None:
                return RunCreation(run=replay, replayed=True)
        attachment_bytes = _attachment_bytes(resources)
        request, attachment_bytes = await self._answer_resources.pin_current_image_links(
            request,
            attachment_bytes,
        )
        acceptance_resources = await build_current_answer_resources(
            links=request.links,
            attachments=request.attachments,
            attachment_loaders=[
                in_memory_attachment_loader(content) for content in attachment_bytes
            ],
        )
        return await self.astart_answer_run(
            owner_id=owner,
            request=await self.aprepare_answer_run_input(
                request,
                resources=acceptance_resources or None,
                idempotency_fingerprint=fingerprint,
            ),
            idempotency_key=idempotency_key,
            attachment_bytes=attachment_bytes,
        )

    async def aanswer_stream(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        history: list[dict[str, Any]] | None = None,
        semantic_highlights: bool = False,
        resources: list[ResourceInput] | None = None,
        idempotency_key: str | None = None,
        owner_id: str | None = None,
    ) -> AsyncGenerator[AnswerRunEvent]:
        """Create one durable answer run and follow its events until it ends.

        Yields the run's durable ``progress``, ``token``, ``reset``, and terminal
        events. Closing this generator detaches this subscriber only; the run
        keeps executing until it finishes or is explicitly cancelled.
        """
        creation = await self.acreate_answer_run(
            query,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            history=history,
            semantic_highlights=semantic_highlights,
            resources=resources,
            idempotency_key=idempotency_key,
            owner_id=owner_id,
        )
        run = creation.run
        async with aclosing(
            await self.asubscribe_answer_run(owner_id=run.owner_id, run_id=run.run_id)
        ) as events:
            async for event in events:
                yield event

    async def _normalized_answer_run_request(
        self,
        query: str,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        history: list[dict[str, Any]] | None,
        semantic_highlights: bool,
        resources: list[ResourceInput] | None,
    ) -> AnswerRunRequest:
        """Normalize one in-process public request without resolved model input."""
        validate_query_workspace_selection(
            all_workspaces=all_workspaces,
            workspace=workspace,
            workspaces=workspaces,
        )
        available = (
            normalize_workspace_ids(await self.corpora.list_workspaces())
            if all_workspaces
            else None
        )
        ws_list = resolve_query_workspaces(
            default_workspace=normalize_workspace(self._config.workspace),
            workspace=normalize_workspace(workspace) if workspace else None,
            workspaces=normalize_workspace_ids(workspaces) if workspaces is not None else None,
            all_workspaces=all_workspaces,
            available_workspaces=available,
        )
        links: list[LinkReference] = []
        attachments: list[AttachmentReference] = []
        for resource in resources or ():
            if resource.url is not None:
                links.append(
                    LinkReference(
                        url=resource.url,
                        filename=resource.filename,
                        ordinal=len(links),
                        mime_type=resource.declared_mime,
                    )
                )
                continue
            if resource.content is None:
                raise ValueError("durable answer resources need inline bytes or an HTTPS link")
            attachments.append(
                AttachmentReference(
                    digest=artifact_digest(resource.content),
                    filename=safe_source_filename(resource.filename),
                    mime_type=resource.declared_mime or "application/octet-stream",
                    ordinal=len(attachments),
                    byte_size=len(resource.content),
                )
            )
        return AnswerRunRequest(
            query=query,
            workspaces=tuple(ws_list),
            history=tuple(dict(message) for message in history or ()),
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters.model_dump(exclude_none=True, mode="json") if filters else None,
            semantic_highlights=semantic_highlights,
            links=tuple(links),
            attachments=tuple(attachments),
        )

    async def _project_answer_run_acceptance(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
    ) -> _AcceptanceProjection:
        """Resolve the exact shared-history envelopes without building the run rig."""
        from dlightrag.answer.evidence import EvidenceLedger
        from dlightrag.answer.tools import compose_research_tools

        if resources:
            await self._capabilities.refresh_vlm()
        model_profiles = self._capabilities.current_profiles()
        models = self._capabilities.request_model_context(model_profiles)
        planner = self.retrieval.planner_for(models.extract)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(models.query))
        resolved = await self._answer_resources.resolve(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._capabilities.confirmed_live_answer_context,
        )
        try:
            ws_list = list(request.workspaces)
            self.retrieval.warm(ws_list)
            models = resolved.models
            model_profiles["extract"] = models.extract
            model_profiles["query"] = models.query
            model_profiles["vlm"] = models.vlm
            image_descriptions = tuple(
                await prepare_query_images(
                    query_images=resolved.current_images,
                    describer=self._answer_models.query_image_describer(),
                )
                if resolved.current_images
                else ()
            )
            schema = await self.retrieval.schema_for(ws_list)
            targets = [
                HistoryProjectionTarget(
                    "planner",
                    models.extract,
                    planner.history_input_measure(
                        request.query,
                        schema=schema,
                        current_image_descriptions=list(image_descriptions) or None,
                        preserve_query=True if resolved.research else None,
                    ),
                )
            ]
            if resolved.research:
                evidence = EvidenceLedger(image_budget=resolved.image_budget)

                async def unused_retrieve(_query: str) -> RetrievalResult:
                    raise RuntimeError("acceptance tool definitions are never executed")

                tools, tool_cache = compose_research_tools(
                    evidence=evidence,
                    trace={},
                    retrieve_knowledge_base=unused_retrieve,
                    search_web=(
                        resolved.web_search.search if resolved.web_search is not None else None
                    ),
                    resource_tools=resolved.resource_tools,
                    register_web_source=(
                        resolved.registry.register_discovered_link
                        if resolved.registry is not None and resolved.web_search is not None
                        else None
                    ),
                )
                try:
                    measure = research_history_input_measure(
                        model_profile=models.query,
                        context_policy=CONTEXT_POLICY,
                        query=request.query,
                        query_images=resolved.query_images,
                        resource_manifest=resolved.resource_manifest,
                        image_budget=resolved.image_budget,
                        tools=tools,
                        retained_tail_tokens=CONTEXT_POLICY.retained_tail_target(models.query),
                    )
                finally:
                    await tool_cache.aclose()
                targets.append(
                    HistoryProjectionTarget(
                        "research_seed",
                        models.query,
                        measure,
                        proactive_compaction=True,
                    )
                )
            else:
                synthesizer = AnswerSynthesizer(
                    image_policy=self._capabilities.answer_image_policy(models.query),
                    model_profile=models.query,
                    context_policy=CONTEXT_POLICY,
                    model_func=None,
                )
                targets.append(
                    HistoryProjectionTarget(
                        "fast_generation",
                        models.query,
                        synthesizer.history_input_measure(request.query),
                    )
                )
            try:
                history = project_history(
                    [dict(message) for message in request.history],
                    targets=targets,
                )
            except HistoryProjectionOverflowError as exc:
                raise AnswerInputOverflowError(str(exc)) from exc
            return _AcceptanceProjection(
                history=history,
                image_descriptions=image_descriptions,
                pinned_models=self._pin_model_profiles(model_profiles),
            )
        finally:
            if resolved.registry is not None:
                await resolved.registry.aclose()

    async def aprepare_answer_run_input(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        idempotency_fingerprint: str,
    ) -> AnswerRunInput:
        """Resolve one normalized public request into immutable durable input."""
        projection = await self._project_answer_run_acceptance(
            request,
            resources=resources,
        )
        return AnswerRunInput(
            query=request.query,
            workspaces=request.workspaces,
            history=tuple(dict(message) for message in projection.history.messages),
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=request.filters,
            semantic_highlights=request.semantic_highlights,
            links=request.links,
            attachments=request.attachments,
            history_attachments=request.history_attachments,
            pinned_models=projection.pinned_models,
            context_policy_revision=CONTEXT_POLICY_REVISION,
            model_catalog_revision=MODEL_CATALOG_REVISION,
            idempotency_fingerprint=idempotency_fingerprint,
            image_descriptions=projection.image_descriptions,
        )

    def _pin_model_profiles(
        self,
        profiles: Mapping[ModelRole, ModelProfile],
    ) -> tuple[PinnedModelProfile, ...]:
        return tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=model_fingerprint(model_settings_for_role(self._config, role)),
                profile=profiles[role],
            )
            for role in MODEL_ROLE_NAMES
        )

    async def aclose(self) -> None:
        """Close all managed WorkspaceRag instances."""
        from dlightrag.observability import shutdown_tracing

        self._health.mark_closed()
        cancellation: asyncio.CancelledError | None = None
        await self.corpora.aclose()
        if self._answer_coordinator is not None:
            coordinator, self._answer_coordinator = self._answer_coordinator, None
            try:
                await coordinator.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close the durable answer coordinator", exc_info=True)

        try:
            await self.retrieval.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close Retrieval service", exc_info=True)

        try:
            await self._workspace_pool.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close workspace pool", exc_info=True)

        try:
            await self._answer_models.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close Answer model runtime", exc_info=True)

        from dlightrag.adapters.postgres._pool import pg_pool

        try:
            await pg_pool.close()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        shutdown_tracing()
        if cancellation is not None:
            raise cancellation


__all__ = [
    "RAGServiceUnavailableError",
    "RAGServiceManager",
]
