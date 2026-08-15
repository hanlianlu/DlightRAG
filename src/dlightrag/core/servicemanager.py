# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGServiceManager — unified multi-workspace RAG coordinator.

Absorbs pool.py workspace management and federation routing into a single
entry point. All API/MCP consumers depend on this class only.
"""

import asyncio
import base64
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig
    from dlightrag.core.request.images import QueryImageDescriber
    from dlightrag.core.resources import ResourceInput, ResourceRegistry
    from dlightrag.core.retrieval.web_search import ExaSearch
    from dlightrag.core.source_download import SourceDownloadTarget
    from dlightrag.storage.file_panel import PGFilePanelStore
    from dlightrag.storage.workspaces import PGWorkspaceRegistry

from dlightrag_agent.tools import AgentTool
from dlightrag_ai.capacity import (
    CONTEXT_POLICY,
    CONTEXT_POLICY_REVISION,
    ModelCapabilityError,
    ModelProfile,
)
from dlightrag_ai.catalog import MODEL_CATALOG_REVISION
from dlightrag_ai.completion import CompletionModel
from dlightrag_ai.fingerprints import model_fingerprint
from dlightrag_ai.media import MAX_DECODE_IMAGE_PIXELS
from dlightrag_ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag_ai.telemetry import safe_log_text
from dlightrag_ai.tool_model import ToolModel
from dlightrag_ai.vision import ImageCapabilityStatus, ImageProbeOutcome, ModelImageCapabilities
from dlightrag_rag.retrieval import MetadataFilter
from PIL import Image

from dlightrag.application import ApplicationHealth
from dlightrag.contracts import VisualAssetSize
from dlightrag.core.agent.orchestrator import (
    AnswerOrchestrator,
    research_history_input_measure,
)
from dlightrag.core.answer.capability import (
    AnswerImageCapability,
    answer_image_capability_summary,
    check_answer_image_capability,
    derive_effective_max_images,
)
from dlightrag.core.answer.errors import (
    AnswerInputError,
    AnswerInputOverflowError,
    AnswerModelCapabilityError,
    AnswerResourceAdmissionError,
    CurrentImagePayloadError,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.core.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.core.answer.images import AnswerImageBudget, AnswerImagePolicy
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.answer_runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.core.answer_runs.models import AgentRunState
from dlightrag.core.answer_runs.results import restore_answer_result, store_answer_result
from dlightrag.core.client_contracts import MAX_QUERY_IMAGES, IngestSpec, SourceType
from dlightrag.core.client_requests import ingest_kwargs_from_payload
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.ingest_job_coordinator import IngestJobCoordinator
from dlightrag.core.ingestion.paths import is_explicit_upload_batch_dir
from dlightrag.core.lightrag_lifecycle import defer_cancellation
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.principal import DEPLOYMENT_OWNER_ID
from dlightrag.core.request.images import prepare_query_images
from dlightrag.core.request.retrieval_planner import RetrievalPlan, RetrievalPlanner
from dlightrag.core.request.workspaces import (
    normalize_query_workspaces,
    resolve_query_workspaces,
    validate_query_workspace_selection,
)
from dlightrag.core.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.service import RAGService
from dlightrag.model_settings import (
    model_profile_for_role,
    model_settings_for_role,
    rerank_scoring_model_settings,
)
from dlightrag.observability import LangfuseTelemetry
from dlightrag.runtime import (
    AnswerRunCancelledError,
    AnswerRunEvent,
    AnswerRunFailedError,
    AnswerRunRecord,
    CancellationOutcome,
    CheckpointError,
    LeaseLostError,
    PendingArtifact,
    PendingArtifactReference,
    RunCancelledError,
    RunCoordinator,
    RunCreation,
    RunExecutionError,
    RunSession,
    answer_run_request_fingerprint,
    artifact_digest,
)
from dlightrag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.storage.answer_runs import PGAnswerRunStore
from dlightrag.storage.ingest_jobs import JOB_STATES_WITH_RESULT
from dlightrag.storage.migrations import SchemaValidationError
from dlightrag.utils import normalize_workspace

logger = logging.getLogger(__name__)
_MAX_RETRY_INTERVAL: float = 300.0
_QUERY_WORKSPACE_MAX_CONCURRENCY = 8
_SCHEMA_CACHE_MAX_ENTRIES = 128


async def _postgres_not_ready_detail(config: DlightragConfig) -> str | None:
    """Project the current PostgreSQL adapter's role-specific readiness."""
    from dlightrag.storage.pool import pg_pool

    try:
        read_only = await pg_pool.run_once(lambda conn: conn.fetchval("SHOW transaction_read_only"))
        if str(read_only).lower() != "off":
            raise RuntimeError("domain pool session is read-only")
    except Exception:
        logger.warning("Domain PostgreSQL readiness probe failed", exc_info=True)
        return "DlightRAG domain database session is not writable"

    if not config.is_reader:
        return None

    from dlightrag.storage.lightrag_readonly import verify_reader_corpus_session

    try:
        await verify_reader_corpus_session()
    except Exception:
        logger.warning("Reader corpus PostgreSQL readiness probe failed", exc_info=True)
        return "Reader corpus database session is not read-only or is unavailable"
    return None


def _attachment_bytes(resources: list[ResourceInput] | None) -> list[bytes]:
    """Return the inline bytes an accepted run must persist with its input."""
    return [resource.content for resource in resources or () if resource.content is not None]


@dataclass(frozen=True)
class _OrchestratorRun:
    """One request resolved into a capability-driven orchestrator and its inputs."""

    orchestrator: AnswerOrchestrator
    image_descriptions: list[str]
    query_images: list[dict[str, Any]] | None
    history: PriorTurns
    current_image_count: int
    ws_list: list[str]
    registry: ResourceRegistry | None


@dataclass(frozen=True, slots=True)
class _AcceptanceProjection:
    history: PriorTurns
    image_descriptions: tuple[str, ...]
    pinned_models: tuple[PinnedModelProfile, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedAnswerResources:
    models: _RequestModelContext
    web_search: ExaSearch | None
    registry: ResourceRegistry | None
    resource_tools: list[AgentTool]
    resource_manifest: tuple[ResourceManifestEntry, ...]
    current_images: list[dict[str, Any]]
    current_image_count: int
    research: bool
    image_budget: AnswerImageBudget | None
    query_images: list[dict[str, Any]] | None


@dataclass(frozen=True, slots=True)
class _RequestModelContext:
    extract: ModelProfile
    query: ModelProfile
    vlm: ModelProfile


def _exa_contents_text(web_search: ExaSearch) -> Callable[[str], Awaitable[str | None]]:
    """Adapt Exa Contents to the registry's provider-neutral URL text fallback.

    Usable passages for the one known URL are folded into a single deterministic
    text once, with the page title preserved as a leading line when it adds
    information. A parked or unreachable provider yields ``None`` so the caller
    keeps the original direct-extraction error rather than fabricating evidence.
    """
    from dlightrag.core.retrieval.web_search import WebSearchUnavailable

    async def _fallback(url: str) -> str | None:
        try:
            result = await web_search.contents(url)
        except WebSearchUnavailable:
            return None
        logger.info("Exa Contents fallback completed; cost_dollars=%.6f", result.cost_dollars)
        title: str | None = None
        passages: list[str] = []
        for hit in result.hits:
            text = hit.text.strip()
            if not text:
                continue
            if title is None and hit.title and hit.title != hit.url:
                title = hit.title.strip() or None
            passages.append(text)
        if not passages:
            return None
        body = "\n\n".join(passages)
        return f"{title}\n\n{body}" if title else body

    return _fallback


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _context_output(contexts: RetrievalContexts) -> dict[str, int]:
    return {
        "context_chunk_count": _context_count(contexts, "chunks"),
        "entity_count": _context_count(contexts, "entities"),
        "relationship_count": _context_count(contexts, "relationships"),
    }


def answer_trace_output(
    answer: str | None, sources: Sequence[Any] | None, contexts: Any
) -> dict[str, Any]:
    """Shape what a pipeline span reports as its answer, streamed or not."""
    from dlightrag.observability import trace_sensitive_enabled

    output: dict[str, Any] = {
        "answer_len": len(answer or ""),
        "source_count": len(sources or []),
        "context_chunk_count": _context_count(contexts, "chunks"),
    }
    if trace_sensitive_enabled():
        output["answer"] = answer or ""
    return output


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _cleanup_paths_for_local_ingest(*, source_type: SourceType, path: str | None) -> list[str]:
    if source_type != "local" or not path:
        return []
    return [path] if is_explicit_upload_batch_dir(Path(path)) else []


class _ManagerAnswerExecutor:
    """Adapt the manager's answer pipeline to the coordinator's executor seam."""

    def __init__(self, manager: RAGServiceManager) -> None:
        self._manager = manager

    async def execute(self, session: RunSession) -> Mapping[str, Any]:
        try:
            return await self._manager._execute_answer_run(session)
        except (
            asyncio.CancelledError,
            RunCancelledError,
            LeaseLostError,
            CheckpointError,
            RunExecutionError,
        ):
            raise
        except Exception as exc:
            logger.warning("Answer run %s execution failed", session.run_id, exc_info=True)
            # Only the Answer taxonomy vets a public message; a foreign attribute is untrusted.
            message = (
                exc.public_message
                if isinstance(exc, AnswerInputError | InvalidToolConfigurationError)
                and exc.public_message
                else "Answer run failed."
            )
            raise RunExecutionError(classify_answer_error(exc), message) from exc


def _fetched_bytes_sink(
    session: RunSession, store: PGAnswerRunStore
) -> Callable[[Any], Awaitable[None]]:
    """Persist each validated fetched resource under this worker's live fence."""

    async def _persist(fetched: Any) -> None:
        artifact = PendingArtifact(content=fetched.content)
        await session.attach_artifacts(
            artifacts=[artifact],
            references=[
                PendingArtifactReference(
                    resource_id=fetched.resource_id,
                    reference_kind="fetched_resource",
                    ordinal=fetched.ordinal,
                    digest=artifact.digest,
                    filename=fetched.filename,
                    mime_type=fetched.mime_type,
                    transform_locator={"url": fetched.url},
                )
            ],
        )

    return _persist


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


class IncompatibleAnswerRunError(RuntimeError):
    """An accepted run cannot execute under this binary or endpoint deployment."""


def _verified_current_image_data_uri(data: bytes, *, max_pixels: int) -> tuple[str, str]:
    from dlightrag_ai.media import image_bytes_to_data_uri, verify_web_image_bytes

    mime = verify_web_image_bytes(data, max_pixels=max_pixels)
    return mime, image_bytes_to_data_uri(data, fallback_mime=mime)


class RAGServiceManager:
    """Multi-workspace RAG coordinator.

    Manages a pool of RAGService instances (one per workspace).
    Routes read operations to single workspace or federation.
    """

    def __init__(self, config: DlightragConfig | None = None) -> None:
        from dlightrag.config import get_config

        # Large document scans are DlightRAG product policy, not an AI package import side effect.
        Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS
        self._config = config or get_config()
        self._services: dict[str, RAGService] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        self._health = ApplicationHealth(
            readiness_probe=lambda: _postgres_not_ready_detail(self._config),
        )

        # Per-workspace backoff: workspace -> (last_error_ts, retry_interval)
        self._backoff: dict[str, tuple[float, float]] = {}

        # In-flight workspace initializations started when a request resolves its scope.
        self._warmups: set[asyncio.Task[None]] = set()

        self._answer_synthesizers_by_profile: dict[ModelProfile, AnswerSynthesizer] = {}
        self._answer_model: CompletionModel | None = None
        self._declared_model_profiles: dict[ModelRole, ModelProfile] = {}
        self._model_profiles: dict[ModelRole, ModelProfile] = {}
        self._ingest_jobs = IngestJobCoordinator(
            self._get_ingest_service,
            input_root=self._config.input_dir_path,
        )
        self._retrieval_planners_by_profile: dict[ModelProfile, RetrievalPlanner] = {}
        self._planner_model: CompletionModel | None = None
        self._vlm_func: Callable[..., Any] | None = None
        self._vlm_model: CompletionModel | None = None
        self._web_search: ExaSearch | None = None
        self._query_tool_model: ToolModel | None = None
        self._vlm_func_lock = asyncio.Lock()
        self._workspace_registry: PGWorkspaceRegistry | None = None
        self._file_panel_store: PGFilePanelStore | None = None
        self._schema_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
        # Image capability is role-specific but cached per resolved model config,
        # so roles that share one model share one probe.
        self._image_capabilities = ModelImageCapabilities(telemetry=LangfuseTelemetry())
        self._rerank_supports_vision: bool | None = None
        self._answer_image_capability: AnswerImageCapability | None = None
        self._vlm_image_status: ImageCapabilityStatus = "unknown"
        self._direct_llm_sem = asyncio.Semaphore(max(1, int(self._config.max_async)))
        self._answer_run_store: PGAnswerRunStore | None = None
        self._answer_coordinator: RunCoordinator | None = None
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
        manager._resolve_model_profiles()
        manager._validate_startup_model_capabilities()
        init_tracing(manager._config)

        # Bind the process-wide domain pool to this service config so the
        # endpoint and role cannot silently diverge from a caller-supplied SDK
        # config that never called set_config().
        from dlightrag.storage.pool import pg_pool

        pg_pool.bind(manager._config)

        default_ws = normalize_workspace(manager._config.workspace)
        default_err: Exception | None = None
        try:
            await manager._initialize_answer_run_store()
            await manager._validate_active_answer_run_compatibility()
            await manager._initialize_workspace_registry()

            # Bind the retrieval-planner LLM during startup; this does not make a model call.
            manager._get_retrieval_planner()

            # ── Vision probes (once at startup, not per workspace) ─────────
            await manager._probe_role_image_capabilities()

            try:
                await manager._get_service(default_ws)
                logger.info("Warmed up default workspace service '%s'", default_ws)
            except SchemaValidationError:
                raise
            except Exception as exc:
                default_err = exc
                logger.warning(
                    "Failed to warm up default workspace '%s'", default_ws, exc_info=True
                )
        except SchemaValidationError, IncompatibleAnswerRunError:
            # Schema/run incompatibility needs operator action; never degrade into it.
            await manager.aclose()
            raise

        await manager._start_ingest_job_recovery()
        await manager._start_answer_runtime()
        if default_ws in manager._services:
            manager._health.mark_ready()
        else:
            detail = getattr(default_err, "detail", str(default_err)) if default_err else "unknown"
            manager._health.mark_degraded(f"Default workspace init failed: {detail}")
            logger.error("RAG service started in degraded mode: %s", detail)
        return manager

    def _resolve_model_profiles(self) -> None:
        """Snapshot every reachable role before startup opens external state."""
        self._declared_model_profiles = {
            role: model_profile_for_role(self._config, role) for role in MODEL_ROLE_NAMES
        }
        self._model_profiles = dict(self._declared_model_profiles)

    def _declared_model_profile(self, role: ModelRole) -> ModelProfile:
        profile = self._declared_model_profiles.get(role)
        if profile is None:
            profile = model_profile_for_role(self._config, role)
            self._declared_model_profiles[role] = profile
            self._model_profiles.setdefault(role, profile)
        return profile

    def _model_profile(self, role: ModelRole) -> ModelProfile:
        profile = self._model_profiles.get(role)
        if profile is None:
            profile = self._declared_model_profile(role)
            self._model_profiles[role] = profile
        return profile

    def _request_model_context(
        self,
        pinned: Mapping[ModelRole, ModelProfile] | None,
    ) -> _RequestModelContext:
        if pinned is not None:
            return _RequestModelContext(
                extract=pinned["extract"],
                query=pinned["query"],
                vlm=pinned["vlm"],
            )
        return _RequestModelContext(
            extract=self._model_profile("extract"),
            query=self._model_profile("query"),
            vlm=self._model_profile("vlm"),
        )

    def _narrow_role_image_profile(
        self,
        role: ModelRole,
        status: ImageCapabilityStatus,
    ) -> None:
        declared = self._declared_model_profile(role)
        self._model_profiles[role] = replace(
            declared,
            supports_images=declared.supports_images and status == "supported",
        )

    def _validate_startup_model_capabilities(self) -> None:
        if self._config.web_search.api_key and not self._model_profile("query").supports_tools:
            raise ModelCapabilityError(role="query", capability="tool calling")

    async def _initialize_workspace_registry(self) -> None:
        """Migrate the durable workspace registry, or validate it on a reader."""
        from dlightrag.storage.workspaces import PGWorkspaceRegistry

        self._workspace_registry = PGWorkspaceRegistry()
        try:
            await self._workspace_registry.initialize(validate_only=self._config.is_reader)
            if not self._config.is_reader:
                await self._workspace_registry.upsert(
                    workspace=normalize_workspace(self._config.workspace),
                    display_name=self._config.workspace,
                    embedding_model=self._config.embedding.model,
                )
        except SchemaValidationError:
            raise
        except Exception as exc:
            self._health.add_warning("Workspace registry unavailable")
            logger.warning("Workspace registry initialization failed: %s", exc)

    async def _initialize_answer_run_store(self) -> None:
        """Migrate the durable operational schema, or validate it on a reader.

        Answer runs are startup state, not first-request state: a process whose
        run schema is absent must fail before readiness rather than accept runs
        it cannot durably record. The Web conversation link table is part of the
        same schema because run retention exempts conversation-linked runs, so
        every process that owns runs also establishes that table.
        """
        from dlightrag.storage.web_conversations import PGWebConversationStore

        try:
            await self._get_answer_run_store()
            await PGWebConversationStore().initialize(validate_only=self._config.is_reader)
        except SchemaValidationError:
            raise
        except Exception as exc:
            self._health.add_warning("Answer run store unavailable")
            logger.warning("Answer run store initialization failed: %s", exc)

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

    async def _get_workspace_registry(self) -> PGWorkspaceRegistry:
        if self._workspace_registry is None:
            await self._initialize_workspace_registry()
        if self._workspace_registry is None:
            raise RuntimeError("Workspace registry unavailable")
        return self._workspace_registry

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

    async def _get_ingest_service(self, workspace: str) -> RAGService:
        # Resolved through self so the coordinator sees the current service lookup.
        return await self._get_service(workspace)

    async def _get_service(self, workspace: str) -> RAGService:
        """Get or create a RAGService for a specific workspace. Async-safe.

        Normalizes the workspace name to a safe PG identifier (lowercase,
        alphanumeric + underscore only) before lookup or creation.
        """
        from dlightrag.utils import normalize_workspace

        workspace = normalize_workspace(workspace)

        if workspace in self._services:
            return self._services[workspace]

        # Check per-workspace backoff
        if workspace in self._backoff:
            last_ts, interval = self._backoff[workspace]
            if time.time() - last_ts < interval:
                raise RAGServiceUnavailableError(
                    detail=f"Workspace '{workspace}' in backoff (retry in {interval:.0f}s)"
                )

        async with self._locks[workspace]:
            # Double-check after acquiring lock
            if workspace in self._services:
                return self._services[workspace]

            if workspace in self._backoff:
                last_ts, interval = self._backoff[workspace]
                if time.time() - last_ts < interval:
                    raise RAGServiceUnavailableError(
                        detail=f"Workspace '{workspace}' in backoff (retry in {interval:.0f}s)"
                    )

            try:
                ws_config = self._config.model_copy(update={"workspace": workspace})
                svc = await RAGService.acreate(
                    config=ws_config,
                    rerank_supports_vision=self._rerank_supports_vision,
                )
                self._services[workspace] = svc

                # Clear backoff on success
                self._backoff.pop(workspace, None)

                logger.info("Created RAGService for workspace '%s'", safe_log_text(workspace))
                return svc
            except SchemaValidationError:
                # Terminal: no backoff or retry can repair an absent schema, and
                # startup must see the exact failure, not a generic unavailable.
                raise
            except Exception as e:
                error_msg = self._actionable_error(e)
                # Per-workspace exponential backoff
                _, prev_interval = self._backoff.get(workspace, (0, 7.5))
                new_interval = min(prev_interval * 2, _MAX_RETRY_INTERVAL)
                self._backoff[workspace] = (time.time(), new_interval)
                logger.error(
                    "RAGService creation failed for '%s': %s. Retry in %ss",
                    safe_log_text(workspace),
                    safe_log_text(error_msg),
                    new_interval,
                )
                raise RAGServiceUnavailableError(detail=error_msg) from e

    # --- Write operations (single workspace) ---

    async def aingest(
        self,
        workspace: str,
        request: IngestSpec,
    ) -> dict[str, Any]:
        """Start an ingest job, wait according to config, and return the result if ready."""
        job = await self.astart_ingest_job(workspace, request)
        row = await self.ajoin_ingest_job(job["job_id"], timeout=self._config.ingest_timeout)
        if row is None:
            raise RAGServiceUnavailableError(detail=f"Ingest job disappeared: {job['job_id']}")
        status = str(row.get("status") or "")
        if status in JOB_STATES_WITH_RESULT:
            result = row.get("result")
            return result if isinstance(result, dict) else {}
        if status == "failed":
            raw_errors = row.get("errors")
            errors = raw_errors if isinstance(raw_errors, list) else []
            detail = "; ".join(str(error) for error in errors) or "Ingest job failed"
            raise RAGServiceUnavailableError(detail=detail)
        return row

    async def aingest_source(
        self,
        workspace: str,
        source: AsyncDataSource,
        *,
        source_type: str = "source",
        documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument] | None = None,
        prefix: str | None = None,
        source_uri_for_key: Callable[[str], str] | None = None,
        download_uri_for_key: Callable[[str], str | None] | None = None,
        replace: bool | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        retain_source_file: bool | None = None,
    ) -> dict[str, Any]:
        """Ingest from an in-memory SDK data source without durable job recovery."""
        svc = await self._get_service(workspace)
        await svc.aregister_workspace()
        return await svc.aingest_source(
            source,
            source_type=source_type,
            documents=documents,
            prefix=prefix,
            source_uri_for_key=source_uri_for_key,
            download_uri_for_key=download_uri_for_key,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            retain_source_file=retain_source_file,
        )

    async def _start_ingest_job_recovery(self) -> None:
        # Recovery resumes corpus writes, so it stays with the writer role.
        if self._config.is_reader:
            return
        try:
            await self._ingest_jobs.start_recovery()
        except Exception:
            self._health.add_warning("Ingest job recovery unavailable")
            logger.warning("Ingest job recovery initialization failed", exc_info=True)

    @property
    def answer_image_capability(self) -> AnswerImageCapability | None:
        """Query-role answer-model image capability, discovered at startup."""
        return self._answer_image_capability

    async def _probe_role_image_capabilities(self) -> None:
        """Resolve the answer, VLM, and rerank image capabilities once at startup."""
        await self._probe_answer_image_capability()
        await self._probe_vlm_image_capability()
        await self._probe_rerank_image_capability()
        answer = self._answer_image_capability
        logger.info(
            "Image capability by role: answer=%s (ceiling=%s, effective=%s) vlm=%s rerank=%s",
            answer.status if answer is not None else "unknown",
            answer.configured_ceiling if answer is not None else 0,
            answer.effective_max_images if answer is not None else 0,
            self._vlm_image_status,
            "not probed" if self._rerank_supports_vision is None else self._rerank_supports_vision,
        )

    async def _probe_answer_image_capability(self) -> None:
        """Discover the query-role answer model's image capability once at startup."""
        if self._answer_image_capability is not None:
            return
        self._cache_answer_image_capability(await self._discover_answer_image_capability())

    def _cache_answer_image_capability(self, capability: AnswerImageCapability) -> None:
        self._answer_image_capability = capability
        self._health.set_answer_image_capability(answer_image_capability_summary(capability))
        self._narrow_role_image_profile("query", capability.status)

    async def _maybe_reprobe_answer_image_capability(self) -> None:
        """Lazily re-probe when the cached answer capability is ``unknown``.

        ``supported``/``unsupported`` are terminal and never re-probed. An
        ``unknown`` verdict (a transient startup probe failure) is retried on
        demand -- when an image request actually needs it -- and the shared probe
        cache bounds that retry to one model call per cooldown window, so a
        genuinely-unreachable model is never hammered.
        """
        capability = self._answer_image_capability
        if capability is not None and capability.status != "unknown":
            return
        self._cache_answer_image_capability(await self._discover_answer_image_capability())

    async def _confirmed_live_answer_image_context(
        self,
        _models: _RequestModelContext,
    ) -> tuple[_RequestModelContext, AnswerImageCapability | None]:
        """Refresh and return one internally consistent live image context."""
        await self._maybe_reprobe_answer_image_capability()
        refreshed = self._request_model_context(None)
        return refreshed, self._answer_image_capability

    async def _pinned_answer_image_context(
        self,
        models: _RequestModelContext,
    ) -> tuple[_RequestModelContext, AnswerImageCapability]:
        """Project the already accepted query capability without a live probe."""
        return models, self._answer_image_capability_from_profile(models.query)

    async def _discover_answer_image_capability(self) -> AnswerImageCapability:
        """Probe ``model_settings_for_role(config, "query")`` and build a capability.

        Probes the model the AnswerSynthesizer actually uses -- not ``llm.default``.
        A non-positive deployment ceiling disables answer images without any model
        call, and without recording that config choice against a model another
        role may share. Best-effort otherwise: failures degrade to ``unknown``.
        """
        ceiling = int(self._config.answer.max_images)
        cfg = model_settings_for_role(self._config, "query")
        if ceiling <= 0:
            outcome = ImageProbeOutcome(status="unsupported", failure_kind="config_disabled")
        elif not self._declared_model_profile("query").supports_images:
            outcome = ImageProbeOutcome(
                status="unsupported",
                failure_kind="profile_declared_unsupported",
            )
        else:
            outcome = await self._image_capabilities.resolve(cfg)
        return AnswerImageCapability(
            status=outcome.status,
            configured_ceiling=ceiling,
            effective_max_images=derive_effective_max_images(outcome.status, ceiling),
            provider=cfg.provider,
            base_url=cfg.base_url,
            model=cfg.model,
            failure_kind=outcome.failure_kind,
        )

    async def _probe_vlm_image_capability(self) -> None:
        """Resolve the VLM role's own image capability.

        The VLM role drives query-image description and ``inspect_resource``; it
        may resolve to a different model than the answer role, so a text-only
        answer model must not withdraw visual inspection (or the reverse). A
        non-positive deployment ceiling leaves no image slot for any role, so it
        settles the role without spending a model call.
        """
        if int(self._config.answer.max_images) <= 0:
            self._vlm_image_status = "unsupported"
            self._narrow_role_image_profile("vlm", self._vlm_image_status)
            return
        if not self._declared_model_profile("vlm").supports_images:
            self._vlm_image_status = "unsupported"
            self._narrow_role_image_profile("vlm", self._vlm_image_status)
            return
        outcome = await self._image_capabilities.resolve(
            model_settings_for_role(self._config, "vlm")
        )
        self._vlm_image_status = outcome.status
        self._narrow_role_image_profile("vlm", self._vlm_image_status)

    async def _maybe_reprobe_vlm_image_capability(self) -> None:
        """Lazily re-probe the VLM role while its capability is still ``unknown``."""
        if self._vlm_image_status != "unknown":
            return
        await self._probe_vlm_image_capability()

    async def _probe_rerank_image_capability(self) -> None:
        """Resolve the rerank scoring model's image capability once at startup.

        Only the ``chat_llm_reranker`` strategy sends image blocks to a scoring
        model, so probing is skipped entirely for other strategies (and when
        reranking is disabled). Stored on this manager instance so SDK callers can
        run multiple managers with different model configs in one process.
        """
        if self._rerank_supports_vision is not None:
            return
        if not (
            self._config.rerank.enabled and self._config.rerank.strategy == "chat_llm_reranker"
        ):
            return  # no rerank model consumes image input; nothing to probe

        outcome = await self._image_capabilities.resolve(
            rerank_scoring_model_settings(self._config)
        )
        self._rerank_supports_vision = {"supported": True, "unsupported": False}.get(outcome.status)

    async def astart_ingest_job(
        self,
        workspace: str,
        request: IngestSpec,
    ) -> dict[str, Any]:
        """Start a background ingest job and return its durable job row."""
        self._config.require_writer("ingestion")
        kwargs = ingest_kwargs_from_payload(request)
        return await self._ingest_jobs.start_job(
            workspace,
            request.source_type,
            cleanup_paths=_cleanup_paths_for_local_ingest(
                source_type=request.source_type,
                path=request.path,
            ),
            **kwargs,
        )

    async def ajoin_ingest_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Wait for an in-process ingest job without cancelling it on timeout."""
        self._config.require_writer("ingest job access")
        return await self._ingest_jobs.await_job(job_id, timeout=timeout)

    async def aget_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        self._config.require_writer("ingest job access")
        return await self._ingest_jobs.get_job(job_id)

    async def acancel_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        """Stop a running ingest job, keeping whatever it already ingested."""
        self._config.require_writer("ingest job cancellation")
        job = await self._ingest_jobs.get_job(job_id)
        if job is None:
            return None
        await self._ingest_jobs.cancel_job(job_id, workspace=str(job.get("workspace", "")))
        return await self._ingest_jobs.get_job(job_id)

    def _get_file_panel_store(self) -> PGFilePanelStore:
        if self._file_panel_store is None:
            from dlightrag.storage.file_panel import PGFilePanelStore

            self._file_panel_store = PGFilePanelStore()
        return self._file_panel_store

    async def acreate_workspace(self, workspace: str, *, display_name: str | None = None) -> None:
        """Initialize a workspace through the public manager API."""
        self._config.require_writer("workspace creation")
        svc = await self._get_service(workspace)
        await svc.aregister_workspace(display_name=display_name)

    async def aget_file_panel_snapshot(self, workspace: str) -> dict[str, Any]:
        """Return files-panel data without warming a cold RAG service."""
        workspace_id = normalize_workspace(workspace)
        files = await self._get_file_panel_store().list_processed_files(workspace_id)

        if workspace_id in self._services:
            pipeline_status = await self._services[workspace_id].aget_pipeline_status()
        elif self._ingest_jobs.has_active_workspace_job(workspace_id):
            pipeline_status = {
                "busy": True,
                "pending_enqueues": 0,
                "latest_message": "Starting ingest...",
            }
        else:
            pipeline_status = {
                "busy": False,
                "pending_enqueues": 0,
                "latest_message": "",
            }

        return {
            "files": files,
            "pipeline_status": pipeline_status,
        }

    async def aprepare_source_download(
        self,
        workspace: str,
        document_id: str,
    ) -> SourceDownloadTarget:
        """Prepare a source download without warming a workspace model service."""
        from dlightrag.core.source_download import SourceDownloadService
        from dlightrag.storage.pg_metadata_index import PGMetadataIndex

        workspace_id = normalize_workspace(workspace)
        index = PGMetadataIndex(workspace=workspace_id)
        service = SourceDownloadService(
            config=self._config,
            metadata_index=index,
            workspace=workspace_id,
        )
        return await service.prepare(document_id)

    async def alist_ingested_files(self, workspace: str) -> list[dict[str, Any]]:
        """List ingested files in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def aget_pipeline_status(self, workspace: str) -> dict[str, Any]:
        """Return pipeline progress for a workspace."""
        svc = await self._get_service(workspace)
        return await svc.aget_pipeline_status()

    async def adelete_files(
        self,
        workspace: str,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Delete files from a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            dry_run=dry_run,
        )

    async def alist_failed_docs(self, workspace: str) -> list[dict[str, Any]]:
        """List FAILED documents in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_failed_docs()

    async def aget_visual_asset(
        self, workspace: str, chunk_id: str, *, size: VisualAssetSize = "full"
    ) -> Any:
        """Resolve a visual chunk asset for browser/API image routes."""
        svc = await self._get_service(workspace)
        return await svc.aget_visual_asset(chunk_id, size=size)

    async def aretry_failed_docs(self, workspace: str) -> dict[str, Any]:
        """Retry all FAILED documents in a specific workspace via re-ingest."""
        svc = await self._get_service(workspace)
        return await svc.aretry_failed_docs()

    async def aget_metadata(self, workspace: str, doc_id: str) -> dict[str, Any]:
        """Get document metadata by ID."""
        svc = await self._get_service(workspace)
        return await svc.aget_metadata(doc_id)

    async def aupdate_metadata(
        self,
        workspace: str,
        doc_id: str,
        data: dict[str, Any],
    ) -> None:
        """Update (merge) document metadata."""
        svc = await self._get_service(workspace)
        await svc.aupdate_metadata(doc_id, data)

    async def asearch_metadata(self, workspace: str, filters: MetadataFilter) -> list[str]:
        """Search metadata by filters, return matching doc_ids."""
        svc = await self._get_service(workspace)
        return await svc.asearch_metadata(filters)

    async def areset(
        self,
        *,
        workspace: str | None = None,
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reset one or all workspaces.

        Drops all storage backends, clears DlightRAG indexes, and optionally
        removes local files. After reset, the service is closed and evicted
        from cache.
        """
        self._config.require_writer("workspace reset")
        if workspace is not None:
            requested_workspace = workspace
            target_workspace = normalize_workspace(workspace)
            known = await self.alist_workspaces()
            if target_workspace not in known and target_workspace not in self._services:
                cancelled_jobs = (
                    0 if dry_run else await self._ingest_jobs.cancel_for_workspace(target_workspace)
                )
                from dlightrag.core.reset import areset_orphaned_workspace

                result = await areset_orphaned_workspace(
                    requested_workspace,
                    keep_files=keep_files,
                    dry_run=dry_run,
                    input_dir=str(self._config.input_dir_path),
                )
                await self._ingest_jobs.attach_reset_result(
                    workspace=target_workspace,
                    result=result,
                    dry_run=dry_run,
                )
                result["ingest_jobs_cancelled"] = cancelled_jobs
                return {
                    "workspaces": {target_workspace: result},
                    "total_errors": len(result.get("errors", [])),
                }
            workspaces = [target_workspace]
        else:
            workspaces = await self.alist_workspaces()

        results: dict[str, Any] = {}
        total_errors = 0

        for ws in workspaces:
            cancelled_jobs = 0 if dry_run else await self._ingest_jobs.cancel_for_workspace(ws)
            try:
                svc = await self._get_service(ws)
                ws_result = await svc.areset(keep_files=keep_files, dry_run=dry_run)
                ws_result["ingest_jobs_cancelled"] = cancelled_jobs
                await self._ingest_jobs.attach_reset_result(
                    workspace=ws,
                    result=ws_result,
                    dry_run=dry_run,
                )
                results[ws] = ws_result
                total_errors += len(ws_result.get("errors", []))
            except Exception as exc:
                results[ws] = {
                    "error": "workspace reset failed",
                    "ingest_jobs_cancelled": cancelled_jobs,
                }
                total_errors += 1
                logger.warning(
                    "Failed to reset workspace '%s': %s",
                    safe_log_text(ws),
                    safe_log_text(exc),
                )

            # Close and evict from cache even after reset errors, but never for a
            # dry run -- a preview must not tear down the live workspace runtime.
            if not dry_run and ws in self._services:
                try:
                    await self._services[ws].aclose()
                except Exception:
                    logger.warning(
                        "Failed to close service for '%s'",
                        safe_log_text(ws),
                        exc_info=True,
                    )
                del self._services[ws]

        return {"workspaces": results, "total_errors": total_errors}

    def _get_answer_synthesizer(
        self,
        model_profile: ModelProfile,
    ) -> AnswerSynthesizer:
        """Lazy-create the AnswerSynthesizer from global config."""
        if self._health.is_closed:
            raise RAGServiceUnavailableError("RAG service manager is closed")
        if cached := self._answer_synthesizers_by_profile.get(model_profile):
            return cached
        synthesizer = AnswerSynthesizer(
            model_func=None,
            image_policy=self._answer_image_policy(model_profile),
            model_profile=model_profile,
            context_policy=CONTEXT_POLICY,
        )
        if self._answer_model is None:
            self._answer_model = CompletionModel(
                model_settings_for_role(self._config, "query"),
                telemetry=LangfuseTelemetry(),
            )
        synthesizer.model_func = self._answer_model
        self._answer_synthesizers_by_profile[model_profile] = synthesizer
        return synthesizer

    def _answer_image_policy(
        self,
        profile: ModelProfile,
    ) -> AnswerImagePolicy:
        """Compose the Answer transport policy for the answer model's own capability."""
        return self._image_policy(
            int(self._config.answer.max_images) if profile.supports_images else 0
        )

    def _answer_image_capability_from_profile(
        self,
        profile: ModelProfile,
    ) -> AnswerImageCapability:
        settings = model_settings_for_role(self._config, "query")
        ceiling = int(self._config.answer.max_images)
        status: ImageCapabilityStatus = "supported" if profile.supports_images else "unsupported"
        return AnswerImageCapability(
            status=status,
            configured_ceiling=ceiling,
            effective_max_images=derive_effective_max_images(status, ceiling),
            provider=settings.provider,
            base_url=settings.base_url,
            model=settings.model,
            failure_kind=None if profile.supports_images else "pinned_profile_unsupported",
        )

    def _vlm_image_policy(
        self,
        profile: ModelProfile,
    ) -> AnswerImagePolicy:
        """Compose the same transport policy for the VLM role's own capability."""
        return self._image_policy(
            int(self._config.answer.max_images) if profile.supports_images else 0
        )

    def _image_policy(self, max_images: int) -> AnswerImagePolicy:
        answer = self._config.answer
        return AnswerImagePolicy(
            max_images=max_images,
            max_total_bytes=answer.image_max_total_bytes,
            max_bytes_per_image=answer.image_max_bytes,
            max_pixels=answer.image_max_pixels,
            max_px=answer.image_max_px,
            min_px=answer.image_min_px,
            quality=answer.image_quality,
            min_quality=answer.image_min_quality,
        )

    @staticmethod
    async def _budget_agent_images(
        current_images: list[dict[str, Any]],
        budget: AnswerImageBudget,
        resource_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        def build() -> list[dict[str, Any]]:
            blocks: list[dict[str, Any]] = []
            for index, image in enumerate(current_images, start=1):
                block = budget.add_user_image(image, label=f"query_image_{index}")
                if block is None:
                    raise CurrentImagePayloadError(
                        f"current image query_image_{index} could not fit the answer image budget"
                    )
                if index <= len(resource_ids):
                    blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"[current image {index} | resource: {resource_ids[index - 1]}]"
                            ),
                        }
                    )
                blocks.append(block)
            return blocks

        return await asyncio.to_thread(build)

    def _sem_bound(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Cap a DlightRAG-owned LLM callable by the direct-LLM concurrency semaphore.

        Replaces the old per-func priority queue: planner/vlm now run inline (so
        they nest under the request span), and this semaphore preserves the
        ``max_async`` concurrency bound the queue used to provide.
        """
        sem = self._direct_llm_sem

        async def _bounded(*args: Any, **kwargs: Any) -> Any:
            async with sem:
                return await func(*args, **kwargs)

        return _bounded

    def _get_retrieval_planner(
        self,
        model_profile: ModelProfile | None = None,
    ) -> RetrievalPlanner:
        """Return the manager-owned RetrievalPlanner, creating it when needed."""
        if self._health.is_closed:
            raise RAGServiceUnavailableError("RAG service manager is closed")
        profile = model_profile or self._model_profile("extract")
        if cached := self._retrieval_planners_by_profile.get(profile):
            return cached
        if self._planner_model is None:
            self._planner_model = CompletionModel(
                model_settings_for_role(self._config, "extract"),
                telemetry=LangfuseTelemetry(),
            )
        planner = RetrievalPlanner(
            llm_func=self._sem_bound(self._planner_model),
            model_profile=profile,
            context_policy=CONTEXT_POLICY,
        )
        self._retrieval_planners_by_profile[profile] = planner
        return planner

    def _get_web_search(self) -> ExaSearch | None:
        """Return the manager-owned web search client, or None when unconfigured."""
        if self._health.is_closed:
            raise RAGServiceUnavailableError("RAG service manager is closed")
        key = self._config.web_search.api_key
        if not key:
            return None
        if self._web_search is None:
            from dlightrag.core.retrieval.web_search import ExaSearch

            self._web_search = ExaSearch(key)
        return self._web_search

    def _get_query_tool_model(self) -> ToolModel:
        """Return the agent control model used by the research answer path."""
        if self._health.is_closed:
            raise RAGServiceUnavailableError("RAG service manager is closed")
        if self._query_tool_model is None:
            self._query_tool_model = ToolModel(
                model_settings_for_role(self._config, "query"),
                telemetry=LangfuseTelemetry(),
            )
        return self._query_tool_model

    def _query_image_describer(self) -> QueryImageDescriber:
        """Build a describer bound to the VLM role's current image capability.

        The describer holds only the shared VLM callable, a policy, and a count,
        so it is composed per request instead of cached: a lazy re-probe that
        settles ``unknown`` then takes effect immediately.
        """
        from dlightrag.core.request.images import QueryImageDescriber

        profile = self._model_profile("vlm")
        return QueryImageDescriber(
            vlm_func=self._get_or_create_vlm_func(),
            max_images=MAX_QUERY_IMAGES if profile.supports_images else 0,
            image_policy=self._vlm_image_policy(profile),
        )

    def _get_or_create_vlm_func(self) -> Callable[..., Any]:
        if self._health.is_closed:
            raise RAGServiceUnavailableError("RAG service manager is closed")
        if self._vlm_func is None:
            self._vlm_model = CompletionModel(
                model_settings_for_role(self._config, "vlm"),
                telemetry=LangfuseTelemetry(),
            )
            self._vlm_func = self._sem_bound(self._vlm_model)
        return self._vlm_func

    async def _get_schema(
        self,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Fetch a metadata schema for the requested workspace set."""
        normalized = tuple(normalize_query_workspaces(workspaces or ())) or (
            normalize_workspace(self._config.workspace),
        )
        ws_key = tuple(sorted(normalized))
        now = time.monotonic()
        cached = self._schema_cache.get(ws_key)
        if cached is not None and now - cached[0] < 300.0:
            return cached[1]

        from dlightrag.storage.pg_metadata_index import PGMetadataIndex

        try:
            schema = await PGMetadataIndex(
                workspace=normalize_workspace(self._config.workspace)
            ).get_field_schema(workspaces=ws_key)
        except Exception:
            logger.debug("Schema lookup failed for workspaces %s", ws_key, exc_info=True)
            # Never cache a failed lookup. Prefer the last-known-good entry
            # even when stale, and otherwise retry on the next request.
            return cached[1] if cached is not None else {}
        if (
            ws_key not in self._schema_cache
            and len(self._schema_cache) >= _SCHEMA_CACHE_MAX_ENTRIES
        ):
            oldest = min(self._schema_cache, key=lambda key: self._schema_cache[key][0])
            self._schema_cache.pop(oldest, None)
        self._schema_cache[ws_key] = (now, schema)
        return schema

    async def _plan_retrieval(
        self,
        query: str,
        *,
        text_history: PriorTurns | None,
        planner: RetrievalPlanner | None = None,
        current_image_descriptions: list[str] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
        preserve_query: bool | None = None,
    ) -> RetrievalPlan:
        """Plan one retrieval query inside the canonical retrieval operation."""
        effective_planner = planner or self._get_retrieval_planner()
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "retrieval_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces or []),
                "history_messages": len(text_history or PriorTurns()),
            },
        ) as trace:
            schema = await self._get_schema(workspaces)
            plan = await effective_planner.plan(
                query,
                conversation_history=text_history,
                schema=schema,
                current_image_descriptions=current_image_descriptions,
                preserve_query=preserve_query,
            )
            trace.update(
                output={
                    "standalone_query": plan.standalone_query,
                    "has_metadata_filter": plan.metadata_filter is not None,
                    "planning_outcome": plan.outcome,
                }
            )
            return plan

    def _start_query_service_warmup(self, workspaces: list[str] | tuple[str, ...]) -> None:
        """Initialize a request's cold workspaces now.

        Retrieval reaches the same services through ``_get_service``, whose
        per-workspace lock lets it join an initialization already running here
        instead of starting after planning or a control turn finishes.
        """
        cold = [
            ws for ws in normalize_query_workspaces(workspaces or ()) if ws not in self._services
        ]
        if not cold:
            return
        task = asyncio.create_task(self._warm_query_services(cold))
        self._warmups.add(task)
        task.add_done_callback(self._finish_query_service_warmup)

    def _finish_query_service_warmup(self, task: asyncio.Task[None]) -> None:
        self._warmups.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            # Retrieval raises the same failure to the caller; this only observes it.
            logger.debug("Workspace warm-up failed", exc_info=error)

    async def _warm_query_services(
        self,
        workspaces: list[str] | tuple[str, ...] | None,
    ) -> None:
        """Initialize only the services selected for an imminent query."""
        selected = tuple(normalize_query_workspaces(workspaces or ())) or (
            normalize_workspace(self._config.workspace),
        )
        semaphore = asyncio.Semaphore(_QUERY_WORKSPACE_MAX_CONCURRENCY)

        async def _warm(workspace: str) -> None:
            async with semaphore:
                await self._get_service(workspace)

        tasks = [asyncio.create_task(_warm(workspace)) for workspace in selected]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _open_query_workspaces(
        self,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
    ) -> list[str]:
        """Resolve a request's workspace scope and start initializing the cold ones.

        Nothing can warm earlier than this: ``all_workspaces`` is a flag, not a
        list, so which services to open is unknown until the registry answers.
        """
        resolved = await self._resolve_query_workspace_scope(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        self._start_query_service_warmup(resolved)
        return resolved

    async def _resolve_query_workspace_scope(
        self,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
    ) -> list[str]:
        """Resolve the stable public workspace set without starting services."""
        validate_query_workspace_selection(
            all_workspaces=all_workspaces,
            workspace=workspace,
            workspaces=workspaces,
        )
        available = await self.alist_workspaces() if all_workspaces else None
        resolved = resolve_query_workspaces(
            default_workspace=self._config.workspace,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            available_workspaces=available,
        )
        return resolved

    # --- Read operations (single or federated) ---

    async def aretrieve(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: list[dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        ``query_images`` are current-request images. VLM descriptions inform
        query planning, and verified image blocks are embedded only when optional
        direct visual retrieval is active. Public retrieval is stateless: it
        accepts neither history nor Web attachment documents.
        """
        current_images = list(query_images or [])
        if len(current_images) > MAX_QUERY_IMAGES:
            raise CurrentImagePayloadError(f"at most {MAX_QUERY_IMAGES} current images are allowed")
        ws_list = await self._open_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        return await self._retrieve(
            query,
            workspaces=ws_list,
            history=None,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            bm25_query=bm25_query,
            filters=filters,
            query_images=current_images,
            image_descriptions=None,
        )

    async def _retrieve(
        self,
        query: str,
        *,
        workspaces: list[str],
        history: PriorTurns | None,
        top_k: int | None,
        chunk_top_k: int | None,
        bm25_query: str | None,
        filters: MetadataFilter | None,
        query_images: list[dict[str, Any]] | None,
        image_descriptions: list[str] | None,
        preserve_query: bool | None = None,
        planner: RetrievalPlanner | None = None,
    ) -> RetrievalResult:
        """Plan and execute one retrieval over already-resolved workspaces."""
        requested_top_k = _positive_int_or_none(top_k)
        requested_chunk_top_k = _positive_int_or_none(chunk_top_k)
        current_images = list(query_images or [])
        kwargs: dict[str, Any] = {
            "top_k": requested_top_k or self._config.top_k,
            "chunk_top_k": requested_chunk_top_k or self._config.chunk_top_k,
        }
        if current_images:
            kwargs["query_image_blocks"] = current_images
        from dlightrag.observability import trace_observation

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_observation(
                    "retrieve",
                    as_type="retriever",
                    input={"query": query},
                    metadata={
                        "workspaces": workspaces,
                        "top_k": kwargs["top_k"],
                        "chunk_top_k": kwargs["chunk_top_k"],
                        "has_filters": filters is not None,
                    },
                ) as trace:
                    descriptions = image_descriptions
                    if descriptions is None:
                        if current_images:
                            await self._maybe_reprobe_vlm_image_capability()
                        descriptions = await prepare_query_images(
                            query_images=current_images,
                            describer=self._query_image_describer(),
                        )
                    plan = await self._plan_retrieval(
                        query,
                        text_history=history,
                        planner=planner,
                        current_image_descriptions=descriptions or None,
                        workspaces=workspaces,
                        preserve_query=preserve_query,
                    )
                    effective_query = plan.standalone_query
                    effective_filters = filters if filters is not None else plan.metadata_filter
                    filter_source = (
                        "explicit" if filters is not None else plan.metadata_filter_source
                    )
                    effective_bm25_query = (bm25_query or "").strip() or plan.bm25_query
                    if effective_filters is not None:
                        kwargs["filters"] = effective_filters
                    if filter_source is not None:
                        kwargs["filter_source"] = filter_source
                    if effective_bm25_query is not None:
                        kwargs["bm25_query"] = effective_bm25_query
                    if len(workspaces) == 1:
                        svc = await self._get_service(workspaces[0])
                        result = await svc.aretrieve(effective_query, **kwargs)
                    else:
                        result = await federated_retrieve(
                            effective_query,
                            workspaces,
                            self._get_service,
                            max_concurrency=_QUERY_WORKSPACE_MAX_CONCURRENCY,
                            **kwargs,
                        )
                    result.image_descriptions = descriptions
                    result.trace["query_image_description_count"] = len(descriptions)
                    trace.update(
                        output={
                            **_context_output(result.contexts),
                            "standalone_query": effective_query,
                            "query_image_description_count": len(descriptions),
                        }
                    )
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def _resolve_answer_resources(
        self,
        resources: list[ResourceInput] | None,
        *,
        models: _RequestModelContext,
        text_window_budget: TextWindowBudget,
        confirm_image_context: Callable[
            [_RequestModelContext],
            Awaitable[tuple[_RequestModelContext, AnswerImageCapability | None]],
        ],
        fetched_bytes_sink: Callable[[Any], Awaitable[None]] | None = None,
    ) -> _ResolvedAnswerResources:
        """Resolve resource capabilities, manifests, tools, and image transport once."""
        if resources and not models.query.supports_tools:
            raise AnswerModelCapabilityError()
        declared_image_count = sum(
            1
            for resource in resources or ()
            if resource.loader is None
            and (resource.declared_mime or "").lower().startswith("image/")
        )
        image_capability: AnswerImageCapability | None = None
        if declared_image_count:
            models, image_capability = await confirm_image_context(models)
            check_answer_image_capability(
                image_count=declared_image_count,
                capability=image_capability,
            )
        (
            current_images,
            remaining_resources,
            current_image_resources,
        ) = await self._prepare_current_images(resources)
        if current_images and not declared_image_count:
            models, image_capability = await confirm_image_context(models)
        check_answer_image_capability(
            image_count=len(current_images),
            capability=image_capability,
        )

        web_search = self._get_web_search()
        registry, resource_tools = self._build_resource_context(
            remaining_resources,
            text_window_budget=text_window_budget,
            web_search=web_search,
            fetched_bytes_sink=fetched_bytes_sink,
            vlm_profile=models.vlm,
        )
        try:
            current_image_resource_ids = (
                tuple(registry.register(resource) for resource in current_image_resources)
                if registry is not None
                else ()
            )
            resource_manifest = registry.manifest() if registry is not None else ()
            research = web_search is not None or bool(resource_manifest)
            image_budget: AnswerImageBudget | None = None
            query_images: list[dict[str, Any]] | None = current_images or None
            if research:
                image_budget = self._answer_image_policy(models.query).new_budget()
                query_images = (
                    await self._budget_agent_images(
                        current_images,
                        image_budget,
                        current_image_resource_ids,
                    )
                    or None
                )
            return _ResolvedAnswerResources(
                models=models,
                web_search=web_search,
                registry=registry,
                resource_tools=resource_tools,
                resource_manifest=resource_manifest,
                current_images=current_images,
                current_image_count=len(current_images),
                research=research,
                image_budget=image_budget,
                query_images=query_images,
            )
        except BaseException:
            if registry is not None:
                await registry.aclose()
            raise

    async def _prepare_orchestrated_run(
        self,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None,
        fetched_bytes_sink: Callable[[Any], Awaitable[None]] | None = None,
        pinned_image_descriptions: tuple[str, ...],
        projected_history: PriorTurns,
        model_profiles: Mapping[ModelRole, ModelProfile],
    ) -> _OrchestratorRun:
        """Build execution collaborators from one immutable accepted run input."""
        history = projected_history
        models = self._request_model_context(model_profiles)
        query_profile = models.query
        extract_profile = models.extract
        ws_list = await self._open_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        planner = self._get_retrieval_planner(extract_profile)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(query_profile))
        resolved = await self._resolve_answer_resources(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._pinned_answer_image_context,
            fetched_bytes_sink=fetched_bytes_sink,
        )
        try:
            image_descriptions = list(pinned_image_descriptions)

            async def retrieve_knowledge_base(search_query: str) -> RetrievalResult:
                return await self._retrieve(
                    search_query,
                    workspaces=ws_list,
                    history=history,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    bm25_query=None,
                    filters=filters,
                    query_images=resolved.current_images,
                    image_descriptions=image_descriptions,
                    preserve_query=True if resolved.research else None,
                    planner=planner,
                )

            model_func: Callable[..., Any] | None = None
            stream_model_func: Callable[..., AsyncIterator[str]] | None = None
            if resolved.research:
                tool_model = self._get_query_tool_model()

                async def _model_func(**kwargs: Any) -> Any:
                    async with self._direct_llm_sem:
                        return await tool_model(**kwargs)

                def _stream_model_func(**kwargs: Any) -> AsyncIterator[str]:
                    async def _bounded() -> AsyncIterator[str]:
                        await self._direct_llm_sem.acquire()
                        inner = tool_model.stream_text(**kwargs)
                        try:
                            async for token in inner:
                                yield token
                        finally:
                            try:
                                close = getattr(inner, "aclose", None)
                                if callable(close):
                                    result = close()
                                    if inspect.isawaitable(result):
                                        await cast(Awaitable[Any], result)
                            finally:
                                self._direct_llm_sem.release()

                    return _bounded()

                model_func = _model_func
                stream_model_func = _stream_model_func

            synthesizer = self._get_answer_synthesizer(query_profile)
            orchestrator = AnswerOrchestrator(
                synthesizer=synthesizer,
                retrieve_knowledge_base=retrieve_knowledge_base,
                search_web=(
                    resolved.web_search.search if resolved.web_search is not None else None
                ),
                model_func=model_func,
                stream_model_func=stream_model_func,
                resource_tools=resolved.resource_tools,
                resource_manifest=resolved.resource_manifest,
                register_web_source=(
                    resolved.registry.register_discovered_link
                    if resolved.registry is not None and resolved.web_search is not None
                    else None
                ),
                image_budget=resolved.image_budget,
                text_window_budget=text_window_budget,
                model_profile=query_profile,
                context_policy=CONTEXT_POLICY,
                max_agent_turns=self._config.max_agent_turns,
                telemetry=LangfuseTelemetry(),
            )

            return _OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=image_descriptions,
                query_images=resolved.query_images,
                history=history,
                current_image_count=resolved.current_image_count,
                ws_list=ws_list,
                registry=resolved.registry,
            )
        except BaseException:
            if resolved.registry is not None:
                await resolved.registry.aclose()
            raise

    async def _prepare_current_images(
        self,
        resources: list[ResourceInput] | None,
    ) -> tuple[list[dict[str, Any]], list[ResourceInput], list[ResourceInput]]:
        """Build current-image blocks while retaining every attachment as a resource.

        Inline bytes that decode as a real image and remote image links
        (materialized under SSRF revalidation) become internal current-image
        blocks fed to VLM description, direct visual retrieval, and final answer
        transport. Their verified bytes also stay in the request-local registry
        for focused ``inspect_resource`` calls, so one preparation pass never
        fetches the same remote image twice. Durable lazy resources (prior
        attachments) and every non-image
        resource remain lazy. Non-image bytes never enter the image chain.
        """
        if not resources:
            return [], [], []
        from dlightrag.core.resources import ResourceInput

        max_pixels = self._config.answer.image_max_pixels
        images: list[dict[str, Any]] = []
        remaining: list[ResourceInput] = []
        image_resources: list[ResourceInput] = []
        for resource in resources:
            data: bytes | None = None
            if resource.loader is not None:
                remaining.append(resource)
                continue
            if resource.content is not None:
                data = resource.content
            elif resource.url is not None and (resource.declared_mime or "").lower().startswith(
                "image/"
            ):
                data = await self._materialize_link_image(resource.url)
            if data is None:
                remaining.append(resource)
                continue
            try:
                mime, data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=max_pixels,
                )
            except ValueError:
                remaining.append(resource)
                continue
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
            image_resource = ResourceInput(
                filename=resource.filename,
                content=data,
                declared_mime=mime,
            )
            remaining.append(image_resource)
            image_resources.append(image_resource)
        return images, remaining, image_resources

    async def _materialize_link_image(self, url: str) -> bytes | None:
        """Fetch a current-image link under SSRF revalidation; None if it fails."""
        from dlightrag.sourcing.url import afetch_public_https_bytes, avalidate_public_https_url

        try:
            await avalidate_public_https_url(url)
            return await afetch_public_https_bytes(
                url,
                max_bytes=self._config.answer.image_max_bytes,
                timeout=120.0,
            )
        except Exception:
            logger.warning("Failed to materialize current image link", exc_info=True)
            return None

    def _build_resource_context(
        self,
        resources: list[ResourceInput] | None,
        *,
        text_window_budget: TextWindowBudget,
        web_search: ExaSearch | None = None,
        fetched_bytes_sink: Callable[[Any], Awaitable[None]] | None = None,
        vlm_profile: ModelProfile,
    ) -> tuple[ResourceRegistry | None, list[AgentTool]]:
        """Register request-local resources and their peer tools.

        When Exa is configured, its Contents endpoint is adapted into the
        registry's provider-neutral URL text fallback so a link whose direct
        fetch or local conversion fails or comes back empty can recover exactly
        one text view. The registry owns admission, SSRF revalidation, and the
        single-fallback contract; it never imports any web-search provider.
        """
        if not resources and web_search is None:
            return None, []
        from dlightrag.core.resources import ResourceRegistry
        from dlightrag.core.resources.visual import ResourceInspector
        from dlightrag.core.tools.resources import build_resource_tools

        answer = self._config.answer
        registry = ResourceRegistry(
            max_attachments=answer.max_attachments,
            max_attachment_bytes=answer.max_attachment_bytes,
            max_total_attachment_bytes=answer.max_total_attachment_bytes,
            url_text_fallback=_exa_contents_text(web_search) if web_search is not None else None,
            fetched_bytes_sink=fetched_bytes_sink,
        )
        try:
            for resource in resources or []:
                registry.register(resource)
        except (ValueError, ResourceRegistryError) as exc:
            raise AnswerResourceAdmissionError() from exc

        # Visual inspection is a VLM-role capability: a text-only answer model
        # must not withdraw it, and a zero effective ceiling leaves no image slot,
        # so an inspector built on that policy could only ever fail.
        vlm_policy = self._vlm_image_policy(vlm_profile)
        visual_supported = vlm_profile.supports_images and vlm_policy.max_images > 0
        inspector: ResourceInspector | None = None
        if visual_supported:
            inspector = ResourceInspector(
                registry,
                vlm_func=self._get_or_create_vlm_func(),
                image_policy=vlm_policy,
            )
        tools = build_resource_tools(
            registry,
            text_window_budget=text_window_budget,
            inspector=inspector,
            visual_supported=visual_supported,
        )
        return registry, tools

    # --- Durable answer runs ---

    async def _get_answer_run_store(self) -> PGAnswerRunStore:
        if self._answer_run_store is not None:
            return self._answer_run_store
        if self._health.is_closed:
            raise RAGServiceUnavailableError("Answer runtime is shutting down")
        async with self._answer_store_lock:
            if self._answer_run_store is None:
                from dlightrag.storage.answer_runs import PGAnswerRunStore as _Store

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
            coordinator = RunCoordinator(
                store=store,
                executor=_ManagerAnswerExecutor(self),
                max_async=int(self._config.max_async),
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

    async def _execute_answer_run(self, session: RunSession) -> Mapping[str, Any]:
        """Execute one claimed run from its immutable input and last checkpoint."""
        from dlightrag.citations.finalization import finalize_answer
        from dlightrag.citations.streaming import aclose_answer_stream
        from dlightrag.core.answer.media import (
            answer_images_from_sources,
        )
        from dlightrag.core.answer_runs.checkpoints import (
            encode_checkpoint_state,
            restore_agent_state,
        )
        from dlightrag.core.answer_runs.execution import (
            AnswerRunInput as _Input,
        )
        from dlightrag.core.answer_runs.execution import (
            SessionBoundaries,
        )
        from dlightrag.core.client_payloads import project_contexts_for_client
        from dlightrag.observability import trace_observation

        store = await self._get_answer_run_store()
        request = _Input.from_request(session.request)
        model_profiles = self._validate_pinned_model_profiles(request)
        # Retrieval planning, resource admission, and image description all run
        # before the first retrieval, so the run reports the planning phase the
        # durable contract names rather than opening on `searching`.
        await session.enter_phase("planning")
        projected_history = PriorTurns([dict(message) for message in request.history])
        run = await self._prepare_orchestrated_run(
            workspace=None,
            workspaces=list(request.workspaces) or None,
            all_workspaces=False,
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
            resources=await self._answer_run_resources(
                request, owner_id=session.owner_id, store=store
            ),
            fetched_bytes_sink=_fetched_bytes_sink(session, store),
            pinned_image_descriptions=request.image_descriptions,
            projected_history=projected_history,
            model_profiles=model_profiles,
        )
        stream: AsyncIterator[str] | None = None
        try:
            async with trace_observation(
                "answer_orchestration",
                as_type="chain",
                input={"query": request.query},
                metadata={
                    "run_id": session.run_id,
                    "research": run.orchestrator.uses_research_path,
                    "workspaces": run.ws_list,
                    "history_turns": len(run.history or []),
                    "query_image_count": run.current_image_count,
                    "semantic_highlights": request.semantic_highlights,
                },
            ) as pipeline_trace:
                prepared = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                if session.checkpoint is not None:
                    await restore_agent_state(
                        prepared.state,
                        {
                            "version": session.checkpoint.version,
                            "completed_turns": session.checkpoint.completed_turns,
                            "state": session.checkpoint.state,
                        },
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                        store=store,
                        expected_completed_turns=session.completed_turns,
                        load_corpus_image=self._load_corpus_image,
                    )

                async def _encode(state: AgentRunState) -> Mapping[str, Any]:
                    return await encode_checkpoint_state(
                        state, owner_id=session.owner_id, run_id=session.run_id, store=store
                    )

                boundaries = SessionBoundaries(session, encode=_encode)
                contexts, stream = await run.orchestrator.answer_stream(
                    request.query,
                    conversation_history=run.history,
                    run=prepared,
                    boundaries=boundaries,
                )
                answer_parts: list[str] = []
                if stream is not None:
                    async for chunk in stream:
                        answer_parts.append(chunk)
                        await session.emit_token(chunk)
                await session.flush_tokens()
                answer_text = getattr(stream, "answer", "") or "".join(answer_parts)
                finalized = finalize_answer(answer_text, contexts)
                if request.semantic_highlights:
                    from dlightrag.core.answer.highlights import enrich_semantic_highlights

                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        config=self._config,
                    )
                trace = dict(getattr(stream, "trace", None) or {})
                trace["query_image_description_count"] = len(run.image_descriptions)
                images = answer_images_from_sources(finalized.sources, contexts=contexts)
                pipeline_trace.update(
                    output=answer_trace_output(finalized.answer, finalized.sources, contexts)
                )
                return store_answer_result(
                    answer=finalized.answer,
                    # Answer images are derived from the raw contexts first; only the
                    # client-safe projection is durable, so no inline image payload or
                    # internal source locator is stored twice.
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    answer_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                )
        finally:
            await aclose_answer_stream(stream)
            if run.registry is not None:
                await run.registry.aclose()

    async def _answer_run_resources(
        self,
        request: AnswerRunInput,
        *,
        owner_id: str,
        store: PGAnswerRunStore,
    ) -> list[ResourceInput] | None:
        """Rebuild the run's ordered links and attachments over durable bytes.

        Image attachments are materialized now because current-turn images are
        verified and promoted into image blocks before the run starts; every
        other attachment, including every prior-turn one, stays lazy and is read
        only through resource tools.
        """
        if not request.links and not request.attachments and not request.history_attachments:
            return None

        async def _load(digest: str) -> bytes:
            content = await store.load_artifact(owner_id=owner_id, digest=digest)
            if content is None:
                raise CheckpointError(
                    "checkpoint_corrupt",
                    "Answer run attachment bytes no longer exist.",
                )
            return content

        def _loader(digest: str) -> Callable[[], Awaitable[bytes]]:
            async def _read() -> bytes:
                return await _load(digest)

            return _read

        resources = await build_current_answer_resources(
            links=request.links,
            attachments=request.attachments,
            attachment_loaders=[_loader(attachment.digest) for attachment in request.attachments],
        )
        resources.extend(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                loader=_loader(attachment.digest),
            )
            for attachment in request.history_attachments
        )
        return resources

    async def _load_corpus_image(self, workspace: str, chunk_id: str) -> str | None:
        """Resolve one knowledge-base visual, or ``None`` when it no longer exists."""
        try:
            asset = await self.aget_visual_asset(workspace, chunk_id)
        except Exception:
            logger.info(
                "Knowledge-base visual for '%s' no longer resolves; dropping the image block",
                safe_log_text(chunk_id),
            )
            return None
        content = getattr(asset, "data", None)
        if not content:
            return None
        return base64.b64encode(content).decode("ascii")

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
    ) -> RetrievalResult:
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
        ws_list = await self._resolve_query_workspace_scope(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
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
        from dlightrag.core.memory.evidence import EvidenceLedger
        from dlightrag.core.tools import compose_research_tools

        if resources:
            await self._maybe_reprobe_vlm_image_capability()
        model_profiles: dict[ModelRole, ModelProfile] = {
            role: self._model_profile(role) for role in MODEL_ROLE_NAMES
        }
        models = self._request_model_context(model_profiles)
        planner = self._get_retrieval_planner(models.extract)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(models.query))
        resolved = await self._resolve_answer_resources(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._confirmed_live_answer_image_context,
        )
        try:
            ws_list = await self._open_query_workspaces(
                workspace=None,
                workspaces=list(request.workspaces),
                all_workspaces=False,
            )
            models = resolved.models
            model_profiles["extract"] = models.extract
            model_profiles["query"] = models.query
            model_profiles["vlm"] = models.vlm
            image_descriptions = tuple(
                await prepare_query_images(
                    query_images=resolved.current_images,
                    describer=self._query_image_describer(),
                )
                if resolved.current_images
                else ()
            )
            schema = await self._get_schema(ws_list)
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
                    image_policy=self._answer_image_policy(models.query),
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

    def _validate_pinned_model_profiles(
        self,
        request: AnswerRunInput,
    ) -> dict[ModelRole, ModelProfile]:
        if request.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise IncompatibleAnswerRunError(
                "answer run context policy revision does not match this binary; "
                "drain active runs before deployment"
            )
        pinned = {item.role: item for item in request.pinned_models}
        if len(request.pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(
            MODEL_ROLE_NAMES
        ):
            raise IncompatibleAnswerRunError(
                "answer run does not contain the complete pinned model role set"
            )
        return {role: pinned[role].profile for role in MODEL_ROLE_NAMES}

    # --- Management ---

    async def alist_workspaces(self) -> list[str]:
        """Discover available workspaces."""
        records = await self.alist_workspace_records()
        return [row["workspace"] for row in records]

    async def alist_workspace_records(self) -> list[dict[str, Any]]:
        """Return registered workspace records for UI/API adapters."""
        try:
            registry = await self._get_workspace_registry()
            rows = await registry.list()
            if rows:
                return [self._serialize_workspace_record(row) for row in rows]
        except Exception as exc:
            logger.warning("Failed to list workspaces from registry: %s", exc)

        return [
            {
                "workspace": normalize_workspace(self._config.workspace),
                "display_name": self._config.workspace,
                "embedding_model": self._config.embedding.model,
                "created_at": None,
                "updated_at": None,
            }
        ]

    @staticmethod
    def _serialize_workspace_record(row: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-safe workspace record."""
        raw_workspace = str(row.get("workspace") or "")
        return {
            "workspace": raw_workspace,
            "display_name": str(row.get("display_name") or raw_workspace),
            "embedding_model": str(row.get("embedding_model") or ""),
            "created_at": _iso_or_none(row.get("created_at")),
            "updated_at": _iso_or_none(row.get("updated_at")),
        }

    async def aclose(self) -> None:
        """Close all managed RAGService instances."""
        from dlightrag.observability import shutdown_tracing

        self._health.mark_closed()
        cancellation: asyncio.CancelledError | None = None
        await self._ingest_jobs.close()
        if self._answer_coordinator is not None:
            coordinator, self._answer_coordinator = self._answer_coordinator, None
            try:
                await coordinator.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close the durable answer coordinator", exc_info=True)

        for warmup in list(self._warmups):
            warmup.cancel()
        if self._warmups:
            await asyncio.gather(*self._warmups, return_exceptions=True)

        async with self._vlm_func_lock:
            self._vlm_func = None
            vlm_model, self._vlm_model = self._vlm_model, None

        answer_model, self._answer_model = self._answer_model, None
        planner_model, self._planner_model = self._planner_model, None
        query_tool_model, self._query_tool_model = self._query_tool_model, None
        web_search, self._web_search = self._web_search, None
        self._answer_synthesizers_by_profile.clear()
        self._retrieval_planners_by_profile.clear()

        for component in (
            query_tool_model,
            answer_model,
            planner_model,
            vlm_model,
            web_search,
        ):
            close = getattr(component, "aclose", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close manager component", exc_info=True)

        for ws, svc in self._services.items():
            try:
                await svc.aclose()
            except asyncio.CancelledError as exc:
                cancellation = defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
        self._services.clear()

        from dlightrag.storage.pool import pg_pool

        try:
            await pg_pool.close()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        shutdown_tracing()
        if cancellation is not None:
            raise cancellation


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError("top-k limits must be positive integers")
    return result


__all__ = [
    "RAGServiceManager",
    "RAGServiceUnavailableError",
    "answer_trace_output",
]
