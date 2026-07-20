# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGServiceManager — unified multi-workspace RAG coordinator.

Absorbs pool.py workspace management and federation routing into a single
entry point. All API/MCP consumers depend on this class only.
"""

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig
    from dlightrag.core.query_images import QueryImageDescriber
    from dlightrag.core.source_download import SourceDownloadTarget
    from dlightrag.storage.file_panel import PGFilePanelStore
    from dlightrag.storage.workspaces import PGWorkspaceRegistry

from dlightrag.core.answer.capability import AnswerImageCapability, derive_effective_max_images
from dlightrag.core.answer.engine import AnswerEngine
from dlightrag.core.answer.errors import (
    ANSWER_IMAGE_CAPABILITY_UNKNOWN,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
)
from dlightrag.core.answer.turn import PreparedAnswerTurn
from dlightrag.core.client_contracts import IngestSpec, SourceType
from dlightrag.core.client_requests import ingest_kwargs_from_payload
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.ingest_job_coordinator import IngestJobCoordinator
from dlightrag.core.ingestion.paths import is_explicit_upload_batch_dir
from dlightrag.core.query_images import (
    PreparedQueryImages,
    prepare_query_images,
)
from dlightrag.core.query_planner import QueryPlan, QueryPlanner
from dlightrag.core.query_workspaces import (
    resolve_query_workspaces,
    validate_query_workspace_selection,
)
from dlightrag.core.retrieval.metadata_fields import MetadataIngestPolicy
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.scope import RequestScope
from dlightrag.core.service import RAGService
from dlightrag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag.utils import log_safe, normalize_workspace

logger = logging.getLogger(__name__)

_MAX_RETRY_INTERVAL: float = 300.0


@dataclass(frozen=True)
class _AnswerLimits:
    candidate_top_k: int
    context_top_k: int


class _ScopedAnswerStream:
    """Async iterator wrapper that releases a semaphore when streaming ends."""

    def __init__(self, inner: AsyncIterator[str], semaphore: asyncio.Semaphore) -> None:
        self._inner = inner
        self._aiter = inner.__aiter__()
        self._semaphore = semaphore
        self._released = False

    def __aiter__(self) -> _ScopedAnswerStream:
        return self

    async def __anext__(self) -> str:
        try:
            return await self._aiter.__anext__()
        except StopAsyncIteration:
            self._release()
            raise
        except BaseException:
            self._release()
            raise

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        try:
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
        finally:
            self._release()

    @property
    def answer(self) -> str:
        return getattr(self._inner, "answer", "")

    @property
    def trace(self) -> Any:
        return getattr(self._inner, "trace", None)

    @trace.setter
    def trace(self, value: Any) -> None:
        self._inner.trace = value  # type: ignore[attr-defined]

    @property
    def image_descriptions(self) -> Any:
        return getattr(self._inner, "image_descriptions", None)

    @image_descriptions.setter
    def image_descriptions(self, value: Any) -> None:
        self._inner.image_descriptions = value  # type: ignore[attr-defined]

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._semaphore.release()


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _normalize_workspaces(workspaces: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for workspace in workspaces or ():
        normalized = normalize_workspace(workspace)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _context_output(contexts: RetrievalContexts) -> dict[str, int]:
    return {
        "context_chunk_count": _context_count(contexts, "chunks"),
        "entity_count": _context_count(contexts, "entities"),
        "relationship_count": _context_count(contexts, "relationships"),
    }


def _answer_output(result: RetrievalResult) -> dict[str, int]:
    return {
        "answer_len": len(result.answer or ""),
        "source_count": len(result.sources or []),
        "context_chunk_count": _context_count(result.contexts, "chunks"),
    }


def _scope_for_workspaces(
    scope: RequestScope | None,
    workspaces: list[str] | tuple[str, ...] | None,
) -> RequestScope:
    base = scope or RequestScope.anonymous()
    normalized = _normalize_workspaces(workspaces)
    return base.for_workspaces(normalized) if normalized else base


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _cleanup_paths_for_local_ingest(*, source_type: SourceType, path: str | None) -> list[str]:
    if source_type != "local" or not path:
        return []
    return [path] if is_explicit_upload_batch_dir(Path(path)) else []


def _merge_schema_rows(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    if not schemas:
        return {}
    if len(schemas) == 1:
        return dict(schemas[0])

    columns: dict[str, dict[str, Any]] = {}
    custom_keys: set[str] = set()
    for schema in schemas:
        for column in schema.get("columns", []) or []:
            if isinstance(column, dict):
                name = str(column.get("name") or "")
                if name and name not in columns:
                    columns[name] = dict(column)
        for key in schema.get("custom_keys", []) or []:
            if key:
                custom_keys.add(str(key))
    return {
        "columns": list(columns.values()),
        "custom_keys": sorted(custom_keys),
    }


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


_ERROR_IMAGES_NOT_SUPPORTED = (
    "Current model does not support image input. Use a vision-capable model or remove query_images."
)
_ERROR_CAPABILITY_UNKNOWN = (
    "Answer-model image capability is unknown: the startup probe did not confirm image support. "
    "Provide a vision-capable query model or retry once the model is reachable."
)
# Lazy re-probe cooldown for an `unknown` answer-image capability (transient recovery).
_ANSWER_CAPABILITY_REPROBE_COOLDOWN_SECONDS = 30.0


def _check_answer_image_capability(
    *,
    query_images: list[dict[str, Any]] | None,
    capability: AnswerImageCapability | None,
) -> None:
    """Reject current images unless the query-role answer model is confirmed to accept them.

    Gates on the capability of ``model_for_role(config, "query")`` -- the model that
    actually receives the images -- not ``llm.default``. Fail-closed: only a confirmed
    ``supported`` verdict is admitted; ``unsupported`` raises
    ``CURRENT_IMAGES_UNSUPPORTED`` and ``unknown``/unprobed raises
    ``ANSWER_IMAGE_CAPABILITY_UNKNOWN``, so every surface returns the same clear error
    instead of a late provider or transport-budget failure.
    """
    if not query_images:
        return
    if capability is None or capability.status == "unknown":
        raise AnswerImageError(
            f"[ANSWER_IMAGE_CAPABILITY_UNKNOWN] {_ERROR_CAPABILITY_UNKNOWN}",
            error_kind=ANSWER_IMAGE_CAPABILITY_UNKNOWN,
        )
    if capability.status == "unsupported":
        raise AnswerImageError(
            f"[IMAGES_NOT_SUPPORTED_BY_MODEL] {_ERROR_IMAGES_NOT_SUPPORTED}",
            error_kind=CURRENT_IMAGES_UNSUPPORTED,
        )


class RAGServiceManager:
    """Multi-workspace RAG coordinator.

    Manages a pool of RAGService instances (one per workspace).
    Routes read operations to single workspace or federation.
    """

    def __init__(self, config: DlightragConfig | None = None) -> None:
        from dlightrag.config import get_config

        self._config = config or get_config()
        self._services: dict[str, RAGService] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Health/error tracking
        self._ready: bool = False
        self._degraded: bool = False
        self._startup_warnings: list[str] = []

        # Per-workspace backoff: workspace -> (last_error_ts, retry_interval)
        self._backoff: dict[str, tuple[float, float]] = {}

        self._answer_engine: AnswerEngine | None = None
        self._ingest_jobs = IngestJobCoordinator(
            self._get_ingest_service,
            input_root=self._config.input_dir_path,
        )
        self._query_planner: QueryPlanner | None = None
        self._query_image_describer: QueryImageDescriber | None = None
        self._workspace_registry: PGWorkspaceRegistry | None = None
        self._file_panel_store: PGFilePanelStore | None = None
        self._schema_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
        self._rerank_supports_vision: bool | None = None
        self._answer_image_capability: AnswerImageCapability | None = None
        self._answer_capability_reprobe_lock = asyncio.Lock()
        self._answer_capability_last_probe: float = 0.0
        self._answer_stream_sem = asyncio.Semaphore(max(1, int(self._config.max_async)))
        self._direct_llm_sem = asyncio.Semaphore(max(1, int(self._config.max_async)))

    @property
    def config(self) -> DlightragConfig:
        """Read-only access to the manager configuration for UI/API adapters."""
        return self._config

    @classmethod
    async def acreate(cls, config: DlightragConfig | None = None) -> RAGServiceManager:
        """Async factory — creates manager and warms the default workspace."""
        from dlightrag.observability import init_tracing

        manager = cls(config=config)
        init_tracing(manager._config)

        await manager._initialize_workspace_registry()

        # Discover all known workspaces for recovery, but only instantiate the
        # default service on startup. Other workspaces are lazy-loaded on demand.
        all_ws = await manager._list_all_workspaces()
        default_ws = normalize_workspace(manager._config.workspace)
        if default_ws not in all_ws:
            all_ws.insert(0, default_ws)

        # Bind the planner LLM during startup; this does not make a model call.
        manager._get_query_planner()

        # ── Vision probe (once at startup, not per workspace) ──────────
        await manager._probe_vision_support()
        await manager._probe_answer_image_capability()

        default_err: Exception | None = None
        try:
            await manager._get_service(default_ws)
            logger.info("Warmed up default workspace service '%s'", default_ws)
        except Exception as exc:
            default_err = exc
            logger.warning("Failed to warm up default workspace '%s'", default_ws, exc_info=True)

        await manager._start_ingest_job_recovery()
        await manager._recover_stalled_docs(all_ws)
        if default_ws in manager._services:
            manager._ready = True
        else:
            manager._degraded = True
            detail = getattr(default_err, "detail", str(default_err)) if default_err else "unknown"
            manager._startup_warnings.append(f"Default workspace init failed: {detail}")
            logger.error("RAG service started in degraded mode: %s", detail)
        return manager

    async def _initialize_workspace_registry(self) -> None:
        """Initialize the durable workspace registry."""
        from dlightrag.storage.workspaces import PGWorkspaceRegistry

        self._workspace_registry = PGWorkspaceRegistry()
        try:
            await self._workspace_registry.initialize()
            await self._workspace_registry.upsert(
                workspace=normalize_workspace(self._config.workspace),
                display_name=self._config.workspace,
                embedding_model=self._config.embedding.model,
            )
        except Exception as exc:
            self._startup_warnings.append("Workspace registry unavailable")
            logger.warning("Workspace registry initialization failed: %s", exc)

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
        if "asyncpg" in type(exc).__module__ if hasattr(type(exc), "__module__") else False:
            return f"{msg}. Check DLIGHTRAG_POSTGRES_HOST/PORT/USER/PASSWORD."
        if "timeout" in text or "timed out" in text:
            return f"{msg}. Service may be overloaded or unreachable."
        if "authentication" in text or "password" in text or "denied" in text:
            return f"{msg}. Check API keys or database credentials."
        return msg

    async def _get_ingest_service(self, workspace: str) -> RAGService:
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

                logger.info("Created RAGService for workspace '%s'", log_safe(workspace))
                return svc
            except Exception as e:
                error_msg = self._actionable_error(e)
                # Per-workspace exponential backoff
                _, prev_interval = self._backoff.get(workspace, (0, 7.5))
                new_interval = min(prev_interval * 2, _MAX_RETRY_INTERVAL)
                self._backoff[workspace] = (time.time(), new_interval)
                logger.error(
                    "RAGService creation failed for '%s': %s. Retry in %ss",
                    log_safe(workspace),
                    log_safe(error_msg),
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
        if status == "succeeded":
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
        metadata_policy: MetadataIngestPolicy | None = None,
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
            metadata_policy=metadata_policy,
            retain_source_file=retain_source_file,
        )

    async def _start_ingest_job_recovery(self) -> None:
        try:
            await self._ingest_jobs.start_recovery()
        except Exception:
            self._startup_warnings.append("Ingest job recovery unavailable")
            logger.warning("Ingest job recovery initialization failed", exc_info=True)

    async def _recover_stalled_docs(self, workspaces: list[str]) -> None:
        """Recover documents stalled in intermediate LightRAG pipeline states.

        Scans ``LIGHTRAG_DOC_STATUS`` for documents stuck in PARSING/ANALYZING/
        PROCESSING whose ``updated_at`` exceeds ``stalled_doc_timeout_seconds``,
        and resets them to PENDING so the next pipeline run retries them.

        Best-effort: failures are logged individually and never block startup.
        """
        timeout = self._config.stalled_doc_timeout_seconds
        if timeout <= 0:
            return  # disabled

        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        total = 0
        for workspace in workspaces:
            try:
                result = await recover_stalled_docs(
                    workspace=workspace,
                    timeout_seconds=timeout,
                    dry_run=False,
                )
                count: int = result.get("reset_count", 0)
                if count:
                    total += count
                    ids: list[str] = result.get("stalled_ids", [])
                    logger.warning(
                        "Recovered %d stalled doc(s) in workspace '%s': %s",
                        count,
                        workspace,
                        ids,
                    )
            except Exception:
                logger.debug(
                    "Stalled doc recovery skipped for workspace '%s'",
                    workspace,
                    exc_info=True,
                )

        if total:
            logger.info("Total stalled docs recovered at startup: %d", total)

    @property
    def answer_image_capability(self) -> AnswerImageCapability | None:
        """Query-role answer-model image capability, discovered at startup."""
        return self._answer_image_capability

    async def _probe_answer_image_capability(self) -> None:
        """Discover the query-role answer model's image capability once at startup."""
        if self._answer_image_capability is not None:
            return
        self._answer_image_capability = await self._discover_answer_image_capability()

    async def _maybe_reprobe_answer_image_capability(self) -> None:
        """Lazily re-probe when the cached capability is ``unknown``.

        ``supported``/``unsupported`` are terminal and never re-probed. An
        ``unknown`` verdict (a transient startup probe failure) is retried on
        demand -- when an image request actually needs it -- at most once per
        cooldown window under a single-flight lock, so a genuinely-unreachable
        model is never hammered.
        """
        capability = self._answer_image_capability
        if capability is not None and capability.status != "unknown":
            return
        async with self._answer_capability_reprobe_lock:
            capability = self._answer_image_capability
            if capability is not None and capability.status != "unknown":
                return
            now = time.monotonic()
            if (
                now - self._answer_capability_last_probe
                < _ANSWER_CAPABILITY_REPROBE_COOLDOWN_SECONDS
            ):
                return
            self._answer_capability_last_probe = now
            self._answer_image_capability = await self._discover_answer_image_capability()

    async def _discover_answer_image_capability(self) -> AnswerImageCapability:
        """Probe ``model_for_role(config, "query")`` and build a tri-state capability.

        Probes the model the AnswerEngine actually uses -- not ``llm.default``.
        Best-effort: failures degrade to ``unknown`` and never block the caller.
        """
        from dlightrag.core.vision_probe import ImageProbeOutcome, probe_image_capability
        from dlightrag.models.llm_roles import model_for_role
        from dlightrag.models.providers import get_provider

        ceiling = int(self._config.answer.max_images)
        cfg = model_for_role(self._config, "query")
        default = self._config.llm.default
        provider: Any = None
        try:
            provider = get_provider(
                cfg.provider,
                api_key=cfg.api_key or default.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
            )
            outcome = await probe_image_capability(provider, model=cfg.model, ceiling=ceiling)
        except Exception:
            logger.debug("Answer image capability probe failed", exc_info=True)
            outcome = ImageProbeOutcome(status="unknown", failure_kind="probe_error")
        finally:
            if provider is not None:
                await provider.aclose()
        capability = AnswerImageCapability(
            status=outcome.status,
            configured_ceiling=ceiling,
            effective_max_images=derive_effective_max_images(outcome.status, ceiling),
            provider=cfg.provider,
            base_url=cfg.base_url,
            model=cfg.model,
            failure_kind=outcome.failure_kind,
        )
        logger.info(
            "Answer image capability: status=%s effective=%d model=%s",
            capability.status,
            capability.effective_max_images,
            cfg.model,
        )
        return capability

    async def _probe_vision_support(self) -> None:
        """Probe the rerank scoring model's image capability once at startup.

        Only the ``chat_llm_reranker`` strategy sends image blocks to a scoring
        model, so probing is skipped entirely for other strategies (and when
        reranking is disabled). Uses the same transport-acceptance probe as the
        answer path. Stored on this manager instance so SDK callers can run
        multiple managers with different model configs in one process.
        """
        if self._rerank_supports_vision is not None:
            return  # already probed
        if not (
            self._config.rerank.enabled and self._config.rerank.strategy == "chat_llm_reranker"
        ):
            return  # no rerank model consumes image input; nothing to probe

        from dlightrag.core.vision_probe import probe_image_capability
        from dlightrag.models.llm import get_chat_rerank_scoring_config
        from dlightrag.models.providers import get_provider

        default = self._config.llm.default
        rerank_model = get_chat_rerank_scoring_config(self._config)
        provider: Any = None
        try:
            provider = get_provider(
                rerank_model.provider,
                api_key=rerank_model.api_key or default.api_key,
                base_url=rerank_model.base_url,
                timeout=rerank_model.timeout,
                max_retries=rerank_model.max_retries,
            )
            outcome = await probe_image_capability(
                provider,
                model=rerank_model.model,
                ceiling=1,
                model_kwargs=rerank_model.model_kwargs,
            )
            self._rerank_supports_vision = {"supported": True, "unsupported": False}.get(
                outcome.status
            )
            logger.info(
                "Rerank model image probe: status=%s (model=%s, provider=%s)",
                outcome.status,
                rerank_model.model,
                rerank_model.provider,
            )
        except Exception:
            logger.debug("Rerank model image probe failed", exc_info=True)
            self._rerank_supports_vision = None
        finally:
            if provider is not None:
                await provider.aclose()

    async def astart_ingest_job(
        self,
        workspace: str,
        request: IngestSpec,
    ) -> dict[str, Any]:
        """Start a background ingest job and return its durable job row."""
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
        return await self._ingest_jobs.await_job(job_id, timeout=timeout)

    async def aget_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._ingest_jobs.get_job(job_id)

    def _get_file_panel_store(self) -> PGFilePanelStore:
        if self._file_panel_store is None:
            from dlightrag.storage.file_panel import PGFilePanelStore

            self._file_panel_store = PGFilePanelStore()
        return self._file_panel_store

    async def acreate_workspace(self, workspace: str, *, display_name: str | None = None) -> None:
        """Initialize a workspace through the public manager API."""
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

    async def aget_visual_asset(self, workspace: str, chunk_id: str, *, size: str = "full") -> Any:
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
        *,
        mode: Literal["merge", "replace"] = "merge",
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> None:
        """Update (merge) document metadata."""
        svc = await self._get_service(workspace)
        await svc.aupdate_metadata(doc_id, data, mode=mode, metadata_policy=metadata_policy)

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
                    log_safe(ws),
                    log_safe(exc),
                )

            # Close and evict from cache even after reset errors.
            if ws in self._services:
                try:
                    await self._services[ws].aclose()
                except Exception:
                    logger.warning(
                        "Failed to close service for '%s'",
                        log_safe(ws),
                        exc_info=True,
                    )
                del self._services[ws]

        return {"workspaces": results, "total_errors": total_errors}

    def _get_answer_engine(self) -> AnswerEngine:
        """Lazy-create AnswerEngine from global config."""
        if self._answer_engine is None:
            from dlightrag.models.llm import get_query_model_func

            answer_cfg = self._config.answer
            capability = self._answer_image_capability
            effective_max_images = capability.effective_max_images if capability is not None else 0
            self._answer_engine = AnswerEngine(
                model_func=get_query_model_func(self._config),
                effective_max_images=effective_max_images,
                image_max_bytes=answer_cfg.image_max_bytes,
                image_max_total_bytes=answer_cfg.image_max_total_bytes,
                image_max_px=answer_cfg.image_max_px,
                image_min_px=answer_cfg.image_min_px,
                image_quality=answer_cfg.image_quality,
                image_min_quality=answer_cfg.image_min_quality,
                context_top_k=answer_cfg.context_top_k,
            )
        return self._answer_engine

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

    def _get_query_planner(self) -> QueryPlanner:
        """Return the manager-owned QueryPlanner, creating it when needed."""
        if self._query_planner is None:
            from dlightrag.models.llm import get_planner_model_func

            self._query_planner = QueryPlanner(
                llm_func=self._sem_bound(get_planner_model_func(self._config)),
                schema_provider=self._get_schema,
                schema_ttl=300.0,
            )
        return self._query_planner

    def _get_query_image_describer(self) -> QueryImageDescriber:
        """Lazy-create the VLM query-image describer."""
        if self._query_image_describer is None:
            from dlightrag.core.query_images import QueryImageDescriber
            from dlightrag.models.llm import get_vlm_model_func

            cfg = self._config.query_images
            transport = self._config.answer
            self._query_image_describer = QueryImageDescriber(
                vlm_func=self._sem_bound(get_vlm_model_func(self._config)),
                max_images=cfg.max_current_images,
                max_total_bytes=transport.image_max_total_bytes,
                max_bytes_per_image=transport.image_max_bytes,
                max_px=transport.image_max_px,
                min_px=transport.image_min_px,
                quality=transport.image_quality,
                min_quality=transport.image_min_quality,
            )
        return self._query_image_describer

    async def _get_schema(
        self,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Fetch a metadata schema for the requested workspace set."""
        ws_key = _normalize_workspaces(workspaces) or (normalize_workspace(self._config.workspace),)
        now = time.monotonic()
        cached = self._schema_cache.get(ws_key)
        if cached is not None and now - cached[0] < 300.0:
            return cached[1]

        schemas: list[dict[str, Any]] = []
        for workspace in ws_key:
            try:
                svc = await self._get_service(workspace)
                metadata_index = getattr(svc, "_metadata_index", None)
                if metadata_index is not None:
                    schema = await metadata_index.get_field_schema()
                    if isinstance(schema, dict):
                        schemas.append(schema)
            except Exception:
                logger.debug(
                    "Schema lookup failed for workspace %s",
                    log_safe(workspace),
                    exc_info=True,
                )
        merged = _merge_schema_rows(schemas)
        self._schema_cache[ws_key] = (now, merged)
        return merged

    async def aplan_query(
        self,
        query: str,
        *,
        text_history: list[dict[str, Any]] | None = None,
        image_catalog: list[dict[str, Any]] | None = None,
        allowed_history_image_count: int = 0,
        current_image_descriptions: list[str] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> QueryPlan:
        """Plan one query using the manager-owned planner.

        Stateless callers pass no history/catalog. The web prepare step passes
        request-local ``text_history`` and, when the answer model supports
        images, an ``image_catalog`` so the planner also selects scoped history
        images in the same call.
        """
        return await self._aplan_query_prepared(
            query,
            text_history=text_history,
            image_catalog=image_catalog,
            allowed_history_image_count=allowed_history_image_count,
            current_image_descriptions=current_image_descriptions,
            workspaces=workspaces,
        )

    async def aplan_web_conversation_query(
        self,
        query: str,
        *,
        text_history: list[dict[str, Any]] | None = None,
        image_catalog: list[dict[str, Any]] | None = None,
        attachment_catalog: list[dict[str, Any]] | None = None,
        current_attachment_catalog: list[dict[str, Any]] | None = None,
        allowed_history_image_count: int = 0,
        allowed_history_attachment_count: int = 0,
        current_image_descriptions: list[str] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> QueryPlan:
        """Plan one Web conversation turn using the manager-owned planner.

        Web ``/web/answer`` only. Routes through the separate
        :meth:`QueryPlanner.plan_web_conversation` contract so scoped history
        documents/images and an attachment query are planned in the same call.
        REST/MCP/SDK/retrieve keep using :meth:`aplan_query`.
        """
        planner = self._get_query_planner()
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "query_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces or []),
                "history_turns": len(text_history or []),
                "history_image_catalog_count": len(image_catalog or []),
                "history_attachment_catalog_count": len(attachment_catalog or []),
            },
        ) as trace:
            schema = await self._get_schema(workspaces)
            plan = await planner.plan_web_conversation(
                query,
                conversation_history=text_history,
                image_catalog=image_catalog,
                attachment_catalog=attachment_catalog,
                current_attachment_catalog=current_attachment_catalog,
                allowed_history_image_count=allowed_history_image_count,
                allowed_history_attachment_count=allowed_history_attachment_count,
                current_image_descriptions=current_image_descriptions,
                max_turns=self._config.max_conversation_turns,
                max_tokens=self._config.max_conversation_tokens,
                schema=schema,
            )
            trace.update(
                output={
                    "standalone_query": plan.standalone_query,
                    "has_metadata_filter": plan.metadata_filter is not None,
                }
            )
            return plan

    async def aget_attachment_chunking_lightrag(
        self, workspaces: list[str] | tuple[str, ...] | None = None
    ) -> Any:
        """Return an initialized LightRAG for Web query-attachment parsing/chunking.

        Web ``/web/answer`` only: the Web layer owns the conversation store and
        its :class:`QueryAttachmentService`, but the parser/chunker must run on a
        real LightRAG. Reuse the primary selected (or manager-default) workspace
        service; parsing writes nothing to any workspace store (the query-
        attachment path neutralises the single workspace-write hook).
        """
        workspace = (list(workspaces or []) or [self._config.workspace])[0]
        svc = await self._get_service(workspace)
        return svc.lightrag

    async def adescribe_query_images(self, images: list[dict[str, Any]]) -> dict[str, str]:
        """Describe current-turn images (ordinal -> text) for image-aware planning.

        The web prepare step calls this before ``aplan_query`` so the planner can
        fold image semantics into the standalone query, bm25, and filters, then
        carries the descriptions on the turn to avoid re-describing in retrieval.
        """
        if not images:
            return {}
        prepared = await prepare_query_images(
            query_images=images,
            describer=self._get_query_image_describer(),
        )
        return prepared.descriptions_by_ordinal

    async def _aplan_query_prepared(
        self,
        query: str,
        *,
        text_history: list[dict[str, Any]] | None,
        image_catalog: list[dict[str, Any]] | None = None,
        allowed_history_image_count: int = 0,
        current_image_descriptions: list[str] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> QueryPlan:
        """Plan a server-prepared query with request-local text history."""
        planner = self._get_query_planner()
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "query_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces or []),
                "history_turns": len(text_history or []),
                "history_image_catalog_count": len(image_catalog or []),
            },
        ) as trace:
            schema = await self._get_schema(workspaces)
            plan = await planner.plan(
                query,
                conversation_history=text_history,
                max_turns=self._config.max_conversation_turns,
                max_tokens=self._config.max_conversation_tokens,
                schema=schema,
                image_catalog=image_catalog,
                allowed_history_image_count=allowed_history_image_count,
                current_image_descriptions=current_image_descriptions,
            )
            trace.update(
                output={
                    "standalone_query": plan.standalone_query,
                    "has_metadata_filter": plan.metadata_filter is not None,
                }
            )
            return plan

    async def _describe_and_plan(
        self,
        query: str,
        *,
        text_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        ws_list: list[str],
    ) -> tuple[QueryPlan, PreparedQueryImages]:
        """Describe current-turn images, then plan an image-aware query.

        Shared by the stateless /retrieve and /answer paths so both consistently
        run the planner (BM25 keyword extraction, metadata-filter inference, and
        image-aware query construction). Describing zero images is a no-op.
        """
        prepared = await prepare_query_images(
            query_images=list(query_images or []),
            describer=self._get_query_image_describer(),
        )
        plan = await self._aplan_query_prepared(
            query,
            text_history=text_history,
            current_image_descriptions=prepared.descriptions or None,
            workspaces=ws_list,
        )
        return plan, prepared

    async def agenerate_stream_from_contexts(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        query_images: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Generate a stateless streaming answer from retrieved contexts."""
        return await self._agenerate_stream_from_contexts_prepared(
            query,
            contexts,
            query_images=query_images,
            text_history=None,
            context_top_k=context_top_k,
        )

    async def _agenerate_stream_from_contexts_prepared(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        query_images: list[dict[str, Any]] | None = None,
        text_history: list[dict[str, Any]] | None,
        context_top_k: int | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Generate from server-prepared contexts and request-local text history."""
        engine = self._get_answer_engine()
        await self._answer_stream_sem.acquire()
        try:
            prepared_contexts, stream = await engine.generate_stream(
                query,
                contexts,
                query_images=query_images,
                conversation_history=text_history,
                context_top_k=context_top_k,
            )
        except BaseException:
            self._answer_stream_sem.release()
            raise
        if stream is None:
            self._answer_stream_sem.release()
            return prepared_contexts, None
        return prepared_contexts, _ScopedAnswerStream(stream, self._answer_stream_sem)

    def _resolve_answer_limits(self, kwargs: dict[str, Any]) -> _AnswerLimits:
        """Resolve answer-specific retrieval and final-prompt chunk limits.

        ``top_k``/``chunk_top_k`` remain retrieval controls. ``context_top_k``
        only controls final answer-prompt packing.
        """
        answer_cfg = self._config.answer
        answer_context_top_k = _positive_int_or_none(kwargs.pop("answer_context_top_k", None))
        requested_top_k = _positive_int_or_none(kwargs.get("top_k"))
        requested_chunk_top_k = _positive_int_or_none(kwargs.get("chunk_top_k"))

        context_top_k = answer_context_top_k or answer_cfg.context_top_k
        candidate_top_k = requested_chunk_top_k or self._config.chunk_top_k
        kwargs["top_k"] = requested_top_k or self._config.top_k
        kwargs["chunk_top_k"] = candidate_top_k
        return _AnswerLimits(candidate_top_k=candidate_top_k, context_top_k=context_top_k)

    async def _resolve_manager_query_workspaces(
        self,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
    ) -> list[str]:
        validate_query_workspace_selection(
            all_workspaces=all_workspaces,
            workspace=workspace,
            workspaces=workspaces,
        )
        available = await self.alist_workspaces() if all_workspaces else None
        return resolve_query_workspaces(
            default_workspace=self._config.workspace,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            available_workspaces=available,
        )

    async def _plan_and_retrieve(
        self,
        current_query: str,
        *,
        retrieval_query: str,
        text_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        ws_list: list[str],
        scope: RequestScope,
        kwargs: dict[str, Any],
        plan: QueryPlan | None = None,
        current_image_descriptions: dict[str, str] | None = None,
        log_plan: bool = False,
    ) -> tuple[QueryPlan, PreparedQueryImages, _AnswerLimits, RetrievalResult]:
        current_images = list(query_images or [])
        if plan is None:
            # Stateless answer: describe current images then plan so BM25,
            # metadata filters, and the rewrite are image-aware.
            plan, prepared = await self._describe_and_plan(
                retrieval_query,
                text_history=text_history,
                query_images=current_images,
                ws_list=ws_list,
            )
        else:
            # Web planned upstream in prepare_answer_turn with these same image
            # descriptions; reuse them (retrieval runs through the planner query).
            described = dict(current_image_descriptions or {})
            prepared = PreparedQueryImages(
                descriptions=list(described.values()),
                descriptions_by_ordinal=described,
            )
        if log_plan:
            logger.info(
                "Query plan: original=%r, standalone=%r",
                plan.original_query[:60],
                plan.standalone_query[:60],
            )

        limits = self._resolve_answer_limits(kwargs)
        retrieval = await self.aretrieve(
            current_query,
            plan=plan,
            workspaces=ws_list,
            scope=scope,
            query_images=current_images,
            **kwargs,
        )
        retrieval.trace["query_image_description_count"] = len(prepared.descriptions)
        retrieval.trace["answer_resolved_chunk_top_k"] = limits.candidate_top_k
        retrieval.trace["answer_context_top_k"] = limits.context_top_k
        return plan, prepared, limits, retrieval

    # --- Read operations (single or federated) ---

    async def aretrieve(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        plan: QueryPlan | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: list[dict[str, Any]] | None = None,
        scope: RequestScope | None = None,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        Args:
            plan: Pre-computed QueryPlan from QueryPlanner. When provided,
                uses plan.standalone_query for search and plan.metadata_filter
                for structured filtering.
        """
        kwargs: dict[str, Any] = {}
        requested_top_k = _positive_int_or_none(top_k)
        requested_chunk_top_k = _positive_int_or_none(chunk_top_k)
        if filters is not None:
            kwargs["filters"] = filters
        if bm25_query is not None:
            kwargs["bm25_query"] = bm25_query
        kwargs["top_k"] = requested_top_k or self._config.top_k
        kwargs["chunk_top_k"] = requested_chunk_top_k or self._config.chunk_top_k
        ws_list = await self._resolve_manager_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        current_images = list(query_images or [])
        if current_images:
            kwargs["query_image_blocks"] = current_images
        descriptions: list[str] = []
        if plan is None:
            # Stateless retrieve always plans so BM25 keyword extraction,
            # metadata-filter inference, and image-aware query construction match
            # the answer path. Describing zero images is a free short-circuit.
            plan, prepared = await self._describe_and_plan(
                query,
                text_history=None,
                query_images=current_images,
                ws_list=ws_list,
            )
            descriptions = prepared.descriptions
        effective_query = plan.standalone_query if plan is not None else query
        if plan is not None:
            kwargs.setdefault("_plan", plan)
        from dlightrag.observability import trace_observation

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_observation(
                    "retrieve",
                    as_type="retriever",
                    input={"query": effective_query},
                    metadata={
                        "workspaces": ws_list,
                        "top_k": kwargs["top_k"],
                        "chunk_top_k": kwargs["chunk_top_k"],
                        "has_filters": "filters" in kwargs,
                    },
                ) as trace:
                    if len(ws_list) == 1:
                        svc = await self._get_service(ws_list[0])
                        result = await svc.aretrieve(effective_query, **kwargs)
                    else:
                        result = await federated_retrieve(
                            effective_query, ws_list, self._get_service, **kwargs
                        )
                    result.image_descriptions = descriptions
                    result.trace["query_image_description_count"] = len(descriptions)
                    trace.update(
                        output={
                            **_context_output(result.contexts),
                            "query_image_description_count": len(descriptions),
                        }
                    )
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def aanswer(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        answer_context_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        query_images: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
        semantic_highlights: bool = False,
        scope: RequestScope | None = None,
    ) -> RetrievalResult:
        """Answer from one or more workspaces: plan -> retrieve -> generate.

        ``query_images`` are user-attached ``image_url`` blocks inlined into the
        answer LLM call; they inform answer generation and (via VLM descriptions)
        the retrieval plan, but are not embedded for retrieval.

        ``history`` is caller-supplied prior turns (``role``/``content`` dicts).
        It is stateless -- the caller owns persistence and passes it per request;
        it feeds the planner's standalone-query rewrite and answer generation.
        """
        turn = PreparedAnswerTurn.stateless(query, query_images)
        ws_list = await self._resolve_manager_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        scoped = _scope_for_workspaces(scope, ws_list)
        kwargs = _drop_none(
            {
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "answer_context_top_k": answer_context_top_k,
                "filters": filters,
            }
        )

        # Guard: reject current images when the query-role answer model can't accept them.
        # Lazily re-probe first so a transient startup `unknown` can recover.
        current_images = list(turn.materialized_query_images)
        if current_images:
            await self._maybe_reprobe_answer_image_capability()
        _check_answer_image_capability(
            query_images=current_images,
            capability=self._answer_image_capability,
        )

        from dlightrag.observability import trace_observation

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_observation(
                    "answer_pipeline",
                    as_type="chain",
                    input={"query": query},
                    metadata={
                        "workspaces": ws_list,
                        "history_turns": len(history or []),
                        "query_image_count": len(turn.materialized_query_images),
                        "semantic_highlights": semantic_highlights,
                    },
                ) as pipeline_trace:
                    plan, prepared, limits, retrieval = await self._plan_and_retrieve(
                        turn.current_query,
                        retrieval_query=turn.retrieval_query,
                        text_history=history,
                        query_images=list(turn.current_query_images),
                        ws_list=ws_list,
                        scope=scoped,
                        kwargs=kwargs,
                        plan=turn.plan,
                        current_image_descriptions=turn.current_image_descriptions,
                    )
                    engine = self._get_answer_engine()
                    async with trace_observation(
                        "answer_generation",
                        as_type="chain",
                        input={"query": plan.standalone_query},
                        metadata={
                            "context_chunks": len(retrieval.contexts.get("chunks", [])),
                            "context_top_k": limits.context_top_k,
                        },
                    ) as generation_trace:
                        result = await engine.generate(
                            plan.standalone_query,
                            retrieval.contexts,
                            query_images=current_images or None,
                            conversation_history=history,
                            context_top_k=limits.context_top_k,
                        )
                        generation_trace.update(output=_answer_output(result))
                    result.trace.update(retrieval.trace)
                    result.image_descriptions = prepared.descriptions
                    if semantic_highlights:
                        from dlightrag.core.answer.highlights import enrich_semantic_highlights

                        result.sources = await enrich_semantic_highlights(
                            result.sources,
                            answer_text=result.answer,
                            config=self._config,
                        )
                        from dlightrag.core.answer.media import (
                            answer_blocks_from_markdown,
                            answer_images_from_sources,
                        )

                        result.answer_images = answer_images_from_sources(
                            result.sources,
                            contexts=result.contexts,
                        )
                        result.answer_blocks = answer_blocks_from_markdown(
                            result.answer,
                            result.answer_images,
                        )
                    pipeline_trace.update(output=_answer_output(result))
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def aanswer_stream(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        answer_context_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        query_images: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
        scope: RequestScope | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer from one or more workspaces: plan -> retrieve -> stream.

        See ``aanswer`` for ``query_images`` and ``history`` semantics.
        """
        return await self._aanswer_stream_prepared(
            PreparedAnswerTurn.stateless(query, query_images, history=history),
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            answer_context_top_k=answer_context_top_k,
            filters=filters,
            scope=scope,
        )

    async def _aanswer_stream_prepared(
        self,
        turn: PreparedAnswerTurn,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        all_workspaces: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        answer_context_top_k: int | None = None,
        filters: MetadataFilter | None = None,
        scope: RequestScope | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Stream one server-prepared Web turn through the stateless core pipeline."""
        ws_list = await self._resolve_manager_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        scoped = _scope_for_workspaces(scope, ws_list)
        kwargs = _drop_none(
            {
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "answer_context_top_k": answer_context_top_k,
                "filters": filters,
            }
        )
        history = list(turn.text_history) or None
        query_images = list(turn.materialized_query_images)

        # Guard: reject current images when the query-role answer model can't accept them.
        # Lazily re-probe first so a transient startup `unknown` can recover.
        if query_images:
            await self._maybe_reprobe_answer_image_capability()
        _check_answer_image_capability(
            query_images=query_images,
            capability=self._answer_image_capability,
        )

        try:
            async with asyncio.timeout(self._config.request_timeout):
                plan, prepared, limits, retrieval = await self._plan_and_retrieve(
                    turn.current_query,
                    retrieval_query=turn.retrieval_query,
                    text_history=history,
                    query_images=list(turn.current_query_images),
                    ws_list=ws_list,
                    scope=scoped,
                    kwargs=kwargs,
                    plan=turn.plan,
                    current_image_descriptions=turn.current_image_descriptions,
                    log_plan=True,
                )
                context_top_k = limits.context_top_k
                if turn.attachment_context_chunks:
                    # Inject request-local Web attachment rows ahead of retrieved
                    # chunks so the single downstream AnswerContextPacker +
                    # AnswerImageBudget owns budgeting for attachment, workspace,
                    # and current images together. Attachment visual rows carry
                    # image_data but no pre-budget flag, so no second budget here.
                    existing = retrieval.contexts.setdefault("chunks", [])
                    retrieval.contexts["chunks"] = [
                        *turn.attachment_context_chunks,
                        *existing,
                    ]
                    # Attachments are additive to the packer's count budget, not a
                    # replacement for workspace context. The packer treats
                    # context_top_k as a pure COUNT cap; because attachment rows are
                    # prepended, a document yielding >= context_top_k chunks would
                    # otherwise fill the whole budget and starve workspace grounding.
                    # Grow the cap by the attachment count so workspace retrieval
                    # keeps its normal allocation. A falsy cap already means "no cap".
                    if isinstance(context_top_k, int) and context_top_k > 0:
                        context_top_k = context_top_k + len(turn.attachment_context_chunks)
                contexts, stream = await self._agenerate_stream_from_contexts_prepared(
                    plan.standalone_query,
                    retrieval.contexts,
                    query_images=query_images or None,
                    text_history=history,
                    context_top_k=context_top_k,
                )
                if stream is not None:
                    stream_meta = cast(Any, stream)
                    stream_meta.image_descriptions = prepared.descriptions_by_ordinal
                    answer_trace = getattr(stream_meta, "trace", None)
                    stream_meta.trace = (
                        {
                            **retrieval.trace,
                            **answer_trace,
                        }
                        if isinstance(answer_trace, dict)
                        else retrieval.trace
                    )
                return contexts, stream
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

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

    async def _list_all_workspaces(self) -> list[str]:
        """Internal: discover all workspaces."""
        return await self.alist_workspaces()

    async def aclose(self) -> None:
        """Close all managed RAGService instances."""
        from dlightrag.observability import shutdown_tracing

        await self._ingest_jobs.close()

        for component in (
            self._answer_engine,
            self._query_planner,
            self._query_image_describer,
        ):
            close = getattr(component, "aclose", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except Exception:
                logger.warning("Failed to close manager component", exc_info=True)

        for ws, svc in self._services.items():
            try:
                await svc.aclose()
            except Exception:
                logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
        self._services.clear()
        self._ready = False

        from dlightrag.storage.pool import pg_pool

        await pg_pool.close()
        shutdown_tracing()

    # --- Health ---

    def is_ready(self) -> bool:
        """Check if manager is ready (default workspace initialized)."""
        return self._ready

    def is_degraded(self) -> bool:
        return self._degraded

    def get_warnings(self) -> list[str]:
        return list(self._startup_warnings)

    def get_error_info(self) -> dict[str, Any]:
        """Get per-workspace backoff state for health checks."""
        return {
            "backoff_workspaces": {
                ws: {"retry_after": interval, "since": ts}
                for ws, (ts, interval) in self._backoff.items()
            },
        }


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
]
