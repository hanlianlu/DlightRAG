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
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig
    from dlightrag.core.request.images import QueryImageDescriber
    from dlightrag.core.resources import ResourceInput, ResourceRegistry
    from dlightrag.core.retrieval.web_search import ExaSearch
    from dlightrag.core.source_download import SourceDownloadTarget
    from dlightrag.models.tool_model import QueryToolModel
    from dlightrag.storage.file_panel import PGFilePanelStore
    from dlightrag.storage.workspaces import PGWorkspaceRegistry

from dlightrag.contracts import VisualAssetSize
from dlightrag.core.agent.orchestrator import AnswerOrchestrator
from dlightrag.core.agent.tool_loop import AgentTool
from dlightrag.core.answer.capability import AnswerImageCapability, derive_effective_max_images
from dlightrag.core.answer.errors import (
    ANSWER_IMAGE_CAPABILITY_UNKNOWN,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
    CurrentImagePayloadError,
)
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.answer.synthesizer import AnswerSynthesizer
from dlightrag.core.answer.turn import PreparedAnswerTurn
from dlightrag.core.client_contracts import IngestSpec, SourceType
from dlightrag.core.client_requests import ingest_kwargs_from_payload
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.ingest_job_coordinator import IngestJobCoordinator
from dlightrag.core.ingestion.paths import is_explicit_upload_batch_dir
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.request.images import (
    PreparedQueryImages,
    prepare_query_images,
)
from dlightrag.core.request.planner import QueryPlan, QueryPlanner
from dlightrag.core.request.workspaces import (
    resolve_query_workspaces,
    validate_query_workspace_selection,
)
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.service import RAGService
from dlightrag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag.storage.ingest_jobs import JOB_STATES_WITH_RESULT
from dlightrag.utils import log_safe, normalize_workspace

logger = logging.getLogger(__name__)

_MAX_RETRY_INTERVAL: float = 300.0
_QUERY_WORKSPACE_MAX_CONCURRENCY = 8


def _defer_cancellation(
    first: asyncio.CancelledError | None,
    current: asyncio.CancelledError,
) -> asyncio.CancelledError:
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()
    return first if first is not None else current


class _ScopedAnswerStream:
    """Async iterator wrapper that releases a semaphore when streaming ends."""

    def __init__(
        self,
        inner: AsyncIterator[str],
        semaphore: asyncio.Semaphore,
        *,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._inner = inner
        self._aiter = inner.__aiter__()
        self._semaphore = semaphore
        self._released = False
        self._on_close = on_close
        self._completed = False

    def __aiter__(self) -> _ScopedAnswerStream:
        return self

    async def __anext__(self) -> str:
        try:
            return await self._aiter.__anext__()
        except StopAsyncIteration:
            await self._acomplete()
            raise
        except BaseException:
            await self._acomplete()
            raise

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        try:
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
        finally:
            await self._acomplete()

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

    async def _acomplete(self) -> None:
        """Await cleanup exactly once on exhaustion, aclose, error, or cancel.

        The registry close is awaited (never fire-and-forget) and the semaphore
        slot is always released, even if the close callback raises.
        """
        if self._completed:
            return
        self._completed = True
        try:
            if self._on_close is not None:
                await self._on_close()
        finally:
            self._release()

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._semaphore.release()


@dataclass(frozen=True)
class _OrchestratorRun:
    """One request resolved into a capability-driven orchestrator and its inputs."""

    orchestrator: AnswerOrchestrator
    answer_query: str
    retrieval_query: str | None
    prepared: PreparedQueryImages
    query_images: list[dict[str, Any]] | None
    history: PriorTurns
    current_image_count: int
    ws_list: list[str]
    registry: ResourceRegistry | None


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

        # In-flight workspace initializations started when a request resolves its scope.
        self._warmups: set[asyncio.Task[None]] = set()

        self._answer_synthesizer: AnswerSynthesizer | None = None
        self._ingest_jobs = IngestJobCoordinator(
            self._get_ingest_service,
            input_root=self._config.input_dir_path,
        )
        self._query_planner: QueryPlanner | None = None
        self._query_image_describer: QueryImageDescriber | None = None
        self._web_search: ExaSearch | None = None
        self._query_tool_model: QueryToolModel | None = None
        self._query_image_describer_lock = asyncio.Lock()
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

        # Bind the process-wide domain pool to this service config so the
        # reader read-only invariant (and endpoint) cannot silently diverge
        # from a caller-supplied SDK config that never called set_config().
        from dlightrag.storage.pool import pg_pool

        pg_pool.bind(manager._config)

        await manager._initialize_workspace_registry()

        default_ws = normalize_workspace(manager._config.workspace)

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

        # Readers attach to a replica and do not recover ingest jobs.
        if not manager._config.is_reader:
            await manager._start_ingest_job_recovery()
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
            await self._workspace_registry.initialize(read_only=self._config.is_reader)
            if not self._config.is_reader:
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
        try:
            await self._ingest_jobs.start_recovery()
        except Exception:
            self._startup_warnings.append("Ingest job recovery unavailable")
            logger.warning("Ingest job recovery initialization failed", exc_info=True)

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

        Probes the model the AnswerSynthesizer actually uses -- not ``llm.default``.
        Best-effort: failures degrade to ``unknown`` and never block the caller.
        """
        from dlightrag.core.vision_probe import ImageProbeOutcome, probe_image_capability
        from dlightrag.models.llm_roles import model_for_role
        from dlightrag.models.providers import get_provider

        ceiling = int(self._config.answer.max_images)
        cfg = model_for_role(self._config, "query")
        provider: Any = None
        try:
            provider = get_provider(
                cfg.provider,
                api_key=cfg.api_key,
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

        rerank_model = get_chat_rerank_scoring_config(self._config)
        provider: Any = None
        try:
            provider = get_provider(
                rerank_model.provider,
                api_key=rerank_model.api_key,
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
                    log_safe(ws),
                    log_safe(exc),
                )

            # Close and evict from cache even after reset errors, but never for a
            # dry run -- a preview must not tear down the live workspace runtime.
            if not dry_run and ws in self._services:
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

    def _get_answer_synthesizer(self) -> AnswerSynthesizer:
        """Lazy-create the AnswerSynthesizer from global config."""
        if self._answer_synthesizer is None:
            from dlightrag.models.llm import get_query_model_func

            answer_cfg = self._config.answer
            capability = self._answer_image_capability
            effective_max_images = capability.effective_max_images if capability is not None else 0
            self._answer_synthesizer = AnswerSynthesizer(
                model_func=get_query_model_func(self._config),
                effective_max_images=effective_max_images,
                image_max_bytes=answer_cfg.image_max_bytes,
                image_max_total_bytes=answer_cfg.image_max_total_bytes,
                image_max_pixels=answer_cfg.image_max_pixels,
                image_max_px=answer_cfg.image_max_px,
                image_min_px=answer_cfg.image_min_px,
                image_quality=answer_cfg.image_quality,
                image_min_quality=answer_cfg.image_min_quality,
                context_window_tokens=answer_cfg.context_window_tokens,
            )
        return self._answer_synthesizer

    def _new_answer_image_budget(self) -> AnswerImageBudget:
        answer = self._config.answer
        capability = self._answer_image_capability
        return AnswerImageBudget(
            max_images=capability.effective_max_images if capability is not None else 0,
            max_total_bytes=answer.image_max_total_bytes,
            max_bytes_per_image=answer.image_max_bytes,
            max_pixels=answer.image_max_pixels,
            max_px=answer.image_max_px,
            min_px=answer.image_min_px,
            quality=answer.image_quality,
            min_quality=answer.image_min_quality,
        )

    @staticmethod
    def _budget_agent_images(
        current_images: list[dict[str, Any]],
        budget: AnswerImageBudget,
        resource_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
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
                        "text": f"[current image {index} | resource: {resource_ids[index - 1]}]",
                    }
                )
            blocks.append(block)
        return blocks

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
            from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE
            from dlightrag.models.llm import get_planner_model_func

            envelope = self._config.answer.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
            self._query_planner = QueryPlanner(
                llm_func=self._sem_bound(get_planner_model_func(self._config)),
                input_token_envelope=max(1, envelope),
            )
        return self._query_planner

    def _get_web_search(self) -> ExaSearch | None:
        """Return the manager-owned web search client, or None when unconfigured."""
        key = self._config.web_search.api_key
        if not key:
            return None
        if self._web_search is None:
            from dlightrag.core.retrieval.web_search import ExaSearch

            self._web_search = ExaSearch(key)
        return self._web_search

    def _get_query_tool_model(self) -> QueryToolModel:
        """Return the agent control model used by the research answer path."""
        if self._query_tool_model is None:
            from dlightrag.models.tool_model import create_query_tool_model

            self._query_tool_model = create_query_tool_model(self._config)
        return self._query_tool_model

    async def _aget_query_image_describer(self) -> QueryImageDescriber:
        """Lazy-create the VLM query-image describer."""
        async with self._query_image_describer_lock:
            if self._query_image_describer is None:
                from dlightrag.core.request.images import QueryImageDescriber
                from dlightrag.models.llm import get_vlm_model_func

                cfg = self._config.query_images
                transport = self._config.answer
                self._query_image_describer = QueryImageDescriber(
                    vlm_func=self._sem_bound(get_vlm_model_func(self._config)),
                    max_images=cfg.max_current_images,
                    max_total_bytes=transport.image_max_total_bytes,
                    max_bytes_per_image=transport.image_max_bytes,
                    max_pixels=transport.image_max_pixels,
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
        self._schema_cache[ws_key] = (now, schema)
        return schema

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
            describer=await self._aget_query_image_describer(),
        )
        return prepared.descriptions_by_ordinal

    async def _aplan_query_prepared(
        self,
        query: str,
        *,
        text_history: PriorTurns | None,
        current_image_descriptions: list[str] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
        preserve_query: bool = False,
    ) -> QueryPlan:
        """Plan one retrieval query, optionally preserving its semantic text."""
        planner = self._get_query_planner()
        from dlightrag.observability import trace_observation

        async with trace_observation(
            "query_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces or []),
                "history_messages": len(text_history or PriorTurns()),
            },
        ) as trace:
            schema = await self._get_schema(workspaces)
            plan = await planner.plan(
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
                    "planner_outcome": plan.planner_outcome,
                }
            )
            return plan

    async def _describe_and_plan(
        self,
        query: str,
        *,
        text_history: PriorTurns | None,
        query_images: list[dict[str, Any]] | None,
        ws_list: list[str],
    ) -> tuple[QueryPlan, PreparedQueryImages]:
        """Describe current-turn images, then plan an image-aware query.

        Used by stateless `/retrieve` and the Answer fast path. Research mode
        invokes planning lazily inside its knowledge-base tool instead.
        Describing zero images is a no-op.
        """
        prepared = await prepare_query_images(
            query_images=list(query_images or []),
            describer=await self._aget_query_image_describer(),
        )
        plan = await self._aplan_query_prepared(
            query,
            text_history=text_history,
            current_image_descriptions=prepared.descriptions or None,
            workspaces=ws_list,
        )
        return plan, prepared

    def _start_query_service_warmup(self, workspaces: list[str] | tuple[str, ...]) -> None:
        """Initialize a request's cold workspaces now.

        Retrieval reaches the same services through ``_get_service``, whose
        per-workspace lock lets it join an initialization already running here
        instead of starting after planning or a control turn finishes.
        """
        cold = [ws for ws in _normalize_workspaces(workspaces) if ws not in self._services]
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
        selected = _normalize_workspaces(workspaces) or (
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
        self._start_query_service_warmup(resolved)
        return resolved

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
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        ``query_images`` are current-request images. VLM descriptions inform
        query planning, and verified image blocks are embedded only when optional
        direct visual retrieval is active. Public retrieval is stateless: it
        accepts neither history nor Web attachment documents.

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
        ws_list = await self._open_query_workspaces(
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
                            effective_query,
                            ws_list,
                            self._get_service,
                            max_concurrency=_QUERY_WORKSPACE_MAX_CONCURRENCY,
                            **kwargs,
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

    async def _prepare_orchestrated_run(
        self,
        turn: PreparedAnswerTurn,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None,
    ) -> _OrchestratorRun:
        """Resolve one answer request into a capability-driven orchestrator."""
        ws_list = await self._open_query_workspaces(
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
        # One window for the whole request: planning and answering must agree on what
        # the conversation is, or a rewrite can cite a turn the answer never sees.
        history = PriorTurns(list(turn.text_history)).recent(
            max_messages=self._config.max_conversation_turns * 2,
            max_tokens=self._config.max_conversation_tokens,
        )
        (
            current_images,
            remaining_resources,
            current_image_resources,
        ) = await self._prepare_current_images(resources)
        if current_images:
            await self._maybe_reprobe_answer_image_capability()
        _check_answer_image_capability(
            query_images=current_images,
            capability=self._answer_image_capability,
        )

        web_search = self._get_web_search()
        registry, resource_tools = self._build_resource_context(
            remaining_resources, web_search=web_search
        )
        current_image_resource_ids = (
            tuple(registry.register(resource) for resource in current_image_resources)
            if registry is not None
            else ()
        )
        resource_manifest = registry.manifest() if registry is not None else ()
        research = web_search is not None or bool(resource_manifest)
        prepared = PreparedQueryImages(descriptions=[], descriptions_by_ordinal={})
        prepared_task: asyncio.Task[PreparedQueryImages] | None = None

        async def prepare_images_once() -> PreparedQueryImages:
            nonlocal prepared_task

            async def prepare() -> PreparedQueryImages:
                if not current_images:
                    return prepared
                result = await prepare_query_images(
                    query_images=current_images,
                    describer=await self._aget_query_image_describer(),
                )
                prepared.descriptions = result.descriptions
                prepared.descriptions_by_ordinal = result.descriptions_by_ordinal
                return prepared

            if prepared_task is None:
                prepared_task = asyncio.create_task(prepare())
            return await prepared_task

        plan: QueryPlan | None = None
        if not research:
            plan, prepared = await self._describe_and_plan(
                turn.retrieval_query,
                text_history=history,
                query_images=current_images,
                ws_list=ws_list,
            )

        async def retrieve_knowledge_base(search_query: str) -> RetrievalResult:
            if research:

                async def plan_agent_query() -> QueryPlan:
                    image_context = await prepare_images_once()
                    return await self._aplan_query_prepared(
                        search_query,
                        text_history=None,
                        current_image_descriptions=image_context.descriptions or None,
                        workspaces=ws_list,
                        preserve_query=True,
                    )

                tool_plan = await plan_agent_query()
                return await self.aretrieve(
                    search_query,
                    plan=tool_plan,
                    workspaces=ws_list,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    filters=filters,
                    query_images=current_images,
                )
            if plan is None:  # pragma: no cover - construction invariant
                raise RuntimeError("Fast answer requires a query plan")
            return await self.aretrieve(
                turn.current_query,
                plan=plan,
                workspaces=ws_list,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                filters=filters,
                query_images=current_images,
            )

        model_func: Callable[..., Any] | None = None
        stream_model_func: Callable[..., AsyncIterator[str]] | None = None
        final_text_func: Callable[..., Awaitable[str]] | None = None
        if research:
            tool_model = self._get_query_tool_model()

            async def _model_func(**kwargs: Any) -> Any:
                async with self._direct_llm_sem:
                    return await tool_model(**kwargs)

            async def _final_text_func(**kwargs: Any) -> str:
                async with self._direct_llm_sem:
                    return await tool_model.complete_text(**kwargs)

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
            final_text_func = _final_text_func

        image_budget: AnswerImageBudget | None = None
        if research:
            image_budget = self._new_answer_image_budget()
            query_images = (
                self._budget_agent_images(
                    current_images,
                    image_budget,
                    current_image_resource_ids,
                )
                or None
            )
        else:
            query_images = current_images or None

        orchestrator = AnswerOrchestrator(
            synthesizer=self._get_answer_synthesizer(),
            retrieve_knowledge_base=retrieve_knowledge_base,
            search_web=web_search.search if web_search is not None else None,
            model_func=model_func,
            stream_model_func=stream_model_func,
            final_text_func=final_text_func,
            resource_tools=resource_tools,
            resource_manifest=resource_manifest,
            register_web_source=(
                registry.register_discovered_link
                if registry is not None and web_search is not None
                else None
            ),
            image_budget=image_budget,
            context_window_tokens=self._config.answer.context_window_tokens,
            max_agent_turns=self._config.max_agent_turns,
        )
        return _OrchestratorRun(
            orchestrator=orchestrator,
            answer_query=turn.current_query if research else cast(QueryPlan, plan).standalone_query,
            retrieval_query=None if research else cast(QueryPlan, plan).standalone_query,
            prepared=prepared,
            query_images=query_images,
            history=history,
            current_image_count=len(current_images),
            ws_list=ws_list,
            registry=registry,
        )

    async def _prepare_current_images(
        self,
        resources: list[ResourceInput] | None,
    ) -> tuple[list[dict[str, Any]], list[ResourceInput], list[ResourceInput]]:
        """Build current-image blocks while retaining every attachment as a resource.

        Inline bytes that decode as a real image and remote image links
        (materialized under SSRF revalidation) become internal current-image
        blocks fed to VLM description, direct visual retrieval, and final answer
        transport. Their verified bytes also stay in the request-local registry
        for focused ``inspect_resource`` calls; a remote image is not fetched
        twice. Durable lazy resources (prior attachments) and every non-image
        resource remain lazy. Non-image bytes never enter the image chain.
        """
        if not resources:
            return [], [], []
        from dlightrag.core.resources import ResourceInput
        from dlightrag.utils.images import image_bytes_to_data_uri, verify_web_image_bytes

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
                mime = verify_web_image_bytes(data, max_pixels=max_pixels)
            except ValueError:
                remaining.append(resource)
                continue
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_bytes_to_data_uri(data, fallback_mime=mime)},
                }
            )
            image_resource = ResourceInput(
                filename=resource.filename,
                content=data,
                declared_mime=mime,
            )
            remaining.append(image_resource)
            image_resources.append(image_resource)
        limit = max(0, int(self._config.query_images.max_current_images))
        if len(images) > limit:
            raise CurrentImagePayloadError(f"at most {limit} current images are allowed")
        return images, remaining, image_resources

    async def _materialize_link_image(self, url: str) -> bytes | None:
        """Fetch a current-image link under SSRF revalidation; None if it fails."""
        from dlightrag.sourcing.url import afetch_public_https_bytes, validate_public_https_url

        try:
            validate_public_https_url(url, resolve_host=True)
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
        web_search: ExaSearch | None = None,
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
        from dlightrag.core.resources.tools import build_resource_tools
        from dlightrag.core.resources.visual import ResourceInspector

        answer = self._config.answer
        registry = ResourceRegistry(
            max_attachments=answer.max_attachments,
            max_attachment_bytes=answer.max_attachment_bytes,
            max_total_attachment_bytes=answer.max_total_attachment_bytes,
            url_text_fallback=_exa_contents_text(web_search) if web_search is not None else None,
        )
        for resource in resources or []:
            registry.register(resource)

        capability = self._answer_image_capability
        visual_supported = capability is not None and capability.status == "supported"
        inspector: ResourceInspector | None = None
        if visual_supported:
            from dlightrag.models.llm import get_vlm_model_func

            inspector = ResourceInspector(
                registry,
                vlm_func=self._sem_bound(get_vlm_model_func(self._config)),
                max_total_bytes=answer.image_max_total_bytes,
                max_bytes_per_image=answer.image_max_bytes,
                max_pixels=answer.image_max_pixels,
                max_px=answer.image_max_px,
                min_px=answer.image_min_px,
                quality=answer.image_quality,
                min_quality=answer.image_min_quality,
            )
        tools = build_resource_tools(
            registry,
            inspector=inspector,
            visual_supported=visual_supported,
        )
        return registry, tools

    async def _aanswer_orchestrated(
        self,
        turn: PreparedAnswerTurn,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        semantic_highlights: bool,
        resources: list[ResourceInput] | None = None,
    ) -> RetrievalResult:
        from dlightrag.observability import trace_observation

        run = await self._prepare_orchestrated_run(
            turn,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            resources=resources,
        )
        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_observation(
                    "answer_orchestration",
                    as_type="chain",
                    input={"query": turn.current_query},
                    metadata={
                        "stream": False,
                        "research": run.orchestrator.uses_research_path,
                        "workspaces": run.ws_list,
                        "history_turns": len(run.history or []),
                        "query_image_count": run.current_image_count,
                        "semantic_highlights": semantic_highlights,
                    },
                ) as pipeline_trace:
                    await self._acquire_answer_slot()
                    try:
                        result = await run.orchestrator.answer(
                            run.answer_query,
                            retrieval_query=run.retrieval_query,
                            conversation_history=run.history,
                            query_images=run.query_images,
                        )
                    finally:
                        self._answer_stream_sem.release()
                    result.trace["query_image_description_count"] = len(run.prepared.descriptions)
                    result.image_descriptions = run.prepared.descriptions
                    self._set_answer_media(result)
                    if semantic_highlights:
                        from dlightrag.core.answer.highlights import enrich_semantic_highlights

                        result.sources = await enrich_semantic_highlights(
                            result.sources,
                            answer_text=result.answer,
                            config=self._config,
                        )
                        self._set_answer_media(result)
                    pipeline_trace.update(
                        output=answer_trace_output(
                            result.answer,
                            result.sources,
                            result.contexts,
                        )
                    )
                    return result
        except TimeoutError as exc:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from exc
        finally:
            if run.registry is not None:
                await run.registry.aclose()

    async def _aanswer_stream_orchestrated(
        self,
        turn: PreparedAnswerTurn,
        *,
        workspace: str | None,
        workspaces: list[str] | None,
        all_workspaces: bool,
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        from dlightrag.observability import trace_observation

        run = await self._prepare_orchestrated_run(
            turn,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            resources=resources,
        )
        registry_transferred = False
        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_observation(
                    "answer_orchestration",
                    as_type="chain",
                    input={"query": turn.current_query},
                    metadata={
                        "stream": True,
                        "research": run.orchestrator.uses_research_path,
                        "workspaces": run.ws_list,
                        "history_turns": len(run.history or []),
                        "query_image_count": run.current_image_count,
                    },
                ) as pipeline_trace:
                    await self._acquire_answer_slot()
                    answer_slot_owned = True
                    try:
                        contexts, stream = await run.orchestrator.answer_stream(
                            run.answer_query,
                            retrieval_query=run.retrieval_query,
                            conversation_history=run.history,
                            query_images=run.query_images,
                        )
                        if stream is None:
                            return contexts, None
                        stream_meta = cast(Any, stream)
                        existing_trace = getattr(stream_meta, "trace", None)
                        merged_trace = (
                            dict(existing_trace) if isinstance(existing_trace, dict) else {}
                        )
                        merged_trace["query_image_description_count"] = len(
                            run.prepared.descriptions
                        )
                        stream_meta.trace = merged_trace
                        stream_meta.image_descriptions = run.prepared.descriptions_by_ordinal
                        wrapped = _ScopedAnswerStream(
                            stream,
                            self._answer_stream_sem,
                            on_close=run.registry.aclose if run.registry is not None else None,
                        )
                        answer_slot_owned = False
                        registry_transferred = True
                        pipeline_trace.update(
                            output={
                                **_context_output(contexts),
                                "agent_turns": merged_trace.get("agent_turns", 0),
                            }
                        )
                        return contexts, wrapped
                    finally:
                        if answer_slot_owned:
                            self._answer_stream_sem.release()
        except TimeoutError as exc:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from exc
        finally:
            if run.registry is not None and not registry_transferred:
                await run.registry.aclose()

    async def _acquire_answer_slot(self) -> None:
        try:
            await asyncio.wait_for(
                self._answer_stream_sem.acquire(),
                timeout=self._config.answer_acquire_timeout,
            )
        except TimeoutError as exc:
            raise RAGServiceUnavailableError("Every answer slot is busy; retry shortly.") from exc

    @staticmethod
    def _set_answer_media(result: RetrievalResult) -> None:
        from dlightrag.core.answer.media import (
            answer_blocks_from_markdown,
            answer_images_from_sources,
        )
        from dlightrag.models.schemas import Reference

        result.references = [
            Reference(id=source.id, title=source.title or "Source") for source in result.sources
        ]
        result.answer_images = answer_images_from_sources(
            result.sources,
            contexts=result.contexts,
        )
        result.answer_blocks = answer_blocks_from_markdown(
            result.answer,
            result.answer_images,
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
    ) -> RetrievalResult:
        """Answer from one or more workspaces through the one answer orchestrator.

        Current-turn images and documents are supplied through ``resources``. The
        manager prepares verified current images as internal image blocks and
        registers every attachment request-locally. Fast answers describe images
        before fixed KB retrieval; research answers show them directly to the
        agent and describe them only if it selects KB search. Attachment text is
        read only through resource tools and never enters query planning.

        ``history`` is caller-supplied prior turns (``role``/``content`` dicts).
        It is stateless -- the caller owns persistence and passes it per request.
        The orchestrator takes the standard-RAG fast path unless the request has
        resources or an open-web capability, in which case it researches.
        """
        return await self._aanswer_orchestrated(
            PreparedAnswerTurn.stateless(query, history=history),
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            semantic_highlights=semantic_highlights,
            resources=resources,
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
        resources: list[ResourceInput] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer from one or more workspaces through the orchestrator.

        See ``aanswer`` for ``resources`` and ``history`` semantics.
        """
        return await self._aanswer_stream_prepared(
            PreparedAnswerTurn.stateless(query, history=history),
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            resources=resources,
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
        filters: MetadataFilter | None = None,
        resources: list[ResourceInput] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Stream one server-prepared turn through the one answer orchestrator."""
        return await self._aanswer_stream_orchestrated(
            turn,
            workspace=workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            filters=filters,
            resources=resources,
        )

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

        cancellation: asyncio.CancelledError | None = None
        await self._ingest_jobs.close()

        for warmup in list(self._warmups):
            warmup.cancel()
        if self._warmups:
            await asyncio.gather(*self._warmups, return_exceptions=True)

        async with self._query_image_describer_lock:
            self._query_image_describer = None

        for component in (
            self._answer_synthesizer,
            self._query_planner,
            self._query_tool_model,
            self._web_search,
        ):
            close = getattr(component, "aclose", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except asyncio.CancelledError as exc:
                cancellation = _defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close manager component", exc_info=True)

        for ws, svc in self._services.items():
            try:
                await svc.aclose()
            except asyncio.CancelledError as exc:
                cancellation = _defer_cancellation(cancellation, exc)
            except Exception:
                logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
        self._services.clear()
        self._ready = False

        from dlightrag.storage.pool import pg_pool

        try:
            await pg_pool.close()
        except asyncio.CancelledError as exc:
            cancellation = _defer_cancellation(cancellation, exc)
        shutdown_tracing()
        if cancellation is not None:
            raise cancellation

    # --- Health ---

    def is_ready(self) -> bool:
        """Check if manager is ready (default workspace initialized)."""
        return self._ready

    def is_degraded(self) -> bool:
        return self._degraded

    def get_warnings(self) -> list[str]:
        return list(self._startup_warnings)


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
