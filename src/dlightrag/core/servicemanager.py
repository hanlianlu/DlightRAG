# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGServiceManager — unified multi-workspace RAG coordinator.

Absorbs pool.py workspace management and federation routing into a single
entry point. All API/MCP consumers depend on this class only.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.ingest_job_coordinator import IngestJobCoordinator
from dlightrag.core.query_image_payloads import PreparedQueryImages, prepare_query_images
from dlightrag.core.query_planner import QueryPlan, QueryPlanner
from dlightrag.core.retrieval.metadata_fields import MetadataIngestPolicy
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.scope import RequestScope
from dlightrag.core.service import RAGService
from dlightrag.utils import normalize_workspace

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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

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


def _scope_for_workspaces(
    scope: RequestScope | None,
    workspaces: list[str] | tuple[str, ...] | None,
) -> RequestScope:
    base = scope or RequestScope.anonymous()
    normalized = _normalize_workspaces(workspaces)
    return base.for_workspaces(normalized) if normalized else base


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

# Module-level vision support — set once at startup by the server lifecycle.
# NOT stored on config.answer because Pydantic extra="forbid" rejects monkey-patching.
_supports_vision: bool | None = None


def set_vision_supported(supported: bool | None) -> None:
    """Store the startup vision probe result (called from server lifespan)."""
    global _supports_vision
    _supports_vision = supported


def _history_has_images(history: list[dict[str, Any]] | None) -> bool:
    """Return True if any message in *history* contains image_url blocks."""
    if not history:
        return False
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


def _check_vision_support(
    *,
    query_images: list[dict[str, Any]] | None,
    conversation_history: list[dict[str, Any]] | None,
) -> None:
    """Raise ValueError if images are present but model doesn't support vision.

    Checks both ``query_images`` and ``conversation_history`` for image
    content.  Reads the startup probe result from the module-level
    ``_supports_vision`` flag (set by ``set_vision_supported()`` in the
    server lifespan).
    """
    has_images = bool(query_images) or _history_has_images(conversation_history)
    if not has_images:
        return
    if _supports_vision is False:
        raise ValueError(f"[IMAGES_NOT_SUPPORTED_BY_MODEL] {_ERROR_IMAGES_NOT_SUPPORTED}")
    # None (unprobed) or True — allow through


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
        self._ingest_jobs = IngestJobCoordinator(lambda workspace: self._get_service(workspace))
        self._query_planner: QueryPlanner | None = None
        self._query_image_enhancer: Any = None
        self._session_images: Any = None
        self._workspace_registry: Any = None
        self._checkpoint: Any = None
        self._schema_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
        self._answer_stream_sem = asyncio.Semaphore(max(1, int(self._config.max_async)))

    @property
    def config(self) -> DlightragConfig:
        """Read-only access to the manager configuration for UI/API adapters."""
        return self._config

    @classmethod
    async def create(cls, config: DlightragConfig | None = None) -> RAGServiceManager:
        """Async factory — creates manager, warms up all known workspaces in parallel."""
        from dlightrag.observability import init_tracing
        from dlightrag.utils.concurrency import bounded_gather

        manager = cls(config=config)
        init_tracing(manager._config)

        await manager._initialize_workspace_registry()

        # Discover all known workspaces and warm them up in parallel
        all_ws = await manager._list_all_workspaces()
        default_ws = normalize_workspace(manager._config.workspace)
        if default_ws not in all_ws:
            all_ws.insert(0, default_ws)

        warmup_coros = [manager._get_service(ws) for ws in all_ws]
        results = await bounded_gather(
            warmup_coros,
            max_concurrent=max(1, int(manager._config.max_async)),
            task_name="warmup",
        )
        failed = [ws for ws, r in zip(all_ws, results, strict=True) if isinstance(r, Exception)]
        ok = len(all_ws) - len(failed)
        logger.info("Warmed up %d/%d workspace services", ok, len(all_ws))
        if failed:
            logger.warning("Failed to warm up: %s", ", ".join(failed))

        await manager._start_ingest_job_recovery()
        await manager._recover_stalled_docs(all_ws)
        await manager._prune_checkpoint_sessions()

        # ── Vision probe (once at startup, not per workspace) ──────────
        await manager._probe_vision_support()

        if default_ws in manager._services:
            manager._ready = True
        else:
            manager._degraded = True
            # Include original error detail for diagnostics
            default_err = next(
                (
                    r
                    for ws, r in zip(all_ws, results, strict=True)
                    if ws == default_ws and isinstance(r, Exception)
                ),
                None,
            )
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

    async def _get_workspace_registry(self) -> Any:
        if self._workspace_registry is None:
            await self._initialize_workspace_registry()
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
                svc = await RAGService.create(config=ws_config)
                self._services[workspace] = svc

                # Clear backoff on success
                self._backoff.pop(workspace, None)

                logger.info("Created RAGService for workspace '%s'", workspace)
                return svc
            except Exception as e:
                error_msg = self._actionable_error(e)
                # Per-workspace exponential backoff
                _, prev_interval = self._backoff.get(workspace, (0, 7.5))
                new_interval = min(prev_interval * 2, _MAX_RETRY_INTERVAL)
                self._backoff[workspace] = (time.time(), new_interval)
                logger.error(
                    "RAGService creation failed for '%s': %s. Retry in %ss",
                    workspace,
                    error_msg,
                    new_interval,
                )
                raise RAGServiceUnavailableError(detail=error_msg) from e

    # --- Write operations (single workspace) ---

    async def aingest(
        self,
        workspace: str,
        source_type: Literal["local", "azure_blob", "s3"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start an ingest job, wait according to config, and return the result if ready."""
        job = await self.astart_ingest_job(workspace, source_type=source_type, **kwargs)
        row = await self.await_ingest_job(job["job_id"], timeout=self._config.ingest_timeout)
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

    async def _prune_checkpoint_sessions(self) -> None:
        """Prune old checkpoint sessions at startup.

        Checkpoint data (conversation turns, context anchors) grows unboundedly.
        Sessions whose ``updated_at`` exceeds ``checkpoint_session_ttl_days``
        are deleted.  Best-effort: the checkpoint SQLite file may not exist yet
        (no sessions have been created), and failures are logged but never
        block startup.
        """
        max_age_days = int(self._config.checkpoint_session_ttl_days)
        try:
            cp = self._get_checkpoint()
            deleted = await cp.prune_old_sessions(max_age_days=max_age_days)
            if deleted:
                logger.info(
                    "Pruned %d checkpoint session(s) older than %d days",
                    deleted,
                    max_age_days,
                )
        except Exception:
            logger.debug("Checkpoint session pruning skipped", exc_info=True)

    async def _probe_vision_support(self) -> None:
        """Probe the chat model's vision capability once at startup.

        Sets the module-level ``_supports_vision`` flag consumed by
        ``_check_vision_support`` on every request.
        """
        global _supports_vision
        if _supports_vision is not None:
            return  # already probed

        from dlightrag.core.vision_probe import probe_vision_support
        from dlightrag.models.providers import get_provider

        default = self._config.llm.default
        provider = get_provider(
            default.provider,
            api_key=default.api_key,
            base_url=default.base_url,
            timeout=default.timeout,
            max_retries=default.max_retries,
        )
        try:
            has_vision = await probe_vision_support(provider, model=default.model)
            _supports_vision = has_vision
            logger.info(
                "Chat model vision probe: %s (model=%s, provider=%s)",
                "supported" if has_vision else "not supported",
                default.model,
                default.provider,
            )
        except Exception:
            logger.debug(
                "Vision probe failed; deferring to per-request check",
                exc_info=True,
            )
        finally:
            await provider.aclose()

    async def astart_ingest_job(
        self,
        workspace: str,
        source_type: Literal["local", "azure_blob", "s3"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a background ingest job and return its durable job row."""
        return await self._ingest_jobs.start_job(workspace, source_type, **kwargs)

    async def await_ingest_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Wait for an in-process ingest job without cancelling it on timeout."""
        return await self._ingest_jobs.await_job(job_id, timeout=timeout)

    async def get_ingest_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._ingest_jobs.get_job(job_id)

    async def acreate_workspace(self, workspace: str, *, display_name: str | None = None) -> None:
        """Initialize a workspace through the public manager API."""
        svc = await self._get_service(workspace)
        await svc.aregister_workspace(display_name=display_name)

    async def list_ingested_files(self, workspace: str) -> list[dict[str, Any]]:
        """List ingested files in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def get_pipeline_status(self, workspace: str) -> dict[str, Any]:
        """Return pipeline progress for a workspace."""
        svc = await self._get_service(workspace)
        return await svc.aget_pipeline_status()

    async def delete_files(self, workspace: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Delete files from a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.adelete_files(**kwargs)

    async def list_failed_docs(self, workspace: str) -> list[dict[str, Any]]:
        """List FAILED documents in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_failed_docs()

    async def aget_visual_asset(self, workspace: str, chunk_id: str, *, size: str = "full") -> Any:
        """Resolve a visual chunk asset for browser/API image routes."""
        svc = await self._get_service(workspace)
        return await svc.aget_visual_asset(chunk_id, size=size)

    async def retry_failed_docs(self, workspace: str) -> dict[str, Any]:
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
            known = await self.list_workspaces()
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
            workspaces = await self.list_workspaces()

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
                results[ws] = {"error": str(exc), "ingest_jobs_cancelled": cancelled_jobs}
                total_errors += 1
                logger.warning("Failed to reset workspace '%s': %s", ws, exc)

            # Close and evict from cache even after reset errors.
            if ws in self._services:
                try:
                    await self._services[ws].close()
                except Exception:
                    logger.warning("Failed to close service for '%s'", ws, exc_info=True)
                del self._services[ws]

        return {"workspaces": results, "total_errors": total_errors}

    def _get_answer_engine(self) -> AnswerEngine:
        """Lazy-create AnswerEngine from global config."""
        if self._answer_engine is None:
            from dlightrag.models.llm import get_query_model_func

            answer_cfg = self._config.answer
            self._answer_engine = AnswerEngine(
                model_func=get_query_model_func(self._config),
                max_images=answer_cfg.max_images,
                max_user_images=answer_cfg.max_user_images,
                image_max_bytes=answer_cfg.image_max_bytes,
                image_max_total_bytes=answer_cfg.image_max_total_bytes,
                image_max_px=answer_cfg.image_max_px,
                image_min_px=answer_cfg.image_min_px,
                image_quality=answer_cfg.image_quality,
                image_min_quality=answer_cfg.image_min_quality,
                context_top_k=answer_cfg.context_top_k,
            )
        return self._answer_engine

    def _get_query_planner(self) -> QueryPlanner:
        """Lazy-create QueryPlanner with schema TTL refresh."""
        if self._query_planner is None:
            from dlightrag.models.llm import get_query_model_func

            self._query_planner = QueryPlanner(
                llm_func=get_query_model_func(self._config),
                schema_provider=self._get_schema,
                schema_ttl=300.0,
            )
        return self._query_planner

    def _get_session_images(self):
        """Lazy-create session image memory."""
        if self._session_images is None:
            from dlightrag.core.session_images import SessionImageStore

            cfg = self._config.query_images
            self._session_images = SessionImageStore(
                max_images_per_session=cfg.session_max_images,
                max_sessions=cfg.session_max_sessions,
                ttl_seconds=self._config.checkpoint_session_ttl_seconds,
            )
        return self._session_images

    async def get_session_image_data(
        self,
        session_id: str | None,
        image_ids: list[str] | None,
        *,
        scope: RequestScope | None = None,
    ) -> list[str]:
        """Return stored query image data for web/API presentation layers."""
        scoped_session_id = scope.session_key(session_id) if scope is not None else session_id
        return self._get_session_images().get(scoped_session_id, image_ids)

    def _get_checkpoint(self):
        """Lazy-create conversation checkpoint store."""
        if self._checkpoint is None:
            from dlightrag.storage.checkpoint_pg import PGCheckpointStore

            self._checkpoint = PGCheckpointStore()
        return self._checkpoint

    async def aget_checkpoint_history(
        self,
        *,
        session_id: str,
        scope: RequestScope | None = None,
        max_turns: int = 50,
    ) -> list[dict[str, Any]]:
        """Return conversation history for a session from the checkpoint store.

        Reconstructs the scoped session key so the same raw ``session_id``
        returns the same history across requests, matching the scoping used
        by ``save_turn_checkpoint``.
        """
        if not session_id:
            return []
        try:
            cp = self._get_checkpoint()
            scoped = _scope_for_workspaces(scope, None)
            scoped_session_id = scoped.session_key(session_id) or session_id
            return await cp.get_history(scoped_session_id, max_turns=max_turns)
        except Exception:
            logger.debug(
                "Failed to load checkpoint history for session %s", session_id, exc_info=True
            )
            return []

    async def adelete_checkpoint_session(
        self,
        *,
        session_id: str,
        scope: RequestScope | None = None,
    ) -> int:
        """Delete all checkpoint rows for a session. Returns count deleted."""
        if not session_id:
            return 0
        try:
            cp = self._get_checkpoint()
            scoped = _scope_for_workspaces(scope, None)
            scoped_session_id = scoped.session_key(session_id) or session_id
            return await cp.delete_session(scoped_session_id)
        except Exception:
            logger.debug("Failed to delete checkpoint session %s", session_id, exc_info=True)
            return 0

    async def save_turn_checkpoint(
        self,
        session_id: str,
        query: str,
        answer: str,
        contexts: dict[str, Any],
        cited_chunk_ids: list[str],
        *,
        query_content: list[dict[str, Any]] | None = None,
        answer_content: list[dict[str, Any]] | None = None,
        workspace: str | None = None,
        scope: RequestScope | None = None,
    ) -> None:
        """Save a complete turn (query + answer + retrieval anchors) to checkpoint."""
        if not session_id:
            return
        try:
            cp = self._get_checkpoint()
            effective_workspace = normalize_workspace(
                workspace
                or (scope.workspaces[0] if scope is not None and scope.workspaces else "")
                or self._config.workspace
            )
            scoped = _scope_for_workspaces(scope, [effective_workspace])
            scoped_session_id = scoped.session_key(session_id) or session_id
            await cp.save_turn_pair(
                scoped_session_id,
                workspace=effective_workspace,
                query=query,
                answer=answer,
                query_content=query_content,
                answer_content=answer_content,
                contexts=contexts,
                cited_chunk_ids=cited_chunk_ids,
            )
        except Exception:
            logger.warning("Failed to save checkpoint for session %s", session_id, exc_info=True)

    def _get_query_image_enhancer(self):
        """Lazy-create VLM query image enhancer."""
        if self._query_image_enhancer is None:
            from dlightrag.core.query_images import QueryImageEnhancer
            from dlightrag.models.llm import get_vlm_model_func

            cfg = self._config.query_images
            self._query_image_enhancer = QueryImageEnhancer(
                vlm_func=get_vlm_model_func(self._config),
                max_images=cfg.max_described_images,
            )
        return self._query_image_enhancer

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
                logger.debug("Schema lookup failed for workspace %s", workspace, exc_info=True)
        merged = _merge_schema_rows(schemas)
        self._schema_cache[ws_key] = (now, merged)
        return merged

    async def aplan_query(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        workspaces: list[str] | tuple[str, ...] | None = None,
    ) -> QueryPlan:
        """Plan a query using the manager-owned planner and config limits."""
        planner = self._get_query_planner()
        schema = await self._get_schema(workspaces)
        return await planner.plan(
            query,
            conversation_history=conversation_history,
            max_turns=self._config.max_conversation_turns,
            max_tokens=self._config.max_conversation_tokens,
            schema=schema,
        )

    async def agenerate_stream_from_contexts(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Generate a streaming answer from already-retrieved contexts."""
        engine = self._get_answer_engine()
        await self._answer_stream_sem.acquire()
        try:
            prepared_contexts, stream = await engine.generate_stream(
                query,
                contexts,
                query_images=query_images,
                conversation_history=conversation_history,
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

    async def _prepare_answer_retrieval(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        ws_list: list[str],
        scope: RequestScope,
        kwargs: dict[str, Any],
        log_plan: bool = False,
    ) -> tuple[QueryPlan, PreparedQueryImages, _AnswerLimits, RetrievalResult]:
        plan = await self.aplan_query(
            query,
            conversation_history=conversation_history,
            workspaces=ws_list,
        )
        if log_plan:
            logger.info(
                "Query plan: original=%r, standalone=%r",
                plan.original_query[:60],
                plan.standalone_query[:60],
            )

        prepared = await prepare_query_images(
            plan.standalone_query,
            query_images=query_images,
            session_id=kwargs.pop("session_id", None),
            referenced_image_ids=kwargs.pop("referenced_image_ids", None)
            or plan.referenced_image_ids,
            store_current=True,
            session_images=self._get_session_images(),
            enhancer=self._get_query_image_enhancer(),
            scope=scope,
        )
        retrieval_plan = replace(plan, standalone_query=prepared.query)
        if prepared.multimodal_content:
            existing_mm = kwargs.get("multimodal_content") or []
            kwargs["multimodal_content"] = [*existing_mm, *prepared.multimodal_content]
        limits = self._resolve_answer_limits(kwargs)
        retrieval = await self.aretrieve(
            query,
            plan=retrieval_plan,
            workspaces=ws_list,
            scope=scope,
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
        plan: QueryPlan | None = None,
        query_images: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        referenced_image_ids: list[str] | None = None,
        scope: RequestScope | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        Args:
            plan: Pre-computed QueryPlan from QueryPlanner. When provided,
                uses plan.standalone_query for search and plan.metadata_filter
                for structured filtering.
        """
        requested_top_k = _positive_int_or_none(kwargs.get("top_k"))
        requested_chunk_top_k = _positive_int_or_none(kwargs.get("chunk_top_k"))
        kwargs["top_k"] = requested_top_k or self._config.top_k
        kwargs["chunk_top_k"] = requested_chunk_top_k or self._config.chunk_top_k
        ws_list = workspaces or [workspace or self._config.workspace]
        scoped = _scope_for_workspaces(scope, ws_list)
        effective_query = plan.standalone_query if plan is not None else query
        prepared = await prepare_query_images(
            effective_query,
            query_images=query_images,
            session_id=session_id,
            referenced_image_ids=referenced_image_ids
            or (plan.referenced_image_ids if plan is not None else None),
            store_current=False,
            session_images=self._get_session_images(),
            enhancer=self._get_query_image_enhancer(),
            scope=scoped,
        )
        effective_query = prepared.query
        if prepared.multimodal_content:
            existing_mm = kwargs.get("multimodal_content") or []
            kwargs["multimodal_content"] = [*existing_mm, *prepared.multimodal_content]
        if plan is not None:
            kwargs.setdefault("_plan", plan)
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline("retrieve", workspaces=ws_list, query=effective_query):
                    if len(ws_list) == 1:
                        svc = await self._get_service(ws_list[0])
                        result = await svc.aretrieve(effective_query, **kwargs)
                    else:
                        result = await federated_retrieve(
                            effective_query, ws_list, self._get_service, **kwargs
                        )
                    result.image_descriptions = prepared.descriptions
                    result.current_image_ids = prepared.current_image_ids
                    result.trace["query_image_description_count"] = len(prepared.descriptions)
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def aanswer(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        query_images: list[dict[str, Any]] | None = None,
        scope: RequestScope | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Answer from one or more workspaces: plan -> retrieve -> generate.

        ``query_images`` are user-attached ``image_url`` blocks inlined into
        the answer LLM call. Distinct
        from retrieval-time ``multimodal_content`` in ``kwargs``: this list
        only affects answer generation, never the retrieval pipeline.
        """
        ws_list = workspaces or [workspace or self._config.workspace]
        scoped = _scope_for_workspaces(scope, ws_list)

        # Guard: reject image input when the model doesn't support it
        _check_vision_support(
            query_images=query_images,
            conversation_history=conversation_history,
        )

        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline("answer_pipeline", workspaces=ws_list, query=query):
                    plan, prepared, limits, retrieval = await self._prepare_answer_retrieval(
                        query,
                        conversation_history=conversation_history,
                        query_images=query_images,
                        ws_list=ws_list,
                        scope=scoped,
                        kwargs=kwargs,
                    )
                    engine = self._get_answer_engine()
                    result = await engine.generate(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
                        conversation_history=conversation_history,
                        context_top_k=limits.context_top_k,
                    )
                    result.trace.update(retrieval.trace)
                    result.image_descriptions = prepared.descriptions
                    result.current_image_ids = prepared.current_image_ids
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def aanswer_stream(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        query_images: list[dict[str, Any]] | None = None,
        scope: RequestScope | None = None,
        **kwargs: Any,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer from one or more workspaces: plan -> retrieve -> stream.

        See ``aanswer`` for ``query_images`` semantics.
        """
        ws_list = workspaces or [workspace or self._config.workspace]
        scoped = _scope_for_workspaces(scope, ws_list)

        # Guard: reject image input when the model doesn't support it
        _check_vision_support(
            query_images=query_images,
            conversation_history=conversation_history,
        )

        scoped = _scope_for_workspaces(scope, ws_list)
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline(
                    "answer_stream_pipeline", workspaces=ws_list, query=query
                ):
                    plan, prepared, limits, retrieval = await self._prepare_answer_retrieval(
                        query,
                        conversation_history=conversation_history,
                        query_images=query_images,
                        ws_list=ws_list,
                        scope=scoped,
                        kwargs=kwargs,
                        log_plan=True,
                    )
                    contexts, stream = await self.agenerate_stream_from_contexts(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
                        conversation_history=conversation_history,
                        context_top_k=limits.context_top_k,
                    )
                    if stream is not None:
                        stream_meta = cast(Any, stream)
                        stream_meta.current_image_ids = prepared.current_image_ids
                        stream_meta.image_descriptions = prepared.descriptions
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

    async def list_workspaces(self) -> list[str]:
        """Discover available workspaces."""
        records = await self.list_workspace_records()
        return [row["workspace"] for row in records]

    async def list_workspace_records(self) -> list[dict[str, Any]]:
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
        return await self.list_workspaces()

    async def close(self) -> None:
        """Close all managed RAGService instances."""
        from dlightrag.observability import shutdown_tracing

        await self._ingest_jobs.close()

        for component in (
            self._answer_engine,
            self._query_planner,
            self._query_image_enhancer,
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
                await svc.close()
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
