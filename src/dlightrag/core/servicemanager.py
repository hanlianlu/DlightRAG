# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGServiceManager — unified multi-workspace RAG coordinator.

Absorbs pool.py workspace management and federation routing into a single
entry point. All API/MCP consumers depend on this class only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.query_planner import QueryPlan, QueryPlanner
from dlightrag.core.retrieval.metadata_fields import MetadataIngestPolicy
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)

_MAX_RETRY_INTERVAL: float = 300.0


@dataclass
class _PreparedQueryImages:
    query: str
    answer_images: list[str | dict[str, Any]]
    multimodal_content: list[dict[str, Any]]
    descriptions: list[str]
    current_image_ids: list[str]


class RAGServiceUnavailableError(Exception):
    """Raised when the RAG service is not ready."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "RAG service is not available"
        super().__init__(self.detail)


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
        self._query_planner: QueryPlanner | None = None
        self._query_image_enhancer: Any = None
        self._session_images: Any = None

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

        # Discover all known workspaces and warm them up in parallel
        all_ws = await manager._list_all_workspaces()
        default_ws = config.workspace if config else manager._config.workspace
        if default_ws not in all_ws:
            all_ws.insert(0, default_ws)

        warmup_coros = [manager._get_service(ws) for ws in all_ws]
        results = await bounded_gather(warmup_coros, max_concurrent=8, task_name="warmup")
        failed = [ws for ws, r in zip(all_ws, results, strict=True) if isinstance(r, Exception)]
        ok = len(all_ws) - len(failed)
        logger.info("Warmed up %d/%d workspace services", ok, len(all_ws))
        if failed:
            logger.warning("Failed to warm up: %s", ", ".join(failed))

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

    def _ensure_writable(self, operation: str) -> None:
        """Fail fast when a mutating API is invoked by a query-role process."""
        if self._config.is_query_role is True:
            raise PermissionError(
                f"{operation} is not available when runtime_role='query'; "
                "route it to an ingest/admin worker connected to primary PostgreSQL"
            )

    async def _wait_after_write(self) -> str | None:
        """Apply configured read-after-write barrier for primary-backed writes."""
        if self._config.read_after_write_mode != "wait_for_replay":
            return None

        from dlightrag.storage.replication import wait_for_current_wal_replay

        try:
            return await wait_for_current_wal_replay(
                self._config,
                timeout=self._config.read_after_write_timeout,
            )
        except TimeoutError as exc:
            raise RAGServiceUnavailableError(detail=str(exc)) from exc

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
        """Ingest documents into a specific workspace."""
        self._ensure_writable("ingest")
        try:
            async with asyncio.timeout(self._config.request_timeout):
                svc = await self._get_service(workspace)
                result = await svc.aingest(source_type=source_type, **kwargs)
                replay_lsn = await self._wait_after_write()
                if replay_lsn is not None and isinstance(result, dict):
                    result["replica_replay_lsn"] = replay_lsn
                return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def acreate_workspace(self, workspace: str) -> None:
        """Initialize a workspace through the public manager API."""
        self._ensure_writable("create workspace")
        await self._get_service(workspace)
        await self._wait_after_write()

    async def list_ingested_files(self, workspace: str) -> list[dict[str, Any]]:
        """List ingested files in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def delete_files(self, workspace: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Delete files from a specific workspace."""
        self._ensure_writable("delete files")
        svc = await self._get_service(workspace)
        result = await svc.adelete_files(**kwargs)
        await self._wait_after_write()
        return result

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
        self._ensure_writable("retry failed docs")
        svc = await self._get_service(workspace)
        result = await svc.aretry_failed_docs()
        await self._wait_after_write()
        return result

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
        self._ensure_writable("metadata update")
        svc = await self._get_service(workspace)
        await svc.aupdate_metadata(doc_id, data, mode=mode, metadata_policy=metadata_policy)
        await self._wait_after_write()

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
        self._ensure_writable("reset")
        if workspace is not None:
            known = await self.list_workspaces()
            if workspace not in known and workspace not in self._services:
                return {"workspaces": {}, "total_errors": 0}
            workspaces = [workspace]
        else:
            workspaces = await self.list_workspaces()

        results: dict[str, Any] = {}
        total_errors = 0

        for ws in workspaces:
            try:
                svc = await self._get_service(ws)
                ws_result = await svc.areset(keep_files=keep_files, dry_run=dry_run)
                results[ws] = ws_result
                total_errors += len(ws_result.get("errors", []))
            except Exception as exc:
                results[ws] = {"error": str(exc)}
                total_errors += 1
                logger.warning("Failed to reset workspace '%s': %s", ws, exc)

            # Close and evict from cache (even on error — stale state)
            if ws in self._services:
                try:
                    await self._services[ws].close()
                except Exception:
                    logger.warning("Failed to close service for '%s'", ws, exc_info=True)
                del self._services[ws]

        if results:
            await self._wait_after_write()

        return {"workspaces": results, "total_errors": total_errors}

    def _get_answer_engine(self) -> AnswerEngine:
        """Lazy-create AnswerEngine from global config."""
        if self._answer_engine is None:
            from dlightrag.models.llm import get_query_model_func

            answer_cfg = self._config.answer
            self._answer_engine = AnswerEngine(
                model_func=get_query_model_func(self._config),
                max_images=answer_cfg.max_images,
                image_max_bytes=answer_cfg.image_max_bytes,
                image_max_total_bytes=answer_cfg.image_max_total_bytes,
                image_max_px=answer_cfg.image_max_px,
                image_quality=answer_cfg.image_quality,
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
                ttl_seconds=cfg.session_ttl_seconds,
            )
        return self._session_images

    def _get_query_image_enhancer(self):
        """Lazy-create VLM query image enhancer."""
        if self._query_image_enhancer is None:
            from dlightrag.core.query_images import QueryImageEnhancer
            from dlightrag.models.llm import get_vlm_model_func

            cfg = self._config.query_images
            self._query_image_enhancer = QueryImageEnhancer(
                vlm_func=get_vlm_model_func(self._config),
                enabled=cfg.semantic_enhancement,
                max_images=cfg.max_described_images,
            )
        return self._query_image_enhancer

    async def _get_schema(self) -> dict[str, Any]:
        """Fetch metadata schema from first available workspace."""
        workspaces = await self.list_workspaces()
        if workspaces:
            svc = await self._get_service(workspaces[0])
            if svc._metadata_index is not None:
                return await svc._metadata_index.get_field_schema()
        return {}

    def get_llm_func(self):
        """Return the global chat model function (for web UI etc.)."""
        from dlightrag.models.llm import get_chat_model_func

        return get_chat_model_func(self._config)

    async def aplan_query(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryPlan:
        """Plan a query using the manager-owned planner and config limits."""
        planner = self._get_query_planner()
        return await planner.plan(
            query,
            conversation_history=conversation_history,
            max_turns=self._config.max_conversation_turns,
            max_tokens=self._config.max_conversation_tokens,
        )

    async def agenerate_stream_from_contexts(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        query_images: list[str | dict[str, Any]] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Generate a streaming answer from already-retrieved contexts."""
        engine = self._get_answer_engine()
        return await engine.generate_stream(
            query,
            contexts,
            query_images=query_images,
        )

    # --- Read operations (single or federated) ---

    async def aretrieve(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        plan: QueryPlan | None = None,
        query_images: list[str | dict[str, Any]] | None = None,
        session_id: str | None = None,
        referenced_image_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        Args:
            plan: Pre-computed QueryPlan from QueryPlanner. When provided,
                uses plan.standalone_query for search and plan.metadata_filter
                for structured filtering.
        """
        effective_query = plan.standalone_query if plan else query
        prepared = await self._prepare_query_images(
            effective_query,
            query_images=query_images,
            session_id=session_id,
            referenced_image_ids=referenced_image_ids
            or (plan.referenced_image_ids if plan else None),
            store_current=False,
        )
        effective_query = prepared.query
        if prepared.multimodal_content:
            existing_mm = kwargs.get("multimodal_content") or []
            kwargs["multimodal_content"] = [*existing_mm, *prepared.multimodal_content]
        if plan is not None:
            kwargs.setdefault("_plan", plan)
        ws_list = workspaces or [workspace or self._config.workspace]
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
        conversation_history: list[dict[str, str]] | None = None,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        query_images: list[str | dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Answer from one or more workspaces: plan -> retrieve -> generate.

        ``query_images`` are user-attached images inlined into the answer LLM
        call as ``image_url`` blocks (URLs or pre-built dict blocks). Distinct
        from retrieval-time ``multimodal_content`` in ``kwargs``: this list
        only affects answer generation, never the retrieval pipeline.
        """
        ws_list = workspaces or [workspace or self._config.workspace]
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline("answer_pipeline", workspaces=ws_list, query=query):
                    plan = await self.aplan_query(
                        query,
                        conversation_history=conversation_history,
                    )
                    prepared = await self._prepare_query_images(
                        plan.standalone_query,
                        query_images=query_images,
                        session_id=kwargs.pop("session_id", None),
                        referenced_image_ids=kwargs.pop("referenced_image_ids", None)
                        or plan.referenced_image_ids,
                        store_current=True,
                    )
                    retrieval_plan = replace(plan, standalone_query=prepared.query)
                    if prepared.multimodal_content:
                        existing_mm = kwargs.get("multimodal_content") or []
                        kwargs["multimodal_content"] = [*existing_mm, *prepared.multimodal_content]
                    retrieval = await self.aretrieve(
                        query,
                        plan=retrieval_plan,
                        workspaces=ws_list,
                        **kwargs,
                    )
                    retrieval.trace["query_image_description_count"] = len(
                        prepared.descriptions
                    )
                    engine = self._get_answer_engine()
                    result = await engine.generate(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
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
        conversation_history: list[dict[str, str]] | None = None,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        query_images: list[str | dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer from one or more workspaces: plan -> retrieve -> stream.

        See ``aanswer`` for ``query_images`` semantics.
        """
        ws_list = workspaces or [workspace or self._config.workspace]
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline(
                    "answer_stream_pipeline", workspaces=ws_list, query=query
                ):
                    plan = await self.aplan_query(
                        query,
                        conversation_history=conversation_history,
                    )
                    logger.info(
                        "Query plan: original=%r, standalone=%r",
                        plan.original_query[:60],
                        plan.standalone_query[:60],
                    )

                    prepared = await self._prepare_query_images(
                        plan.standalone_query,
                        query_images=query_images,
                        session_id=kwargs.pop("session_id", None),
                        referenced_image_ids=kwargs.pop("referenced_image_ids", None)
                        or plan.referenced_image_ids,
                        store_current=True,
                    )
                    retrieval_plan = replace(plan, standalone_query=prepared.query)
                    if prepared.multimodal_content:
                        existing_mm = kwargs.get("multimodal_content") or []
                        kwargs["multimodal_content"] = [*existing_mm, *prepared.multimodal_content]
                    retrieval = await self.aretrieve(
                        query,
                        plan=retrieval_plan,
                        workspaces=ws_list,
                        **kwargs,
                    )
                    retrieval.trace["query_image_description_count"] = len(
                        prepared.descriptions
                    )
                    contexts, stream = await self.agenerate_stream_from_contexts(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
                    )
                    if stream is not None:
                        stream_meta = cast(Any, stream)
                        stream_meta.current_image_ids = prepared.current_image_ids
                        stream_meta.image_descriptions = prepared.descriptions
                        stream_meta.trace = retrieval.trace
                    return contexts, stream
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def _prepare_query_images(
        self,
        query: str,
        *,
        query_images: list[str | dict[str, Any]] | None,
        session_id: str | None,
        referenced_image_ids: list[str] | None,
        store_current: bool,
    ) -> _PreparedQueryImages:
        """Resolve session images and create semantic/direct image query inputs."""
        current_images = list(query_images or [])
        current_storable = _storable_image_strings(current_images)
        current_ids = (
            self._get_session_images().store(session_id, current_storable)
            if store_current and current_storable
            else []
        )
        historical = self._get_session_images().get(session_id, referenced_image_ids)
        answer_images: list[str | dict[str, Any]] = [*historical, *current_images]

        enhancer = self._get_query_image_enhancer()
        enhanced = await enhancer.enhance(query, answer_images)
        multimodal_content = _images_to_multimodal_content(answer_images)
        return _PreparedQueryImages(
            query=enhanced.query,
            answer_images=answer_images,
            multimodal_content=multimodal_content,
            descriptions=enhanced.descriptions,
            current_image_ids=current_ids,
        )

    # --- Management ---

    async def list_workspaces(self) -> list[str]:
        """Discover available workspaces."""
        return await self._list_all_workspaces()

    async def _list_all_workspaces(self) -> list[str]:
        """Internal: discover all workspaces."""
        config = self._config

        try:
            import asyncpg

            conn = await asyncpg.connect(**config.pg_connection_kwargs())
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT workspace FROM dlightrag_workspace_meta ORDER BY workspace"
                )
                workspaces = [row["workspace"] for row in rows]
                return workspaces if workspaces else [config.workspace]
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("Failed to list workspaces from PG: %s", exc)
            return [config.workspace]

    async def close(self) -> None:
        """Close all managed RAGService instances."""
        from dlightrag.observability import shutdown_tracing

        for ws, svc in self._services.items():
            try:
                await svc.close()
            except Exception:
                logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
        self._services.clear()
        self._ready = False
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


def _storable_image_strings(images: list[str | dict[str, Any]]) -> list[str]:
    return [image for image in images if isinstance(image, str) and image.strip()]


def _images_to_multimodal_content(images: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw base64/data URI images to direct visual retrieval inputs."""
    from dlightrag.utils.images import split_data_uri

    items: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, str):
            continue
        text = image.strip()
        if not text or text.startswith(("http://", "https://")):
            continue
        _, payload = split_data_uri(text)
        if payload:
            items.append({"type": "image", "data": payload})
    return items


__all__ = [
    "RAGServiceManager",
    "RAGServiceUnavailableError",
]
