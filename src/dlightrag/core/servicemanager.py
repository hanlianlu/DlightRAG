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


@dataclass(frozen=True)
class _AnswerLimits:
    candidate_top_k: int
    context_top_k: int


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


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
        self._workspace_registry: Any = None
        self._checkpoint: Any = None

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
        default_ws = manager._config.workspace
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

    async def _initialize_workspace_registry(self) -> None:
        """Initialize the durable workspace registry."""
        from dlightrag.storage.workspaces import PGWorkspaceRegistry

        self._workspace_registry = PGWorkspaceRegistry(target=self._config.pg_target_for_runtime())
        try:
            await self._workspace_registry.initialize(read_only=self._config.is_query_role is True)
            if self._config.is_query_role is not True:
                await self._workspace_registry.upsert(
                    workspace=self._config.workspace,
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
        timeout = self._config.ingest_timeout
        try:
            if timeout is None:
                svc = await self._get_service(workspace)
                await svc.aregister_workspace()
                result = await svc.aingest(source_type=source_type, **kwargs)
                replay_lsn = await self._wait_after_write()
                if replay_lsn is not None and isinstance(result, dict):
                    result["replica_replay_lsn"] = replay_lsn
                return result
            else:
                async with asyncio.timeout(timeout):
                    svc = await self._get_service(workspace)
                    await svc.aregister_workspace()
                    result = await svc.aingest(source_type=source_type, **kwargs)
                    replay_lsn = await self._wait_after_write()
                    if replay_lsn is not None and isinstance(result, dict):
                        result["replica_replay_lsn"] = replay_lsn
                    return result
        except TimeoutError as e:
            raise RAGServiceUnavailableError(detail=f"Request timed out after {timeout}s") from e

    async def acreate_workspace(self, workspace: str, *, display_name: str | None = None) -> None:
        """Initialize a workspace through the public manager API."""
        self._ensure_writable("create workspace")
        svc = await self._get_service(workspace)
        await svc.aregister_workspace(display_name=display_name)
        await self._wait_after_write()

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
                from dlightrag.core.reset import areset_orphaned_workspace

                result = await areset_orphaned_workspace(
                    workspace,
                    keep_files=keep_files,
                    dry_run=dry_run,
                    input_dir=str(self._config.input_dir_path),
                )
                return {
                    "workspaces": {workspace: result},
                    "total_errors": len(result.get("errors", [])),
                }
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
                ttl_seconds=cfg.session_ttl_seconds,
            )
        return self._session_images

    def _get_checkpoint(self):
        """Lazy-create conversation checkpoint store."""
        if self._checkpoint is None:
            from dlightrag.core.checkpoint import ConversationCheckpoint

            db_path = self._config.working_dir_path / "checkpoints.db"
            self._checkpoint = ConversationCheckpoint(db_path)
        return self._checkpoint

    async def save_turn_checkpoint(
        self,
        session_id: str,
        query: str,
        answer: str,
        contexts: dict[str, Any],
        cited_chunk_ids: list[str],
    ) -> None:
        """Save a complete turn (query + answer + retrieval anchors) to checkpoint."""
        if not session_id:
            return
        try:
            cp = self._get_checkpoint()
            await cp.ensure_session(session_id)
            turn_number = await cp.next_turn_number(session_id)
            await cp.save_turn(
                session_id,
                turn_number,
                role="user",
                content=query,
            )
            await cp.save_turn(
                session_id,
                turn_number,
                role="assistant",
                content=answer,
                cited_chunk_ids=cited_chunk_ids,
            )
            chunks = contexts.get("chunks", [])
            if chunks:
                await cp.save_anchors(session_id, turn_number, chunks)
                await cp.mark_cited(session_id, turn_number, cited_chunk_ids)
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
        context_top_k: int | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Generate a streaming answer from already-retrieved contexts."""
        engine = self._get_answer_engine()
        return await engine.generate_stream(
            query,
            contexts,
            query_images=query_images,
            context_top_k=context_top_k,
        )

    def _resolve_answer_limits(self, kwargs: dict[str, Any]) -> _AnswerLimits:
        """Resolve answer-specific retrieval and final-prompt chunk limits.

        ``top_k``/``chunk_top_k`` remain retrieval controls. Answer generation
        defaults to over-fetching candidates, then ``AnswerContextPacker``
        deterministically trims to the final prompt budget.
        """
        answer_cfg = self._config.answer
        answer_candidate_top_k = _positive_int_or_none(kwargs.pop("answer_candidate_top_k", None))
        answer_context_top_k = _positive_int_or_none(kwargs.pop("answer_context_top_k", None))
        requested_top_k = _positive_int_or_none(kwargs.get("top_k"))
        requested_chunk_top_k = _positive_int_or_none(kwargs.get("chunk_top_k"))

        candidate_top_k = (
            answer_candidate_top_k or requested_chunk_top_k or answer_cfg.candidate_top_k
        )
        context_top_k = answer_context_top_k or answer_cfg.context_top_k
        kwargs["top_k"] = requested_top_k or self._config.top_k
        kwargs["chunk_top_k"] = candidate_top_k
        return _AnswerLimits(candidate_top_k=candidate_top_k, context_top_k=context_top_k)

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
                    limits = self._resolve_answer_limits(kwargs)
                    retrieval = await self.aretrieve(
                        query,
                        plan=retrieval_plan,
                        workspaces=ws_list,
                        **kwargs,
                    )
                    retrieval.trace["query_image_description_count"] = len(prepared.descriptions)
                    engine = self._get_answer_engine()
                    result = await engine.generate(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
                        context_top_k=limits.context_top_k,
                    )
                    retrieval.trace["answer_candidate_top_k"] = limits.candidate_top_k
                    retrieval.trace["answer_context_top_k"] = limits.context_top_k
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
                    limits = self._resolve_answer_limits(kwargs)
                    retrieval = await self.aretrieve(
                        query,
                        plan=retrieval_plan,
                        workspaces=ws_list,
                        **kwargs,
                    )
                    retrieval.trace["query_image_description_count"] = len(prepared.descriptions)
                    contexts, stream = await self.agenerate_stream_from_contexts(
                        plan.standalone_query,
                        retrieval.contexts,
                        query_images=prepared.answer_images or None,
                        context_top_k=limits.context_top_k,
                    )
                    retrieval.trace["answer_candidate_top_k"] = limits.candidate_top_k
                    retrieval.trace["answer_context_top_k"] = limits.context_top_k
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
                "workspace": self._config.workspace,
                "display_name": self._config.workspace,
                "embedding_model": self._config.embedding.model,
                "created_at": None,
                "updated_at": None,
            }
        ]

    @staticmethod
    def _serialize_workspace_record(row: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-safe workspace record."""
        return {
            "workspace": str(row.get("workspace") or ""),
            "display_name": str(row.get("display_name") or row.get("workspace") or ""),
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


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError("answer top-k limits must be positive integers")
    return result


__all__ = [
    "RAGServiceManager",
    "RAGServiceUnavailableError",
]
