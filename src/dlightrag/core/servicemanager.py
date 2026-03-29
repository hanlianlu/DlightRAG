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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.federation import federated_retrieve
from dlightrag.core.query_planner import QueryPlan, QueryPlanner
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)

_MAX_RETRY_INTERVAL: float = 300.0


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
        try:
            async with asyncio.timeout(self._config.request_timeout):
                svc = await self._get_service(workspace)
                return await svc.aingest(source_type=source_type, **kwargs)
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    async def list_ingested_files(self, workspace: str) -> list[dict[str, Any]]:
        """List ingested files in a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.alist_ingested_files()

    async def delete_files(self, workspace: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Delete files from a specific workspace."""
        svc = await self._get_service(workspace)
        return await svc.adelete_files(**kwargs)

    async def aget_metadata(self, workspace: str, doc_id: str) -> dict[str, Any]:
        """Get document metadata by ID."""
        svc = await self._get_service(workspace)
        return await svc.aget_metadata(doc_id)

    async def aupdate_metadata(self, workspace: str, doc_id: str, data: dict[str, Any]) -> None:
        """Update (merge) document metadata."""
        svc = await self._get_service(workspace)
        await svc.aupdate_metadata(doc_id, data)

    async def asearch_metadata(self, workspace: str, filters: dict[str, Any]) -> list[str]:
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

        return {"workspaces": results, "total_errors": total_errors}

    def _get_answer_engine(self) -> AnswerEngine:
        """Lazy-create AnswerEngine from global config."""
        if self._answer_engine is None:
            from dlightrag.models.llm import get_chat_model_func

            self._answer_engine = AnswerEngine(
                model_func=get_chat_model_func(self._config),
            )
        return self._answer_engine

    def _get_query_planner(self) -> QueryPlanner:
        """Lazy-create QueryPlanner with schema TTL refresh."""
        if self._query_planner is None:
            from dlightrag.models.llm import get_chat_model_func

            self._query_planner = QueryPlanner(
                llm_func=get_chat_model_func(self._config),
                schema_provider=self._get_schema,
                schema_ttl=300.0,
            )
        return self._query_planner

    async def _get_schema(self) -> dict[str, Any]:
        """Fetch metadata schema from first available workspace."""
        workspaces = await self.list_workspaces()
        if workspaces:
            svc = await self._get_service(workspaces[0])
            if svc._metadata_index is not None:
                return await svc._metadata_index.get_table_schema()
        return {}

    def get_llm_func(self):
        """Return the global chat model function (for web UI etc.)."""
        from dlightrag.models.llm import get_chat_model_func

        return get_chat_model_func(self._config)

    # --- Read operations (single or federated) ---

    async def aretrieve(
        self,
        query: str,
        *,
        workspace: str | None = None,
        workspaces: list[str] | None = None,
        plan: QueryPlan | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve from one or more workspaces (federated if multiple).

        Args:
            plan: Pre-computed QueryPlan from QueryPlanner. When provided,
                uses plan.standalone_query for search and plan.metadata_filter
                for structured filtering.
        """
        effective_query = plan.standalone_query if plan else query
        if plan and plan.metadata_filter is not None:
            kwargs.setdefault("filters", plan.metadata_filter)
        ws_list = workspaces or [workspace or self._config.workspace]
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline("retrieve", workspaces=ws_list, query=effective_query):
                    if len(ws_list) == 1:
                        svc = await self._get_service(ws_list[0])
                        return await svc.aretrieve(effective_query, **kwargs)
                    return await federated_retrieve(
                        effective_query, ws_list, self._get_service, **kwargs
                    )
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
        **kwargs: Any,
    ) -> RetrievalResult:
        """Answer from one or more workspaces: plan -> retrieve -> generate."""
        ws_list = workspaces or [workspace or self._config.workspace]
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline("answer_pipeline", workspaces=ws_list, query=query):
                    planner = self._get_query_planner()
                    plan = await planner.plan(
                        query,
                        conversation_history=conversation_history,
                        max_turns=self._config.max_conversation_turns,
                        max_tokens=self._config.max_conversation_tokens,
                    )

                    retrieval = await self.aretrieve(query, plan=plan, workspaces=ws_list, **kwargs)
                    engine = self._get_answer_engine()
                    return await engine.generate(plan.standalone_query, retrieval.contexts)
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
        **kwargs: Any,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer from one or more workspaces: plan -> retrieve -> stream."""
        ws_list = workspaces or [workspace or self._config.workspace]
        from dlightrag.observability import trace_pipeline

        try:
            async with asyncio.timeout(self._config.request_timeout):
                async with trace_pipeline(
                    "answer_stream_pipeline", workspaces=ws_list, query=query
                ):
                    planner = self._get_query_planner()
                    plan = await planner.plan(
                        query,
                        conversation_history=conversation_history,
                        max_turns=self._config.max_conversation_turns,
                        max_tokens=self._config.max_conversation_tokens,
                    )
                    logger.info(
                        "Query plan: original=%r, standalone=%r",
                        plan.original_query[:60],
                        plan.standalone_query[:60],
                    )

                    retrieval = await self.aretrieve(query, plan=plan, workspaces=ws_list, **kwargs)
                    engine = self._get_answer_engine()
                    return await engine.generate_stream(plan.standalone_query, retrieval.contexts)
        except TimeoutError as e:
            raise RAGServiceUnavailableError(
                detail=f"Request timed out after {self._config.request_timeout}s"
            ) from e

    # --- Management ---

    async def list_workspaces(self, *, compatible_only: bool = False) -> list[str]:
        """Discover available workspaces.

        Args:
            compatible_only: If True, filter to workspaces whose embedding
                model matches the current global config (PG-only).
        """
        all_ws = await self._list_all_workspaces()
        if not compatible_only:
            return all_ws
        return await self._filter_compatible(all_ws)

    async def _list_all_workspaces(self) -> list[str]:
        """Internal: discover all workspaces regardless of compatibility."""
        config = self._config

        if config.kv_storage.startswith("PG"):
            try:
                import asyncpg

                conn = await asyncpg.connect(
                    host=config.postgres_host,
                    port=config.postgres_port,
                    user=config.postgres_user,
                    password=config.postgres_password,
                    database=config.postgres_database,
                )
                try:
                    rows = await conn.fetch(
                        "SELECT DISTINCT workspace FROM dlightrag_file_hashes ORDER BY workspace"
                    )
                    workspaces = [row["workspace"] for row in rows]
                    return workspaces if workspaces else [config.workspace]
                finally:
                    await conn.close()
            except Exception as exc:
                logger.warning("Failed to list workspaces from PG: %s", exc)
                return [config.workspace]

        _fs_backends = {
            "JsonKVStorage",
            "JsonDocStatusStorage",
            "NanoVectorDBStorage",
            "NetworkXStorage",
            "FaissVectorDBStorage",
        }
        if config.kv_storage in _fs_backends or config.vector_storage in _fs_backends:
            return self._discover_filesystem_workspaces()

        cached = list(self._services.keys())
        if config.workspace not in cached:
            cached.append(config.workspace)
        return sorted(cached)

    async def _filter_compatible(self, workspaces: list[str]) -> list[str]:
        """Filter workspaces to those using the same embedding model."""
        if not self._config.kv_storage.startswith("PG"):
            return workspaces  # non-PG: no metadata, all compatible
        compatible = []
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=self._config.postgres_host,
                port=self._config.postgres_port,
                user=self._config.postgres_user,
                password=self._config.postgres_password,
                database=self._config.postgres_database,
            )
            try:
                for ws in workspaces:
                    row = await conn.fetchrow(
                        "SELECT embedding_model FROM dlightrag_workspace_meta WHERE workspace = $1",
                        ws,
                    )
                    if row is None or row["embedding_model"] == self._config.embedding.model:
                        compatible.append(ws)
            finally:
                await conn.close()
        except Exception:
            logger.debug("workspace compatibility filter failed, returning all", exc_info=True)
            return workspaces
        return compatible

    def _discover_filesystem_workspaces(self) -> list[str]:
        """Scan working_dir for subdirectories containing LightRAG data files."""
        working_dir = self._config.working_dir_path
        if not working_dir.exists():
            return [self._config.workspace]
        workspaces = []
        for entry in working_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if (
                any(entry.glob("kv_store_*.json"))
                or any(entry.glob("vdb_*.json"))
                or any(entry.glob("graph_*.graphml"))
                or any(entry.glob("file_content_hashes.json"))
            ):
                workspaces.append(entry.name)
        return sorted(workspaces) if workspaces else [self._config.workspace]

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


__all__ = [
    "RAGServiceManager",
    "RAGServiceUnavailableError",
]
