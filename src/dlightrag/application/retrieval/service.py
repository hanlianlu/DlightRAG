# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Inline retrieval use case over authorized canonical workspaces."""

import asyncio
import copy
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalContexts, RetrievalResult
from dlightrag.engine.rag.retrieval.federation import federated_retrieve
from dlightrag.engine.rag.retrieval.planner import RetrievalPlan, RetrievalPlanner
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.ports import (
    CorpusUnavailableError as _EngineCorpusUnavailableError,
)

logger = logging.getLogger(__name__)

type SchemaLookup = Callable[[Sequence[str]], Awaitable[dict[str, Any]]]
type QueryImagePreparer = Callable[[Sequence[Mapping[str, Any]]], Awaitable[list[str]]]
type RetrievalProjection = Callable[[RetrievalResult, "RetrieveProjection"], "ProjectedRetrieval"]


class RetrievalTimeoutError(RuntimeError):
    """One inline retrieval did not finish within its request budget."""


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    default_top_k: int
    default_chunk_top_k: int
    timeout_seconds: float
    query_image_limit: int
    workspace_fanout_concurrency: int = 8


@dataclass(frozen=True, slots=True)
class RetrieveProjection:
    """Already-authorized reader scope for client-safe projection."""

    downloadable_workspaces: frozenset[str] | None
    visual_workspaces: frozenset[str] | None
    include_download_links: bool = False
    image_url_prefix: str | None = "/images"


@dataclass(frozen=True, slots=True)
class RetrieveRequest:
    """One trusted inline retrieval request with concrete canonical workspaces."""

    query: str
    workspaces: tuple[str, ...]
    projection: RetrieveProjection
    top_k: int | None = None
    chunk_top_k: int | None = None
    bm25_query: str | None = None
    filters: MetadataFilter | None = None
    query_images: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectedRetrieval:
    contexts: RetrievalContexts
    sources: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RetrieveResponse:
    contexts: RetrievalContexts
    sources: tuple[Mapping[str, Any], ...]
    trace: Mapping[str, Any]
    image_descriptions: tuple[str, ...]


class PlannerProvider(Protocol):
    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner: ...

    async def aclose(self) -> None: ...


class RetrievalService:
    """Plan, retrieve, and project one caller-awaited result."""

    def __init__(
        self,
        *,
        pool: WorkspacePool,
        planners: PlannerProvider,
        schema_lookup: SchemaLookup,
        image_preparer: QueryImagePreparer,
        projector: RetrievalProjection,
        settings: RetrievalSettings,
        telemetry: Telemetry,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pool = pool
        self._planners = planners
        self._schema_lookup = schema_lookup
        self._image_preparer = image_preparer
        self._projector = projector
        self._settings = settings
        self._telemetry = telemetry
        self._clock = clock
        self._schema_cache: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
        self._schema_refreshes: dict[tuple[str, ...], asyncio.Task[dict[str, Any]]] = {}
        self._warmups: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def _acquire(self, workspace: str) -> Any:
        """Acquire one workspace and translate Engine availability errors."""
        try:
            return await self._pool.acquire(workspace)
        except _EngineCorpusUnavailableError as exc:
            raise CorpusUnavailableError(str(exc)) from exc

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner:
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        return self._planners.planner_for(model_profile)

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]:
        key = tuple(sorted(workspaces))
        now = self._clock()
        cached = self._schema_cache.get(key)
        if cached is not None and now - cached[0] < 300.0:
            return copy.deepcopy(cached[1])
        refresh = self._schema_refreshes.get(key)
        if refresh is None:
            refresh = asyncio.create_task(self._refresh_schema(key))
            self._schema_refreshes[key] = refresh
            refresh.add_done_callback(
                lambda task, refresh_key=key: self._finish_schema_refresh(refresh_key, task)
            )
        try:
            schema = await asyncio.shield(refresh)
        except Exception:
            logger.debug("Schema lookup failed for workspaces %s", key, exc_info=True)
            return copy.deepcopy(cached[1]) if cached is not None else {}
        if key not in self._schema_cache and len(self._schema_cache) >= 128:
            oldest = min(self._schema_cache, key=lambda item: self._schema_cache[item][0])
            self._schema_cache.pop(oldest, None)
        cached_schema = copy.deepcopy(schema)
        self._schema_cache[key] = (self._clock(), cached_schema)
        return copy.deepcopy(cached_schema)

    async def _refresh_schema(self, key: tuple[str, ...]) -> dict[str, Any]:
        return await self._schema_lookup(key)

    def _finish_schema_refresh(
        self,
        key: tuple[str, ...],
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        if self._schema_refreshes.get(key) is task:
            self._schema_refreshes.pop(key, None)
        if not task.cancelled() and task.exception() is not None:
            logger.debug("Schema refresh failed for workspaces %s", key, exc_info=task.exception())

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        if not request.workspaces:
            raise ValueError("At least one canonical workspace is required")
        if len(request.query_images) > self._settings.query_image_limit:
            raise ValueError(
                f"at most {self._settings.query_image_limit} current images are allowed"
            )

        self.warm(request.workspaces)
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                return await self._retrieve(request)
        except TimeoutError as exc:
            raise RetrievalTimeoutError(
                f"Retrieval timed out after {self._settings.timeout_seconds:g}s"
            ) from exc

    def warm(self, workspaces: Sequence[str]) -> None:
        """Start pool-owned initialization for an imminent request."""
        if self._closed:
            return
        warmup = asyncio.create_task(self._pool.warm(workspaces))
        self._warmups.add(warmup)
        warmup.add_done_callback(self._observe_warmup)

    async def _retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        images = tuple(dict(image) for image in request.query_images)
        descriptions = await self._image_preparer(images) if images else []
        result = await self.retrieve_result(
            request.query,
            workspaces=request.workspaces,
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            bm25_query=request.bm25_query,
            filters=request.filters,
            query_images=images,
            image_descriptions=descriptions,
        )
        projected = self._projector(result, request.projection)
        return RetrieveResponse(
            contexts=projected.contexts,
            sources=projected.sources,
            trace=dict(result.trace),
            image_descriptions=tuple(descriptions),
        )

    async def retrieve_result(
        self,
        query: str,
        *,
        workspaces: Sequence[str],
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: Sequence[Mapping[str, Any]] = (),
        image_descriptions: Sequence[str] = (),
        preserve_query: bool | None = None,
        model_profile: ModelProfile | None = None,
        planner: RetrievalPlanner | None = None,
    ) -> RetrievalResult:
        """Plan and execute raw retrieval without inline timeout or reader projection."""
        if self._closed:
            raise CorpusUnavailableError("Retrieval service is closed")
        if not workspaces:
            raise ValueError("At least one canonical workspace is required")
        active_planner = planner or self.planner_for(model_profile)
        async with self._telemetry.observe(
            "retrieval_planning",
            as_type="chain",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces),
                "history_messages": len(conversation_history or ()),
            },
        ) as planning_observation:
            schema = await self.schema_for(workspaces)
            plan: RetrievalPlan = await active_planner.plan(
                query,
                conversation_history=conversation_history,
                schema=schema,
                current_image_descriptions=list(image_descriptions) or None,
                preserve_query=preserve_query,
            )
            planning_observation.update(
                output={
                    "standalone_query": plan.standalone_query,
                    "has_metadata_filter": plan.metadata_filter is not None,
                    "planning_outcome": plan.outcome,
                }
            )

        effective_top_k = _positive_int_or_none(top_k) or self._settings.default_top_k
        effective_chunk_top_k = (
            _positive_int_or_none(chunk_top_k) or self._settings.default_chunk_top_k
        )
        kwargs: dict[str, Any] = {
            "top_k": effective_top_k,
            "chunk_top_k": effective_chunk_top_k,
        }
        if query_images:
            kwargs["query_image_blocks"] = [dict(image) for image in query_images]
        effective_filters = filters if filters is not None else plan.metadata_filter
        if effective_filters is not None:
            kwargs["filters"] = effective_filters
        filter_source = "explicit" if filters is not None else plan.metadata_filter_source
        if filter_source is not None:
            kwargs["filter_source"] = filter_source
        effective_bm25_query = (bm25_query or "").strip() or plan.bm25_query
        if effective_bm25_query is not None:
            kwargs["bm25_query"] = effective_bm25_query

        async with self._telemetry.observe(
            "retrieve",
            as_type="retriever",
            input={"query": query},
            metadata={
                "workspaces": list(workspaces),
                "top_k": effective_top_k,
                "chunk_top_k": effective_chunk_top_k,
                "has_filters": effective_filters is not None,
            },
        ) as observation:
            if len(workspaces) == 1:
                runtime = await self._acquire(workspaces[0])
                result = await runtime.aretrieve(plan.standalone_query, **kwargs)
            else:
                result = await federated_retrieve(
                    plan.standalone_query,
                    list(workspaces),
                    self._acquire,
                    max_concurrency=self._settings.workspace_fanout_concurrency,
                    **kwargs,
                )
            result.image_descriptions = list(image_descriptions)
            result.trace["query_image_description_count"] = len(image_descriptions)
            observation.update(
                output={
                    **_context_output(result.contexts),
                    "standalone_query": plan.standalone_query,
                    "query_image_description_count": len(image_descriptions),
                }
            )
            return result

    async def aclose(self) -> None:
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(self._close_resources())
            self._close_task = close_task
        await await_shared_cleanup(close_task)

    async def _close_resources(self) -> None:
        for warmup in self._warmups:
            warmup.cancel()
        for refresh in self._schema_refreshes.values():
            refresh.cancel()
        if self._warmups:
            await asyncio.gather(*self._warmups, return_exceptions=True)
        if self._schema_refreshes:
            await asyncio.gather(*self._schema_refreshes.values(), return_exceptions=True)
            self._schema_refreshes.clear()
        await self._planners.aclose()

    def _observe_warmup(self, task: asyncio.Task[None]) -> None:
        self._warmups.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.debug("Workspace warm-up failed", exc_info=error)


def _positive_int_or_none(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


def _context_output(contexts: RetrievalContexts) -> dict[str, int]:
    return {
        "chunk_count": len(contexts.get("chunks", [])),
        "entity_count": len(contexts.get("entities", [])),
        "relationship_count": len(contexts.get("relationships", [])),
    }


__all__ = [
    "CorpusUnavailableError",
    "ProjectedRetrieval",
    "QueryImagePreparer",
    "RetrieveProjection",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievalService",
    "RetrievalSettings",
    "RetrievalTimeoutError",
    "SchemaLookup",
]
