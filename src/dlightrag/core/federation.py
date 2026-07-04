# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.utils.concurrency import bounded_gather

if TYPE_CHECKING:
    from dlightrag.core.service import RAGService

logger = logging.getLogger(__name__)

# Type alias for RBAC hook: given requested workspaces, return accessible subset
WorkspaceFilter = Callable[[list[str]], Awaitable[list[str]]]


def merge_results(
    results: list[RetrievalResult],
    workspaces: list[str],
    chunk_top_k: int | None = None,
) -> RetrievalResult:
    """Merge multiple RetrievalResults via round-robin interleaving.

    Each chunk/entity/relation is tagged with ``_workspace`` to identify
    its source. Results are interleaved: ws_a[0], ws_b[0], ws_a[1], ws_b[1]...
    then truncated to ``chunk_top_k``.
    """
    per_ws_chunks: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces, strict=True):
        chunks = result.contexts.get("chunks", [])
        tagged = []
        for chunk in chunks:
            c = dict(chunk)
            c["_workspace"] = ws
            tagged.append(c)
        per_ws_chunks.append(tagged)

    # Round-robin interleave
    merged_chunks: list[dict[str, Any]] = []
    max_len = max((len(cs) for cs in per_ws_chunks), default=0)
    for i in range(max_len):
        for ws_chunks in per_ws_chunks:
            if i < len(ws_chunks):
                merged_chunks.append(ws_chunks[i])

    # Dedup by chunk_id, keeping highest-scored occurrence (already ordered)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in merged_chunks:
        cid = c.get("chunk_id", "")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        deduped.append(c)
    merged_chunks = deduped

    if chunk_top_k is not None:
        merged_chunks = merged_chunks[:chunk_top_k]

    # Re-canonicalize reference_id under the federation namespace so
    # citations like [3-2] map to one chunk across the merged answer.
    from dlightrag.core.retrieval import canonicalize_reference_ids

    merged_chunks = canonicalize_reference_ids(merged_chunks, federated=True)

    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    return RetrievalResult(
        answer=None,
        contexts={
            "chunks": merged_chunks,
            "entities": merged_entities,
            "relationships": merged_relations,
        },
        trace={
            "federated": True,
            "workspaces": workspaces,
            "per_workspace": {
                ws: getattr(result, "trace", {})
                for ws, result in zip(workspaces, results, strict=True)
            },
            "merged_chunk_count": len(merged_chunks),
        },
    )


def _round_robin_merge_key(
    results: list[RetrievalResult],
    workspaces: list[str],
    key: str,
) -> list[dict[str, Any]]:
    """Round-robin merge a specific context key across results."""
    per_ws: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces, strict=True):
        items = result.contexts.get(key, [])
        tagged = [dict(item, _workspace=ws) for item in items]
        per_ws.append(tagged)

    merged: list[dict[str, Any]] = []
    max_len = max((len(items) for items in per_ws), default=0)
    for i in range(max_len):
        for ws_items in per_ws:
            if i < len(ws_items):
                merged.append(ws_items[i])
    return merged


async def federated_retrieve(
    query: str,
    workspaces: list[str],
    get_service: Callable[[str], Awaitable[RAGService]],
    *,
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_concurrency: int = 8,
    per_workspace_timeout: float | None = None,
    workspace_filter: WorkspaceFilter | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated retrieval across multiple workspaces.

    Args:
        query: The search query.
        workspaces: List of workspace names to search.
        get_service: Async callable that returns a RAGService for a workspace name.
        top_k: Per-workspace top_k for vector search.
        chunk_top_k: Final merged chunk count limit.
        max_concurrency: Maximum concurrent workspace queries (default 8).
        per_workspace_timeout: Optional timeout (seconds) for each workspace
            query. A workspace exceeding this bound is treated as a partial
            failure (logged + dropped from the merge), so a single slow
            backend cannot block the federation. ``None`` disables the cap.
        workspace_filter: Optional RBAC filter — given requested workspaces,
            returns the accessible subset.
        **kwargs: Additional kwargs passed to each RAGService.aretrieve().
    """
    # Apply RBAC filter if provided
    if workspace_filter is not None:
        workspaces = await workspace_filter(workspaces)

    if not workspaces:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
        )

    # Single workspace — no federation overhead
    if len(workspaces) == 1:
        svc = await get_service(workspaces[0])
        result = await svc.aretrieve(query=query, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs)
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        result.trace.setdefault("workspace", workspaces[0])
        result.trace.setdefault("federated", False)
        return result

    # Query planning belongs to the caller (RAGServiceManager). Federation is
    # only responsible for parallel workspace retrieval and merge semantics.
    # This avoids duplicate planner calls and keeps all DlightRAG-owned LLM
    # calls behind the manager's role-level concurrency budget.
    shared_plan = kwargs.pop("_plan", None)

    # Bounded parallel queries with per-workspace timing + optional timeout.
    # A slow/hanging workspace is treated as a partial failure rather than
    # blocking the merge, matching ArtRAG's federation graceful-degradation
    # contract.
    async def _query_workspace(ws: str) -> RetrievalResult:
        start = time.monotonic()
        try:
            svc = await get_service(ws)
            coro = svc.aretrieve(
                query=query,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                _plan=shared_plan,
                **kwargs,
            )
            if per_workspace_timeout is not None:
                result = await asyncio.wait_for(coro, timeout=per_workspace_timeout)
            else:
                result = await coro
            elapsed = time.monotonic() - start
            logger.info(
                "Federation workspace '%s' retrieved %d chunks in %.2fs",
                ws,
                len(result.contexts.get("chunks", [])),
                elapsed,
            )
            return result
        except TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "Federation workspace '%s' timed out after %.2fs (cap=%.2fs)",
                ws,
                elapsed,
                per_workspace_timeout or 0.0,
            )
            raise

    coros = [_query_workspace(ws) for ws in workspaces]
    raw_results = await bounded_gather(
        coros, max_concurrent=max_concurrency, task_name="federation"
    )

    # Filter out failed workspaces (errors, timeouts) — partial result is
    # returned to the caller rather than raising.
    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    failed_workspaces: list[str] = []
    for ws, result in zip(workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            logger.warning("Federated retrieval failed for workspace '%s': %s", ws, result)
            failed_workspaces.append(ws)
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if failed_workspaces:
        logger.warning(
            "Federated query partial: %d/%d workspaces failed (%s)",
            len(failed_workspaces),
            len(workspaces),
            ", ".join(failed_workspaces),
        )

    if not successful_results:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)
