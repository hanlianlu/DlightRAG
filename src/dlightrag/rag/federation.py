# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces."""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from dlightrag.ai.concurrency import bounded_gather
from dlightrag.ai.telemetry import safe_log_text
from dlightrag.rag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class WorkspaceRetriever(Protocol):
    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...


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

    # Chunk IDs are content-derived and can repeat across workspace namespaces.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for c in merged_chunks:
        cid = str(c.get("chunk_id") or "")
        identity = (str(c.get("_workspace") or ""), cid)
        if cid and identity in seen:
            continue
        if cid:
            seen.add(identity)
        deduped.append(c)
    merged_chunks = deduped

    if chunk_top_k is not None:
        merged_chunks = merged_chunks[:chunk_top_k]

    # Re-canonicalize reference_id under the federation namespace so
    # citations like [3-2] map to one chunk across the merged answer.
    from dlightrag.rag.retrieval.references import canonicalize_reference_ids

    merged_chunks = canonicalize_reference_ids(merged_chunks, federated=True)

    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    return RetrievalResult(
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
    get_service: Callable[[str], Awaitable[WorkspaceRetriever]],
    *,
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_concurrency: int = 8,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated retrieval across multiple workspaces.

    Args:
        query: The search query.
        workspaces: List of workspace names to search.
        get_service: Async callable that returns a WorkspaceRag for a workspace id.
        top_k: Per-workspace top_k for vector search.
        chunk_top_k: Final merged chunk count limit.
        max_concurrency: Maximum concurrent workspace queries (default 8).
        **kwargs: Additional kwargs passed to each WorkspaceRag.aretrieve().

    The caller owns empty/single-workspace routing and authorization; this
    function receives two or more already-authorized workspaces.
    """
    if len(workspaces) < 2:
        raise ValueError("federated_retrieve requires at least two workspaces")

    async def _query_workspace(ws: str) -> RetrievalResult:
        start = time.monotonic()
        svc = await get_service(ws)
        result = await svc.aretrieve(
            query=query,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            **kwargs,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "Federation workspace '%s' retrieved %d chunks in %.2fs",
            safe_log_text(ws),
            len(result.contexts.get("chunks", [])),
            elapsed,
        )
        return result

    coros = [_query_workspace(ws) for ws in workspaces]
    raw_results = await bounded_gather(
        coros, max_concurrent=max_concurrency, task_name="federation"
    )

    # Filter out failed workspaces (errors, timeouts) — partial result is
    # returned to the caller rather than raising.
    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    failed_workspaces: list[str] = []
    failures: list[Exception] = []
    for ws, result in zip(workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                "Federated retrieval failed for workspace '%s': %s",
                safe_log_text(ws),
                safe_log_text(result),
            )
            failed_workspaces.append(ws)
            failures.append(result)
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if failed_workspaces:
        logger.warning(
            "Federated query partial: %d/%d workspaces failed (%s)",
            len(failed_workspaces),
            len(workspaces),
            safe_log_text(", ".join(failed_workspaces)),
        )

    if not successful_results:
        raise failures[0]

    merged = merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)
    if failed_workspaces:
        merged.trace["failed_workspaces"] = failed_workspaces
    return merged
