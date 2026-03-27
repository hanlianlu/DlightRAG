# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces.

Orchestrates parallel queries to multiple RAGService instances (one per
workspace) and merges results. Supports two merge strategies:
- Pure round-robin (fair, no quality signals needed)
- Quality-weighted seat allocation + round-robin (when quality scores available)

Backported L1-weighted merge from ArtRAG's federation module.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.utils.concurrency import bounded_gather

logger = logging.getLogger(__name__)

# Type alias for RBAC hook: given requested workspaces, return accessible subset
WorkspaceFilter = Callable[[list[str]], Awaitable[list[str]]]

DEFAULT_QUALITY_SCORE = 0.5


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

    if chunk_top_k is not None:
        merged_chunks = merged_chunks[:chunk_top_k]

    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    return RetrievalResult(
        answer=None,
        contexts={
            "chunks": merged_chunks,
            "entities": merged_entities,
            "relationships": merged_relations,
        },
    )


def merge_results_weighted(
    results: list[RetrievalResult],
    workspaces: list[str],
    *,
    quality_scores: dict[str, float | None],
    chunk_top_k: int | None = None,
) -> RetrievalResult:
    """Merge results with quality-weighted seat allocation + round-robin.

    Each workspace gets seats proportional to its quality score.
    Higher quality workspaces get more representation in the final result.
    Minimum 1 seat per workspace guarantees representation.
    """
    if not results:
        return RetrievalResult(
            answer=None, contexts={"chunks": [], "entities": [], "relationships": []}
        )

    top_k = chunk_top_k or sum(len(r.contexts.get("chunks", [])) for r in results)

    # Resolve quality scores (None -> DEFAULT_QUALITY_SCORE)
    resolved: dict[str, float] = {}
    for ws in workspaces:
        score = quality_scores.get(ws)
        resolved[ws] = score if score is not None else DEFAULT_QUALITY_SCORE

    # When num_workspaces > top_k, drop lowest quality workspaces
    if len(workspaces) > top_k:
        sorted_ws = sorted(workspaces, key=lambda w: resolved[w], reverse=True)
        kept = set(sorted_ws[:top_k])
        paired = [(r, w) for r, w in zip(results, workspaces, strict=False) if w in kept]
        results = [r for r, _ in paired]
        workspaces = [w for _, w in paired]

    # Seat allocation proportional to quality
    total_score = sum(resolved[w] for w in workspaces)
    if total_score <= 0:
        total_score = float(len(workspaces))

    seats: dict[str, int] = {}
    for ws in workspaces:
        raw = round(resolved[ws] / total_score * top_k)
        seats[ws] = max(1, raw)

    # Adjust total seats to match top_k
    total_seats = sum(seats.values())
    if total_seats > top_k:
        sorted_by_seats = sorted(seats, key=lambda w: seats[w], reverse=True)
        diff = total_seats - top_k
        for ws in sorted_by_seats:
            reduce = min(diff, seats[ws] - 1)
            seats[ws] -= reduce
            diff -= reduce
            if diff <= 0:
                break
    elif total_seats < top_k:
        best = max(workspaces, key=lambda w: resolved[w])
        seats[best] += top_k - total_seats

    # Tag and collect per-workspace chunks (top N per workspace)
    ws_chunks: dict[str, list[dict[str, Any]]] = {}
    for result, ws in zip(results, workspaces, strict=True):
        chunks = result.contexts.get("chunks", [])
        tagged = [dict(c, _workspace=ws) for c in chunks]
        ws_chunks[ws] = tagged[: seats[ws]]

    # Round-robin interleave (sorted by quality score desc)
    sorted_ws = sorted(workspaces, key=lambda w: resolved[w], reverse=True)
    merged: list[dict[str, Any]] = []
    rnd = 0
    while len(merged) < top_k:
        added = False
        for ws in sorted_ws:
            if rnd < len(ws_chunks.get(ws, [])):
                merged.append(ws_chunks[ws][rnd])
                added = True
                if len(merged) >= top_k:
                    break
        if not added:
            break
        rnd += 1

    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    return RetrievalResult(
        answer=None,
        contexts={
            "chunks": merged,
            "entities": merged_entities,
            "relationships": merged_relations,
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
    get_service: Callable[[str], Awaitable[Any]],
    *,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_concurrency: int = 8,
    workspace_filter: WorkspaceFilter | None = None,
    get_quality_score: Callable[[str], Awaitable[float | None]] | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated retrieval across multiple workspaces.

    Args:
        query: The search query.
        workspaces: List of workspace names to search.
        get_service: Async callable that returns a RAGService for a workspace name.
        mode: LightRAG query mode.
        top_k: Per-workspace top_k for vector search.
        chunk_top_k: Final merged chunk count limit.
        max_concurrency: Maximum concurrent workspace queries (default 8).
        workspace_filter: Optional RBAC filter — given requested workspaces,
            returns the accessible subset.
        get_quality_score: Optional async callable returning a quality score
            (0-1) for a workspace. When provided, uses weighted seat allocation
            instead of pure round-robin. None = pure round-robin.
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
        result = await svc.aretrieve(
            query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
        )
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        return result

    # Pre-compute query analysis ONCE to avoid redundant LLM calls per workspace
    shared_plan = None
    first_svc = await get_service(workspaces[0])
    if first_svc._metadata_index is not None:
        try:
            from dlightrag.core.retrieval.query_analyzer import QueryAnalyzer

            lr = first_svc._lightrag or (
                getattr(first_svc.rag, "lightrag", None) if first_svc.rag else None
            )
            llm_func = getattr(lr, "llm_model_func", None) if lr else None
            schema = getattr(first_svc, "_table_schema", None)
            analyzer = QueryAnalyzer(llm_func=llm_func, schema=schema)
            shared_plan = await analyzer.analyze(query, explicit_filters=kwargs.get("filters"))
            logger.info(
                "Federation: shared plan computed, query=%r",
                (shared_plan.query or "")[:60],
            )
        except Exception as exc:
            logger.warning("Federation: shared plan computation failed (non-fatal): %s", exc)

    # Bounded parallel queries
    async def _query_workspace(ws: str) -> RetrievalResult:
        svc = await get_service(ws)
        return await svc.aretrieve(
            query=query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            _plan=shared_plan,
            **kwargs,
        )

    coros = [_query_workspace(ws) for ws in workspaces]
    raw_results = await bounded_gather(
        coros, max_concurrent=max_concurrency, task_name="federation"
    )

    # Filter out failed workspaces
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

    # Weighted merge when quality scores available, otherwise pure round-robin
    if get_quality_score is not None:
        scores: dict[str, float | None] = {}
        for ws in successful_workspaces:
            try:
                scores[ws] = await get_quality_score(ws)
            except Exception:
                scores[ws] = None
        return merge_results_weighted(
            successful_results,
            successful_workspaces,
            quality_scores=scores,
            chunk_top_k=chunk_top_k,
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)
