# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces."""

import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from dlightrag.engine.ai.concurrency import bounded_gather
from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.rag.retrieval.rerank_fallback import rerank_with_fallback
from dlightrag.engine.rag.retrieval.visual import PreparedVisualQuery, VisualEmbeddingDomain

logger = logging.getLogger(__name__)


class WorkspaceRetriever(Protocol):
    @property
    def visual_embedding_domain(self) -> VisualEmbeddingDomain | None: ...

    async def prepare_visual_query(
        self, query_image_blocks: list[dict[str, Any]]
    ) -> PreparedVisualQuery | None: ...

    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult: ...


type FederatedVisualPreparer = Callable[
    [WorkspaceRetriever, VisualEmbeddingDomain, Sequence[Mapping[str, Any]]],
    Awaitable[PreparedVisualQuery | None],
]


class FederatedReranker(Protocol):
    """One cross-workspace rerank pass over the merged candidate pool.

    Built from the product-configured reranker settings; its presence at the
    federation call site is the enable switch, so no separate flag exists at
    this seam. Errors surface as exceptions; callers reuse the shared
    ``rerank_with_fallback`` semantics for deterministic degradation.
    """

    async def __call__(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class FederationMergePolicy:
    """Facts for one federated merge: the output budget and the fairness floor.

    ``chunk_top_k`` is the effective per-request output budget (the configured
    value or the resolved default). ``min_chunks_per_workspace`` guarantees
    each contributing workspace its top-N chunks when the budget would squeeze
    them below N; zero disables the floor.
    """

    chunk_top_k: int | None = None
    min_chunks_per_workspace: int = 7


def _resolve_cap(n_non_empty: int, policy: FederationMergePolicy) -> int | None:
    """Resolve the merged-output truncation for one federation merge.

    The output keeps at least the effective chunk budget and at least
    ``min_chunks_per_workspace`` per contributing workspace, so fan-out never
    squeezes a workspace below its quality frontier. The configured budget is
    honored directly — there is no separate hardcoded default.
    """
    if policy.chunk_top_k is None:
        return None
    if policy.min_chunks_per_workspace > 0:
        return max(policy.chunk_top_k, policy.min_chunks_per_workspace * n_non_empty)
    return policy.chunk_top_k


def _chunk_count_by_workspace(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        workspace = str(chunk.get("_workspace") or "")
        counts[workspace] = counts.get(workspace, 0) + 1
    return counts


def merge_results(
    results: list[RetrievalResult],
    workspaces: list[str],
    *,
    policy: FederationMergePolicy,
) -> RetrievalResult:
    """Merge multiple RetrievalResults via round-robin interleaving.

    Each chunk/entity/relation is tagged with ``_workspace`` to identify
    its source. Results are interleaved: ws_a[0], ws_b[0], ws_a[1], ws_b[1]...
    then truncated per the merge policy's resolved cap.
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

    n_non_empty = sum(1 for ws_chunks in per_ws_chunks if ws_chunks)
    cap = _resolve_cap(n_non_empty, policy)
    if cap is not None:
        merged_chunks = merged_chunks[:cap]

    # Re-canonicalize reference_id under the federation namespace so
    # citations like [3-2] map to one chunk across the merged answer.
    from dlightrag.engine.rag.retrieval.references import canonicalize_reference_ids

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
            "federation_cap": cap,
            "per_workspace_chunk_count": _chunk_count_by_workspace(merged_chunks),
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
    policy: FederationMergePolicy,
    max_concurrency: int = 8,
    query_image_blocks: Sequence[Mapping[str, Any]] = (),
    prepare_visual_query: FederatedVisualPreparer | None = None,
    reranker: FederatedReranker | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute bounded federated retrieval over already-authorized workspaces.

    Services are acquired first so compatible visual domains can share one
    preparation. Only typed prepared vectors cross into workspace retrieval;
    raw query-image blocks are never fanned out to workspace ``aretrieve``.

    Without a ``reranker`` the merged list is truncated by the policy's
    resolved cap (fairness floor applied). With one, the full interleaved
    pool is reranked in one pass and the policy budget selects the final
    list; failures degrade to the capped interleave.
    """
    if len(workspaces) < 2:
        raise ValueError("federated_retrieve requires at least two workspaces")

    starts = {workspace: time.monotonic() for workspace in workspaces}

    async def _acquire_workspace(workspace: str) -> WorkspaceRetriever:
        return await get_service(workspace)

    raw_services = await bounded_gather(
        [_acquire_workspace(workspace) for workspace in workspaces],
        max_concurrent=max_concurrency,
        task_name="federation-acquire",
    )

    services: dict[str, WorkspaceRetriever] = {}
    failures_by_workspace: dict[str, Exception] = {}
    for workspace, service in zip(workspaces, raw_services, strict=True):
        if isinstance(service, Exception):
            failures_by_workspace[workspace] = service
        else:
            services[workspace] = service

    prepared_by_domain: dict[VisualEmbeddingDomain, PreparedVisualQuery | None] = {}
    if query_image_blocks and prepare_visual_query is not None and services:
        representatives: dict[VisualEmbeddingDomain, WorkspaceRetriever] = {}
        for service in services.values():
            domain = getattr(service, "visual_embedding_domain", None)
            if isinstance(domain, VisualEmbeddingDomain):
                representatives.setdefault(domain, service)

        domains = list(representatives)

        async def _prepare_domain(domain: VisualEmbeddingDomain) -> PreparedVisualQuery | None:
            return await prepare_visual_query(representatives[domain], domain, query_image_blocks)

        raw_prepared = await bounded_gather(
            [_prepare_domain(domain) for domain in domains],
            max_concurrent=max_concurrency,
            task_name="federation-visual-prepare",
        )
        for domain, prepared in zip(domains, raw_prepared, strict=True):
            if isinstance(prepared, Exception):
                logger.warning("Federated visual query preparation failed", exc_info=prepared)
                prepared_by_domain[domain] = None
            elif prepared is not None and prepared.domain != domain:
                logger.warning("Federated visual query preparation returned a mismatched domain")
                prepared_by_domain[domain] = None
            else:
                prepared_by_domain[domain] = prepared

    async def _query_workspace(workspace: str) -> RetrievalResult:
        service = services[workspace]
        call_kwargs = dict(kwargs)
        domain = getattr(service, "visual_embedding_domain", None)
        if isinstance(domain, VisualEmbeddingDomain) and domain in prepared_by_domain:
            # Passing None is intentional after a failed/empty attempt: the
            # workspace must not fall back to embedding the raw blocks itself.
            call_kwargs["prepared_visual_query"] = prepared_by_domain[domain]
        result = await service.aretrieve(
            query=query,
            top_k=top_k,
            chunk_top_k=policy.chunk_top_k,
            **call_kwargs,
        )
        elapsed = time.monotonic() - starts[workspace]
        logger.info(
            "Federation workspace '%s' retrieved %d chunks in %.2fs",
            safe_log_text(workspace),
            len(result.contexts.get("chunks", [])),
            elapsed,
        )
        return result

    query_workspaces = [workspace for workspace in workspaces if workspace in services]
    raw_results = await bounded_gather(
        [_query_workspace(workspace) for workspace in query_workspaces],
        max_concurrent=max_concurrency,
        task_name="federation",
    )
    for workspace, result in zip(query_workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            failures_by_workspace[workspace] = result

    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    for workspace, result in zip(query_workspaces, raw_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                "Federated retrieval failed for workspace '%s': %s",
                safe_log_text(workspace),
                safe_log_text(result),
            )
            continue
        successful_results.append(result)
        successful_workspaces.append(workspace)

    failed_workspaces = [
        workspace for workspace in workspaces if workspace in failures_by_workspace
    ]
    if failed_workspaces:
        logger.warning(
            "Federated query partial: %d/%d workspaces failed (%s)",
            len(failed_workspaces),
            len(workspaces),
            safe_log_text(", ".join(failed_workspaces)),
        )

    if not successful_results:
        raise failures_by_workspace[failed_workspaces[0]]

    if reranker is not None and policy.chunk_top_k:
        merged = merge_results(
            successful_results,
            successful_workspaces,
            policy=replace(policy, chunk_top_k=None),
        )
        pool = merged.contexts["chunks"]
        n_non_empty = sum(1 for result in successful_results if result.contexts.get("chunks"))
        cap = _resolve_cap(n_non_empty, policy)
        capped = pool[:cap] if cap is not None else pool
        outcome = await rerank_with_fallback(
            query=query,
            chunks=pool,
            top_k=policy.chunk_top_k,
            rerank_func=reranker,
        )
        merged.contexts["chunks"] = list(outcome.chunks) if outcome.reranked else list(capped)
        merged.trace["federation_cap"] = policy.chunk_top_k if outcome.reranked else cap
        merged.trace["per_workspace_chunk_count"] = _chunk_count_by_workspace(
            merged.contexts["chunks"]
        )
        merged.trace["federated_rerank"] = {
            "pool_chunk_count": len(pool),
            "reranked": outcome.reranked,
            "output_chunk_count": len(merged.contexts["chunks"]),
        }
        if outcome.error_type:
            merged.trace["federated_rerank"]["error_type"] = outcome.error_type
    else:
        merged = merge_results(successful_results, successful_workspaces, policy=policy)

    if failed_workspaces:
        merged.trace["failed_workspaces"] = failed_workspaces
    return merged


__all__ = [
    "FederatedReranker",
    "FederatedVisualPreparer",
    "FederationMergePolicy",
    "WorkspaceRetriever",
    "federated_retrieve",
    "merge_results",
]
