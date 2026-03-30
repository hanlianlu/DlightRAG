# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multimodal rerank strategies for DlightRAG.

All strategies accept chunks with text content + optional image_data
and return chunks with rerank_score added, sorted descending.

Unified interface:
    async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]

For LightRAG caption-mode compatibility, use ``build_lightrag_rerank_adapter``
to wrap the multimodal reranker into LightRAG's (query, documents) signature.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx

from dlightrag.utils import image_data_uri

if TYPE_CHECKING:
    from dlightrag.config import RerankConfig

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 7

_LISTWISE_PROMPT = """\
You are a relevance judge. Given a query and {n} items (each may be an image, text, or both), \
score each item's relevance to the query.

Respond with ONLY a JSON array of {n} scores in order: [<float>, <float>, ...]
Each score is 0.00 (completely irrelevant) to 1.00 (perfectly relevant).

Query: {query}"""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_listwise_scores(text: str, expected: int) -> list[float]:
    """Parse a JSON array of scores from VLM response."""
    text = text.strip()
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_clamp(float(s)) for s in data[:expected]]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: extract all floats from text
    matches = re.findall(r"\d+\.\d+", text)
    if matches:
        return [_clamp(float(m)) for m in matches[:expected]]
    logger.warning("Could not parse listwise scores from: %s", text[:200])
    return [0.0] * expected


# ── VLM listwise reranker ────────────────────────────────────────


async def _chat_llm_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    vlm_func: Callable[..., Any],
    max_concurrency: int = 4,
    score_threshold: float = 0.5,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """VLM listwise reranker. Scores chunks in batches for relative comparison."""
    if not chunks:
        return []

    all_results: list[tuple[dict[str, Any], float]] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        prompt = _LISTWISE_PROMPT.format(query=query, n=len(batch))

        # Build multimodal content: interleave items with labels
        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})

        for i, chunk in enumerate(batch):
            content.append({"type": "text", "text": f"\n--- Item {i + 1} ---"})
            img = chunk.get("image_data")
            if img:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri(img)},
                    }
                )
            if chunk.get("content"):
                text_preview = chunk["content"][:500]
                content.append({"type": "text", "text": text_preview})

        messages = [{"role": "user", "content": content}]
        try:
            resp = await vlm_func(messages=messages)
            scores = _parse_listwise_scores(resp, len(batch))
            logger.info(
                "Rerank batch [%d..%d] (%d items): scores=%s",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
                [f"{s:.3f}" for s in scores],
            )
        except Exception:
            logger.warning("VLM listwise rerank failed for batch %d", batch_start, exc_info=True)
            scores = [0.0] * len(batch)

        for chunk, score in zip(batch, scores, strict=False):
            all_results.append((chunk, score))

    # Apply threshold
    scored = []
    for chunk, score in all_results:
        if score >= score_threshold:
            out = chunk.copy()
            out["rerank_score"] = score
            scored.append(out)

    # Fallback: if threshold filtered everything, keep top-5 by score
    if not scored and all_results:
        by_score = sorted(all_results, key=lambda x: x[1], reverse=True)
        for chunk, score in by_score[:5]:
            out = chunk.copy()
            out["rerank_score"] = score
            scored.append(out)
        logger.info(
            "Rerank: all below threshold %.2f, kept top-%d (best=%.3f)",
            score_threshold,
            len(scored),
            scored[0]["rerank_score"] if scored else 0,
        )

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


# ── HTTP-based rerankers (Jina, Aliyun, local) ──────────────────


async def _http_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    url: str,
    model: str,
    api_key: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Shared HTTP reranker for Jina, Aliyun, and self-hosted endpoints."""
    if not chunks:
        return []

    documents: list[dict[str, str]] = []
    for c in chunks:
        img = c.get("image_data")
        documents.append({"image": image_data_uri(img)} if img else {"text": c.get("content", "")})

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    scored = []
    all_results = data.get("results", [])
    for r in all_results:
        idx = r["index"]
        score = r["relevance_score"]
        if score >= score_threshold:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            scored.append(chunk)

    # Fallback: if threshold filtered everything, keep top-5 by score
    if not scored and all_results:
        by_score = sorted(all_results, key=lambda r: r["relevance_score"], reverse=True)
        for r in by_score[:5]:
            chunk = chunks[r["index"]].copy()
            chunk["rerank_score"] = r["relevance_score"]
            scored.append(chunk)
        logger.info(
            "Rerank: all below threshold %.2f, kept top-%d (best=%.3f)",
            score_threshold,
            len(scored),
            scored[0]["rerank_score"] if scored else 0,
        )

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


async def _jina_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    api_key: str,
    model: str = "jina-reranker-m0",
    base_url: str = "https://api.jina.ai/v1/rerank",
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Jina multimodal reranker via /v1/rerank API."""
    return await _http_rerank(
        query,
        chunks,
        top_k,
        url=base_url,
        model=model,
        api_key=api_key,
        score_threshold=score_threshold,
    )


async def _aliyun_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    api_key: str,
    model: str = "gte-rerank",
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/rerank",
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Aliyun DashScope reranker."""
    return await _http_rerank(
        query,
        chunks,
        top_k,
        url=base_url,
        model=model,
        api_key=api_key,
        score_threshold=score_threshold,
    )


async def _local_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    model: str,
    base_url: str,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Self-hosted OpenAI-compatible /rerank endpoint."""
    rerank_url = base_url.rstrip("/") + "/rerank"
    return await _http_rerank(
        query,
        chunks,
        top_k,
        url=rerank_url,
        model=model,
        score_threshold=score_threshold,
    )


# ── Azure Cohere (text-only, enterprise) ─────────────────────────


async def _azure_cohere_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Azure AI Services Cohere reranker (text-only)."""
    if not chunks:
        return []

    rerank_url = (
        endpoint
        if "/providers/cohere/" in endpoint or endpoint.endswith("/rerank")
        else f"{endpoint}/providers/cohere/v2/rerank"
    )
    documents = [c.get("content", "") for c in chunks]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            rerank_url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json={"model": deployment, "query": query, "documents": documents},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

    scored = []
    for r in results:
        idx = r["index"]
        score = r["relevance_score"]
        if score >= score_threshold:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            scored.append(chunk)

    if not scored and results:
        by_score = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
        for r in by_score[:5]:
            chunk = chunks[r["index"]].copy()
            chunk["rerank_score"] = r["relevance_score"]
            scored.append(chunk)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


# ── Factory ──────────────────────────────────────────────────────


def build_rerank_func(
    rc: RerankConfig,
    ingest_func: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Build a multimodal rerank callable from RerankConfig.

    Returns a callable with signature:
        (query: str, chunks: list[dict], top_k: int) -> list[dict]
    """
    from dlightrag.observability import wrap_rerank_func

    strategy = rc.strategy
    fn: Callable[..., Any]

    if strategy == "chat_llm_reranker":
        if ingest_func is None:
            raise ValueError("chat_llm_reranker requires ingest_func (VLM)")
        fn = partial(
            _chat_llm_rerank,
            vlm_func=ingest_func,
            max_concurrency=rc.max_concurrency,
            score_threshold=rc.score_threshold,
            batch_size=rc.batch_size,
        )
    elif strategy == "jina_reranker":
        if not rc.api_key:
            raise ValueError("jina_reranker requires api_key")
        fn = partial(
            _jina_rerank,
            api_key=rc.api_key,
            model=rc.model or "jina-reranker-m0",
            base_url=rc.base_url or "https://api.jina.ai/v1/rerank",
            score_threshold=rc.score_threshold,
        )
    elif strategy == "aliyun_reranker":
        if not rc.api_key:
            raise ValueError("aliyun_reranker requires api_key")
        fn = partial(
            _aliyun_rerank,
            api_key=rc.api_key,
            model=rc.model or "gte-rerank",
            base_url=rc.base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/rerank",
            score_threshold=rc.score_threshold,
        )
    elif strategy == "local_reranker":
        if not rc.base_url:
            raise ValueError("local_reranker requires base_url")
        fn = partial(
            _local_rerank,
            model=rc.model or "default",
            base_url=rc.base_url,
            score_threshold=rc.score_threshold,
        )
    elif strategy == "azure_cohere":
        if not rc.base_url or not rc.api_key:
            raise ValueError("azure_cohere requires base_url and api_key")
        fn = partial(
            _azure_cohere_rerank,
            endpoint=rc.base_url,
            api_key=rc.api_key,
            deployment=rc.model or "cohere-rerank-v3.5",
            score_threshold=rc.score_threshold,
        )
    else:
        raise ValueError(f"Unknown rerank strategy: {strategy}")

    return wrap_rerank_func(fn, name=f"rerank/{strategy}")


# ── LightRAG adapter ────────────────────────────────────────────


def build_lightrag_rerank_adapter(rerank_func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap multimodal reranker for LightRAG's text-only interface.

    LightRAG calls rerank as::

        await rerank_func(query=query, documents=document_texts, top_n=top_n)

    This adapter converts text documents to chunk dicts, calls the
    multimodal reranker, then maps back to LightRAG's index-based format:
    ``[{"index": int, "relevance_score": float}, ...]``
    """

    async def adapter(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        top_k = top_n if top_n is not None else len(documents)
        chunks = [{"content": doc, "_orig_idx": i} for i, doc in enumerate(documents)]
        scored = await rerank_func(query=query, chunks=chunks, top_k=top_k)
        return [
            {"index": c["_orig_idx"], "relevance_score": c.get("rerank_score", 0.0)} for c in scored
        ]

    return adapter


__all__ = [
    "build_rerank_func",
    "build_lightrag_rerank_adapter",
    "_parse_listwise_scores",
]
