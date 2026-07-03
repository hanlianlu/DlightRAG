# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multimodal rerank strategies for DlightRAG.

All strategies accept chunks with text content + optional image_data
and return chunks with rerank_score added, sorted descending.

Runtime interface:
    async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]

Reranking is handled by DlightRAG after retrieval fusion and provenance
hydration, NOT through LightRAG's ``rerank_model_func``. DlightRAG's final
candidate set can include BM25, metadata-injected, and multimodal chunks that
LightRAG's native reranker cannot see as one list.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx

from dlightrag.prompts import LISTWISE_RERANK_PROMPT
from dlightrag.utils.image_budget import ImagePayloadBudget

if TYPE_CHECKING:
    from dlightrag.config import RerankConfig

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 7
_DEFAULT_IMAGE_MAX_BYTES = 1_500_000
_DEFAULT_IMAGE_MAX_TOTAL_BYTES = 8_000_000
_DEFAULT_IMAGE_MAX_PX = 1280
_DEFAULT_IMAGE_MIN_PX = 768
_DEFAULT_IMAGE_QUALITY = 86
_DEFAULT_IMAGE_MIN_QUALITY = 76


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_scored_chunks(
    chunks: list[dict[str, Any]],
    indexed_scores: list[dict[str, Any]],
    *,
    score_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Parse HTTP reranker results into scored, thresholded, top-k chunks.

    Shared by ``_http_rerank`` and ``_azure_cohere_rerank``. Both decode
    the same response contract: ``{"results": [{"index": …, "relevance_score": …}, …]}``.
    """
    scored: list[dict[str, Any]] = []
    for r in indexed_scores:
        idx = r["index"]
        score = r["relevance_score"]
        if score >= score_threshold:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            scored.append(chunk)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def _parse_listwise_scores(text: str, expected: int) -> list[float]:
    """Parse a JSON array of scores from a model response."""
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


# ── LLM/VLM listwise reranker ────────────────────────────────────


async def _chat_llm_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    scoring_func: Callable[..., Any],
    max_concurrency: int = 4,
    score_threshold: float = 0.5,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    multimodal: bool = True,
    image_max_bytes: int = _DEFAULT_IMAGE_MAX_BYTES,
    image_max_total_bytes: int = _DEFAULT_IMAGE_MAX_TOTAL_BYTES,
    image_max_px: int = _DEFAULT_IMAGE_MAX_PX,
    image_min_px: int = _DEFAULT_IMAGE_MIN_PX,
    image_quality: int = _DEFAULT_IMAGE_QUALITY,
    image_min_quality: int = _DEFAULT_IMAGE_MIN_QUALITY,
) -> list[dict[str, Any]]:
    """LLM/VLM listwise reranker. Scores chunks in bounded-concurrency batches."""
    if not chunks:
        return []

    async def _score_batch(
        batch_start: int,
        batch: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        prompt = LISTWISE_RERANK_PROMPT.format(query=query, n=len(batch))

        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        image_budget = ImagePayloadBudget(
            max_total_bytes=image_max_total_bytes,
            max_bytes_per_image=image_max_bytes,
            max_px=image_max_px,
            min_px=image_min_px,
            quality=image_quality,
            min_quality=image_min_quality,
        )

        for i, chunk in enumerate(batch):
            content.append({"type": "text", "text": f"\n--- Item {i + 1} ---"})
            image_included = False
            if multimodal:
                img = chunk.get("image_data")
                if img:
                    bounded = image_budget.add_base64(
                        img,
                        label=f"rerank:{batch_start + i}",
                    )
                    if bounded is not None:
                        uri, _ = bounded
                        content.append({"type": "image_url", "image_url": {"url": uri}})
                        image_included = True
            if chunk.get("content"):
                # image_included=True → image carries rich signal, keep text short.
                # image_included=False → text-only path (either multimodal disabled,
                # or image exceeded budget, or chunk had no image): send full VLM
                # text description for accurate reranking.
                text_limit = 500 if image_included else None
                text_preview = (
                    chunk["content"] if text_limit is None else chunk["content"][:text_limit]
                )
                content.append({"type": "text", "text": text_preview})

        messages = [{"role": "user", "content": content}]
        try:
            resp = await scoring_func(messages=messages)
            scores = _parse_listwise_scores(resp, len(batch))
            logger.info(
                "Rerank batch [%d..%d] (%d items): scores=%s",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
                [f"{s:.3f}" for s in scores],
            )
        except Exception:
            logger.warning(
                "LLM/VLM listwise rerank failed for batch %d", batch_start, exc_info=True
            )
            raise

        return list(zip(batch, scores, strict=False))

    from dlightrag.utils.concurrency import bounded_gather

    batches = [
        (batch_start, chunks[batch_start : batch_start + batch_size])
        for batch_start in range(0, len(chunks), batch_size)
    ]
    logger.info(
        "Rerank listwise schedule: chunks=%d batch_size=%d batches=%d max_concurrency=%d active_batches=%d",
        len(chunks),
        batch_size,
        len(batches),
        max(1, max_concurrency),
        min(max(1, max_concurrency), len(batches)),
    )
    batch_results = await bounded_gather(
        [_score_batch(batch_start, batch) for batch_start, batch in batches],
        max_concurrent=max(1, max_concurrency),
        task_name="rerank-batch",
    )

    all_results: list[tuple[dict[str, Any], float]] = []
    for result in batch_results:
        if isinstance(result, Exception):
            continue
        all_results.extend(result)
    if not all_results and chunks:
        raise RuntimeError("All rerank batches failed")

    # Apply threshold
    scored = []
    for chunk, score in all_results:
        if score >= score_threshold:
            out = chunk.copy()
            out["rerank_score"] = score
            scored.append(out)

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
    client: httpx.AsyncClient | None = None,
    multimodal: bool = True,
    image_max_bytes: int = _DEFAULT_IMAGE_MAX_BYTES,
    image_max_total_bytes: int = _DEFAULT_IMAGE_MAX_TOTAL_BYTES,
    image_max_px: int = _DEFAULT_IMAGE_MAX_PX,
    image_min_px: int = _DEFAULT_IMAGE_MIN_PX,
    image_quality: int = _DEFAULT_IMAGE_QUALITY,
    image_min_quality: int = _DEFAULT_IMAGE_MIN_QUALITY,
) -> list[dict[str, Any]]:
    """Shared HTTP reranker for Jina, Aliyun, and self-hosted endpoints.

    Reuses *client* when provided (pooled by ``build_rerank_func``);
    falls back to an ephemeral client otherwise so the function stays
    callable in tests and ad-hoc scripts.
    """
    if not chunks:
        return []

    documents: list[dict[str, str]] = []
    image_budget = ImagePayloadBudget(
        max_total_bytes=image_max_total_bytes,
        max_bytes_per_image=image_max_bytes,
        max_px=image_max_px,
        min_px=image_min_px,
        quality=image_quality,
        min_quality=image_min_quality,
    )
    for idx, c in enumerate(chunks):
        img = c.get("image_data")
        if multimodal and img:
            bounded = image_budget.add_base64(img, label=f"rerank:{idx}")
            documents.append({"image": bounded[0]} if bounded else {"text": c.get("content", "")})
        else:
            documents.append({"text": c.get("content", "")})

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if client is not None:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    else:
        async with httpx.AsyncClient(timeout=60.0) as ephemeral:
            resp = await ephemeral.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

    all_results = data.get("results", [])
    return _build_scored_chunks(chunks, all_results, score_threshold=score_threshold, top_k=top_k)


# ── HTTP /rerank strategy registry ───────────────────────────────
# Three strategies share `_http_rerank`'s shape; only defaults differ.
# Tuple = (default_base_url, default_model, append_/rerank_to_base, requires_api_key)
_HTTP_RERANK_DEFAULTS: dict[str, tuple[str | None, str, bool, bool]] = {
    "jina_reranker": (
        "https://api.jina.ai/v1/rerank",
        "jina-reranker-m0",
        False,
        True,
    ),
    "aliyun_reranker": (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/rerank",
        "gte-rerank",
        False,
        True,
    ),
    "local_reranker": (None, "default", True, False),
}


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
    client: httpx.AsyncClient | None = None,
    multimodal: bool = True,  # accepted for uniform interface; Azure Cohere is text-only
) -> list[dict[str, Any]]:
    """Azure AI Services Cohere reranker (text-only).

    The *multimodal* parameter is accepted for interface consistency with
    the other rerank strategies. When False, the behaviour is unchanged
    because Cohere's REST API only accepts text documents — ``image_data``
    is always ignored.

    Reuses *client* when provided; ephemeral fallback otherwise.
    """
    if not chunks:
        return []

    rerank_url = (
        endpoint
        if "/providers/cohere/" in endpoint or endpoint.endswith("/rerank")
        else f"{endpoint}/providers/cohere/v2/rerank"
    )
    documents = [c.get("content", "") for c in chunks]
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    json_body = {"model": deployment, "query": query, "documents": documents}

    if client is not None:
        resp = await client.post(rerank_url, headers=headers, json=json_body)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    else:
        async with httpx.AsyncClient(timeout=30.0) as ephemeral:
            resp = await ephemeral.post(rerank_url, headers=headers, json=json_body)
            resp.raise_for_status()
            results = resp.json().get("results", [])

    return _build_scored_chunks(chunks, results, score_threshold=score_threshold, top_k=top_k)


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
            raise ValueError("chat_llm_reranker requires a messages-first scoring callable")
        fn = partial(
            _chat_llm_rerank,
            scoring_func=ingest_func,
            max_concurrency=rc.max_concurrency,
            score_threshold=rc.score_threshold,
            batch_size=rc.batch_size,
            multimodal=rc.multimodal,
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
    elif strategy in _HTTP_RERANK_DEFAULTS:
        default_url, default_model, append_path, needs_key = _HTTP_RERANK_DEFAULTS[strategy]
        url = rc.base_url or default_url
        if url is None:
            raise ValueError(f"{strategy} requires base_url")
        if append_path:
            url = url.rstrip("/") + "/rerank"
        if needs_key and not rc.api_key:
            raise ValueError(f"{strategy} requires api_key")
        # Pool the httpx client across calls; per-call client construction
        # wastes connection setup.
        client = httpx.AsyncClient(timeout=60.0)
        fn = partial(
            _http_rerank,
            url=url,
            model=rc.model or default_model,
            api_key=rc.api_key,
            score_threshold=rc.score_threshold,
            client=client,
            multimodal=rc.multimodal,
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
    elif strategy == "azure_cohere":
        if not rc.base_url or not rc.api_key:
            raise ValueError("azure_cohere requires base_url and api_key")
        client = httpx.AsyncClient(timeout=30.0)
        fn = partial(
            _azure_cohere_rerank,
            endpoint=rc.base_url,
            api_key=rc.api_key,
            deployment=rc.model or "cohere-rerank-v3.5",
            score_threshold=rc.score_threshold,
            client=client,
            multimodal=rc.multimodal,
        )
    else:
        raise ValueError(f"Unknown rerank strategy: {strategy}")

    return wrap_rerank_func(fn, name=f"rerank/{strategy}")


__all__ = [
    "build_rerank_func",
    "_parse_listwise_scores",
]
