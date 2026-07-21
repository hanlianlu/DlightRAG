# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multimodal rerank strategies for DlightRAG.

All strategies accept chunks with text content + optional image_data
and return chunks with rerank_score added, sorted descending.

Runtime interface:
    async def rerank_func(query: str, chunks: list[dict], top_k: int) -> list[dict]

Reranking is handled by DlightRAG after retrieval fusion and provenance
hydration, NOT through LightRAG's ``rerank_model_func``. DlightRAG's final
candidate set can include BM25, LightRAG, and multimodal chunks that LightRAG's
native reranker cannot see as one list.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx

from dlightrag.models.providers.rerank_base import (
    PreparedDocument,
    RerankProvider,
    ResolvedInputModality,
    resolve_rerank_input_modality,
)
from dlightrag.models.providers.rerank_providers import RERANK_PROVIDERS
from dlightrag.prompts import LISTWISE_RERANK_PROMPT
from dlightrag.utils.image_budget import ImagePayloadBudget
from dlightrag.utils.images import MODEL_IMAGE_MAX_PIXELS

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
    score_threshold: float | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """Parse indexed reranker results into scored, thresholded, top-k chunks."""
    scored: list[dict[str, Any]] = []
    for r in indexed_scores:
        idx = r["index"]
        score = r["relevance_score"]
        if score_threshold is None or score >= score_threshold:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = score
            scored.append(chunk)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_consumes_images(rc: RerankConfig, *, supports_vision: bool | None) -> bool:
    """Whether the configured reranker reads chunk ``image_data``.

    Mirrors :func:`build_rerank_func`'s modality resolution so callers can skip
    the expensive pre-rerank image hydration for a text-only reranker. Errs
    toward ``True`` (keep hydration) only for the chat_llm_reranker unknown-probe
    case, so a wrong verdict never starves a multimodal reranker of image bytes.
    """
    if not rc.enabled:
        return False
    if rc.strategy == "chat_llm_reranker":
        if rc.input_modality == "text":
            return False
        if rc.input_modality == "multimodal":
            return True
        return supports_vision is not False  # auto: probe unknown/True -> True (safe)
    provider = RERANK_PROVIDERS.get(rc.strategy)
    if provider is None:
        return False
    return (
        resolve_rerank_input_modality(rc.input_modality) == "multimodal" and provider.accepts_images
    )


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("content") or "")


def _prepare_documents(
    chunks: list[dict[str, Any]],
    modality: ResolvedInputModality,
    budget_factory: Callable[[], ImagePayloadBudget],
) -> list[PreparedDocument]:
    """Build ``(text, bounded_image_uri | None)`` per chunk, off the event loop.

    Text mode carries no image. Multimodal mode bounds each image once (PIL
    decode/resize/re-encode is CPU-bound); a chunk whose image is missing or
    exceeds the shared per-request budget degrades to text.

    Each chunk maps to exactly one document -- never two competing entries.
    Downstream, multimodal HTTP providers render each document as image XOR
    text (``{"image": uri} if uri else text``), so a chunk carrying an image is
    reranked by that image alone and its VLM ``content`` is omitted from the
    rerank payload (it still feeds embedding, retrieval, and answer). Only the
    listwise VLM reranker fuses image + description into one scored candidate.
    """
    if modality == "text":
        return [(_chunk_text(chunk), None) for chunk in chunks]

    budget = budget_factory()
    prepared: list[PreparedDocument] = []
    for idx, chunk in enumerate(chunks):
        uri: str | None = None
        img = chunk.get("image_data")
        if img:
            bounded = budget.add_base64(img, label=f"rerank:{idx}")
            if bounded is not None:
                uri = bounded[0]
        prepared.append((_chunk_text(chunk), uri))
    return prepared


def _parse_listwise_scores(text: str, expected: int) -> list[float]:
    """Parse a JSON array of scores from a model response."""
    text = text.strip()
    # Try JSON array
    with suppress(json.JSONDecodeError, ValueError, TypeError):
        data = json.loads(text)
        if isinstance(data, list):
            return [_clamp(float(s)) for s in data[:expected]]
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
    score_threshold: float | None = None,
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

        def _build_content() -> list[dict[str, Any]]:
            # PIL decode/resize/JPEG re-encode is CPU-bound; build the payload
            # (including image bounding) off the event loop. The budget is local
            # to this batch, so running it in a worker thread is safe.
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            image_budget = ImagePayloadBudget(
                max_total_bytes=image_max_total_bytes,
                max_bytes_per_image=image_max_bytes,
                max_pixels=MODEL_IMAGE_MAX_PIXELS,
                max_px=image_max_px,
                min_px=image_min_px,
                quality=image_quality,
                min_quality=image_min_quality,
            )
            for i, chunk in enumerate(batch):
                content.append({"type": "text", "text": f"\nCandidate {i + 1}:"})
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
                if chunk.get("content"):
                    content.append({"type": "text", "text": chunk["content"]})
            return content

        content = await asyncio.to_thread(_build_content)
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

    scored = []
    for chunk, score in all_results:
        if score_threshold is None or score >= score_threshold:
            out = chunk.copy()
            out["rerank_score"] = score
            scored.append(out)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


# HTTP transport: shared POST helper + the provider-driven rerank driver.


async def _post_rerank_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    client: httpx.AsyncClient | None,
) -> dict[str, Any]:
    if client is not None:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
    async with httpx.AsyncClient(timeout=60.0) as ephemeral:
        resp = await ephemeral.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _run_http_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    provider: RerankProvider,
    model: str,
    base_url: str | None,
    api_key: str | None,
    modality: ResolvedInputModality,
    score_threshold: float | None,
    client: httpx.AsyncClient | None,
    budget_factory: Callable[[], ImagePayloadBudget],
) -> list[dict[str, Any]]:
    """Drive one HTTP rerank request against any :class:`RerankProvider`.

    Shared across every provider: prepare documents (bounding images off the
    event loop), POST, then score/threshold/top-k. Only the wire shape -- URL,
    auth, body, response key -- comes from the provider.
    """
    if not chunks:
        return []

    prepared = await asyncio.to_thread(_prepare_documents, chunks, modality, budget_factory)
    payload = provider.build_payload(
        model=model,
        query=query,
        documents=prepared,
        top_n=min(top_k, len(prepared)),
    )
    data = await _post_rerank_json(
        url=provider.request_url(base_url, model),
        payload=payload,
        headers=provider.request_headers(api_key),
        client=client,
    )
    return _build_scored_chunks(
        chunks, provider.parse_results(data), score_threshold=score_threshold, top_k=top_k
    )


# ── Factory ──────────────────────────────────────────────────────


class _RerankCallable:
    """Callable that owns its httpx client lifecycle.

    Wraps the strategy function (partial + traced) and the httpx client
    so RAGService can ``await rerank_func.aclose()`` at shutdown.
    """

    __slots__ = ("_fn", "_client")

    def __init__(self, fn: Callable[..., Any], *, client: httpx.AsyncClient | None = None) -> None:
        self._fn = fn
        self._client = client

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self._fn(*args, **kwargs)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


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

    if strategy == "chat_llm_reranker":
        if ingest_func is None:
            raise ValueError("chat_llm_reranker requires a messages-first scoring callable")
        fn = partial(
            _chat_llm_rerank,
            scoring_func=ingest_func,
            max_concurrency=rc.max_concurrency,
            score_threshold=rc.score_threshold,
            batch_size=rc.batch_size,
            multimodal=rc.input_modality != "text",
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
        return _RerankCallable(fn=wrap_rerank_func(fn, name="rerank/chat_llm_reranker"))

    provider = RERANK_PROVIDERS.get(strategy)
    if provider is None:
        raise ValueError(f"Unknown rerank strategy: {strategy}")
    if provider.requires_api_key and not rc.api_key:
        raise ValueError(f"{strategy} requires api_key")
    if provider.requires_base_url and not rc.base_url:
        raise ValueError(f"{strategy} requires base_url")

    modality = resolve_rerank_input_modality(rc.input_modality)
    if modality == "multimodal" and not provider.accepts_images:
        raise ValueError(
            f"{strategy} is text-only and cannot honor input_modality='multimodal'; "
            "set rerank.input_modality to 'text' or 'auto'"
        )

    budget_factory = partial(
        ImagePayloadBudget,
        max_total_bytes=rc.image_max_total_bytes,
        max_bytes_per_image=rc.image_max_bytes,
        max_pixels=MODEL_IMAGE_MAX_PIXELS,
        max_px=rc.image_max_px,
        min_px=rc.image_min_px,
        quality=rc.image_quality,
        min_quality=rc.image_min_quality,
    )
    client = httpx.AsyncClient(timeout=60.0)
    fn = partial(
        _run_http_rerank,
        provider=provider,
        model=rc.model or provider.default_model,
        base_url=rc.base_url,
        api_key=rc.api_key,
        modality=modality,
        score_threshold=rc.score_threshold,
        client=client,
        budget_factory=budget_factory,
    )
    return _RerankCallable(fn=wrap_rerank_func(fn, name=f"rerank/{strategy}"), client=client)


__all__ = [
    "build_rerank_func",
    "_parse_listwise_scores",
]
