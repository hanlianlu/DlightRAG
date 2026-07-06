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

import json
import logging
import re
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

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
_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v3"
_ALIYUN_RERANK_MODEL = "qwen3-rerank"
_VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
_VOYAGE_RERANK_MODEL = "rerank-2.5"
_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
_COHERE_RERANK_MODEL = "rerank-v4.0-pro"
_AZURE_COHERE_RERANK_MODEL = "cohere-rerank-v4.0-pro"
InputModality = Literal["auto", "text", "multimodal"]
ResolvedInputModality = Literal["text", "multimodal"]
# Official rerank APIs that accept image documents as of the current docs.
_MULTIMODAL_RERANK_MODELS: frozenset[str] = frozenset(
    {
        "jina-reranker-m0",
        "qwen3-vl-rerank",
    }
)
_TEXT_RERANK_MODELS = frozenset(
    {
        "jina-reranker-v3",
        "qwen3-rerank",
    }
)
_TEXT_RERANK_MODEL_FAMILIES = ("rerank-2.5", "rerank-v4.0", "cohere-rerank-v4.0")


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


def _resolve_input_modality(input_modality: InputModality, model: str) -> ResolvedInputModality:
    model_id = model.lower()
    if model_id in _TEXT_RERANK_MODELS or any(
        model_id == family or model_id.startswith(f"{family}-")
        for family in _TEXT_RERANK_MODEL_FAMILIES
    ):
        return "text"
    if input_modality != "auto":
        return input_modality
    return "multimodal" if model_id in _MULTIMODAL_RERANK_MODELS else "text"


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("content") or "")


def _build_rerank_documents(
    chunks: list[dict[str, Any]],
    *,
    input_modality: ResolvedInputModality,
    image_max_bytes: int = _DEFAULT_IMAGE_MAX_BYTES,
    image_max_total_bytes: int = _DEFAULT_IMAGE_MAX_TOTAL_BYTES,
    image_max_px: int = _DEFAULT_IMAGE_MAX_PX,
    image_min_px: int = _DEFAULT_IMAGE_MIN_PX,
    image_quality: int = _DEFAULT_IMAGE_QUALITY,
    image_min_quality: int = _DEFAULT_IMAGE_MIN_QUALITY,
) -> list[str | dict[str, str]]:
    if input_modality == "text":
        return [_chunk_text(chunk) for chunk in chunks]

    documents: list[str | dict[str, str]] = []
    image_budget = ImagePayloadBudget(
        max_total_bytes=image_max_total_bytes,
        max_bytes_per_image=image_max_bytes,
        max_px=image_max_px,
        min_px=image_min_px,
        quality=image_quality,
        min_quality=image_min_quality,
    )
    for idx, chunk in enumerate(chunks):
        img = chunk.get("image_data")
        if img:
            bounded = image_budget.add_base64(img, label=f"rerank:{idx}")
            if bounded is not None:
                documents.append({"image": bounded[0]})
                continue
        documents.append(_chunk_text(chunk))
    return documents


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
            content.append({"type": "text", "text": f"\nCandidate {i + 1}:"})
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

    scored = []
    for chunk, score in all_results:
        if score_threshold is None or score >= score_threshold:
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
    score_threshold: float | None = None,
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


# ── Provider rerankers ──────────────────────────────────────────


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


async def _jina_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    url: str = _JINA_RERANK_URL,
    model: str = _JINA_RERANK_MODEL,
    api_key: str,
    input_modality: InputModality = "auto",
    score_threshold: float | None = None,
    client: httpx.AsyncClient | None = None,
    image_max_bytes: int = _DEFAULT_IMAGE_MAX_BYTES,
    image_max_total_bytes: int = _DEFAULT_IMAGE_MAX_TOTAL_BYTES,
    image_max_px: int = _DEFAULT_IMAGE_MAX_PX,
    image_min_px: int = _DEFAULT_IMAGE_MIN_PX,
    image_quality: int = _DEFAULT_IMAGE_QUALITY,
    image_min_quality: int = _DEFAULT_IMAGE_MIN_QUALITY,
) -> list[dict[str, Any]]:
    """Jina reranker endpoint; v3 is text-only, m0 accepts image documents."""
    if not chunks:
        return []

    documents = _build_rerank_documents(
        chunks,
        input_modality=_resolve_input_modality(input_modality, model),
        image_max_bytes=image_max_bytes,
        image_max_total_bytes=image_max_total_bytes,
        image_max_px=image_max_px,
        image_min_px=image_min_px,
        image_quality=image_quality,
        image_min_quality=image_min_quality,
    )
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = await _post_rerank_json(url=url, payload=payload, headers=headers, client=client)
    return _build_scored_chunks(
        chunks, data.get("results", []), score_threshold=score_threshold, top_k=top_k
    )


async def _aliyun_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    url: str,
    model: str = _ALIYUN_RERANK_MODEL,
    api_key: str,
    input_modality: InputModality = "auto",
    score_threshold: float | None = None,
    client: httpx.AsyncClient | None = None,
    image_max_bytes: int = _DEFAULT_IMAGE_MAX_BYTES,
    image_max_total_bytes: int = _DEFAULT_IMAGE_MAX_TOTAL_BYTES,
    image_max_px: int = _DEFAULT_IMAGE_MAX_PX,
    image_min_px: int = _DEFAULT_IMAGE_MIN_PX,
    image_quality: int = _DEFAULT_IMAGE_QUALITY,
    image_min_quality: int = _DEFAULT_IMAGE_MIN_QUALITY,
) -> list[dict[str, Any]]:
    """Alibaba Model Studio reranker; qwen3 text and qwen3-vl use different bodies."""
    if not chunks:
        return []

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    top_n = min(top_k, len(chunks))
    if model == "qwen3-rerank":
        payload = {
            "model": model,
            "query": query,
            "documents": [_chunk_text(chunk) for chunk in chunks],
            "top_n": top_n,
        }
        data = await _post_rerank_json(url=url, payload=payload, headers=headers, client=client)
        results = data.get("results", [])
    else:
        documents = _build_rerank_documents(
            chunks,
            input_modality=_resolve_input_modality(input_modality, model),
            image_max_bytes=image_max_bytes,
            image_max_total_bytes=image_max_total_bytes,
            image_max_px=image_max_px,
            image_min_px=image_min_px,
            image_quality=image_quality,
            image_min_quality=image_min_quality,
        )
        payload = {
            "model": model,
            "input": {"query": {"text": query}, "documents": documents},
            "parameters": {"top_n": top_n},
        }
        data = await _post_rerank_json(url=url, payload=payload, headers=headers, client=client)
        results = data.get("output", {}).get("results", [])

    return _build_scored_chunks(chunks, results, score_threshold=score_threshold, top_k=top_k)


async def _voyage_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    url: str = _VOYAGE_RERANK_URL,
    model: str = _VOYAGE_RERANK_MODEL,
    api_key: str,
    score_threshold: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Voyage reranker endpoint: text documents, top_k."""
    if not chunks:
        return []

    documents = [c.get("content", "") for c in chunks]
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_k": min(top_k, len(documents)),
        "truncation": True,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = await _post_rerank_json(url=url, payload=payload, headers=headers, client=client)

    return _build_scored_chunks(
        chunks,
        data.get("data", []),
        score_threshold=score_threshold,
        top_k=top_k,
    )


async def _cohere_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    url: str = _COHERE_RERANK_URL,
    model: str = _COHERE_RERANK_MODEL,
    api_key: str,
    score_threshold: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Cohere public reranker endpoint: text documents, top_n."""
    if not chunks:
        return []

    documents = [c.get("content", "") for c in chunks]
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = await _post_rerank_json(url=url, payload=payload, headers=headers, client=client)
    results = data.get("results", [])

    return _build_scored_chunks(chunks, results, score_threshold=score_threshold, top_k=top_k)


# ── Self-hosted /rerank strategy registry ───────────────────────
# Local endpoints keep the old generic document-object shape.
# Tuple = (default_base_url, default_model, append_/rerank_to_base, requires_api_key)
_HTTP_RERANK_DEFAULTS: dict[str, tuple[str | None, str, bool, bool]] = {
    "local_reranker": (None, "default", True, False),
}


# ── Azure Cohere (text-only, enterprise) ─────────────────────────


def _azure_cohere_rerank_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/rerank"):
        return base
    if base.endswith("/providers/cohere"):
        return f"{base}/v2/rerank"
    if _is_azure_ai_services_host(base):
        return f"{base}/providers/cohere/v2/rerank"
    return f"{base}/v1/rerank"


def _is_azure_ai_services_host(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").rstrip(".").lower()
    return host == "services.ai.azure.com" or host.endswith(".services.ai.azure.com")


async def _azure_cohere_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    score_threshold: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Azure AI Services Cohere reranker (text-only)."""
    if not chunks:
        return []

    rerank_url = _azure_cohere_rerank_url(endpoint)
    documents = [c.get("content", "") for c in chunks]
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    json_body = {
        "model": deployment,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

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
    score_threshold = rc.score_threshold
    fn: Callable[..., Any]
    client: httpx.AsyncClient | None = None

    if strategy == "chat_llm_reranker":
        if ingest_func is None:
            raise ValueError("chat_llm_reranker requires a messages-first scoring callable")
        fn = partial(
            _chat_llm_rerank,
            scoring_func=ingest_func,
            max_concurrency=rc.max_concurrency,
            score_threshold=score_threshold,
            batch_size=rc.batch_size,
            multimodal=rc.input_modality != "text",
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
    elif strategy == "jina_reranker":
        if not rc.api_key:
            raise ValueError("jina_reranker requires api_key")
        client = httpx.AsyncClient(timeout=60.0)
        fn = partial(
            _jina_rerank,
            url=rc.base_url or _JINA_RERANK_URL,
            model=rc.model or _JINA_RERANK_MODEL,
            api_key=rc.api_key,
            input_modality=rc.input_modality,
            score_threshold=score_threshold,
            client=client,
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
    elif strategy == "aliyun_reranker":
        if not rc.api_key or not rc.base_url:
            raise ValueError("aliyun_reranker requires base_url and api_key")
        client = httpx.AsyncClient(timeout=60.0)
        fn = partial(
            _aliyun_rerank,
            url=rc.base_url,
            model=rc.model or _ALIYUN_RERANK_MODEL,
            api_key=rc.api_key,
            input_modality=rc.input_modality,
            score_threshold=score_threshold,
            client=client,
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
            score_threshold=score_threshold,
            client=client,
            multimodal=(
                _resolve_input_modality(rc.input_modality, rc.model or default_model)
                == "multimodal"
            ),
            image_max_bytes=rc.image_max_bytes,
            image_max_total_bytes=rc.image_max_total_bytes,
            image_max_px=rc.image_max_px,
            image_min_px=rc.image_min_px,
            image_quality=rc.image_quality,
            image_min_quality=rc.image_min_quality,
        )
    elif strategy == "voyage_reranker":
        if not rc.api_key:
            raise ValueError("voyage_reranker requires api_key")
        client = httpx.AsyncClient(timeout=60.0)
        fn = partial(
            _voyage_rerank,
            url=rc.base_url or _VOYAGE_RERANK_URL,
            model=rc.model or _VOYAGE_RERANK_MODEL,
            api_key=rc.api_key,
            score_threshold=score_threshold,
            client=client,
        )
    elif strategy == "cohere_reranker":
        if not rc.api_key:
            raise ValueError("cohere_reranker requires api_key")
        client = httpx.AsyncClient(timeout=60.0)
        fn = partial(
            _cohere_rerank,
            url=rc.base_url or _COHERE_RERANK_URL,
            model=rc.model or _COHERE_RERANK_MODEL,
            api_key=rc.api_key,
            score_threshold=score_threshold,
            client=client,
        )
    elif strategy == "azure_cohere":
        if not rc.base_url or not rc.api_key:
            raise ValueError("azure_cohere requires base_url and api_key")
        client = httpx.AsyncClient(timeout=30.0)
        fn = partial(
            _azure_cohere_rerank,
            endpoint=rc.base_url,
            api_key=rc.api_key,
            deployment=rc.model or _AZURE_COHERE_RERANK_MODEL,
            score_threshold=score_threshold,
            client=client,
        )
    else:
        raise ValueError(f"Unknown rerank strategy: {strategy}")

    return _RerankCallable(
        fn=wrap_rerank_func(fn, name=f"rerank/{strategy}"),
        client=client,
    )


__all__ = [
    "build_rerank_func",
    "_parse_listwise_scores",
]
