# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified reranker strategies.

All reranker implementations live here — LLM listwise, API-based (Cohere, Jina,
Aliyun, Azure Cohere), and the factory function that dispatches based on config.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 7
_DEFAULT_SCORE_THRESHOLD = 0.3

_LISTWISE_PROMPT = """\
You are a relevance judge. Given a query and {n} text chunks, \
score each chunk's relevance to the query.

Respond with ONLY a JSON array of {n} scores in order: [<float>, <float>, ...]
Each score is 0.00 (completely irrelevant) to 1.00 (perfectly relevant).

Query: {query}"""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_listwise_scores(text: str, expected: int) -> list[float]:
    """Parse a JSON array of scores from LLM response."""
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


def _fallback_ranking(n: int) -> list[dict[str, float]]:
    """Return original-order ranking as fallback."""
    return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(n)]


def build_llm_rerank_func(
    ingest_func: Callable,
    domain_knowledge_hints: str = "",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> Callable:
    """LLM-based listwise reranker with batched scoring.

    Scores chunks in batches for reliable relative comparison.
    Uses a simple JSON array prompt instead of structured JSON objects.

    Args:
        ingest_func: LightRAG-adapted callable (prompt, system_prompt, ...).
        domain_knowledge_hints: Optional domain context appended to prompt.
        batch_size: Number of chunks per LLM call (default 7).
        score_threshold: Minimum score to include in results (default 0.3).
    """

    async def rerank_func(
        query: str,
        documents: list[str],
        domain_knowledge: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        effective_hints = domain_knowledge or domain_knowledge_hints
        all_scores: list[tuple[int, float]] = []  # (original_index, score)

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start : batch_start + batch_size]
            prompt_parts = [_LISTWISE_PROMPT.format(query=query, n=len(batch))]
            if effective_hints:
                prompt_parts.append(f"\nContext: {effective_hints}")
            for i, doc in enumerate(batch):
                prompt_parts.append(f"\n--- Chunk {i + 1} ---\n{doc[:500]}")
            user_content = "".join(prompt_parts)

            try:
                result_str = await ingest_func(user_content, system_prompt="")
                scores = _parse_listwise_scores(result_str, len(batch))
                logger.info(
                    "Rerank batch [%d..%d]: scores=%s",
                    batch_start,
                    batch_start + len(batch) - 1,
                    [f"{s:.3f}" for s in scores],
                )
            except Exception as exc:
                logger.warning("Rerank failed for batch %d: %s", batch_start, exc)
                scores = [0.0] * len(batch)

            for i, score in enumerate(scores):
                all_scores.append((batch_start + i, score))

        # Apply threshold
        results = [
            {"index": idx, "relevance_score": score}
            for idx, score in all_scores
            if score >= score_threshold
        ]

        # Fallback: if threshold filtered everything, keep top-5 by score
        if not results and all_scores:
            by_score = sorted(all_scores, key=lambda x: x[1], reverse=True)
            results = [{"index": idx, "relevance_score": score} for idx, score in by_score[:5]]
            logger.info(
                "Rerank: all below threshold %.2f, kept top-%d (best=%.3f)",
                score_threshold,
                len(results),
                results[0]["relevance_score"] if results else 0,
            )

        # Sort by score descending
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        return results or _fallback_ranking(len(documents))

    return rerank_func


def build_api_rerank_func(rc: Any) -> Callable:
    """Build API-based rerank function (cohere, jina, aliyun, azure_cohere)."""
    if rc.strategy == "azure_cohere":
        return _build_azure_cohere_rerank_func(rc)

    from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank

    rerank_functions = {
        "cohere": cohere_rerank,
        "jina": jina_rerank,
        "aliyun": ali_rerank,
    }
    selected_func = rerank_functions[rc.strategy]
    kwargs: dict[str, Any] = {"api_key": rc.api_key, "model": rc.model}
    if rc.base_url:
        kwargs["base_url"] = rc.base_url

    logger.info(
        "Reranker: strategy=%s, model=%s, base_url=%s",
        rc.strategy,
        rc.model,
        rc.base_url or "(provider default)",
    )
    return partial(selected_func, **kwargs)


def _build_azure_cohere_rerank_func(rc: Any) -> Callable:
    """Azure AI Services Cohere reranker via REST API."""
    import httpx

    endpoint = rc.base_url
    api_key = rc.api_key
    deployment = rc.model

    if not endpoint or not api_key:
        raise ValueError("base_url and api_key required for azure_cohere strategy")
    rerank_url = (
        endpoint
        if "/providers/cohere/" in endpoint or endpoint.endswith("/rerank")
        else f"{endpoint}/providers/cohere/v2/rerank"
    )

    async def azure_cohere_rerank(
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            try:
                response = await http_client.post(
                    rerank_url,
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                    json={"model": deployment, "query": query, "documents": documents},
                )
                response.raise_for_status()
                return response.json().get("results", [])
            except httpx.HTTPStatusError as e:
                logger.error("Azure Cohere rerank error: %s", e.response.status_code)
                raise
            except Exception as e:
                logger.exception("Azure Cohere rerank failed: %s", e)
                raise

    return azure_cohere_rerank


def build_rerank_func(config: Any) -> Callable | None:
    """Factory: build reranker from config.

    Returns None if rerank.enabled is False.
    """
    from dlightrag.models.llm import get_ingest_model_func_for_lightrag

    rc = config.rerank
    if not rc.enabled:
        return None

    if rc.strategy == "llm_listwise":
        ingest_func = get_ingest_model_func_for_lightrag(config)
        return build_llm_rerank_func(
            ingest_func,
            config.domain_knowledge_hints,
            batch_size=getattr(rc, "batch_size", _DEFAULT_BATCH_SIZE),
            score_threshold=getattr(rc, "score_threshold", _DEFAULT_SCORE_THRESHOLD),
        )
    if rc.strategy == "vlm_pointwise":
        raise NotImplementedError("vlm_pointwise rerank not yet implemented")

    return build_api_rerank_func(rc)


__all__ = [
    "build_rerank_func",
    "build_llm_rerank_func",
    "build_api_rerank_func",
    "_fallback_ranking",
    "_parse_listwise_scores",
]
