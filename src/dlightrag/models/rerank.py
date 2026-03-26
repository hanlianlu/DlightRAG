# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified reranker strategies.

All reranker implementations live here — LLM listwise, API-based (Cohere, Jina,
Aliyun, Azure Cohere), and the factory function that dispatches based on config.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)


def _fallback_ranking(n: int) -> list[dict[str, float]]:
    """Return original-order ranking as fallback."""
    return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(n)]


def build_llm_rerank_func(
    ingest_func: Callable,
    domain_knowledge_hints: str = "",
) -> Callable:
    """LLM-based listwise reranker.

    Args:
        ingest_func: LightRAG-adapted callable (prompt, system_prompt, ...).
        domain_knowledge_hints: Optional domain context appended to system prompt.
    """
    from dlightrag.models.schemas import RerankResult
    from dlightrag.utils.text import extract_json

    async def rerank_func(
        query: str,
        documents: list[str],
        domain_knowledge: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        effective_domain_knowledge = domain_knowledge or domain_knowledge_hints
        doc_lines = "\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(documents)])

        system_parts = [
            "You are a reranker. Given a query and a list of chunks, rank them by relevance.",
            '\nRespond with JSON only: {"ranked_chunks": [{"index": 0, "relevance_score": 0.95}, ...]}',
        ]
        if effective_domain_knowledge:
            system_parts.append(f"\n{effective_domain_knowledge}")

        user_content = f"Query: {query}\n\nChunks:\n{doc_lines}"

        try:
            result_str = await ingest_func(
                user_content,
                system_prompt="".join(system_parts),
                response_format={"type": "json_object"},
            )
            parsed = RerankResult.model_validate_json(extract_json(result_str))

            seen: set[int] = set()
            results = []
            for chunk in parsed.ranked_chunks:
                if 0 <= chunk.index < len(documents) and chunk.index not in seen:
                    seen.add(chunk.index)
                    results.append({"index": chunk.index, "relevance_score": chunk.relevance_score})

            if len(results) < len(documents):
                min_score = results[-1]["relevance_score"] if results else 0.5
                for idx in range(len(documents)):
                    if idx not in seen:
                        results.append(
                            {"index": idx, "relevance_score": max(0.0, min_score - 0.01)}
                        )

            return results or _fallback_ranking(len(documents))
        except Exception as exc:
            logger.warning("Rerank failed: %s", exc)
            return _fallback_ranking(len(documents))

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
        return build_llm_rerank_func(ingest_func, config.domain_knowledge_hints)
    if rc.strategy == "vlm_pointwise":
        raise NotImplementedError("vlm_pointwise rerank not yet implemented")

    return build_api_rerank_func(rc)


__all__ = [
    "build_rerank_func",
    "build_llm_rerank_func",
    "build_api_rerank_func",
    "_fallback_ranking",
]
