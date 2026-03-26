# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model factory functions.

Uses the provider registry (openai, anthropic, gemini) to build callables.
Provides _adapt_for_lightrag() to bridge to LightRAG's (prompt, system_prompt) signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.providers import get_provider

logger = logging.getLogger(__name__)


def _adapt_for_lightrag(completion_func: Callable) -> Callable:
    """Wrap messages-first callable for LightRAG's (prompt, system_prompt, ...) signature."""

    async def wrapper(
        prompt: str,
        *,
        system_prompt: str | None = None,
        hashing_kv: Any = None,  # noqa: ARG001
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,  # noqa: ARG001
        enable_cot: bool = False,  # noqa: ARG001
        token_tracker: Any = None,  # noqa: ARG001
        use_azure: bool = False,  # noqa: ARG001
        azure_deployment: str | None = None,  # noqa: ARG001
        api_version: str | None = None,  # noqa: ARG001
        **kwargs: Any,
    ) -> Any:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        return await completion_func(messages=messages, **kwargs)

    return wrapper


def _make_completion_func(cfg: ModelConfig, fallback_api_key: str | None = None) -> partial:
    """Build a messages-first completion callable from config."""
    api_key = cfg.api_key or fallback_api_key
    provider = get_provider(
        cfg.provider,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )

    async def completion_wrapper(messages: list[dict[str, Any]], **kw: Any) -> str:
        return await provider.complete(
            messages=messages,
            model=cfg.model,
            temperature=cfg.temperature,
            model_kwargs={**cfg.model_kwargs, **kw},
        )

    return partial(completion_wrapper)


def get_chat_model_func(config: DlightragConfig) -> Callable:
    """Messages-first chat callable (for DlightRAG direct use)."""
    return _make_completion_func(config.chat)


def get_chat_model_func_for_lightrag(config: DlightragConfig) -> Callable:
    """LightRAG-compatible chat callable with adapter."""
    return _adapt_for_lightrag(get_chat_model_func(config))


def get_ingest_model_func(config: DlightragConfig) -> Callable:
    """Messages-first ingest callable, fallback to chat."""
    cfg = config.ingest or config.chat
    fallback_key = config.chat.api_key
    return _make_completion_func(cfg, fallback_api_key=fallback_key)


def get_ingest_model_func_for_lightrag(config: DlightragConfig) -> Callable:
    """LightRAG-compatible ingest callable with adapter."""
    return _adapt_for_lightrag(get_ingest_model_func(config))


def get_embedding_func(config: DlightragConfig) -> Any:
    """Build LightRAG EmbeddingFunc from config."""
    from lightrag.utils import EmbeddingFunc

    from dlightrag.models.embedding import httpx_embed
    from dlightrag.models.providers.embed_providers import detect_embed_provider

    cfg = config.embedding
    embed_provider = detect_embed_provider(
        model=cfg.model,
        base_url=cfg.base_url,
    )

    async def embed_func(texts: list[str]) -> list[list[float]]:
        return await httpx_embed(
            texts,
            model=cfg.model,
            api_key=cfg.api_key or "",
            base_url=cfg.base_url or "",
            provider=embed_provider,
        )

    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=embed_func,
        model_name=cfg.model,
    )


def get_rerank_func(config: DlightragConfig) -> Callable | None:
    """Build rerank callable.

    LLM backend uses ingest model (cheaper). API-based backends
    (cohere, jina, aliyun, azure_cohere) use their dedicated clients.
    """
    rc = config.rerank
    if not rc.enabled:
        return None

    if rc.strategy == "llm_listwise":
        ingest_func = get_ingest_model_func_for_lightrag(config)
        return _build_llm_rerank_func(ingest_func, config)
    if rc.strategy == "vlm_pointwise":
        raise NotImplementedError("vlm_pointwise rerank not yet implemented")

    return _build_api_rerank_func(rc)


def _build_llm_rerank_func(ingest_func: Callable, config: DlightragConfig) -> Callable:
    """LLM-based listwise reranker."""
    from dlightrag.models.schemas import RerankResult
    from dlightrag.utils.text import extract_json

    default_domain_knowledge = config.domain_knowledge_hints

    async def rerank_func(
        query: str,
        documents: list[str],
        domain_knowledge: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        effective_domain_knowledge = domain_knowledge or default_domain_knowledge
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


def _fallback_ranking(n: int) -> list[dict[str, float]]:
    return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(n)]


def _build_api_rerank_func(rc: Any) -> Callable:
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
        raise ValueError("base_url and api_key required for azure_cohere backend")
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


__all__ = [
    "get_chat_model_func",
    "get_chat_model_func_for_lightrag",
    "get_embedding_func",
    "get_ingest_model_func",
    "get_ingest_model_func_for_lightrag",
    "get_rerank_func",
]
