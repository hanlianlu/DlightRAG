"""Model factory functions.

Builds messages-first callables from config with 2-track dispatch:
- provider=openai  -> AsyncOpenAI SDK
- provider=litellm -> LiteLLM

Provides _adapt_for_lightrag() to bridge to LightRAG's (prompt, system_prompt) signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

from openai import AsyncOpenAI

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.completion import _litellm_completion, _openai_completion
from dlightrag.models.embedding import _litellm_embedding, _openai_embedding

logger = logging.getLogger(__name__)

# LightRAG-internal kwargs that must not leak to API calls
_LIGHTRAG_STRIP_KWARGS = {
    "keyword_extraction", "token_tracker",
    "use_azure", "azure_deployment", "api_version",
}


def _adapt_for_lightrag(completion_func: Callable) -> Callable:
    """Wrap messages-first callable for LightRAG compatibility.

    - Converts (prompt, system_prompt=...) to messages array
    - Strips LightRAG-internal kwargs that would cause API errors
    - Handles ``hashing_kv`` for LightRAG's response caching
    """
    from lightrag.utils import compute_args_hash

    async def wrapper(prompt: str, *, system_prompt: str | None = None, **kwargs: Any) -> Any:
        hashing_kv = kwargs.pop("hashing_kv", None)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in _LIGHTRAG_STRIP_KWARGS}

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Check LightRAG cache
        if hashing_kv is not None:
            args_hash = compute_args_hash(
                "", prompt, system_prompt or "", str(clean_kwargs)
            )
            cached = await hashing_kv.get_by_id(args_hash)
            if cached:
                return cached["return"]

        result = await completion_func(messages=messages, **clean_kwargs)

        # Store in LightRAG cache
        if hashing_kv is not None and isinstance(result, str):
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": "", "prompt": prompt}}
            )

        return result

    return wrapper


# ------------------------------------------------------------------ #
# Factory helpers                                                      #
# ------------------------------------------------------------------ #


def _make_completion_func(cfg: ModelConfig, fallback_api_key: str | None = None) -> partial:
    """Build a messages-first completion callable from config."""
    api_key = cfg.api_key or fallback_api_key
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        extra: dict[str, Any] = {**cfg.model_kwargs}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(
            _openai_completion,
            model=cfg.model, api_key=api_key,
            base_url=cfg.base_url, _client=client,
            **extra,
        )
    else:  # litellm
        extra = {**cfg.model_kwargs, "num_retries": cfg.max_retries}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(
            _litellm_completion,
            model=cfg.model, api_key=api_key,
            base_url=cfg.base_url, timeout=cfg.timeout,
            **extra,
        )


# ------------------------------------------------------------------ #
# Public factory functions                                             #
# ------------------------------------------------------------------ #


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

    cfg = config.embedding
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        func = partial(
            _openai_embedding,
            model=cfg.model, api_key=cfg.api_key,
            base_url=cfg.base_url, _client=client,
        )
    else:
        func = partial(
            _litellm_embedding,
            model=cfg.model, api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=func,
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

    if rc.backend == "llm":
        # LLM-based reranking uses the ingest model
        ingest_func = get_ingest_model_func_for_lightrag(config)
        return _build_llm_rerank_func(ingest_func, config)

    # API-based rerankers
    return _build_api_rerank_func(rc)


def _build_llm_rerank_func(ingest_func: Callable, config: DlightragConfig) -> Callable:
    """LLM-based listwise reranker.

    3-layer defense:
    1. response_format=json_object (universal, replaces per-provider kwargs)
    2. Prompt always instructs JSON format
    3. Pydantic model_validate_json() as safety net
    """
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

            # Append any chunks LLM didn't return
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
    """Return original order ranking as fallback."""
    return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(n)]


def _build_api_rerank_func(rc: Any) -> Callable:
    """Build API-based rerank function (cohere, jina, aliyun, azure_cohere)."""
    if rc.backend == "azure_cohere":
        return _build_azure_cohere_rerank_func(rc)

    from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank

    rerank_functions = {
        "cohere": cohere_rerank,
        "jina": jina_rerank,
        "aliyun": ali_rerank,
    }
    selected_func = rerank_functions[rc.backend]
    kwargs: dict[str, Any] = {"api_key": rc.api_key, "model": rc.model}
    if rc.base_url:
        kwargs["base_url"] = rc.base_url

    logger.info(
        "Reranker: backend=%s, model=%s, base_url=%s",
        rc.backend,
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
                logger.error(f"Azure Cohere rerank error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.exception(f"Azure Cohere rerank failed: {e}")
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
