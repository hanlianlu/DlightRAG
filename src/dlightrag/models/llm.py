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

import numpy as np

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

    async def completion_wrapper(messages: list[dict[str, Any]], **kw: Any) -> Any:
        stream = kw.pop("stream", False)
        model_kwargs = {**cfg.model_kwargs, **kw}
        if stream:
            return provider.stream(
                messages=messages,
                model=cfg.model,
                temperature=cfg.temperature,
                model_kwargs=model_kwargs,
            )
        return await provider.complete(
            messages=messages,
            model=cfg.model,
            temperature=cfg.temperature,
            model_kwargs=model_kwargs,
        )

    from dlightrag.observability import wrap_chat_func

    traced_func = wrap_chat_func(completion_wrapper, name=f"llm_{cfg.model}")

    return partial(traced_func)


def get_chat_model_func(config: DlightragConfig) -> Callable:
    """Messages-first chat callable (for DlightRAG direct use)."""
    return _make_completion_func(config.chat)


def get_ingest_model_func(config: DlightragConfig) -> Callable:
    """Messages-first ingest callable, fallback to chat."""
    cfg = config.ingest or config.chat
    fallback_key = config.chat.api_key
    return _make_completion_func(cfg, fallback_api_key=fallback_key)


def get_query_model_func(config: DlightragConfig) -> Callable:
    """Messages-first query callable for AnswerEngine and QueryPlanner.

    Uses ``config.query`` if set, otherwise falls back to ``config.chat``.
    """
    cfg = config.query or config.chat
    fallback_key = config.chat.api_key
    return _make_completion_func(cfg, fallback_api_key=fallback_key)


def get_vlm_model_func(config: DlightragConfig) -> Callable:
    """Messages-first VLM callable for vlm_parser, multimodal_query, and unified extractor.

    Uses ``config.vlm`` if set, otherwise falls back to ``config.chat`` (which must
    be vision-capable). Centralises the VLM choice so users can pin a single
    vision-strong model for all DlightRAG visual paths without affecting text
    LLM roles.
    """
    cfg = config.vlm or config.chat
    fallback_key = config.chat.api_key
    return _make_completion_func(cfg, fallback_api_key=fallback_key)


def _lightrag_adapted(
    factory: Callable[[DlightragConfig], Callable],
) -> Callable[[DlightragConfig], Callable]:
    """Wrap a messages-first model factory to produce a LightRAG-compatible callable."""

    def wrapper(config: DlightragConfig) -> Callable:
        return _adapt_for_lightrag(factory(config))

    wrapper.__doc__ = f"LightRAG-adapted version of {factory.__name__}."
    wrapper.__name__ = f"{factory.__name__}_for_lightrag"
    return wrapper


get_chat_model_func_for_lightrag = _lightrag_adapted(get_chat_model_func)


# DlightRAG config field name → upstream LightRAG role name (1.5.0 ROLES
# registry uses singular ``keyword``; we surface ``keywords`` in DlightRAG
# config to match the broader codebase convention).
_DLIGHTRAG_TO_LIGHTRAG_ROLE: tuple[tuple[str, str], ...] = (
    ("extract", "extract"),
    ("keywords", "keyword"),
    ("query", "query"),
)


def build_role_llm_configs(config: DlightragConfig) -> dict[str, Any] | None:
    """Build LightRAG ``role_llm_configs`` dict from per-role ModelConfig fields.

    Each DlightRAG config field listed in ``_DLIGHTRAG_TO_LIGHTRAG_ROLE`` is
    translated into a ``RoleLLMConfig`` if set. Unset fields are *omitted*
    from the returned dict — do NOT insert ``RoleLLMConfig(func=None, ...)``
    here, as LightRAG's resolver treats an explicit ``func=None`` differently
    from a missing entry. When no overrides are configured, this function
    returns ``None`` so the LightRAG constructor skips role registration
    entirely.
    """
    from lightrag import RoleLLMConfig

    overrides: dict[str, Any] = {}
    for dlightrag_field, lightrag_role in _DLIGHTRAG_TO_LIGHTRAG_ROLE:
        role_cfg: ModelConfig | None = getattr(config, dlightrag_field)
        if role_cfg is None:
            continue
        completion = _make_completion_func(role_cfg, fallback_api_key=config.chat.api_key)
        overrides[lightrag_role] = RoleLLMConfig(
            func=_adapt_for_lightrag(completion),
            timeout=int(role_cfg.timeout),
        )

    return overrides or None


def get_embedding_func(config: DlightragConfig) -> Any:
    """Build LightRAG EmbeddingFunc from config.

    Uses a shared httpx client for connection pooling across requests.
    """
    from lightrag.utils import EmbeddingFunc

    from dlightrag.models.embedding import create_embed_client, httpx_embed
    from dlightrag.models.providers.embed_providers import detect_embed_provider

    cfg = config.embedding
    embed_provider = detect_embed_provider(
        model=cfg.model,
        base_url=cfg.base_url,
    )
    client = create_embed_client(cfg.api_key or "", timeout=float(config.embedding_request_timeout))

    async def embed_func(texts: list[str]) -> np.ndarray:
        result = await httpx_embed(
            texts,
            model=cfg.model,
            api_key=cfg.api_key or "",
            base_url=cfg.base_url or "",
            provider=embed_provider,
            client=client,
        )
        # LightRAG's EmbeddingFunc validates via result.size — requires numpy
        return np.array(result)

    from dlightrag.observability import wrap_embedding_func

    traced_embed_func = wrap_embedding_func(embed_func, name=f"embed_{cfg.model}")

    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=traced_embed_func,
        model_name=cfg.model,
    )


def get_rerank_func(config: DlightragConfig) -> Callable | None:
    """Build multimodal rerank callable from config.

    For chat_llm_reranker: uses an independent rerank model if provider/model
    are set in rerank config, otherwise falls back to chat. In unified mode,
    whichever callable is selected must support image inputs because chunks
    may include page images.
    """
    from dlightrag.models.rerank import build_rerank_func

    rc = config.rerank
    if not rc.enabled:
        return None

    # For chat_llm_reranker: build scoring callable (messages-first, NOT LightRAG-adapted)
    if rc.strategy == "chat_llm_reranker":
        if rc.provider and rc.model:
            # Independent model for reranking
            scoring_func = _make_completion_func(
                ModelConfig(
                    provider=rc.provider,
                    model=rc.model,
                    api_key=rc.api_key,
                    base_url=rc.base_url,
                    temperature=rc.temperature or 0.0,
                    model_kwargs=rc.model_kwargs,
                ),
                fallback_api_key=config.chat.api_key,
            )
        else:
            # Fall back to chat. Do not implicitly consume config.vlm here:
            # reranker-specific model choices belong under rerank.*.
            scoring_func = get_chat_model_func(config)
        return build_rerank_func(rc, ingest_func=scoring_func)

    return build_rerank_func(rc)


__all__ = [
    "build_role_llm_configs",
    "get_chat_model_func",
    "get_chat_model_func_for_lightrag",
    "get_embedding_func",
    "get_ingest_model_func",
    "get_query_model_func",
    "get_rerank_func",
    "get_vlm_model_func",
]
