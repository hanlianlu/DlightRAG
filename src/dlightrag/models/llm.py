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

    async def embed_func(texts: list[str]) -> np.ndarray:
        result = await httpx_embed(
            texts,
            model=cfg.model,
            api_key=cfg.api_key or "",
            base_url=cfg.base_url or "",
            provider=embed_provider,
        )
        return np.array(result)

    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=embed_func,
        model_name=cfg.model,
    )


def get_rerank_func(config: DlightragConfig) -> Callable | None:
    """Build rerank callable. Delegates to models.rerank.build_rerank_func."""
    from dlightrag.models.rerank import build_rerank_func

    return build_rerank_func(config)


__all__ = [
    "get_chat_model_func",
    "get_chat_model_func_for_lightrag",
    "get_embedding_func",
    "get_ingest_model_func",
    "get_ingest_model_func_for_lightrag",
    "get_rerank_func",
]
