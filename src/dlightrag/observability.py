# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized Langfuse observability — wrappers for LLM, embedding, and reranking.

All Langfuse interaction is contained in this module. The rest of DlightRAG
never imports langfuse directly. When tracing is disabled (default),
every wrapper returns the original function unchanged (zero overhead).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None  # Langfuse client when enabled, None otherwise


def init_tracing(config: Any) -> None:
    """Initialize Langfuse client from DlightragConfig.

    No-op if disabled. Gracefully falls back to disabled if keys are missing
    or client creation fails — never crashes the application.
    """
    global _client
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.info("Langfuse tracing disabled (keys missing in config)")
        return
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        logger.info("Langfuse tracing enabled → %s", config.langfuse_host)
    except Exception:
        _client = None
        logger.warning(
            "Langfuse enabled but initialization failed. Falling back to tracing disabled.",
            exc_info=True,
        )


def shutdown_tracing() -> None:
    """Flush pending events. Safe to call when disabled."""
    if _client is not None:
        _client.flush()


# ---------------------------------------------------------------------------
# Factory wrappers — called once at build time, not per request
# ---------------------------------------------------------------------------


def wrap_chat_func(fn: Callable[..., Any], *, name: str = "llm") -> Callable[..., Any]:
    """Wrap an async LLM completion callable with Langfuse generation tracking."""
    if _client is None:
        return fn

    async def _traced(messages: Any, **kwargs: Any) -> Any:
        # Streaming calls return async generators — can't be wrapped with obs context.
        # Pass through directly; only trace non-streaming (complete) calls.
        if kwargs.get("stream"):
            return await fn(messages, **kwargs)

        try:
            with _client.trace(
                name=name, metadata={k: v for k, v in kwargs.items() if k != "messages"}
            ):
                result = await fn(messages, **kwargs)
                return result
        except Exception:
            logger.debug("Langfuse tracing error (non-fatal), falling back", exc_info=True)
            return await fn(messages, **kwargs)

    return _traced


def wrap_embedding_func(fn: Callable[..., Any], *, name: str = "embedding") -> Callable[..., Any]:
    """Wrap an async embedding callable."""
    if _client is None:
        return fn

    async def _traced(inputs: Any, **kwargs: Any) -> Any:
        try:
            with _client.trace(name=name, metadata=kwargs):
                return await fn(inputs, **kwargs)
        except Exception:
            logger.debug("Langfuse tracing error (non-fatal), falling back", exc_info=True)
            return await fn(inputs, **kwargs)

    return _traced


def wrap_rerank_func(fn: Callable[..., Any], *, name: str = "reranking") -> Callable[..., Any]:
    """Wrap an async reranker callable."""
    if _client is None:
        return fn

    async def _traced(query: str, chunks: list[Any], top_k: int, **kwargs: Any) -> Any:
        try:
            with _client.trace(
                name=name,
                metadata={"query": query[:200], "chunk_count": len(chunks), "top_k": top_k},
            ):
                return await fn(query, chunks, top_k, **kwargs)
        except Exception:
            logger.debug("Langfuse tracing error (non-fatal), falling back", exc_info=True)
            return await fn(query, chunks, top_k, **kwargs)

    return _traced


@asynccontextmanager
async def trace_pipeline(name: str, **metadata: Any) -> AsyncIterator[None]:
    """Mark a pipeline stage as a Langfuse span. Nested calls auto-link as children."""
    if _client is None:
        yield
        return
    try:
        with _client.trace(name=name, metadata=metadata):
            yield
    except Exception:
        logger.debug("Langfuse pipeline trace error (non-fatal)", exc_info=True)
        yield
