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
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None  # Langfuse client when enabled, None otherwise
_LANGFUSE_TRACER_SCOPE = "langfuse-sdk"


def init_tracing(config: Any) -> None:
    """Initialize Langfuse client from DlightragConfig.

    No-op if disabled. Gracefully falls back to disabled if keys are missing
    or client creation fails — never crashes the application.
    """
    global _client
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        _client = None
        logger.info("Langfuse tracing disabled (keys missing in config)")
        return
    try:
        from langfuse import Langfuse

        kwargs: dict[str, Any] = {
            "public_key": config.langfuse_public_key,
            "secret_key": config.langfuse_secret_key,
            "base_url": config.langfuse_host,
        }
        if not getattr(config, "langfuse_export_external_spans", False):
            kwargs["should_export_span"] = _is_dlight_observation_span

        _client = Langfuse(**kwargs)
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


def _is_dlight_observation_span(span: Any) -> bool:
    """Export only DlightRAG-created Langfuse observations by default.

    Langfuse v4 can also export GenAI/LLM spans from third-party OTEL
    instrumentation. DlightRAG manually records model calls, so the default is
    to avoid external spans that can double-count LLM calls.
    """
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == _LANGFUSE_TRACER_SCOPE


def _truncate_text(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def _summarize_content(content: Any) -> Any:
    if isinstance(content, str):
        return _truncate_text(content)
    if not isinstance(content, list):
        return content

    summarized: list[Any] = []
    for block in content:
        if isinstance(block, str):
            summarized.append(_truncate_text(block))
        elif isinstance(block, dict) and block.get("type") == "text":
            summarized.append({**block, "text": _truncate_text(str(block.get("text", "")))})
        elif isinstance(block, dict) and block.get("type") == "image_url":
            summarized.append({"type": "image_url", "image_url": "[image omitted]"})
        else:
            summarized.append(block)
    return summarized


def _summarize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    return [
        {
            **msg,
            "content": _summarize_content(msg.get("content")),
        }
        if isinstance(msg, dict)
        else msg
        for msg in messages
    ]


def _safe_output(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    return value


def _embedding_output_summary(result: Any) -> dict[str, int]:
    if hasattr(result, "shape"):
        shape = result.shape
        return {"embedding_count": int(shape[0]) if shape else 0}
    try:
        return {"embedding_count": len(result)}
    except TypeError:
        return {"embedding_count": 1}


def _safe_update(observation: Any, **kwargs: Any) -> None:
    try:
        observation.update(**kwargs)
    except Exception:
        logger.debug("Langfuse observation update failed (non-fatal)", exc_info=True)


def _exit_observation(
    cm: Any,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: TracebackType | None,
) -> None:
    try:
        cm.__exit__(exc_type, exc, tb)
    except Exception:
        logger.debug("Langfuse observation close failed (non-fatal)", exc_info=True)


async def _run_with_observation(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    observation_kwargs: dict[str, Any],
    output_builder: Callable[[Any], Any] = _safe_output,
) -> Any:
    """Run an async callable inside a Langfuse observation without retrying it."""
    if _client is None:
        return await fn(*args, **kwargs)

    try:
        cm = _client.start_as_current_observation(**observation_kwargs)
        observation = cm.__enter__()
    except Exception:
        logger.debug("Langfuse observation start failed (non-fatal)", exc_info=True)
        return await fn(*args, **kwargs)

    exc_type: type[BaseException] | None = None
    exc: BaseException | None = None
    tb: TracebackType | None = None
    try:
        try:
            result = await fn(*args, **kwargs)
        except BaseException as caught:
            exc_type = type(caught)
            exc = caught
            tb = caught.__traceback__
            _safe_update(observation, level="ERROR", status_message=str(caught))
            raise
        _safe_update(observation, output=output_builder(result))
        return result
    finally:
        _exit_observation(cm, exc_type, exc, tb)


async def _stream_with_observation(
    fn: Callable[..., Any],
    messages: Any,
    kwargs: dict[str, Any],
    *,
    observation_kwargs: dict[str, Any],
) -> AsyncIterator[str]:
    if _client is None:
        stream = await fn(messages, **kwargs)
        async for chunk in stream:
            yield chunk
        return

    try:
        cm = _client.start_as_current_observation(**observation_kwargs)
        observation = cm.__enter__()
    except Exception:
        logger.debug("Langfuse streaming observation start failed (non-fatal)", exc_info=True)
        stream = await fn(messages, **kwargs)
        async for chunk in stream:
            yield chunk
        return

    chunks: list[str] = []
    exc_type: type[BaseException] | None = None
    exc: BaseException | None = None
    tb: TracebackType | None = None
    try:
        try:
            stream = await fn(messages, **kwargs)
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except BaseException as caught:
            exc_type = type(caught)
            exc = caught
            tb = caught.__traceback__
            _safe_update(observation, level="ERROR", status_message=str(caught))
            raise
        _safe_update(observation, output=_truncate_text("".join(chunks)))
    finally:
        _exit_observation(cm, exc_type, exc, tb)


# ---------------------------------------------------------------------------
# Factory wrappers — called once at build time, not per request
# ---------------------------------------------------------------------------


def wrap_chat_func(
    fn: Callable[..., Any],
    *,
    name: str = "llm",
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Wrap an async LLM completion callable with Langfuse generation tracking."""
    if _client is None:
        return fn

    async def _traced(messages: Any, **kwargs: Any) -> Any:
        metadata = {k: v for k, v in kwargs.items() if k not in {"messages", "stream"}}
        observation_kwargs: dict[str, Any] = {
            "as_type": "generation",
            "name": name,
            "input": _summarize_messages(messages),
            "metadata": metadata,
        }
        if model is not None:
            observation_kwargs["model"] = model
        merged_model_parameters = {
            **(model_parameters or {}),
            **{k: v for k, v in metadata.items() if isinstance(v, str | int | float | bool)},
        }
        if merged_model_parameters:
            observation_kwargs["model_parameters"] = merged_model_parameters

        if kwargs.get("stream"):
            return _stream_with_observation(
                fn,
                messages,
                kwargs,
                observation_kwargs=observation_kwargs,
            )

        return await _run_with_observation(
            fn,
            (messages,),
            kwargs,
            observation_kwargs=observation_kwargs,
        )

    return _traced


def wrap_embedding_func(fn: Callable[..., Any], *, name: str = "embedding") -> Callable[..., Any]:
    """Wrap an async embedding callable."""
    if _client is None:
        return fn

    async def _traced(inputs: Any, **kwargs: Any) -> Any:
        metadata = {**kwargs}
        if isinstance(inputs, list):
            metadata["input_count"] = len(inputs)
        observation_kwargs = {
            "as_type": "embedding",
            "name": name,
            "input": {"input_count": len(inputs)} if isinstance(inputs, list) else None,
            "metadata": metadata,
        }
        return await _run_with_observation(
            fn,
            (inputs,),
            kwargs,
            observation_kwargs=observation_kwargs,
            output_builder=_embedding_output_summary,
        )

    return _traced


def wrap_rerank_func(fn: Callable[..., Any], *, name: str = "reranking") -> Callable[..., Any]:
    """Wrap an async reranker callable."""
    if _client is None:
        return fn

    async def _traced(query: str, chunks: list[Any], top_k: int, **kwargs: Any) -> Any:
        observation_kwargs = {
            "as_type": "span",
            "name": name,
            "input": {"query": _truncate_text(query, limit=1000)},
            "metadata": {"query": query[:200], "chunk_count": len(chunks), "top_k": top_k},
        }
        return await _run_with_observation(
            fn,
            (query, chunks, top_k),
            kwargs,
            observation_kwargs=observation_kwargs,
        )

    return _traced


@asynccontextmanager
async def trace_pipeline(name: str, **metadata: Any) -> AsyncIterator[None]:
    """Mark a pipeline stage as a Langfuse span. Nested calls auto-link as children."""
    if _client is None:
        yield
        return
    as_type = "retriever" if "retrieve" in name else "chain" if "pipeline" in name else "span"
    try:
        cm = _client.start_as_current_observation(
            as_type=as_type,
            name=name,
            metadata=metadata,
        )
        observation = cm.__enter__()
    except Exception:
        logger.debug("Langfuse pipeline observation start failed (non-fatal)", exc_info=True)
        yield
        return

    exc_type: type[BaseException] | None = None
    exc: BaseException | None = None
    tb: TracebackType | None = None
    try:
        try:
            yield
        except BaseException as caught:
            exc_type = type(caught)
            exc = caught
            tb = caught.__traceback__
            _safe_update(observation, level="ERROR", status_message=str(caught))
            raise
    finally:
        _exit_observation(cm, exc_type, exc, tb)
