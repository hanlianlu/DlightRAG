# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized Langfuse observability — wrappers for LLM, embedding, and reranking.

All Langfuse interaction is contained in this module. The rest of DlightRAG
never imports langfuse directly. When tracing is disabled (default),
every wrapper returns the original function unchanged (zero overhead).
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

try:
    from opentelemetry import context as _otel_context_api
except ImportError:  # pragma: no cover - opentelemetry ships with langfuse
    _otel_context_api = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_client: Any = None  # Langfuse client when enabled, None otherwise
_LANGFUSE_TRACER_SCOPE = "langfuse-sdk"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "secret",
    "password",
    "token",
    "authorization",
    "connection_string",
    "account_key",
    "sas_token",
)


def init_tracing(config: Any) -> None:
    """Initialize Langfuse client from DlightragConfig.

    No-op if disabled. Langfuse's SDK performs export asynchronously; DlightRAG
    avoids calling the SDK's blocking ``auth_check()`` in production startup.
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
            "mask": _mask_langfuse_payload,
        }
        optional_kwargs = {
            "environment": getattr(config, "langfuse_environment", None),
            "release": getattr(config, "langfuse_release", None),
            "sample_rate": getattr(config, "langfuse_sample_rate", None),
            "timeout": getattr(config, "langfuse_timeout", None),
            "flush_at": getattr(config, "langfuse_flush_at", None),
            "flush_interval": getattr(config, "langfuse_flush_interval", None),
        }
        kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})
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
    """Flush pending events and stop SDK background resources."""
    global _client
    client = _client
    _client = None
    if client is None:
        return
    try:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        logger.debug("Langfuse shutdown failed (non-fatal)", exc_info=True)


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


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _mask_langfuse_payload(data: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """SDK-level Langfuse mask for secrets, large text, and inline media."""
    if isinstance(data, dict):
        if data.get("type") == "image_url":
            return {"type": "image_url", "image_url": "[image omitted]"}
        return {
            key: "[redacted]" if _is_sensitive_key(str(key)) else _mask_langfuse_payload(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_mask_langfuse_payload(item) for item in data]
    if isinstance(data, tuple):
        return [_mask_langfuse_payload(item) for item in data]
    if isinstance(data, bytes):
        return f"[bytes omitted: {len(data)}]"
    if isinstance(data, str):
        return _truncate_text(data)
    return data


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


class _ObservationHandle:
    def __init__(self, observation: Any | None) -> None:
        self._observation = observation

    def update(self, **kwargs: Any) -> None:
        if self._observation is not None:
            _safe_update(self._observation, **kwargs)


# Provider usage key synonyms → Langfuse canonical usage types. Langfuse sums
# every usageDetails value into `total` unless `total` is provided, so forwarding
# raw provider usage (which mixes component counters, an aggregate, and cache
# breakdowns) triple-counts tokens. Normalize to input/output/total.
_USAGE_INPUT_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
_USAGE_OUTPUT_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")
_USAGE_TOTAL_KEYS = ("total_tokens", "total_token_count")
_USAGE_CACHED_INPUT_KEYS = (
    "prompt_tokens_details.cached_tokens",  # OpenAI
    "prompt_cache_hit_tokens",  # DeepSeek
    "cache_read_input_tokens",  # Anthropic
    "cached_content_token_count",  # Gemini
)


def _langfuse_usage_details(raw: dict[str, int]) -> dict[str, int]:
    """Map provider token usage to Langfuse canonical usage types.

    Langfuse sums every usageDetails value into ``total`` unless ``total`` is
    provided, so an explicit ``total`` is always emitted when known to avoid
    triple-counting. Cached input tokens are surfaced as ``input_cached_tokens``
    (informational; reported cost comes from cost_details or a model price).
    """

    def _first(keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    inp = _first(_USAGE_INPUT_KEYS)
    out = _first(_USAGE_OUTPUT_KEYS)
    total = _first(_USAGE_TOTAL_KEYS)
    cached = _first(_USAGE_CACHED_INPUT_KEYS)
    if total is None and (inp is not None or out is not None):
        total = (inp or 0) + (out or 0)

    details: dict[str, int] = {}
    if inp is not None:
        details["input"] = inp
    if out is not None:
        details["output"] = out
    if total is not None:
        details["total"] = total
    if cached:
        details["input_cached_tokens"] = cached
    return details or raw


def _usage_cost_update(
    usage_details: dict[str, int] | None,
    cost_details: dict[str, float] | None,
) -> dict[str, Any]:
    """Normalized usage/cost fields for a Langfuse observation update."""
    update: dict[str, Any] = {}
    if usage_details:
        update["usage_details"] = _langfuse_usage_details(usage_details)
    if cost_details:
        update["cost_details"] = cost_details
    return update


def _observation_update_kwargs(
    result: Any,
    *,
    output_builder: Callable[[Any], Any],
) -> dict[str, Any]:
    update: dict[str, Any] = {"output": output_builder(result)}
    update.update(
        _usage_cost_update(
            getattr(result, "usage_details", None),
            getattr(result, "cost_details", None),
        )
    )
    return update


def _attach_stream_usage(observation: Any, usage_holder: dict[str, Any]) -> None:
    """Attach usage/cost the provider recorded into the per-call holder.

    Streaming yields text only, so usage (when the provider/model reports it)
    arrives out of band in ``usage_holder``, populated during iteration.
    """
    update = _usage_cost_update(usage_holder.get("usage_details"), usage_holder.get("cost_details"))
    if update:
        _safe_update(observation, **update)


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


_CONTEXT_CARRIER_KEY = "__langfuse_otel_context__"


def _attach_context(otel_context: Any) -> Any:
    """Make a captured OTEL context current; return a detach token (or None)."""
    if otel_context is None or _otel_context_api is None:
        return None
    return _otel_context_api.attach(otel_context)


def _detach_context(token: Any) -> None:
    """Detach an OTEL token, tolerating async cross-task teardown (non-fatal)."""
    if token is None or _otel_context_api is None:
        return
    try:
        _otel_context_api.detach(token)
    except Exception:
        logger.debug("Langfuse context detach mismatch (non-fatal)", exc_info=True)


async def _run_with_observation(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    observation_kwargs: dict[str, Any],
    output_builder: Callable[[Any], Any] = _safe_output,
    otel_context: Any = None,
) -> Any:
    """Run an async callable inside a Langfuse observation without retrying it."""
    if _client is None:
        return await fn(*args, **kwargs)

    token = _attach_context(otel_context)
    try:
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
            _safe_update(
                observation,
                **_observation_update_kwargs(result, output_builder=output_builder),
            )
            return result
        finally:
            _exit_observation(cm, exc_type, exc, tb)
    finally:
        _detach_context(token)


async def _stream_with_observation(
    fn: Callable[..., Any],
    messages: Any,
    kwargs: dict[str, Any],
    *,
    observation_kwargs: dict[str, Any],
    otel_context: Any = None,
) -> AsyncIterator[str]:
    # Fresh per-call holder: the provider records streaming usage into it, so
    # there is no shared provider state (safe under concurrency).
    usage_holder: dict[str, Any] = {}
    call_kwargs = {**kwargs, "usage_holder": usage_holder}
    if _client is None:
        stream = await fn(messages, **call_kwargs)
        async for chunk in stream:
            yield chunk
        return

    token = _attach_context(otel_context)
    try:
        try:
            cm = _client.start_as_current_observation(**observation_kwargs)
            observation = cm.__enter__()
        except Exception:
            logger.debug("Langfuse streaming observation start failed (non-fatal)", exc_info=True)
            stream = await fn(messages, **call_kwargs)
            async for chunk in stream:
                yield chunk
            return

        chunks: list[str] = []
        exc_type: type[BaseException] | None = None
        exc: BaseException | None = None
        tb: TracebackType | None = None
        try:
            try:
                stream = await fn(messages, **call_kwargs)
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
            _attach_stream_usage(observation, usage_holder)
        finally:
            _exit_observation(cm, exc_type, exc, tb)
    finally:
        _detach_context(token)


# ---------------------------------------------------------------------------
# Factory wrappers — called once at build time, not per request
# ---------------------------------------------------------------------------


def capture_context() -> Any | None:
    """Snapshot the active OpenTelemetry context for restoring in a detached task.

    Used to nest post-response work (e.g. streamed-answer semantic highlights,
    which runs after the pipeline span has closed) back under the request trace.
    ``None`` when tracing (or its OpenTelemetry dependency) is unavailable, so
    callers stay a no-op without Langfuse.
    """
    if _client is None or _otel_context_api is None:
        return None
    return _otel_context_api.get_current()


def bind_trace_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Restore the caller's tracing context when *fn* later runs detached.

    DlightRAG runs its LLM calls in LightRAG's persistent worker-pool queues,
    whose workers do not inherit the request's async context. Without this, a
    generation observation created in a worker (or in the SSE task that consumes
    a streamed answer) would detach into its own Langfuse trace. This captures
    the active OpenTelemetry context at call time and carries it to the
    observation, which re-attaches it so the generation nests under the request
    trace as a true child. No-op when tracing (or its OpenTelemetry dependency)
    is unavailable, so callers work unchanged when Langfuse is absent.
    """
    if _client is None or _otel_context_api is None:
        return fn

    async def _bound(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault(_CONTEXT_CARRIER_KEY, capture_context())
        return await fn(*args, **kwargs)

    return _bound


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
        otel_context = kwargs.pop(_CONTEXT_CARRIER_KEY, None)
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
                otel_context=otel_context,
            )

        return await _run_with_observation(
            fn,
            (messages,),
            kwargs,
            observation_kwargs=observation_kwargs,
            otel_context=otel_context,
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
async def trace_observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    metadata: Any | None = None,
    parent_context: Any = None,
) -> AsyncIterator[_ObservationHandle]:
    """Mark a DlightRAG operation as a Langfuse v4 observation.

    ``parent_context`` optionally re-attaches a context captured via
    :func:`capture_context` so the observation nests under a request trace even
    when it runs in a detached task (e.g. streamed-answer post-processing after
    the pipeline span has already closed).
    """
    if _client is None:
        yield _ObservationHandle(None)
        return
    observation_kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
    if input is not None:
        observation_kwargs["input"] = input
    if metadata is not None:
        observation_kwargs["metadata"] = metadata
    token = _attach_context(parent_context)
    try:
        try:
            cm = _client.start_as_current_observation(**observation_kwargs)
            observation = cm.__enter__()
        except Exception:
            logger.debug("Langfuse observation start failed (non-fatal)", exc_info=True)
            yield _ObservationHandle(None)
            return

        exc_type: type[BaseException] | None = None
        exc: BaseException | None = None
        tb: TracebackType | None = None
        try:
            try:
                yield _ObservationHandle(observation)
            except BaseException as caught:
                exc_type = type(caught)
                exc = caught
                tb = caught.__traceback__
                _safe_update(observation, level="ERROR", status_message=str(caught))
                raise
        finally:
            _exit_observation(cm, exc_type, exc, tb)
    finally:
        _detach_context(token)
