# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Root Langfuse adapter for neutral telemetry observations.

All Langfuse interaction is contained here. Core packages receive the neutral
``Telemetry`` protocol, while product orchestration may also open explicit
request and pipeline observations through ``trace_observation``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, ExitStack, asynccontextmanager
from types import TracebackType
from typing import Any

from dlightrag_ai.telemetry import Observation, bounded_telemetry_text

logger = logging.getLogger(__name__)

_client: Any = None  # Langfuse client when enabled, None otherwise
_trace_sensitive: bool = True  # Attach sensitive data (query, error text, raw IDs) to traces
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
    global _client, _trace_sensitive
    _trace_sensitive = bool(getattr(config, "langfuse_trace_sensitive_data", True))
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


def trace_sensitive_enabled() -> bool:
    """Whether sensitive request data may be attached to traces (config switch)."""
    return _trace_sensitive


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
        return bounded_telemetry_text(data)
    return data


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
            usage_details = kwargs.pop("usage_details", None)
            cost_details = kwargs.pop("cost_details", None)
            kwargs.update(_usage_cost_update(usage_details, cost_details))
            _safe_update(self._observation, **kwargs)


class LangfuseTelemetry:
    """Root adapter from neutral telemetry to DlightRAG's Langfuse state."""

    @property
    def capture_sensitive_data(self) -> bool:
        return trace_sensitive_enabled()

    def observe(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: Any | None = None,
        session_id: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Observation]:
        return trace_observation(
            name,
            as_type=as_type,
            input=input,
            metadata=metadata,
            session_id=session_id,
            model=model,
            model_parameters=model_parameters,
        )


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


@asynccontextmanager
async def trace_observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    metadata: Any | None = None,
    session_id: str | None = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> AsyncIterator[_ObservationHandle]:
    """Mark a DlightRAG operation as a Langfuse v4 observation."""
    if _client is None:
        yield _ObservationHandle(None)
        return
    observation_kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
    if input is not None and _trace_sensitive:
        observation_kwargs["input"] = input
    if metadata is not None:
        observation_kwargs["metadata"] = metadata
    if model is not None:
        observation_kwargs["model"] = model
    if model_parameters is not None:
        observation_kwargs["model_parameters"] = model_parameters
    stack = ExitStack()
    try:
        if session_id is not None:
            from langfuse import propagate_attributes

            # Entered first: Langfuse only propagates to spans opened afterwards.
            stack.enter_context(propagate_attributes(session_id=session_id))
        observation = stack.enter_context(
            _client.start_as_current_observation(**observation_kwargs)
        )
    except Exception:
        stack.close()
        logger.debug("Langfuse observation start failed (non-fatal)", exc_info=True)
        yield _ObservationHandle(None)
        return

    exc_type: type[BaseException] | None = None
    exc: BaseException | None = None
    tb: TracebackType | None = None
    try:
        try:
            yield _ObservationHandle(observation)
        except asyncio.CancelledError, GeneratorExit:
            raise
        except BaseException as caught:
            exc_type = type(caught)
            exc = caught
            tb = caught.__traceback__
            status = str(caught) if _trace_sensitive else "error"
            _safe_update(observation, level="ERROR", status_message=status)
            raise
    finally:
        _exit_observation(stack, exc_type, exc, tb)
