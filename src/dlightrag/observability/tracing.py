# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Neutral observation adapter over the process Langfuse client."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, ExitStack, asynccontextmanager
from types import TracebackType
from typing import Any

from dlightrag_ai.telemetry import Observation

from dlightrag.observability.langfuse import current_client, trace_sensitive_enabled

logger = logging.getLogger(__name__)


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
    """Adapter from neutral telemetry to DlightRAG's Langfuse state."""

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


_USAGE_INPUT_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
_USAGE_OUTPUT_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")
_USAGE_TOTAL_KEYS = ("total_tokens", "total_token_count")
_USAGE_CACHED_INPUT_KEYS = (
    "prompt_tokens_details.cached_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_input_tokens",
    "cached_content_token_count",
)


def _langfuse_usage_details(raw: dict[str, int]) -> dict[str, int]:
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
    """Mark a DlightRAG operation as a Langfuse observation."""
    client = current_client()
    sensitive = trace_sensitive_enabled()
    if client is None:
        yield _ObservationHandle(None)
        return
    observation_kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
    if input is not None and sensitive:
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

            stack.enter_context(propagate_attributes(session_id=session_id))
        observation = stack.enter_context(client.start_as_current_observation(**observation_kwargs))
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
            status = str(caught) if sensitive else "error"
            _safe_update(observation, level="ERROR", status_message=status)
            raise
    finally:
        _exit_observation(stack, exc_type, exc, tb)


__all__ = ["LangfuseTelemetry", "trace_observation"]
