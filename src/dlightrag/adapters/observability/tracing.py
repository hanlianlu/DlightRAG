# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Neutral observation adapter over the process Langfuse client."""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
)
from types import TracebackType
from typing import Any

from dlightrag.adapters.observability.langfuse import (
    current_client,
    trace_attributes,
    trace_sensitive_enabled,
)
from dlightrag.engine.ai.providers.base import (
    provider_cache_hit_tokens,
    provider_input_tokens,
    provider_output_tokens,
    provider_total_tokens,
)
from dlightrag.engine.ai.telemetry import SPAN_TYPES, Observation, SpanName, SpanType

logger = logging.getLogger(__name__)


def _safe_update(observation: Any, **kwargs: Any) -> None:
    try:
        observation.update(**kwargs)
    except Exception:
        logger.debug("Langfuse observation update failed (non-fatal)", exc_info=True)


class _ObservationHandle:
    def __init__(self, observation: Any | None) -> None:
        self._observation = observation
        self.level_set = False

    def update(self, **kwargs: Any) -> None:
        if self._observation is not None:
            if kwargs.get("level") is not None:
                # A caller that names the level knows the outcome better than
                # the generic "an exception left the body" mapping below.
                self.level_set = True
            if not trace_sensitive_enabled():
                kwargs.pop("input", None)
                kwargs.pop("output", None)
            usage_details = kwargs.pop("usage_details", None)
            cost_details = kwargs.pop("cost_details", None)
            kwargs.update(_usage_cost_update(usage_details, cost_details))
            _safe_update(self._observation, **kwargs)


class LangfuseTelemetry:
    """Adapter from neutral telemetry to DlightRAG's Langfuse state."""

    @property
    def capture_sensitive_data(self) -> bool:
        return trace_sensitive_enabled()

    def trace(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AbstractContextManager[None]:
        return trace_attributes(session_id=session_id, user_id=user_id)

    def observe(
        self,
        name: SpanName,
        *,
        input: Any | None = None,
        metadata: Any | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Observation]:
        return trace_observation(
            name,
            as_type=SPAN_TYPES[name],
            input=input,
            metadata=metadata,
            model=model,
            model_parameters=model_parameters,
        )


def _langfuse_usage_details(raw: Mapping[str, Any]) -> dict[str, int]:
    """Map one provider's counters onto Langfuse's mutually exclusive usage keys.

    Langfuse prices every key on its own and takes ``total`` as their sum, so a token
    sits in exactly one key: ``input`` is the prompt without the tokens a prefix cache
    served, and those are ``input_cached_tokens``. Which counter holds what, in each
    provider dialect, is the provider helpers' knowledge alone.
    """
    prompt = provider_input_tokens(raw)
    cached = provider_cache_hit_tokens(raw) or 0
    output = provider_output_tokens(raw)

    details: dict[str, int] = {}
    if prompt is not None:
        details["input"] = prompt - cached
    if cached:
        details["input_cached_tokens"] = cached
    if output is not None:
        details["output"] = output
    total = provider_total_tokens(raw)
    if not details and total is None:
        # An unknown provider dialect must not put arbitrary keys on the span:
        # Langfuse derives cost only from the keys it has prices for.
        logger.debug("No recognized usage keys in provider usage payload")
        return details
    details["total"] = sum(details.values()) if total is None else total
    return details


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
    as_type: SpanType,
    input: Any | None = None,
    metadata: Any | None = None,
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
        observation = stack.enter_context(client.start_as_current_observation(**observation_kwargs))
    except Exception:
        stack.close()
        logger.debug("Langfuse observation start failed (non-fatal)", exc_info=True)
        yield _ObservationHandle(None)
        return

    exc_type: type[BaseException] | None = None
    exc: BaseException | None = None
    tb: TracebackType | None = None
    handle = _ObservationHandle(observation)
    try:
        try:
            yield handle
        except asyncio.CancelledError, GeneratorExit:
            raise
        except BaseException as caught:
            exc_type = type(caught)
            exc = caught
            tb = caught.__traceback__
            status = str(caught) if sensitive else "error"
            if not handle.level_set:
                _safe_update(observation, level="ERROR", status_message=status)
            raise
    finally:
        _exit_observation(stack, exc_type, exc, tb)


__all__ = ["LangfuseTelemetry", "trace_observation"]
