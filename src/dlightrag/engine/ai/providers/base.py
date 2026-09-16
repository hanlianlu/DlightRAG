# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for LLM completion providers."""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any

from dlightrag.engine.ai.messages import (
    AssistantTurn,
    ToolCallingUnavailableError,
    ToolChoice,
    ToolDefinition,
    ToolStopReason,
)

#: Provider rejection texts that mean the request exceeded the model's
#: context window. Kept provider-generic: the classifier walks the full
#: exception cause chain and matches either an HTTP status or this text.
_OVERFLOW_MESSAGE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",
        r"request_too_large",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"input \(\d+ tokens\) is longer than",
        r"exceeds the available context size",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
    )
)


def is_provider_context_overflow(exc: BaseException) -> bool:
    """Return whether one exception chain is a provider context-window rejection.

    Matches both explicit API error surfaces (a ``status_code`` of 400/413
    on the same object) and provider text in any cause, so OpenAI-compatible,
    Anthropic, and Gemini-shaped failures classify the same way.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if any(pattern.search(message) for pattern in _OVERFLOW_MESSAGE_PATTERNS):
            return True
        status = getattr(current, "status_code", None)
        if status in {400, 413} and _overflow_status_message(message):
            return True
        current = current.__cause__ or current.__context__
    return False


def provider_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status a provider rejected with, when the chain carries one.

    Status codes are the one part of a provider failure that is safe to record:
    they classify the failure without persisting prompt content.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and 400 <= status <= 599:
            return status
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and 400 <= status <= 599:
            return status
        current = current.__cause__ or current.__context__
    return None


def _overflow_status_message(message: str) -> bool:
    lowered = message.lower()
    markers = ("token", "context", "prompt", "input")
    return any(marker in lowered for marker in markers)


def usage_mapping(usage: Any) -> dict[str, Any]:
    """Flatten a usage payload to a plain mapping, including SDK extra fields.

    Handles OpenAI/Anthropic/Gemini SDK objects (declared fields in
    ``__dict__``, provider extras in ``model_extra``), plain dicts, and simple
    namespaces uniformly. ``_``-prefixed keys are dropped so mock or
    SDK-internal attributes never leak into usage.
    """
    if isinstance(usage, dict):
        raw: dict[str, Any] = usage
    else:
        base = getattr(usage, "__dict__", None)
        extra = getattr(usage, "model_extra", None)
        if not isinstance(extra, dict):
            extra = None
        if not base and not extra:
            return {}
        raw = {**(base or {}), **(extra or {})}
    return {k: v for k, v in raw.items() if not (isinstance(k, str) and k.startswith("_"))}


def provider_input_tokens(usage: Mapping[str, int] | None) -> int | None:
    """Return the total prompt tokens one provider billed, or None when unstated.

    The counters are provider-specific dialects of the same fact. A total
    ``prompt_tokens``/``prompt_token_count`` already includes whatever the
    provider cached; Anthropic's ``input_tokens`` excludes its cache siblings, so
    the cache reads and writes are added back. This is the number a prefix cache
    is measured against, never an approximation of it.
    """
    if not usage:
        return None
    total = _first_int(usage, ("prompt_tokens", "prompt_token_count"))
    if total is not None:
        return total
    uncached = _first_int(usage, ("input_tokens",))
    if uncached is None:
        return None
    return uncached + sum(
        value
        for value in (
            _first_int(usage, ("cache_read_input_tokens", "prompt_cache_hit_tokens")),
            _first_int(usage, ("cache_creation_input_tokens", "prompt_cache_miss_tokens")),
        )
        if value is not None
    )


def _first_int(usage: Mapping[str, int], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def provider_cache_hit_tokens(usage: Mapping[str, int] | None) -> int | None:
    """Return the prompt tokens one provider served from a prefix cache.

    None means the provider does not report caching at all, which is a different
    fact from a reported zero.
    """
    if not usage:
        return None
    return _first_int(
        usage,
        (
            "prompt_tokens_details.cached_tokens",
            "prompt_cache_hit_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
        ),
    )


def usage_to_dict(usage: Any) -> dict[str, int] | None:
    """Extract integer token counters from a provider usage payload.

    Allow-list-free and provider-agnostic: every integer field is captured, so
    flat counters (DeepSeek ``prompt_cache_hit_tokens``, Anthropic
    ``cache_read_input_tokens``, Gemini ``thoughts_token_count``) and future
    fields surface automatically. One level of nested ``*_details`` /
    ``cache_creation`` objects is flattened to ``parent.child`` keys.
    """
    if usage is None:
        return None
    result: dict[str, int] = {}
    for key, value in usage_mapping(usage).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            result[key] = value
            continue
        for sub_key, sub_value in usage_mapping(value).items():
            if isinstance(sub_value, int) and not isinstance(sub_value, bool):
                result[f"{key}.{sub_key}"] = sub_value
    return result or None


def capture_stream_usage(
    holder: dict[str, Any] | None,
    usage: Any,
    *,
    cost_fn: Callable[[Any], dict[str, float] | None] | None = None,
) -> None:
    """Record a streaming call's usage/cost into a per-call holder, best-effort.

    ``holder`` is created once per streaming request by the tracing layer, so
    there is no shared provider state (safe under concurrency). Parsing failures
    are swallowed — streaming usage is optional observability and must never
    break the stream.
    """
    if holder is None or usage is None:
        return
    try:
        details = usage_to_dict(usage)
    except AttributeError, TypeError:
        details = None
    if details:
        holder["usage_details"] = details
    if cost_fn is None:
        return
    try:
        cost = cost_fn(usage)
    except AttributeError, TypeError:
        cost = None
    if cost:
        holder["cost_details"] = cost


class CompletionOutput(str):
    """Completion text with optional observability metadata.

    The value behaves as a plain string for existing callers while allowing
    providers to attach token usage, cost, and normalized stop-reason metadata.
    """

    usage_details: dict[str, int] | None
    cost_details: dict[str, float] | None
    stop_reason: ToolStopReason | None

    def __new__(
        cls,
        text: str,
        *,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        stop_reason: ToolStopReason | None = None,
    ) -> CompletionOutput:
        value = str.__new__(cls, text)
        value.usage_details = usage_details
        value.cost_details = cost_details
        value.stop_reason = stop_reason
        return value


class CompletionProvider(ABC):
    """Abstract base for LLM completion providers.

    All providers accept messages in OpenAI format:
    [{"role": "system"|"user"|"assistant", "content": str | list}]

    Each implementation converts internally to its SDK's native format.

    Lifecycle: callers SHOULD ``await provider.aclose()`` when done to
    release SDK clients and connection pools.  Skipping it is safe for
    ephemeral use (the SDK eventually garbage-collects), but long-lived
    server processes that create many provider instances will leak
    connections.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self.last_reasoning: str = ""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> CompletionOutput:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:  # type: ignore[return]
        raise NotImplementedError
        yield ""  # pragma: no cover

    async def complete_tool_turn(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        """Run one structured tool-capable turn when the provider supports it."""
        raise ToolCallingUnavailableError(
            f"{type(self).__name__} does not implement tool-capable turns"
        )

    async def complete_tool_turn_streaming(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[ToolDefinition],
        emit_text: Callable[[str], Awaitable[None]],
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        """Complete one tool-capable turn, publishing provider text deltas when supported.

        The provider must return the same text that its callbacks concatenate. The
        default deliberately emits nothing, preserving a truthful non-streaming
        fallback for providers without a normalized tool-turn stream.
        """
        del emit_text
        return await self.complete_tool_turn(
            messages,
            model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )

    async def stream_tool_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:  # type: ignore[return]
        """Stream final text while replaying a provider-native tool transcript."""
        raise ToolCallingUnavailableError(
            f"{type(self).__name__} does not implement tool transcript streaming"
        )
        yield ""  # pragma: no cover

    async def aclose(self) -> None:
        """Release SDK clients, connection pools, and other resources.

        The default is a no-op. Providers that hold persistent clients
        (OpenAI, Anthropic, Gemini, etc.) override this.
        """
        return


__all__ = [
    "CompletionOutput",
    "CompletionProvider",
    "capture_stream_usage",
    "is_provider_context_overflow",
    "provider_cache_hit_tokens",
    "provider_input_tokens",
    "provider_status_code",
    "usage_mapping",
    "usage_to_dict",
]
