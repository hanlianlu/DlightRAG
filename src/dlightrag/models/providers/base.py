# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for LLM completion providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any


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
    providers to attach token usage and cost details for tracing integrations.
    """

    usage_details: dict[str, int] | None
    cost_details: dict[str, float] | None

    def __new__(
        cls,
        text: str,
        *,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
    ) -> CompletionOutput:
        value = str.__new__(cls, text)
        value.usage_details = usage_details
        value.cost_details = cost_details
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

    supports_native_json_schema: bool = False
    """Whether this provider natively supports JSON schema structured output.

    Set to ``True`` on providers whose SDK accepts a JSON Schema directly
    (Anthropic, Gemini).  Providers without native support fall back to
    ``json_object`` mode or a strict-mode json_schema via the OpenAI API.
    """

    supports_vision: bool | None = None
    """Whether this provider's model accepts ``image_url`` content blocks.

    ``None`` means "unprobed" — call :func:`probe_vision_support` to
    determine at startup.  ``True`` / ``False`` is the known answer.
    Per-instance so that a probe result on one instance doesn't leak
    to others via the class attribute.
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
        self.supports_vision: bool | None = None
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

    async def aclose(self) -> None:
        """Release SDK clients, connection pools, and other resources.

        The default is a no-op. Providers that hold persistent clients
        (OpenAI, Anthropic, Gemini, etc.) override this.
        """
        return
