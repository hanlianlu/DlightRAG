# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for LLM completion providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


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
    ) -> str: ...

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
    ) -> AsyncGenerator[str, None]: ...  # type: ignore[return]
