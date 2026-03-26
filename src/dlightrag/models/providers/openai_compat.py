# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible completion provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from dlightrag.models.providers.base import CompletionProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(CompletionProvider):
    """OpenAI, Azure OpenAI, Ollama, Xinference, MiniMax, Qwen, OpenRouter.

    Any endpoint that speaks the OpenAI chat completions protocol.
    model_kwargs are routed to extra_body for provider extensions
    (DeepSeek thinking, Kimi partial, etc.).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> str:
        call_kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if model_kwargs:
            call_kwargs["extra_body"] = model_kwargs
        response = await self._get_client().chat.completions.create(**call_kwargs)
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if model_kwargs:
            call_kwargs["extra_body"] = model_kwargs
        response = await self._get_client().chat.completions.create(**call_kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content
