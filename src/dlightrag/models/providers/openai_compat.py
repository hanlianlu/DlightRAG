# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible completion provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from dlightrag.models.providers.base import CompletionProvider


class OpenAICompatProvider(CompletionProvider):
    """OpenAI, Azure OpenAI, Ollama, Xinference, MiniMax, Qwen, OpenRouter."""

    async def complete(self, messages: list[dict[str, Any]], model: str, **kwargs: Any) -> str:
        raise NotImplementedError("Task 2")

    async def stream(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Task 2")
        yield  # pragma: no cover
