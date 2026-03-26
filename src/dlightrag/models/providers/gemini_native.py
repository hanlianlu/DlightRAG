# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Google Gemini native completion provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from dlightrag.models.providers.base import CompletionProvider


class GeminiProvider(CompletionProvider):
    """Google Gemini models via google-genai SDK."""

    async def complete(self, messages: list[dict[str, Any]], model: str, **kwargs: Any) -> str:
        raise NotImplementedError("Task 4")

    async def stream(
        self, messages: list[dict[str, Any]], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Task 4")
        yield  # pragma: no cover
