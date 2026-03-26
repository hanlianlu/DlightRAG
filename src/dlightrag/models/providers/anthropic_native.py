# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Anthropic native completion provider."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from dlightrag.models.providers.base import CompletionProvider

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment, misc]

_ANTHROPIC_TOP_LEVEL_KEYS = frozenset({"thinking", "metadata", "extra_headers"})
_DATA_URI_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)
_JSON_SYSTEM_HINT = "Respond with valid JSON only. Do not include any text outside the JSON object."


def _extract_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system messages from OpenAI-format list; return (system, non-system)."""
    system_parts: list[str] = []
    filtered: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
        else:
            filtered.append(msg)
    return ("\n\n".join(system_parts) if system_parts else None), filtered


def _convert_content(content: str | list[Any]) -> str | list[dict[str, Any]]:
    """Convert OpenAI content blocks to Anthropic format."""
    if isinstance(content, str):
        return content
    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            result.append({"type": "text", "text": block})
        elif block.get("type") == "text":
            result.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url":
            url = (
                block["image_url"]["url"]
                if isinstance(block["image_url"], dict)
                else block["image_url"]
            )
            m = _DATA_URI_RE.match(url)
            if m:
                result.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)},
                    }
                )
        else:
            result.append(block)
    return result


class AnthropicProvider(CompletionProvider):
    """Anthropic Claude models via native SDK.

    System messages extracted as ``system`` top-level parameter.
    Image data-URI converted to Anthropic base64 source format.
    model_kwargs keys ``thinking``, ``metadata``, ``extra_headers`` routed to SDK top-level.
    JSON response_format injected as system instruction.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if AsyncAnthropic is None:
                raise ImportError(
                    "anthropic SDK not installed. Install with: pip install dlightrag[anthropic]"
                )
            self._client = AsyncAnthropic(
                api_key=self._api_key,
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
        system, non_system = _extract_system(messages)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m["role"], "content": _convert_content(m.get("content", ""))}
                for m in non_system
            ],
            "max_tokens": max_tokens or 8192,
        }
        if system:
            call_kwargs["system"] = system
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        if response_format and response_format.get("type") == "json_object":
            call_kwargs["system"] = (call_kwargs.get("system") or "") + "\n\n" + _JSON_SYSTEM_HINT

        if model_kwargs:
            for key in _ANTHROPIC_TOP_LEVEL_KEYS:
                if key in model_kwargs:
                    call_kwargs[key] = model_kwargs[key]
            extra = {k: v for k, v in model_kwargs.items() if k not in _ANTHROPIC_TOP_LEVEL_KEYS}
            if extra:
                call_kwargs["extra_body"] = extra

        response = await self._get_client().messages.create(**call_kwargs)
        return "".join(block.text for block in response.content)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:  # type: ignore
        system, non_system = _extract_system(messages)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m["role"], "content": _convert_content(m.get("content", ""))}
                for m in non_system
            ],
            "max_tokens": max_tokens or 8192,
            "stream": True,
        }
        if system:
            call_kwargs["system"] = system
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        if response_format and response_format.get("type") == "json_object":
            call_kwargs["system"] = (call_kwargs.get("system") or "") + "\n\n" + _JSON_SYSTEM_HINT

        if model_kwargs:
            for key in _ANTHROPIC_TOP_LEVEL_KEYS:
                if key in model_kwargs:
                    call_kwargs[key] = model_kwargs[key]
            extra = {k: v for k, v in model_kwargs.items() if k not in _ANTHROPIC_TOP_LEVEL_KEYS}
            if extra:
                call_kwargs["extra_body"] = extra

        response = await self._get_client().messages.create(**call_kwargs)
        async for event in response:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield event.delta.text
