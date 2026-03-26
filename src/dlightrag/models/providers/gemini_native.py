# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Google Gemini native completion provider."""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from dlightrag.models.providers.base import CompletionProvider

logger = logging.getLogger(__name__)

_GEMINI_CONFIG_KEYS = frozenset({"safety_settings", "cached_content", "thinking_config"})
_ROLE_MAP = {"user": "user", "assistant": "model"}
_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]


def _convert_content(content: str | list[Any]) -> str | list[Any]:
    """Convert OpenAI content blocks to Gemini Part-compatible dicts."""
    if isinstance(content, str):
        return content
    parts: list[Any] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "image_url":
            url = (
                block["image_url"]["url"]
                if isinstance(block["image_url"], dict)
                else block["image_url"]
            )
            m = _DATA_URI_RE.match(url)
            if m:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": m.group(1),
                            "data": base64.b64decode(m.group(2)),
                        },
                    }
                )
            else:
                parts.append({"file_data": {"file_uri": url}})
        else:
            parts.append(block)
    return parts


class GeminiProvider(CompletionProvider):
    """Google Gemini models via google-genai SDK.

    System messages extracted as ``system_instruction``.
    Role mapping: ``assistant`` -> ``model``.
    model_kwargs keys ``safety_settings``, ``cached_content``,
    ``thinking_config`` routed to config dict.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if genai is None:
                raise ImportError(
                    "google-genai SDK not installed. Install with: pip install dlightrag[gemini]"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_args(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        model_kwargs: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        # Extract system messages
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                c = msg.get("content", "")
                if isinstance(c, str):
                    system_parts.append(c)
            else:
                role = _ROLE_MAP.get(msg["role"], msg["role"])
                non_system.append({"role": role, "parts": _convert_content(msg.get("content", ""))})

        config: dict[str, Any] = {}
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if response_format and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"

        if model_kwargs:
            for key in _GEMINI_CONFIG_KEYS:
                if key in model_kwargs:
                    config[key] = model_kwargs[key]
            extra = {k: v for k, v in model_kwargs.items() if k not in _GEMINI_CONFIG_KEYS}
            if extra:
                config.update(extra)

        return model, non_system, config

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
        model_id, contents, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model_kwargs=model_kwargs,
        )
        response = await self._get_client().models.generate_content(
            model=model_id, contents=contents, config=config
        )
        return response.text or ""

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
        model_id, contents, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model_kwargs=model_kwargs,
        )
        config["stream"] = True
        response = await self._get_client().models.generate_content(
            model=model_id, contents=contents, config=config
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text
