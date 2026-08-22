# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Google Gemini native completion provider."""

import base64
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from google import genai

from dlightrag.ai.messages import (
    AssistantTurn,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolStopReason,
)
from dlightrag.ai.providers.base import (
    CompletionOutput,
    CompletionProvider,
    capture_stream_usage,
    usage_to_dict,
)
from dlightrag.ai.structured import json_schema_from_response_format

logger = logging.getLogger(__name__)

_GEMINI_CONFIG_KEYS = frozenset({"safety_settings", "cached_content", "thinking_config"})
_ROLE_MAP = {"user": "user", "assistant": "model"}
_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _gemini_usage(response: Any) -> dict[str, int] | None:
    """Extract token usage from a Gemini SDK response.

    Uses the shared allow-list-free extractor, then maps Gemini's
    ``*_token_count`` names to the ``*_tokens`` convention used by the other
    providers so counters (incl. ``cached_content`` and ``thoughts``) stay
    consistent in observability.
    """
    raw = usage_to_dict(getattr(response, "usage_metadata", None))
    if not raw:
        return None
    return {key.replace("_token_count", "_tokens"): value for key, value in raw.items()}


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


def _gemini_tool_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            parts: list[Any] = []
            converted = _convert_content(message.get("content", ""))
            if isinstance(converted, str):
                if converted:
                    parts.append(converted)
            else:
                parts.extend(converted)
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                part = {
                    "function_call": {
                        "id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "args": arguments if isinstance(arguments, dict) else {},
                    }
                }
                if call.get("thought_signature") is not None:
                    part["thought_signature"] = call["thought_signature"]
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            part = {
                "function_response": {
                    "id": str(message.get("tool_call_id") or ""),
                    "name": str(message.get("name") or ""),
                    "response": {
                        "output": str(message.get("content") or ""),
                        "is_error": bool(message.get("is_error", False)),
                    },
                }
            }
            if (
                contents
                and contents[-1].get("role") == "user"
                and all("function_response" in item for item in contents[-1].get("parts", []))
            ):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue
        contents.append(
            {
                "role": _ROLE_MAP.get(str(role), str(role or "user")),
                "parts": _convert_content(message.get("content", "")),
            }
        )
    return contents


class GeminiProvider(CompletionProvider):
    """Google Gemini models via google-genai SDK.

    System messages extracted as ``system_instruction``.
    Role mapping: ``assistant`` -> ``model``.
    model_kwargs keys ``safety_settings``, ``cached_content``,
    ``thinking_config`` routed to config dict.
    """

    supports_native_json_schema: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    def thinking_off_kwargs(self) -> dict[str, Any]:
        # Gemini thinking is opt-in: a None ``thinking_config`` keeps the SDK
        # default (disabled). The key still wins over any agentic override.
        return {"thinking_config": None}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aio.aclose()
            self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            http_opts: dict[str, Any] = {}
            if self._base_url is not None:
                http_opts["base_url"] = self._base_url
            if self._timeout:
                http_opts["timeout"] = int(self._timeout * 1000)
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=genai.types.HttpOptions(**http_opts) if http_opts else None,
            )
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
        schema = json_schema_from_response_format(response_format)
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema
        elif response_format and response_format.get("type") == "json_object":
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
    ) -> CompletionOutput:
        model_id, contents, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model_kwargs=model_kwargs,
        )
        response = await self._get_client().aio.models.generate_content(
            model=model_id, contents=contents, config=config
        )
        return CompletionOutput(
            response.text or "",
            usage_details=_gemini_usage(response),
        )

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
        model_id, contents, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            model_kwargs=model_kwargs,
        )
        contents = _gemini_tool_contents(
            [message for message in messages if message.get("role") != "system"]
        )
        if tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tools
                    ]
                }
            ]
            config["tool_config"] = {
                "function_calling_config": {
                    "mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}[tool_choice]
                }
            }
        response = await self._get_client().aio.models.generate_content(
            model=model_id,
            contents=contents,
            config=config,
        )
        candidate = response.candidates[0]
        parts = getattr(getattr(candidate, "content", None), "parts", None) or ()
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        for index, part in enumerate(parts):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                (reasoning_parts if getattr(part, "thought", False) else text_parts).append(text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                calls.append(
                    ToolCall(
                        id=str(getattr(function_call, "id", None) or f"gemini-{index}"),
                        name=str(getattr(function_call, "name", "") or ""),
                        arguments=dict(getattr(function_call, "args", None) or {}),
                        thought_signature=getattr(part, "thought_signature", None),
                    )
                )
        reasoning = "".join(reasoning_parts)
        self.last_reasoning = reasoning
        return AssistantTurn(
            text="".join(text_parts),
            reasoning=reasoning,
            tool_calls=tuple(calls),
            stop_reason=_gemini_stop_reason(
                getattr(candidate, "finish_reason", None), has_tool_calls=bool(calls)
            ),
            usage_details=_gemini_usage(response),
        )

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
    ) -> AsyncGenerator[str]:  # type: ignore
        model_id, contents, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model_kwargs=model_kwargs,
        )
        async for token in self._stream_generated(
            model_id,
            contents,
            config,
            usage_holder=usage_holder,
        ):
            yield token

    async def stream_tool_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:  # type: ignore[override]
        model_id, _, config = self._build_args(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=None,
            model_kwargs=model_kwargs,
        )
        contents = _gemini_tool_contents(
            [message for message in messages if message.get("role") != "system"]
        )
        async for token in self._stream_generated(
            model_id,
            contents,
            config,
            usage_holder=usage_holder,
        ):
            yield token

    async def _stream_generated(
        self,
        model_id: str,
        contents: list[dict[str, Any]],
        config: dict[str, Any],
        *,
        usage_holder: dict[str, Any] | None,
    ) -> AsyncGenerator[str]:
        response = await self._get_client().aio.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        )
        usage: Any = None
        async for chunk in response:
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if chunk.text:
                yield chunk.text
        capture_stream_usage(usage_holder, usage)


def _gemini_stop_reason(reason: Any, *, has_tool_calls: bool) -> ToolStopReason:
    if has_tool_calls:
        return "tool_use"
    name = str(getattr(reason, "name", reason) or "")
    if name == "MAX_TOKENS":
        return "length"
    return "stop"
