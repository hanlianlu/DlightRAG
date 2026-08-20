# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Anthropic native completion provider."""

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from anthropic import AsyncAnthropic

from dlightrag_ai.messages import (
    AssistantTurn,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolStopReason,
)
from dlightrag_ai.providers.base import CompletionOutput, CompletionProvider, usage_to_dict
from dlightrag_ai.structured import json_schema_from_response_format

logger = logging.getLogger(__name__)

_ANTHROPIC_TOP_LEVEL_KEYS = frozenset({"thinking", "metadata", "extra_headers"})
_DATA_URI_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)


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
            elif isinstance(url, str) and url.startswith("https://"):
                result.append({"type": "image", "source": {"type": "url", "url": url}})
        else:
            result.append(block)
    return result


def _anthropic_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            state = message.get("provider_state")
            if isinstance(state, dict):
                thinking_blocks = state.get("thinking_blocks")
                if isinstance(thinking_blocks, list):
                    blocks.extend(
                        dict(block)
                        for block in thinking_blocks
                        if isinstance(block, dict)
                        and block.get("type") in {"thinking", "redacted_thinking"}
                    )
            content = _convert_content(message.get("content", ""))
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            else:
                blocks.extend(content)
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": arguments if isinstance(arguments, dict) else {},
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
                "content": str(message.get("content") or ""),
                "is_error": bool(message.get("is_error", False)),
            }
            if (
                converted
                and converted[-1].get("role") == "user"
                and isinstance(converted[-1].get("content"), list)
                and all(item.get("type") == "tool_result" for item in converted[-1]["content"])
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        converted.append(
            {
                "role": str(role or "user"),
                "content": _convert_content(message.get("content", "")),
            }
        )
    return converted


def _apply_response_format(
    call_kwargs: dict[str, Any],
    response_format: dict[str, Any] | None,
) -> None:
    if response_format and response_format.get("type") == "json_object":
        raise ValueError("Anthropic native structured output requires json_schema")
    schema = json_schema_from_response_format(response_format)
    if schema is not None:
        call_kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }


class AnthropicProvider(CompletionProvider):
    """Anthropic Claude models via native SDK.

    System messages extracted as ``system`` top-level parameter.
    Image data-URI converted to Anthropic base64 source format.
    model_kwargs keys ``thinking``, ``metadata``, ``extra_headers`` routed to SDK top-level.
    JSON schema response_format routed to ``output_config.format``.
    """

    supports_native_json_schema: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None

    def thinking_off_kwargs(self) -> dict[str, Any]:
        # Native Anthropic disable: the ``thinking`` top-level key accepts a
        # config dict whose ``type`` is ``disabled``.
        return {"thinking": {"type": "disabled"}}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
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
    ) -> CompletionOutput:
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

        _apply_response_format(call_kwargs, response_format)

        if model_kwargs:
            for key in _ANTHROPIC_TOP_LEVEL_KEYS:
                if key in model_kwargs:
                    call_kwargs[key] = model_kwargs[key]
            extra = {k: v for k, v in model_kwargs.items() if k not in _ANTHROPIC_TOP_LEVEL_KEYS}
            if extra:
                call_kwargs["extra_body"] = extra

        response = await self._get_client().messages.create(**call_kwargs)
        content = response.content
        self.last_reasoning = "".join(
            b.thinking for b in content if getattr(b, "type", None) == "thinking"
        )
        text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
        return CompletionOutput(text, usage_details=usage_to_dict(getattr(response, "usage", None)))

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
        system, non_system = _extract_system(messages)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": _anthropic_tool_messages(non_system),
            "max_tokens": max_tokens or 8192,
        }
        if system:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]
            call_kwargs["tool_choice"] = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }[tool_choice]
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if model_kwargs:
            for key in _ANTHROPIC_TOP_LEVEL_KEYS:
                if key in model_kwargs:
                    call_kwargs[key] = model_kwargs[key]
            extra = {
                key: value
                for key, value in model_kwargs.items()
                if key not in _ANTHROPIC_TOP_LEVEL_KEYS
            }
            if extra:
                call_kwargs["extra_body"] = extra

        response = await self._get_client().messages.create(**call_kwargs)
        content = response.content
        reasoning = "".join(
            str(getattr(block, "thinking", ""))
            for block in content
            if getattr(block, "type", None) == "thinking"
        )
        self.last_reasoning = reasoning
        thinking_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": str(getattr(block, "thinking", "")),
                        "signature": str(getattr(block, "signature", "")),
                    }
                )
            elif block_type == "redacted_thinking":
                thinking_blocks.append(
                    {
                        "type": "redacted_thinking",
                        "data": str(getattr(block, "data", "")),
                    }
                )
        calls = tuple(
            ToolCall(
                id=str(getattr(block, "id", "") or ""),
                name=str(getattr(block, "name", "") or ""),
                arguments=dict(getattr(block, "input", None) or {}),
            )
            for block in content
            if getattr(block, "type", None) == "tool_use"
        )
        text = "".join(
            str(getattr(block, "text", ""))
            for block in content
            if getattr(block, "type", None) == "text"
        )
        return AssistantTurn(
            text=text,
            reasoning=reasoning,
            tool_calls=calls,
            stop_reason=_anthropic_stop_reason(
                getattr(response, "stop_reason", None), has_tool_calls=bool(calls)
            ),
            usage_details=usage_to_dict(getattr(response, "usage", None)),
            provider_state={"thinking_blocks": thinking_blocks} if thinking_blocks else None,
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
    ) -> AsyncGenerator[str]:  # type: ignore[override]
        system, non_system = _extract_system(messages)
        normalized: list[dict[str, Any]] = _anthropic_tool_messages(non_system)
        if system:
            normalized.insert(0, {"role": "system", "content": system})
        async for token in self.stream(
            normalized,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            usage_holder=usage_holder,
        ):
            yield token

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

        _apply_response_format(call_kwargs, response_format)

        if model_kwargs:
            for key in _ANTHROPIC_TOP_LEVEL_KEYS:
                if key in model_kwargs:
                    call_kwargs[key] = model_kwargs[key]
            extra = {k: v for k, v in model_kwargs.items() if k not in _ANTHROPIC_TOP_LEVEL_KEYS}
            if extra:
                call_kwargs["extra_body"] = extra

        response = await self._get_client().messages.create(**call_kwargs)
        reasoning_parts: list[str] = []
        usage_start: Any = None
        usage_delta: Any = None
        async for event in response:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                usage_start = getattr(getattr(event, "message", None), "usage", None)
            elif etype == "message_delta":
                event_usage = getattr(event, "usage", None)
                if event_usage is not None:
                    usage_delta = event_usage
            if etype != "content_block_delta":
                continue
            delta = event.delta
            if delta.type == "text_delta":
                yield delta.text
            elif delta.type == "thinking_delta":
                reasoning_parts.append(delta.thinking)
        self.last_reasoning = "".join(reasoning_parts)
        if usage_holder is not None:
            merged: dict[str, int] = {}
            for src in (usage_start, usage_delta):
                if src is None:
                    continue
                try:
                    part = usage_to_dict(src)
                except AttributeError, TypeError:
                    part = None
                if part:
                    merged.update(part)
            if merged:
                usage_holder["usage_details"] = merged


def _anthropic_stop_reason(reason: Any, *, has_tool_calls: bool) -> ToolStopReason:
    if has_tool_calls or reason == "tool_use":
        return "tool_use"
    if reason == "max_tokens":
        return "length"
    return "stop"
