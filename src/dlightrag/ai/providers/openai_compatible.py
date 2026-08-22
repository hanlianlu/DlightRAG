# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible completion provider."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIStatusError, AsyncOpenAI

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
    usage_mapping,
    usage_to_dict,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_PARAMETER_MARKERS = (
    "invalid parameter",
    "not permitted",
    "not supported",
    "unsupported",
    "unrecognized",
    "unknown",
    "unexpected",
)


def _rejects_stream_options(exc: APIStatusError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    raw_error = body.get("error")
    error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
    parameter = exc.param or body.get("param") or error.get("param")
    if str(parameter or "").casefold() == "stream_options":
        return True
    details = body.get("detail")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            location = detail.get("loc")
            if isinstance(location, list) and any(
                str(part).casefold() == "stream_options" for part in location
            ):
                return True
    messages = (exc.message, body.get("message"), error.get("message"))
    text = " ".join(str(message) for message in messages if message).casefold()
    return "stream_options" in text and any(
        marker in text for marker in _UNSUPPORTED_PARAMETER_MARKERS
    )


def _cost_to_dict(usage: Any) -> dict[str, float] | None:
    if usage is None:
        return None
    cost = usage_mapping(usage).get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return {"total": float(cost)}
    return None


def _openai_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        native = {
            key: value
            for key, value in message.items()
            if key not in {"provider_state", "is_error"}
        }
        state = message.get("provider_state")
        if message.get("role") == "assistant" and isinstance(state, dict):
            for key in ("reasoning_content", "reasoning_details"):
                if key in state:
                    native[key] = state[key]
        converted.append(native)
    return converted


def _openai_provider_state(message: Any) -> dict[str, Any] | None:
    extras = getattr(message, "model_extra", None) or {}
    state = {
        key: extras[key] for key in ("reasoning_content", "reasoning_details") if key in extras
    }
    return state or None


class OpenAICompatibleProvider(CompletionProvider):
    """OpenAI, Azure OpenAI, Ollama, Xinference, MiniMax, Qwen, OpenRouter.

    Any endpoint that speaks the OpenAI chat completions protocol.
    model_kwargs are routed to extra_body for provider extensions
    (DeepSeek thinking, Kimi partial, etc.).

    Tracks ``last_reasoning`` — the ``reasoning_content`` from the most
    recent completion or stream call.  It is exposed for optional display
    or observability only and is never fed back into conversation history:
    re-injecting reasoning wastes context and some providers (DeepSeek's
    legacy reasoner) reject a ``reasoning_content`` field in input.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: AsyncOpenAI | None = None

    def thinking_off_kwargs(self) -> dict[str, Any]:
        # The OpenAI-compatible convention used across endpoints (OpenRouter,
        # xAI, DeepSeek): ``reasoning.enabled`` in the extra body.
        return {"reasoning": {"enabled": False}}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _extract_reasoning(message: Any) -> str:
        """Extract ``reasoning_content`` from an OpenAI SDK message object."""
        extras = getattr(message, "model_extra", None) or {}
        rc = extras.get("reasoning_content")
        return str(rc) if isinstance(rc, str) else ""

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
    ) -> CompletionOutput:
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
        message = response.choices[0].message
        self.last_reasoning = self._extract_reasoning(message)
        return CompletionOutput(
            message.content or "",
            usage_details=usage_to_dict(getattr(response, "usage", None)),
            cost_details=_cost_to_dict(getattr(response, "usage", None)),
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
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": _openai_tool_messages(messages),
        }
        if tools:
            call_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            call_kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if model_kwargs:
            call_kwargs["extra_body"] = model_kwargs

        response = await self._get_client().chat.completions.create(**call_kwargs)
        choice = response.choices[0]
        message = choice.message
        self.last_reasoning = self._extract_reasoning(message)
        normalized_calls = tuple(
            _openai_tool_call(call) for call in (getattr(message, "tool_calls", None) or ())
        )
        stop_reason = _openai_stop_reason(
            getattr(choice, "finish_reason", None),
            has_tool_calls=bool(normalized_calls),
        )
        usage = getattr(response, "usage", None)
        return AssistantTurn(
            text=str(getattr(message, "content", None) or ""),
            reasoning=self.last_reasoning,
            tool_calls=normalized_calls,
            stop_reason=stop_reason,
            usage_details=usage_to_dict(usage),
            cost_details=_cost_to_dict(usage),
            provider_state=_openai_provider_state(message),
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
        async for token in self.stream(
            _openai_tool_messages(messages),
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            usage_holder=usage_holder,
        ):
            yield token

    async def _open_stream(self, call_kwargs: dict[str, Any]) -> Any:
        """Open a streaming completion, requesting token usage when supported.

        ``stream_options={"include_usage": True}`` is the OpenAI-standard way to
        get a final usage chunk, but not every OpenAI-compatible endpoint accepts
        it. Fall back to a plain stream (no usage) rather than fail the call.
        """
        client = self._get_client()
        try:
            return await client.chat.completions.create(
                **call_kwargs, stream_options={"include_usage": True}
            )
        except APIStatusError as exc:
            if exc.status_code not in {400, 422} or not _rejects_stream_options(exc):
                raise
            # Endpoint rejected stream_options — stream again without usage metadata.
            logger.debug("stream_options unsupported; streaming without usage", exc_info=True)
            return await client.chat.completions.create(**call_kwargs)

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
        response = await self._open_stream(call_kwargs)
        reasoning_parts: list[str] = []
        usage: Any = None
        async for chunk in response:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            extras = getattr(delta, "model_extra", None) or {}
            rc = extras.get("reasoning_content")
            if isinstance(rc, str) and rc:
                reasoning_parts.append(rc)
            if delta.content is not None:
                yield delta.content
        self.last_reasoning = "".join(reasoning_parts)
        capture_stream_usage(usage_holder, usage, cost_fn=_cost_to_dict)


def _openai_tool_call(raw: Any) -> ToolCall:
    function = getattr(raw, "function", None)
    name = str(getattr(function, "name", "") or "")
    encoded = str(getattr(function, "arguments", "") or "")
    try:
        parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise TypeError("tool arguments must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolCall(
            id=str(getattr(raw, "id", "") or ""),
            name=name,
            arguments={},
            argument_error=str(exc),
        )
    return ToolCall(
        id=str(getattr(raw, "id", "") or ""),
        name=name,
        arguments=parsed,
    )


def _openai_stop_reason(reason: Any, *, has_tool_calls: bool) -> ToolStopReason:
    if has_tool_calls or reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if reason == "length":
        return "length"
    return "stop"
