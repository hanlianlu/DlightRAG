# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible completion provider."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from dlightrag.models.providers.base import (
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
