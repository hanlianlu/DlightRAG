"""Messages-first LLM completion callables.

Two tracks:
- _openai_completion: AsyncOpenAI SDK (OpenAI, Azure, Qwen, MiniMax, etc.)
- _litellm_completion: LiteLLM (Anthropic, Gemini, Ollama, OpenRouter, etc.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI


async def _openai_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str | None = None,
    stream: bool = False,
    response_format: Any = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    _client: AsyncOpenAI | None = None,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """OpenAI SDK completion."""
    client = _client or AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    call_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    call_kwargs.update(kwargs)

    if stream:
        response = await client.chat.completions.create(**call_kwargs, stream=True)

        async def _stream() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

        return _stream()
    else:
        response = await client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content or ""


async def _litellm_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str | None = None,
    stream: bool = False,
    response_format: Any = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """LiteLLM completion."""
    import litellm

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "timeout": timeout,
    }
    if base_url:
        call_kwargs["api_base"] = base_url
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    call_kwargs.update(kwargs)

    if stream:
        response = await litellm.acompletion(**call_kwargs, stream=True)

        async def _stream() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

        return _stream()
    else:
        response = await litellm.acompletion(**call_kwargs)
        return response.choices[0].message.content or ""


__all__ = ["_openai_completion", "_litellm_completion"]
