"""Embedding callables.

Two tracks matching the completion module:
- _openai_embedding: AsyncOpenAI SDK
- _litellm_embedding: LiteLLM
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


async def _openai_embedding(
    texts: list[str],
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    _client: AsyncOpenAI | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """OpenAI SDK embedding."""
    client = _client or AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.embeddings.create(model=model, input=texts, **kwargs)
    return [d.embedding for d in response.data]


async def _litellm_embedding(
    texts: list[str],
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """LiteLLM embedding."""
    import litellm

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "api_key": api_key}
    if base_url:
        call_kwargs["api_base"] = base_url
    call_kwargs.update(kwargs)
    response = await litellm.aembedding(**call_kwargs)
    return [d["embedding"] for d in response.data]


__all__ = ["_openai_embedding", "_litellm_embedding"]
