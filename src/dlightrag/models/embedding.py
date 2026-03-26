# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Embedding callables using httpx with EmbedProvider strategy."""

from __future__ import annotations

from typing import Any

import httpx


async def httpx_embed(
    texts: list[str],
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    provider: Any = None,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Embed texts via httpx POST to an embedding endpoint.

    Uses OpenAICompatEmbedProvider by default.
    """
    if not texts:
        return []

    prov = provider
    url = (base_url.rstrip("/") if base_url else "https://api.openai.com") + prov.endpoint
    payload = prov.build_payload(model, texts)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    return prov.parse_response(data)


__all__ = ["httpx_embed"]
