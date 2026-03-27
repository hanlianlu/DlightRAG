# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Embedding callables using httpx with EmbedProvider strategy."""

from __future__ import annotations

from typing import Any

import httpx

_MAX_RETRIES = 3


def create_embed_client(api_key: str, timeout: float = 120.0) -> httpx.AsyncClient:
    """Create a persistent httpx client for embedding requests (connection pooling)."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        transport=httpx.AsyncHTTPTransport(retries=2),
    )


async def _post_with_retry(
    url: str,
    payload: dict,
    api_key: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """POST with 429 retry. Reuses client if provided, else creates ephemeral one."""
    import asyncio

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _do_post(c: httpx.AsyncClient) -> dict:
        for attempt in range(_MAX_RETRIES):
            resp = await c.post(url, json=payload, headers=headers)
            if resp.status_code == 429 and attempt < _MAX_RETRIES - 1:
                retry_after = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                await asyncio.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Embedding API: max retries exceeded")

    if client is not None:
        return await _do_post(client)
    async with httpx.AsyncClient(timeout=120.0) as ephemeral:
        return await _do_post(ephemeral)


async def httpx_embed(
    texts: list[str],
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    provider: Any = None,
    timeout: float = 120.0,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """Embed texts via httpx POST to an embedding endpoint.

    When *client* is provided, reuses it for connection pooling.
    Otherwise creates an ephemeral client per call.
    """
    if not texts:
        return []

    prov = provider
    url = (base_url.rstrip("/") if base_url else "https://api.openai.com") + prov.endpoint
    payload = prov.build_payload(model, texts)
    data = await _post_with_retry(url, payload, api_key, client=client)
    return prov.parse_response(data)


__all__ = ["create_embed_client", "httpx_embed"]
