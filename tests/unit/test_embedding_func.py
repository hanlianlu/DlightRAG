# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for httpx_embed embedding callable."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dlightrag.models.embedding import httpx_embed
from dlightrag.models.providers.embed_providers import OpenAICompatEmbedProvider


def _make_response(status_code: int, json_data: dict) -> httpx.Response:
    """Create an httpx.Response with a dummy request (needed for raise_for_status)."""
    resp = httpx.Response(status_code, json=json_data)
    resp._request = httpx.Request("POST", "https://test")
    return resp


def _mock_async_client(mock_response: httpx.Response) -> MagicMock:
    """Create a mock httpx.AsyncClient that works with `async with`."""
    instance = MagicMock()
    instance.post = AsyncMock(return_value=mock_response)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_cls = MagicMock(return_value=ctx)
    return mock_cls, instance


class TestHttpxEmbed:
    @pytest.mark.asyncio
    async def test_empty_texts(self):
        result = await httpx_embed([], model="test", provider=OpenAICompatEmbedProvider())
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts(self):
        provider = OpenAICompatEmbedProvider()
        response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_response = _make_response(200, response_data)
        mock_cls, _ = _mock_async_client(mock_response)

        with patch("dlightrag.models.embedding.httpx.AsyncClient", mock_cls):
            result = await httpx_embed(
                ["hello", "world"],
                model="text-embedding-3-large",
                api_key="sk-test",
                base_url="https://api.openai.com",
                provider=provider,
            )

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @pytest.mark.asyncio
    async def test_auth_header_set(self):
        provider = OpenAICompatEmbedProvider()
        mock_response = _make_response(200, {"data": [{"embedding": [1.0]}]})
        mock_cls, instance = _mock_async_client(mock_response)

        with patch("dlightrag.models.embedding.httpx.AsyncClient", mock_cls):
            await httpx_embed(
                ["test"],
                model="m",
                api_key="sk-secret",
                base_url="https://api.openai.com",
                provider=provider,
            )

            call_kwargs = instance.post.call_args
            assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-secret"
