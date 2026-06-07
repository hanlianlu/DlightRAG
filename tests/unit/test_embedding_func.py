# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for httpx_embed embedding callable."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dlightrag.config import DlightragConfig, EmbeddingConfig
from dlightrag.models.embedding import httpx_embed
from dlightrag.models.llm import get_embedding_func
from dlightrag.models.providers.embed_providers import OpenAICompatibleEmbedProvider


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
        result = await httpx_embed([], model="test", provider=OpenAICompatibleEmbedProvider())
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts(self):
        provider = OpenAICompatibleEmbedProvider()
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
        provider = OpenAICompatibleEmbedProvider()
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


def test_embedding_func_enables_asymmetric_by_default_for_capable_provider() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        )
    )

    embedding_func = get_embedding_func(cfg)

    assert embedding_func.supports_asymmetric is True


def test_embedding_func_uses_lightrag_symmetric_fallback_for_unsupported_auto() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="qwen_openai_compatible",
            model="qwen3-vl-embedding-2b",
            api_key="sk-test",
            dim=2048,
            startup_probe=False,
        )
    )

    embedding_func = get_embedding_func(cfg)

    assert embedding_func.supports_asymmetric is False


async def test_embedding_func_can_reuse_service_embedder() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            api_key="",
            dim=3,
            startup_probe=False,
        )
    )
    embedder = MagicMock()
    embedder.supports_asymmetric = False
    embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    embedding_func = get_embedding_func(cfg, embedder=embedder)
    result = await embedding_func.func(["hello"], context="query")

    assert result.tolist() == [[0.1, 0.2, 0.3]]
    embedder.embed_texts.assert_awaited_once_with(["hello"], context="query")


def test_embedding_func_rejects_required_asymmetric_for_unsupported_provider() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="qwen_openai_compatible",
            model="qwen3-vl-embedding-2b",
            api_key="sk-test",
            dim=2048,
            asymmetric="require",
            startup_probe=False,
        )
    )

    with pytest.raises(ValueError, match="does not support asymmetric"):
        get_embedding_func(cfg)
