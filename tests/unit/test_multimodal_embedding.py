# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for context-aware multimodal embedding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.models.multimodal_embedding import MultimodalEmbedder
from dlightrag.models.providers.embed_providers import (
    GeminiEmbedProvider,
    QwenOpenAICompatEmbedProvider,
    VoyageEmbedProvider,
)


def test_image_embedder_auto_falls_back_for_unsupported_provider() -> None:
    provider = QwenOpenAICompatEmbedProvider()
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://localhost:1234/v1",
        api_key="key",
        dim=2048,
        provider=provider,
        asymmetric="auto",
    )

    image = Image.new("RGB", (1, 1), "white")
    index_payload = embedder.build_image_payload_for_test(image, context="document")
    query_payload = embedder.build_image_payload_for_test(image, context="query")

    assert embedder.supports_asymmetric is False
    assert "input_type" not in index_payload
    assert "input_type" not in query_payload
    assert index_payload["input"][0].startswith("data:image/png;base64,")


def test_image_embedder_uses_asymmetric_by_default_for_capable_provider() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        asymmetric="auto",
    )

    image = Image.new("RGB", (1, 1), "white")
    index_payload = embedder.build_image_payload_for_test(image, context="document")
    query_payload = embedder.build_image_payload_for_test(image, context="query")

    assert embedder.supports_asymmetric is True
    assert index_payload["input_type"] == "document"
    assert query_payload["input_type"] == "query"


def test_image_embedder_can_disable_asymmetric_for_capable_provider() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        asymmetric="disable",
    )

    payload = embedder.build_image_payload_for_test(
        Image.new("RGB", (1, 1), "white"), context="query"
    )

    assert embedder.supports_asymmetric is False
    assert "input_type" not in payload


def test_gemini_embedding_2_uses_symmetric_fallback() -> None:
    embedder = MultimodalEmbedder(
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="key",
        dim=1536,
        provider=GeminiEmbedProvider(),
        asymmetric="auto",
    )

    payload = embedder.build_image_payload_for_test(
        Image.new("RGB", (1, 1), "white"), context="query"
    )

    assert embedder.supports_asymmetric is False
    assert "task_type" not in payload


def test_dimension_mismatch_raises() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    with pytest.raises(ValueError, match="Expected embedding dim 1024"):
        embedder.validate_vectors_for_test([[0.1, 0.2, 0.3]])


async def test_embed_texts_posts_document_context() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    embedder._client.post = AsyncMock(return_value=response)

    result = await embedder.embed_texts(["hello"], context="document")

    assert result == [[0.1, 0.2, 0.3]]
    payload = embedder._client.post.call_args.kwargs["json"]
    assert payload["input_type"] == "document"


async def test_probe_image_embedding_uses_index_context() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    embedder.embed_index_images = AsyncMock(return_value=[[0.1, 0.2, 0.3]])  # type: ignore[method-assign]

    await embedder.probe_image_embedding()

    embedder.embed_index_images.assert_awaited_once()
