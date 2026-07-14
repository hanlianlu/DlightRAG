# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for context-aware multimodal embedding."""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dlightrag.models.multimodal_embedding import (
    EmbeddingInputModality,
    MultimodalEmbedder,
    ResolvedEmbeddingInputModality,
    resolve_embedding_input_modality,
)
from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.models.providers.embed_providers import (
    GeminiEmbedProvider,
    OllamaEmbedProvider,
    OpenAICompatibleEmbedProvider,
    VoyageEmbedProvider,
)
from dlightrag.utils.images import decode_image_base64


@pytest.mark.parametrize(
    ("provider", "configured", "resolved"),
    [
        (VoyageEmbedProvider(), "auto", "multimodal"),
        (VoyageEmbedProvider(), "text", "text"),
        (VoyageEmbedProvider(), "multimodal", "multimodal"),
        (OpenAICompatibleEmbedProvider(), "auto", "text"),
        (OpenAICompatibleEmbedProvider(), "text", "text"),
        (OpenAICompatibleEmbedProvider(), "multimodal", "multimodal"),
        (OllamaEmbedProvider(), "auto", "text"),
        (OllamaEmbedProvider(), "text", "text"),
    ],
)
def test_resolve_embedding_input_modality(
    provider: EmbedProvider,
    configured: EmbeddingInputModality,
    resolved: ResolvedEmbeddingInputModality,
) -> None:
    assert resolve_embedding_input_modality(provider, configured) == resolved


def test_ollama_rejects_required_multimodal_input() -> None:
    with pytest.raises(
        ValueError,
        match="OllamaEmbedProvider cannot satisfy input_modality='multimodal'",
    ):
        resolve_embedding_input_modality(OllamaEmbedProvider(), "multimodal")


async def test_native_multimodal_provider_can_be_forced_to_text() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
        input_modality="text",
    )

    try:
        assert embedder.supports_images is False
        with pytest.raises(ValueError, match="does not support image embeddings"):
            embedder.build_image_payload_for_test(
                Image.new("RGB", (1, 1), "white"), context="document"
            )
    finally:
        await embedder.aclose()


async def test_openai_compatible_auto_is_text_only() -> None:
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        dim=2048,
        provider=OpenAICompatibleEmbedProvider(),
        input_modality="auto",
    )

    try:
        assert embedder.supports_images is False
    finally:
        await embedder.aclose()


async def test_openai_compatible_explicit_multimodal_enables_images() -> None:
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        dim=2048,
        provider=OpenAICompatibleEmbedProvider(),
        input_modality="multimodal",
    )

    try:
        assert embedder.supports_images is True
    finally:
        await embedder.aclose()


def test_image_embedder_auto_falls_back_for_unsupported_provider() -> None:
    provider = GeminiEmbedProvider()
    embedder = MultimodalEmbedder(
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="key",
        dim=1536,
        provider=provider,
        asymmetric="auto",
    )

    image = Image.new("RGB", (1, 1), "white")
    index_payload = embedder.build_image_payload_for_test(image, context="document")
    query_payload = embedder.build_image_payload_for_test(image, context="query")

    assert embedder.supports_asymmetric is False
    assert "input_type" not in index_payload
    assert "input_type" not in query_payload
    assert index_payload["content"]["parts"][0]["inline_data"]["mime_type"] == "image/png"


def test_voyage_embedder_exposes_fused_multimodal_support() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    assert embedder.supports_fused_multimodal is True
    payload = embedder.build_fused_payload_for_test(
        "a bar chart", Image.new("RGB", (2, 2), "white"), context="document"
    )
    content = payload["inputs"][0]["content"]
    assert content[0] == {"type": "text", "text": "a bar chart"}
    assert content[1]["type"] == "image_base64"
    assert content[1]["image_base64"].startswith("data:image/")


def test_fused_payload_degrades_to_image_only_when_description_blank() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    payload = embedder.build_fused_payload_for_test(
        "   ", Image.new("RGB", (2, 2), "white"), context="document"
    )
    content = payload["inputs"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image_base64"


def test_image_capable_but_non_fused_provider_rejects_fused_payload() -> None:
    # Gemini can embed images but does not fuse text+image into one vector,
    # so the embedder must not offer the fused path (keeps LightRAG's native VLM->text).
    embedder = MultimodalEmbedder(
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="key",
        dim=1536,
        provider=GeminiEmbedProvider(),
    )

    assert embedder.supports_images is True
    assert embedder.supports_fused_multimodal is False
    with pytest.raises(ValueError, match="does not support fused"):
        embedder.build_fused_payload_for_test(
            "x", Image.new("RGB", (2, 2), "white"), context="document"
        )


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


def test_image_embedder_bounds_oversized_images_before_send() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=1024,
        provider=VoyageEmbedProvider(),
    )

    payload = embedder.build_image_payload_for_test(
        Image.new("RGB", (6000, 5000), "white"), context="document"
    )

    data_uri = payload["inputs"][0]["content"][0]["image_base64"]
    raw, _ = decode_image_base64(data_uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 4096
        assert decoded.width * decoded.height <= 15_000_000


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


async def test_text_only_embedder_keeps_text_embedding_and_rejects_images() -> None:
    embedder = MultimodalEmbedder(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        api_key="",
        dim=3,
        provider=OllamaEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    embedder._client.post = AsyncMock(return_value=response)

    result = await embedder.embed_texts(["hello"], context="document")

    assert result == [[0.1, 0.2, 0.3]]
    assert embedder.supports_images is False
    with pytest.raises(ValueError, match="does not support image embeddings"):
        await embedder.embed_index_images([Image.new("RGB", (1, 1), "white")])


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


async def test_image_probe_rejects_wrong_vector_count() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    embedder._client.post = AsyncMock(return_value=response)

    with pytest.raises(ValueError, match="Expected 1 embedding vector"):
        await embedder.probe_image_embedding()


async def test_embed_texts_rejects_wrong_vector_count() -> None:
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

    with pytest.raises(ValueError, match="Expected 2 embedding vectors"):
        await embedder.embed_texts(["hello", "world"])
