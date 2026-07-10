# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal embedding provider payload contracts."""

from typing import Any, cast

import pytest

from dlightrag.models.embedding_inputs import (
    ImageEmbeddingInput,
    MultimodalEmbeddingInput,
    TextEmbeddingInput,
)
from dlightrag.models.providers import embed_providers
from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.models.providers.embed_providers import (
    DashScopeQwenEmbedProvider,
    GeminiEmbedProvider,
    JinaEmbedProvider,
    OllamaEmbedProvider,
    OpenAICompatibleEmbedProvider,
    VoyageEmbedProvider,
)


class TestEmbedProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, EmbedProvider)()


def test_voyage_payload_omits_input_type_in_symmetric_mode() -> None:
    payload = VoyageEmbedProvider().build_payload(
        "voyage-multimodal-3.5",
        [TextEmbeddingInput(text="hello")],
        context="query",
        asymmetric=False,
        output_dimension=1024,
    )

    assert "input_type" not in payload
    assert payload["inputs"][0]["content"] == [{"type": "text", "text": "hello"}]


def test_voyage_payload_sets_input_type_when_asymmetric_resolves_active() -> None:
    payload = VoyageEmbedProvider().build_payload(
        "voyage-multimodal-3.5",
        [TextEmbeddingInput(text="hello")],
        context="query",
        asymmetric=True,
        output_dimension=1024,
    )

    assert payload["input_type"] == "query"


def test_dashscope_qwen_payload_sets_dimension_and_image() -> None:
    payload = DashScopeQwenEmbedProvider().build_payload(
        "qwen3-vl-embedding",
        [ImageEmbeddingInput(data_uri="data:image/png;base64,abc")],
        context="document",
        asymmetric=False,
        output_dimension=2048,
    )

    assert payload["parameters"]["dimension"] == 2048
    assert payload["input"]["contents"] == [{"image": "data:image/png;base64,abc"}]


def test_dashscope_qwen_fused_payload_sets_enable_fusion() -> None:
    payload = DashScopeQwenEmbedProvider().build_payload(
        "qwen3-vl-embedding",
        [
            MultimodalEmbeddingInput(
                parts=[
                    TextEmbeddingInput(text="white running shoe"),
                    ImageEmbeddingInput(data_uri="data:image/png;base64,abc"),
                ]
            )
        ],
        context="document",
        asymmetric=False,
        output_dimension=2048,
    )

    assert payload["input"]["contents"] == [
        {"text": "white running shoe"},
        {"image": "data:image/png;base64,abc"},
    ]
    assert payload["parameters"] == {"dimension": 2048, "enable_fusion": True}


def test_gemini_embedding_2_payload_is_multimodal_without_task_type() -> None:
    provider = GeminiEmbedProvider()
    payload = provider.build_payload(
        "gemini-embedding-2",
        [ImageEmbeddingInput(data_uri="data:image/png;base64,abc")],
        context="document",
        asymmetric=True,
        output_dimension=1536,
    )

    assert provider.supports_asymmetric is False
    assert payload["output_dimensionality"] == 1536
    assert "task_type" not in payload
    assert payload["content"]["parts"] == [
        {"inline_data": {"mime_type": "image/png", "data": "abc"}}
    ]


def test_jina_payload_maps_context_only_when_asymmetric() -> None:
    symmetric_payload = JinaEmbedProvider().build_payload(
        "jina-embeddings-v4",
        [ImageEmbeddingInput(data_uri="data:image/jpeg;base64,abc")],
        context="query",
        asymmetric=False,
        output_dimension=2048,
    )
    payload = JinaEmbedProvider().build_payload(
        "jina-embeddings-v4",
        [ImageEmbeddingInput(data_uri="data:image/jpeg;base64,abc")],
        context="query",
        asymmetric=True,
        output_dimension=2048,
    )

    assert "task" not in symmetric_payload
    assert payload["task"] == "retrieval.query"
    assert payload["input"] == [{"bytes": "abc"}]


def test_jina_payload_uses_url_for_remote_images() -> None:
    payload = JinaEmbedProvider().build_payload(
        "jina-embeddings-v4",
        [ImageEmbeddingInput(url="https://example.com/image.png")],
        context="document",
        asymmetric=True,
        output_dimension=2048,
    )

    assert payload["input"] == [{"url": "https://example.com/image.png"}]


def test_openai_compatible_payload_preserves_image_data_uri_without_task_hint() -> None:
    provider = OpenAICompatibleEmbedProvider()
    payload = provider.build_payload(
        "qwen3-vl-embedding-2b",
        [ImageEmbeddingInput(data_uri="data:image/png;base64,abc")],
        context="document",
        asymmetric=True,
        output_dimension=2048,
    )

    assert provider.supports_asymmetric is False
    assert payload["input"] == ["data:image/png;base64,abc"]
    assert payload["encoding_format"] == "float"
    assert "input_type" not in payload


@pytest.mark.parametrize(
    ("provider_name", "provider_type"),
    [
        ("voyage", VoyageEmbedProvider),
        ("dashscope_qwen", DashScopeQwenEmbedProvider),
        ("gemini", GeminiEmbedProvider),
        ("jina", JinaEmbedProvider),
        ("openai_compatible", OpenAICompatibleEmbedProvider),
        ("ollama", OllamaEmbedProvider),
    ],
)
def test_get_embed_provider_uses_explicit_registry(
    provider_name: str,
    provider_type: type[EmbedProvider],
) -> None:
    assert isinstance(embed_providers.get_embed_provider(provider_name), provider_type)


def test_get_embed_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider 'openai'"):
        embed_providers.get_embed_provider("openai")
