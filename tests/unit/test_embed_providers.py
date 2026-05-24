# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal embedding provider payload contracts."""

from __future__ import annotations

import pytest

from dlightrag.models.embedding_inputs import ImageEmbeddingInput, TextEmbeddingInput
from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.models.providers.embed_providers import (
    DashScopeQwenEmbedProvider,
    GeminiEmbedProvider,
    JinaEmbedProvider,
    OllamaEmbedProvider,
    OpenAICompatEmbedProvider,
    QwenOpenAICompatEmbedProvider,
    VoyageEmbedProvider,
    detect_embed_provider,
)


class TestEmbedProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            EmbedProvider()


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
    assert payload["input"] == [{"image": "data:image/jpeg;base64,abc"}]


def test_qwen_openai_compat_payload_preserves_image_data_uri_without_task_hint() -> None:
    provider = QwenOpenAICompatEmbedProvider()
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


class TestDetectProvider:
    def test_explicit_openai_compatible(self) -> None:
        p = detect_embed_provider("any-model", provider="openai_compatible")
        assert isinstance(p, OpenAICompatEmbedProvider)

    def test_openai_alias_is_not_an_embedding_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown embed provider"):
            detect_embed_provider("any-model", provider="openai")

    def test_explicit_voyage(self) -> None:
        p = detect_embed_provider("any-model", provider="voyage")
        assert isinstance(p, VoyageEmbedProvider)

    def test_explicit_qwen_openai_compatible(self) -> None:
        p = detect_embed_provider("any-model", provider="qwen_openai_compatible")
        assert isinstance(p, QwenOpenAICompatEmbedProvider)

    def test_auto_detect_voyage_from_model(self) -> None:
        p = detect_embed_provider("voyage-multimodal-3.5")
        assert isinstance(p, VoyageEmbedProvider)

    def test_auto_detect_jina_from_model(self) -> None:
        p = detect_embed_provider("jina-embeddings-v4")
        assert isinstance(p, JinaEmbedProvider)

    def test_auto_detect_dashscope_qwen(self) -> None:
        p = detect_embed_provider("qwen3-vl-embedding")
        assert isinstance(p, DashScopeQwenEmbedProvider)

    def test_auto_detect_ollama_from_base_url(self) -> None:
        p = detect_embed_provider("nomic-embed", base_url="http://localhost:11434")
        assert isinstance(p, OllamaEmbedProvider)
