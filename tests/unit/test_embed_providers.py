# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for EmbedProvider ABC and implementations."""

from __future__ import annotations

import pytest

from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.models.providers.embed_providers import (
    OpenAICompatEmbedProvider,
    VoyageEmbedProvider,
    JinaEmbedProvider,
    OllamaEmbedProvider,
    detect_embed_provider,
)


class TestEmbedProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            EmbedProvider()


class TestOpenAICompatEmbed:
    def test_endpoint(self):
        p = OpenAICompatEmbedProvider()
        assert p.endpoint == "/embeddings"

    def test_build_payload_text(self):
        p = OpenAICompatEmbedProvider()
        payload = p.build_payload("text-embedding-3-large", ["hello", "world"])
        assert payload["model"] == "text-embedding-3-large"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"

    def test_parse_response(self):
        p = OpenAICompatEmbedProvider()
        data = {"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]}
        result = p.parse_response(data)
        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestVoyageEmbed:
    def test_endpoint(self):
        p = VoyageEmbedProvider()
        assert p.endpoint == "/multimodalembeddings"

    def test_build_payload_text(self):
        p = VoyageEmbedProvider()
        payload = p.build_payload("voyage-multimodal-3", ["hello"])
        assert payload["model"] == "voyage-multimodal-3"
        assert payload["inputs"][0]["content"][0]["type"] == "text"


class TestDetectProvider:
    def test_explicit_openai(self):
        p = detect_embed_provider("any-model", provider="openai")
        assert isinstance(p, OpenAICompatEmbedProvider)

    def test_explicit_voyage(self):
        p = detect_embed_provider("any-model", provider="voyage")
        assert isinstance(p, VoyageEmbedProvider)

    def test_auto_detect_voyage_from_model(self):
        p = detect_embed_provider("voyage-multimodal-3")
        assert isinstance(p, VoyageEmbedProvider)

    def test_auto_detect_ollama_from_base_url(self):
        p = detect_embed_provider("nomic-embed", base_url="http://localhost:11434")
        assert isinstance(p, OllamaEmbedProvider)

    def test_default_is_openai(self):
        p = detect_embed_provider("text-embedding-3-large")
        assert isinstance(p, OpenAICompatEmbedProvider)
