# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider_supports_structured."""

from __future__ import annotations

from dlightrag.models.llm import provider_supports_structured


class TestProviderSupportsStructured:
    """Test unified structured output support check."""

    def test_vision_cloud_providers_supported(self) -> None:
        for provider in (
            "openai",
            "azure_openai",
            "anthropic",
            "google_gemini",
            "qwen",
            "minimax",
            "openrouter",
        ):
            assert provider_supports_structured(provider, vision=True) is True, provider

    def test_vision_local_providers_not_supported(self) -> None:
        for provider in ("ollama", "xinference"):
            assert provider_supports_structured(provider, vision=True) is False, provider

    def test_text_openai_compatible_supported(self) -> None:
        for provider in (
            "openai",
            "azure_openai",
            "qwen",
            "minimax",
            "openrouter",
            "xinference",
        ):
            assert provider_supports_structured(provider, vision=False) is True, provider

    def test_text_non_openai_not_supported(self) -> None:
        for provider in ("anthropic", "google_gemini", "ollama"):
            assert provider_supports_structured(provider, vision=False) is False, provider

    def test_unknown_provider_not_supported(self) -> None:
        assert provider_supports_structured("unknown", vision=True) is False
        assert provider_supports_structured("unknown", vision=False) is False
