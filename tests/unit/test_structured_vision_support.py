# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider_supports_structured_vision."""

from __future__ import annotations

from dlightrag.models.llm import provider_supports_structured_vision


class TestProviderSupportsStructuredVision:
    def test_cloud_providers_supported(self) -> None:
        for provider in (
            "openai",
            "azure_openai",
            "anthropic",
            "google_gemini",
            "qwen",
            "minimax",
            "openrouter",
        ):
            assert provider_supports_structured_vision(provider) is True, provider

    def test_local_providers_not_supported(self) -> None:
        for provider in ("ollama", "xinference"):
            assert provider_supports_structured_vision(provider) is False, provider

    def test_unknown_provider_not_supported(self) -> None:
        assert provider_supports_structured_vision("unknown") is False
