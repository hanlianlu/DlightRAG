# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for provider ABC, registry, and concrete implementations."""

from __future__ import annotations

import pytest

from dlightrag.models.providers import get_provider
from dlightrag.models.providers.base import CompletionProvider


class TestCompletionProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CompletionProvider(api_key="k", timeout=10.0, max_retries=1)


class TestProviderRegistry:
    def test_get_openai_provider(self):
        p = get_provider("openai", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_get_anthropic_provider(self):
        p = get_provider("anthropic", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_get_gemini_provider(self):
        p = get_provider("gemini", api_key="test-key")
        assert isinstance(p, CompletionProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="openai"):
            get_provider("bad")
