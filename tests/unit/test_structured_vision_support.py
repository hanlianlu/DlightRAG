# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for supports_structured function attributes on model funcs."""

from __future__ import annotations

from unittest.mock import patch, MagicMock


class TestTextLLMFuncStructuredAttribute:
    """Test that get_llm_model_func tags supports_structured correctly."""

    def _make_config(self, provider: str) -> MagicMock:
        cfg = MagicMock()
        cfg.llm_provider = provider
        cfg.chat_model_name = "test-model"
        cfg._get_provider_api_key.return_value = "fake-key"
        cfg._get_url.return_value = "http://localhost"
        cfg.ollama_base_url = "http://localhost:11434"
        return cfg

    @patch("dlightrag.models.llm.partial")
    def test_openai_compatible_tagged_true(self, mock_partial) -> None:
        from dlightrag.models.llm import get_llm_model_func
        mock_partial.return_value = MagicMock()
        for provider in ("openai", "qwen", "minimax", "openrouter", "xinference"):
            func = get_llm_model_func(self._make_config(provider))
            assert getattr(func, "supports_structured", False) is True, provider

    @patch("dlightrag.models.llm.partial")
    def test_azure_openai_tagged_true(self, mock_partial) -> None:
        from dlightrag.models.llm import get_llm_model_func
        mock_partial.return_value = MagicMock()
        func = get_llm_model_func(self._make_config("azure_openai"))
        assert getattr(func, "supports_structured", False) is True

    def test_ollama_tagged_false(self) -> None:
        from dlightrag.models.llm import get_llm_model_func
        with patch("dlightrag.models.llm.partial"), \
             patch.dict("sys.modules", {"lightrag.llm.ollama": MagicMock()}):
            func = get_llm_model_func(self._make_config("ollama"))
        assert getattr(func, "supports_structured", False) is False

    def test_anthropic_defaults_false(self) -> None:
        from dlightrag.models.llm import get_llm_model_func
        with patch.dict("sys.modules", {"lightrag.llm.anthropic": MagicMock()}):
            func = get_llm_model_func(self._make_config("anthropic"))
        # anthropic returns a bare partial() — no supports_structured attribute set
        assert getattr(func, "supports_structured", False) is False

    def test_google_gemini_defaults_false(self) -> None:
        from dlightrag.models.llm import get_llm_model_func
        with patch.dict("sys.modules", {"lightrag.llm.gemini": MagicMock()}):
            func = get_llm_model_func(self._make_config("google_gemini"))
        # google_gemini returns a bare partial() — no supports_structured attribute set
        assert getattr(func, "supports_structured", False) is False


class TestVisionFuncStructuredAttribute:
    """Test that get_vision_model_func tags supports_structured correctly."""

    def _make_config(self, provider: str) -> MagicMock:
        cfg = MagicMock()
        cfg.effective_vision_provider = provider
        cfg.vision_model_name = "test-model"
        cfg._get_provider_api_key.return_value = "fake-key"
        cfg._get_url.return_value = "http://localhost"
        cfg.vision_model_kwargs = {}
        cfg.vision_temperature = 0.7
        cfg.llm_request_timeout = 30
        cfg.llm_max_retries = 2
        return cfg

    def test_openai_compatible_vision_tagged_true(self) -> None:
        from dlightrag.models.llm import get_vision_model_func
        for provider in ("openai", "azure_openai", "qwen", "minimax", "openrouter"):
            func = get_vision_model_func(self._make_config(provider))
            assert func is not None
            assert getattr(func, "supports_structured", False) is True, provider

    def test_xinference_vision_tagged_false(self) -> None:
        from dlightrag.models.llm import get_vision_model_func
        func = get_vision_model_func(self._make_config("xinference"))
        assert func is not None
        assert getattr(func, "supports_structured", False) is False

    def test_ollama_vision_tagged_false(self) -> None:
        from dlightrag.models.llm import get_vision_model_func
        func = get_vision_model_func(self._make_config("ollama"))
        assert func is not None
        assert getattr(func, "supports_structured", False) is False

    def test_anthropic_vision_tagged_true(self) -> None:
        from dlightrag.models.llm import get_vision_model_func
        func = get_vision_model_func(self._make_config("anthropic"))
        assert func is not None
        assert getattr(func, "supports_structured", False) is True

    def test_google_gemini_vision_tagged_true(self) -> None:
        from dlightrag.models.llm import get_vision_model_func
        func = get_vision_model_func(self._make_config("google_gemini"))
        assert func is not None
        assert getattr(func, "supports_structured", False) is True


class TestLogAnswerProviderOptional:
    """Test that log_answer_llm_output works with and without provider."""

    def test_log_without_provider(self) -> None:
        from dlightrag.utils.logging import log_answer_llm_output
        log_answer_llm_output("test", structured=False, query="q")

    def test_log_with_provider(self) -> None:
        from dlightrag.utils.logging import log_answer_llm_output
        log_answer_llm_output("test", structured=False, provider="openai", query="q")
