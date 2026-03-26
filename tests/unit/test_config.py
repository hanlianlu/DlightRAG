# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the new nested provider config schema."""

import pytest
from pydantic import ValidationError

from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig, RerankConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model="gpt-4.1-mini")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4.1-mini"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.temperature is None
        assert cfg.timeout == 120.0
        assert cfg.max_retries == 3
        assert cfg.model_kwargs == {}

    def test_anthropic_provider(self):
        cfg = ModelConfig(provider="anthropic", model="claude-3-5-sonnet")
        assert cfg.provider == "anthropic"

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="invalid", model="test")

    def test_model_kwargs(self):
        cfg = ModelConfig(model="gpt-4.1-mini", model_kwargs={"top_p": 0.9})
        assert cfg.model_kwargs == {"top_p": 0.9}


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "text-embedding-3-large"
        assert cfg.dim == 1024
        assert cfg.max_token_size == 8192
        assert cfg.provider is None

    def test_custom(self):
        cfg = EmbeddingConfig(model="custom-embed", dim=768, max_token_size=4096)
        assert cfg.dim == 768
        assert cfg.max_token_size == 4096


class TestRerankConfig:
    def test_defaults(self):
        cfg = RerankConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "llm_listwise"
        assert cfg.model is None
        assert cfg.score_threshold == 0.5

    def test_cohere_strategy(self):
        cfg = RerankConfig(strategy="cohere", model="rerank-v4.0-pro", api_key="key")
        assert cfg.strategy == "cohere"


class TestDlightragConfigNested:
    def test_minimal_config(self):
        """Only chat + embedding required."""
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.chat.model == "gpt-4.1-mini"
        assert cfg.ingest is None
        assert cfg.embedding.model == "text-embedding-3-large"

    def test_chat_defaults(self):
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.chat.model == "qwen/qwen3.5-flash-02-23"
        assert cfg.chat.temperature == 0.5

    def test_ingest_fallback(self):
        """When ingest is None, consumers should fall back to chat."""
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.ingest is None

    def test_ingest_explicit(self):
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            ingest=ModelConfig(provider="anthropic", model="claude-3-5-sonnet"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.ingest.provider == "anthropic"
        assert cfg.ingest.model == "claude-3-5-sonnet"

    def test_env_var_nested(self, monkeypatch):
        """Test env var override with __ delimiter."""
        monkeypatch.setenv("DLIGHTRAG_CHAT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_CHAT__API_KEY", "sk-env")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "sk-emb")
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(api_key="sk-emb"),
        )
        assert cfg.chat.model == "gpt-4.1"
        assert cfg.chat.api_key == "sk-env"

    def test_legacy_field_detection(self):
        """Legacy field names should raise clear error."""
        with pytest.raises(ValidationError, match="Legacy config fields"):
            DlightragConfig(
                openai_api_key="sk-old",
                embedding=EmbeddingConfig(api_key="sk-test"),
            )
