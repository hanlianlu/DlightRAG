# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the new nested provider config schema."""

import pytest
from pydantic import ValidationError

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
    RerankConfig,
)


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
        cfg = EmbeddingConfig(provider="voyage", model="voyage-multimodal-3.5")
        assert cfg.provider == "voyage"
        assert cfg.model == "voyage-multimodal-3.5"
        assert cfg.dim == 1024
        assert cfg.max_token_size == 8192
        assert cfg.asymmetric == "auto"
        assert cfg.startup_probe is True

    def test_custom(self):
        cfg = EmbeddingConfig(
            provider="jina",
            model="jina-embeddings-v4",
            dim=2048,
            max_token_size=4096,
            asymmetric="require",
            startup_probe=False,
        )
        assert cfg.dim == 2048
        assert cfg.max_token_size == 4096
        assert cfg.asymmetric == "require"
        assert cfg.startup_probe is False


class TestRerankConfig:
    def test_defaults(self):
        cfg = RerankConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "chat_llm_reranker"
        assert cfg.model is None
        assert cfg.score_threshold == 0.5
        assert cfg.batch_size == 7

    def test_jina_strategy(self):
        cfg = RerankConfig(strategy="jina_reranker", model="jina-reranker-m0", api_key="key")
        assert cfg.strategy == "jina_reranker"


class TestDlightragConfigNested:
    def test_minimal_config(self):
        """Only embedding required; llm defaults are nested under llm.default."""
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )
        assert cfg.llm.default.model == "google/gemini-2.5-flash-lite"
        assert cfg.embedding.model == "voyage-multimodal-3.5"

    def test_chat_defaults(self):
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )
        assert cfg.llm.default.model == "google/gemini-2.5-flash-lite"
        assert cfg.llm.default.temperature == 0.5

    def test_env_var_nested(self, monkeypatch):
        """Test env var override with __ delimiter."""
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__API_KEY", "sk-env")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "sk-emb")
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-emb",
                startup_probe=False,
            ),
        )
        assert cfg.llm.default.model == "gpt-4.1"
        assert cfg.llm.default.api_key == "sk-env"

    def test_unknown_field_rejected(self):
        """Unknown fields (legacy schema, typos) should raise via extra=forbid."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            DlightragConfig(
                openai_api_key="sk-old",  # legacy flat-schema field
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
            )


def test_storage_backends_are_postgres_only() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
    )

    assert cfg.vector_storage == "PGVectorStorage"
    assert cfg.graph_storage == "PGGraphStorage"
    assert cfg.kv_storage == "PGKVStorage"
    assert cfg.doc_status_storage == "PGDocStatusStorage"
    assert cfg.embedding.asymmetric == "auto"
    assert cfg.parser.rules == "*:native-iteP,*:mineru-iteP,*:legacy-R"
    assert cfg.extraction.use_json is True
    assert cfg.metadata.default_ingest_policy == "validate"
    assert cfg.metadata.allow_ad_hoc_json is True
    assert cfg.pg_vector_index_type == "HNSW"
    assert cfg.pg_hnsw_m == 32
    assert cfg.pg_hnsw_ef_construction == 256
    assert cfg.pg_hnsw_ef_search == 256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vector_storage", "ExternalVectorStorage"),
        ("graph_storage", "ExternalGraphStorage"),
        ("kv_storage", "FileKVStorage"),
        ("doc_status_storage", "FileDocStatusStorage"),
    ],
)
def test_non_postgres_storage_rejected(field: str, value: str) -> None:
    kwargs = {
        "embedding": EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        field: value,
    }
    with pytest.raises(ValidationError):
        DlightragConfig(**kwargs)


def test_pgvector_value_type_is_derived_from_index_type() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
    )

    assert cfg.pg_vector_index_type == "HNSW"
    assert not hasattr(cfg, "pg_vector_value_type")


def test_halfvec_requires_explicit_index_type() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=3072,
            startup_probe=False,
        ),
        pg_vector_index_type="HNSW_HALFVEC",
    )

    assert cfg.pg_vector_index_type == "HNSW_HALFVEC"


def test_role_config_uses_lightrag_role_names() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        llm=LLMConfig(
            default=ModelConfig(provider="openai", model="gpt-4.1"),
            roles=LLMRolesConfig(
                extract=ModelConfig(provider="openai", model="gpt-4.1-mini"),
                keyword=ModelConfig(provider="openai", model="gpt-4.1-mini"),
                query=ModelConfig(provider="openai", model="gpt-4.1"),
                vlm=ModelConfig(provider="gemini", model="gemini-2.5-flash"),
            ),
        ),
    )

    assert cfg.llm.roles.keyword is not None
    assert cfg.llm.roles.keyword.model == "gpt-4.1-mini"


@pytest.mark.parametrize("legacy_field", ["ingest", "keywords"])
def test_legacy_llm_fields_rejected(legacy_field: str) -> None:
    kwargs = {
        "embedding": {
            "provider": "voyage",
            "model": "voyage-multimodal-3.5",
            "api_key": "sk-test",
            "dim": 1024,
            "startup_probe": False,
        },
        legacy_field: {"provider": "openai", "model": "gpt-4.1-mini"},
    }

    with pytest.raises(ValidationError):
        DlightragConfig(**kwargs)


def test_legacy_parser_engine_process_options_rejected() -> None:
    with pytest.raises(ValidationError):
        DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            parser={"engine": "mineru", "process_options": "teP"},
        )
