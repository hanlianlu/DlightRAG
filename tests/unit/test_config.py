# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the new nested provider config schema."""

import os

import pytest
from pydantic import ValidationError

from dlightrag.config import (
    AnswerConfig,
    CitationHighlightConfig,
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
    RerankConfig,
    load_config,
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


class TestCitationHighlightConfig:
    def test_defaults_disabled_for_latency(self):
        cfg = CitationHighlightConfig()
        assert cfg.enabled is False
        assert cfg.timeout == 5.0
        assert cfg.max_concurrency == 4


class TestAnswerConfig:
    def test_defaults_keep_retrieval_candidates_separate_from_prompt_contexts(self):
        cfg = AnswerConfig()
        assert cfg.candidate_top_k == 60
        assert cfg.context_top_k == 30
        assert cfg.candidate_top_k >= cfg.context_top_k


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
        """Unknown fields and typos should raise via extra=forbid."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            DlightragConfig(
                openai_api_key="sk-old",
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
    assert cfg.parser.rules == "docx:native-iteP,*:mineru-iteP"
    assert cfg.extraction.use_json is True
    assert cfg.extraction.language == "English"
    assert cfg.parser_sidecars.vlm.enabled is True
    assert cfg.parser_sidecars.vlm.max_image_bytes == 5_242_880
    assert cfg.parser_sidecars.mineru.api_mode == "local"
    assert cfg.parser_sidecars.mineru.local_endpoint == "http://127.0.0.1:8210"
    assert cfg.parser_sidecars.mineru.local_backend == "hybrid-auto-engine"
    assert cfg.parser_sidecars.mineru.enable_table is True
    assert cfg.parser_sidecars.mineru.enable_formula is True
    assert cfg.metadata.default_ingest_policy == "validate"
    assert cfg.metadata.allow_ad_hoc_json is True
    assert cfg.pg_vector_index_type == "HNSW"
    assert cfg.pg_hnsw_m == 32
    assert cfg.pg_hnsw_ef_construction == 256
    assert cfg.pg_hnsw_ef_search == 256
    assert cfg.postgres_server_settings_dict() == {"hnsw.ef_search": "256"}
    assert cfg.runtime_role == "ingest"
    assert cfg.pg_target_for_runtime() == "primary"
    assert cfg.citations.highlights.enabled is False
    assert cfg.answer.max_images == 6
    assert cfg.answer.image_max_bytes == 3_000_000
    assert cfg.answer.image_max_total_bytes == 24_000_000
    assert cfg.answer.image_max_px == 1536
    assert cfg.answer.image_min_px == 1024
    assert cfg.answer.image_quality == 88
    assert cfg.answer.image_min_quality == 72
    assert cfg.query_images.semantic_enhancement is True
    assert cfg.query_images.max_described_images == 3
    assert cfg.visual_assets.thumb_max_px == 300
    for removed_shortcut in ("chat", "extract", "keywords", "query", "vlm"):
        assert not hasattr(cfg, removed_shortcut)


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


def test_pg_connection_kwargs_uses_primary_fields_by_default() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        postgres_host="primary",
        postgres_port=6543,
        postgres_user="writer",
        postgres_password="writer-pass",
        postgres_database="dlight",
    )

    assert cfg.pg_connection_kwargs() == {
        "host": "primary",
        "port": 6543,
        "user": "writer",
        "password": "writer-pass",
        "database": "dlight",
    }


def test_query_runtime_uses_replica_with_primary_fallback() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        runtime_role="query",
        postgres_host="primary",
        postgres_user="writer",
        postgres_replica_host="replica",
        postgres_replica_user="reader",
    )

    assert cfg.pg_target_for_runtime() == "replica"
    assert cfg.pg_connection_kwargs() == {
        "host": "replica",
        "port": cfg.postgres_port,
        "user": "reader",
        "password": cfg.postgres_password,
        "database": cfg.postgres_database,
    }


def test_read_after_write_primary_route_is_rejected() -> None:
    kwargs: dict[str, object] = {
        "embedding": EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        "read_after_write_mode": "primary_route",
    }
    with pytest.raises(ValidationError):
        DlightragConfig(**kwargs)


@pytest.mark.parametrize("removed_field", ["ingest", "keywords"])
def test_removed_top_level_llm_fields_rejected(removed_field: str) -> None:
    kwargs = {
        "embedding": {
            "provider": "voyage",
            "model": "voyage-multimodal-3.5",
            "api_key": "sk-test",
            "dim": 1024,
            "startup_probe": False,
        },
        removed_field: {"provider": "openai", "model": "gpt-4.1-mini"},
    }

    with pytest.raises(ValidationError):
        DlightragConfig(**kwargs)


def test_removed_parser_engine_process_options_rejected() -> None:
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


def test_unknown_postgres_shadow_fields_rejected() -> None:
    removed_shadow_field = "postgres_" + "primary_host"
    with pytest.raises(ValidationError):
        DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
            **{removed_shadow_field: "primary.internal"},
        )


def test_dotenv_allows_upstream_lightrag_parser_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "VLM_PROCESS_ENABLE=true",
                "VLM_MAX_IMAGE_BYTES=5242880",
                "MINERU_API_MODE=local",
                "MINERU_LOCAL_ENDPOINT=http://127.0.0.1:8210",
                "MINERU_LOCAL_BACKEND=hybrid-auto-engine",
                "MINERU_LOCAL_PARSE_METHOD=auto",
                "MINERU_LOCAL_IMAGE_ANALYSIS=true",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "VLM_PROCESS_ENABLE",
        "VLM_MAX_IMAGE_BYTES",
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_LOCAL_BACKEND",
        "MINERU_LOCAL_PARSE_METHOD",
        "MINERU_LOCAL_IMAGE_ANALYSIS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert cfg.llm.default.api_key == "sk-env"
    assert "vlm_process_enable" not in cfg.model_fields_set
    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://127.0.0.1:8210"


def test_typed_parser_sidecar_config_exports_lightrag_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "VLM_PROCESS_ENABLE",
        "VLM_MAX_IMAGE_BYTES",
        "SURROUNDING_LEADING_MAX_TOKENS",
        "SURROUNDING_TRAILING_MAX_TOKENS",
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_LOCAL_BACKEND",
        "MINERU_LOCAL_PARSE_METHOD",
        "MINERU_LOCAL_IMAGE_ANALYSIS",
        "MINERU_ENABLE_TABLE",
        "MINERU_ENABLE_FORMULA",
        "MINERU_LANGUAGE",
        "MINERU_POLL_INTERVAL_SECONDS",
        "MINERU_MAX_POLLS",
    ):
        monkeypatch.delenv(key, raising=False)

    DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars={
            "vlm": {
                "enabled": True,
                "max_image_bytes": 7_000_000,
                "surrounding_leading_max_tokens": 123,
                "surrounding_trailing_max_tokens": 456,
            },
            "mineru": {
                "api_mode": "local",
                "local_endpoint": "http://shared-mineru.local:8210",
                "local_backend": "pipeline",
                "local_parse_method": "ocr",
                "local_image_analysis": False,
                "enable_table": False,
                "enable_formula": True,
                "language": "ch,en",
                "poll_interval_seconds": 3,
                "max_polls": 42,
            },
        },
    )

    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["VLM_MAX_IMAGE_BYTES"] == "7000000"
    assert os.environ["SURROUNDING_LEADING_MAX_TOKENS"] == "123"
    assert os.environ["SURROUNDING_TRAILING_MAX_TOKENS"] == "456"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://shared-mineru.local:8210"
    assert os.environ["MINERU_LOCAL_BACKEND"] == "pipeline"
    assert os.environ["MINERU_LOCAL_PARSE_METHOD"] == "ocr"
    assert os.environ["MINERU_LOCAL_IMAGE_ANALYSIS"] == "false"
    assert os.environ["MINERU_ENABLE_TABLE"] == "false"
    assert os.environ["MINERU_ENABLE_FORMULA"] == "true"
    assert os.environ["MINERU_LANGUAGE"] == "ch,en"
    assert os.environ["MINERU_POLL_INTERVAL_SECONDS"] == "3"
    assert os.environ["MINERU_MAX_POLLS"] == "42"


def test_sidecar_env_loader_does_not_export_service_helper_keys(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "MINERU_SERVICE_VENV=/tmp/mineru",
                "MINERU_INSTALL_EXTRAS=core,vllm",
                "MINERU_API_HOST=0.0.0.0",
                "MINERU_API_PORT=9001",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "MINERU_SERVICE_VENV",
        "MINERU_INSTALL_EXTRAS",
        "MINERU_API_HOST",
        "MINERU_API_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert "MINERU_SERVICE_VENV" not in os.environ
    assert "MINERU_INSTALL_EXTRAS" not in os.environ
    assert "MINERU_API_HOST" not in os.environ
    assert "MINERU_API_PORT" not in os.environ


def test_postgres_session_settings_merge_hnsw_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_SERVER_SETTINGS", raising=False)
    monkeypatch.delenv("POSTGRES_STATEMENT_CACHE_SIZE", raising=False)

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        pg_hnsw_ef_search=384,
        postgres_session_settings={
            "application_name": "dlightrag api",
            "statement_timeout": "60000",
        },
        postgres_statement_cache_size=256,
    )

    assert cfg.postgres_server_settings_dict() == {
        "hnsw.ef_search": "384",
        "application_name": "dlightrag api",
        "statement_timeout": "60000",
    }
    assert os.environ["POSTGRES_SERVER_SETTINGS"] == (
        "hnsw.ef_search=384&application_name=dlightrag+api&statement_timeout=60000"
    )
    assert os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] == "256"


def test_lightrag_workspace_env_is_not_globalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """LightRAG workspace must come from each instance, not process env."""
    monkeypatch.setenv("POSTGRES_WORKSPACE", "stale_workspace")

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        workspace="fresh_workspace",
    )
    cfg.apply_lightrag_backend_env(force=True)

    assert "POSTGRES_WORKSPACE" not in os.environ


def test_dotenv_rejects_unknown_dlightrag_keys(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    removed_key = "DLIGHTRAG_REMOVED__API_KEY"
    env_file.write_text(f"{removed_key}=sk-old\n", encoding="utf-8")
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    with pytest.raises(ValidationError, match="removed"):
        DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )


def test_load_config_uses_explicit_env_file_without_global_dotenv(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-explicit",
                "DLIGHTRAG_API_PORT=9900",
                "MINERU_API_MODE=local",
                "MINERU_LOCAL_ENDPOINT=http://127.0.0.1:8210",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DLIGHTRAG_API_PORT", raising=False)
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)

    cfg = load_config(
        env_file,
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert cfg.api_port == 9900
    assert cfg.llm.default.api_key == "sk-explicit"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://127.0.0.1:8210"


def test_blank_sidecar_values_do_not_override_typed_defaults(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "MINERU_API_MODE=local",
                "MINERU_LOCAL_ENDPOINT=",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://127.0.0.1:8210"
