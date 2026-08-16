# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the new nested provider config schema."""

import os
import ssl
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.config import (
    AnswerConfig,
    CitationHighlightConfig,
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    MinerUSidecarConfig,
    ModelCapacityOverrideConfig,
    ModelConfig,
    ParserSidecarsConfig,
    RerankConfig,
    VisualAssetsConfig,
    load_config,
)
from dlightrag.model_settings import model_settings_for_role


@pytest.fixture(autouse=True)
def _clean_dlightrag_config_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep config unit tests independent from developer shell env and repo .env."""
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)


def _settings_config(**kwargs: Any) -> DlightragConfig:
    return cast(Any, DlightragConfig)(**kwargs)


def _default_test_config() -> DlightragConfig:
    return _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
    )


def _bridge_lightrag_env(config: DlightragConfig) -> None:
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(config)


class TestJwtAudience:
    @staticmethod
    def _config(value: Any) -> DlightragConfig:
        return _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            jwt_audience=value,
        )

    def test_single_string_preserved(self) -> None:
        assert self._config("api://dlightrag").jwt_audience == "api://dlightrag"

    def test_list_preserved(self) -> None:
        assert self._config(["api://dlightrag", "proxy-id"]).jwt_audience == [
            "api://dlightrag",
            "proxy-id",
        ]

    def test_json_array_string_parsed(self) -> None:
        assert self._config('["a", "b"]').jwt_audience == ["a", "b"]

    def test_blank_becomes_none(self) -> None:
        assert self._config("   ").jwt_audience is None


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model="gpt-5.4-mini")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.4-mini"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.temperature is None
        assert cfg.timeout == 240.0
        assert cfg.max_retries == 3
        assert cfg.structured_output == "auto"
        assert cfg.model_kwargs == {}
        assert cfg.agentic_model_kwargs == {}

    def test_anthropic_provider(self):
        cfg = ModelConfig(provider="anthropic", model="claude-3-5-sonnet")
        assert cfg.provider == "anthropic"

    def test_provider_name_is_case_insensitive(self):
        assert ModelConfig(provider=cast(Any, "OpenAI"), model="x").provider == "openai"
        assert ModelConfig(provider=cast(Any, " GEMINI "), model="x").provider == "gemini"
        assert ModelConfig(provider=cast(Any, "Anthropic"), model="x").provider == "anthropic"

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider=cast(Any, "invalid"), model="test")

    def test_model_kwargs(self):
        cfg = ModelConfig(
            model="gpt-5.4-mini",
            model_kwargs={"thinking": {"type": "disabled"}},
            agentic_model_kwargs={"thinking": {"type": "enabled"}},
        )
        assert cfg.model_kwargs == {"thinking": {"type": "disabled"}}
        assert cfg.agentic_model_kwargs == {"thinking": {"type": "enabled"}}

    def test_structured_output_mode(self):
        cfg = ModelConfig(model="deepseek-v4-flash", structured_output="json_object")
        assert cfg.structured_output == "json_object"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model": "gpt-5.4-mini", "temperature": -0.1},
            {"model": "gpt-5.4-mini", "timeout": 0},
            {"model": "gpt-5.4-mini", "max_retries": -1},
            {"model": "gpt-5.4-mini", "structured_output": "json_yaml"},
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4",
                "structured_output": "json_object",
            },
        ],
    )
    def test_rejects_invalid_model_config_values(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(**kwargs)


class TestModelCapacityOverrideConfig:
    def test_accepts_complete_per_model_capacity_facts(self) -> None:
        override = ModelCapacityOverrideConfig(
            provider="openai",
            model="private-model",
            base_url="http://localhost:8888/v1",
            context_window_tokens=262_144,
            max_input_tokens=200_000,
            max_output_tokens=32_768,
            supports_images=True,
            supports_tools=True,
            supports_reasoning=False,
        )

        config = _settings_config(model_capacity_overrides=[override])

        assert config.model_capacity_overrides == [override]

    def test_rejects_duplicate_model_fingerprints(self) -> None:
        override = {
            "provider": "openai",
            "model": "private-model",
            "base_url": "HTTPS://LOCALHOST:443/v1/../v1",
            "context_window_tokens": 262_144,
        }

        with pytest.raises(ValidationError, match="duplicate model capacity override"):
            _settings_config(
                model_capacity_overrides=[
                    override,
                    {
                        **override,
                        "base_url": "https://localhost/v1",
                    },
                ]
            )


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "voyage"
        assert cfg.model == "voyage-multimodal-3.5"
        assert cfg.base_url == "https://api.voyageai.com/v1"
        assert cfg.dim == 1024
        assert cfg.max_token_size == 8192
        assert cfg.input_modality == "auto"
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

    @pytest.mark.parametrize("input_modality", ["auto", "text", "multimodal"])
    def test_accepts_input_modality(self, input_modality: str) -> None:
        cfg = EmbeddingConfig(
            provider="openai_compatible",
            model="embedding-model",
            input_modality=cast(Any, input_modality),
        )

        assert cfg.input_modality == input_modality

    @pytest.mark.parametrize("kwargs", [{"dim": 0}, {"max_token_size": 0}])
    def test_rejects_invalid_numeric_bounds(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="voyage", model="voyage-multimodal-3.5", **kwargs)

    def test_rejects_retired_model_kwargs(self) -> None:
        with pytest.raises(ValidationError):
            cast(Any, EmbeddingConfig)(model_kwargs={"truncation": False})


class TestRerankConfig:
    def test_defaults(self):
        cfg = RerankConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "chat_llm_reranker"
        assert cfg.model is None
        assert cfg.input_modality == "auto"
        assert cfg.score_threshold is None
        assert cfg.max_concurrency == 8
        assert cfg.batch_size == 8

    def test_jina_strategy(self):
        cfg = RerankConfig(strategy="jina_reranker", model="jina-reranker-v3", api_key="key")
        assert cfg.strategy == "jina_reranker"

    def test_input_modality_accepts_text_override(self):
        cfg = RerankConfig(input_modality="text")
        assert cfg.input_modality == "text"

    def test_voyage_strategy(self):
        cfg = RerankConfig(strategy="voyage_reranker", model="rerank-2.5", api_key="key")
        assert cfg.strategy == "voyage_reranker"

    def test_cohere_strategy(self):
        cfg = RerankConfig(strategy="cohere_reranker", model="rerank-v4.0-fast", api_key="key")
        assert cfg.strategy == "cohere_reranker"

    def test_provider_name_is_case_insensitive(self):
        assert RerankConfig(provider=cast(Any, "OpenAI")).provider == "openai"
        assert RerankConfig(provider=cast(Any, " Gemini ")).provider == "gemini"
        assert RerankConfig().provider is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_concurrency": 0},
            {"batch_size": 0},
            {"score_threshold": -0.1},
            {"temperature": -0.1},
        ],
    )
    def test_rejects_invalid_numeric_bounds(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            RerankConfig(**kwargs)


class TestCitationHighlightConfig:
    def test_defaults_enabled_for_web_source_panel_enrichment(self):
        cfg = CitationHighlightConfig()
        assert cfg.enabled is True
        assert cfg.timeout == 10.0
        assert cfg.max_concurrency == 8
        assert cfg.batch_size == 8


class TestAnswerConfig:
    def test_defaults_keep_prompt_context_controls(self):
        cfg = AnswerConfig()
        assert cfg.image_max_pixels == 40_000_000
        assert cfg.image_quality == 89
        assert cfg.image_min_quality == 79

    def test_attachment_admission_defaults(self) -> None:
        cfg = AnswerConfig()
        assert cfg.max_attachments == 6
        assert cfg.max_attachment_bytes == 100 * 1024 * 1024
        assert cfg.max_total_attachment_bytes == 128 * 1024 * 1024

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_images": -1},
            {"image_max_total_bytes": 0},
            {"image_max_pixels": 0},
            {"image_min_px": 0},
            {"image_min_quality": 96},
            {"max_attachments": -1},
            {"max_attachment_bytes": 0},
            {"max_total_attachment_bytes": 0},
        ],
    )
    def test_rejects_invalid_numeric_bounds(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            AnswerConfig(**kwargs)

    def test_zero_image_slots_are_valid_disable_knob(self) -> None:
        cfg = AnswerConfig(max_images=0)
        assert cfg.max_images == 0


class TestQueryAndVisualConfig:
    @pytest.mark.parametrize(
        ("cls", "kwargs"),
        [
            (VisualAssetsConfig, {"thumb_max_px": 0}),
            (VisualAssetsConfig, {"thumb_cache_size": 0}),
        ],
    )
    def test_rejects_invalid_numeric_bounds(self, cls: type, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            cast(Any, cls)(**kwargs)


class TestDlightragConfigNested:
    def test_shipped_model_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cfg = DlightragConfig()

        assert cfg.llm.default == ModelConfig(
            provider="openai",
            model="z-ai/glm-5.2",
            base_url="https://openrouter.ai/api/v1",
            structured_output="json_schema",
            temperature=0.4,
            timeout=240.0,
        )
        assert cfg.llm.roles.extract is None
        assert model_settings_for_role(cfg, "extract").model == cfg.llm.default.model
        assert cfg.llm.roles.keyword is None
        assert model_settings_for_role(cfg, "keyword").model == cfg.llm.default.model
        assert cfg.llm.roles.query is None
        assert model_settings_for_role(cfg, "query").model == cfg.llm.default.model
        assert cfg.llm.roles.vlm is None
        assert model_settings_for_role(cfg, "vlm").model == cfg.llm.default.model
        assert cfg.embedding == EmbeddingConfig()
        assert cfg.rerank.strategy == "chat_llm_reranker"
        assert cfg.rerank.model is None
        assert cfg.rerank.input_modality == "auto"

    def test_api_defaults_to_loopback_for_local_dev(self):
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )

        assert cfg.api_host == "127.0.0.1"
        assert cfg.auth_mode == "none"

    @pytest.mark.parametrize(
        "listener",
        ["rest", "mcp"],
    )
    def test_public_http_listener_without_auth_is_refused(self, listener: str):
        listener_config = (
            {"api_host": "0.0.0.0"}
            if listener == "rest"
            else {"mcp_transport": "streamable-http", "mcp_host": "0.0.0.0"}
        )
        with pytest.raises(ValueError, match="non-loopback"):
            _settings_config(
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
                auth_mode="none",
                **listener_config,
            )

    @pytest.mark.parametrize("listener", ["rest", "mcp"])
    def test_public_http_listener_without_auth_allows_explicit_override(self, listener: str):
        listener_config = (
            {"api_host": "0.0.0.0"}
            if listener == "rest"
            else {"mcp_transport": "streamable-http", "mcp_host": "0.0.0.0"}
        )
        with pytest.warns(UserWarning, match="allow_insecure_no_auth"):
            cfg = _settings_config(
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
                auth_mode="none",
                allow_insecure_no_auth=True,
                **listener_config,
            )
        assert cfg.allow_insecure_no_auth is True

    def test_minimal_config(self, tmp_path, monkeypatch):
        """Model defaults are nested and can be overridden by callers."""
        monkeypatch.chdir(tmp_path)
        cfg = DlightragConfig()
        assert cfg.llm.default.model == "z-ai/glm-5.2"
        assert cfg.embedding.model == "voyage-multimodal-3.5"

    def test_chat_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = DlightragConfig()
        assert cfg.llm.default.model == "z-ai/glm-5.2"
        assert cfg.llm.default.temperature == 0.4

    def test_langfuse_v4_observability_defaults(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )

        assert cfg.langfuse_environment is None
        assert cfg.langfuse_release is None
        assert cfg.langfuse_sample_rate == 1.0
        assert cfg.langfuse_timeout is None
        assert cfg.langfuse_flush_at is None
        assert cfg.langfuse_flush_interval is None

    def test_langfuse_sample_rate_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            DlightragConfig(
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
                langfuse_sample_rate=1.5,
            )

    def test_env_var_nested(self, monkeypatch):
        """Test env var override with __ delimiter."""
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-5.4-mini")
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
        assert cfg.llm.default.model == "gpt-5.4-mini"
        assert cfg.llm.default.api_key == "sk-env"

    def test_unknown_field_rejected(self):
        """Unknown fields and typos should raise via extra=forbid."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            _settings_config(
                openai_api_key="sk-old",
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
            )

    def test_jwt_mode_accepts_verification_key(self) -> None:
        cfg = DlightragConfig(
            auth_mode="jwt",
            jwt_verification_key="test-jwt-verification-key",
            cors_allow_origins=["http://localhost:3000"],
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )

        assert cfg.jwt_verification_key == "test-jwt-verification-key"

    def test_jwt_mode_accepts_jwks_url_with_issuer_and_audience(self) -> None:
        cfg = DlightragConfig(
            auth_mode="jwt",
            jwt_jwks_url="https://login.example.com/discovery/keys",
            jwt_issuer="https://login.example.com/tenant/v2.0",
            jwt_audience="api://dlightrag",
            jwt_algorithm="RS256",
            cors_allow_origins=["http://localhost:3000"],
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )

        assert cfg.jwt_jwks_url == "https://login.example.com/discovery/keys"

    def test_mcp_oauth_resource_server_config(self) -> None:
        cfg = _settings_config(
            auth_mode="jwt",
            jwt_verification_key="test-jwt-verification-key",
            jwt_issuer="https://auth.example.com",
            jwt_audience="https://rag.example.com",
            mcp_transport="streamable-http",
            mcp_resource_server_url="https://rag.example.com",
            cors_allow_origins=["https://app.example.com"],
            embedding=_default_test_config().embedding,
        )

        assert cfg.mcp_resource_server_url == "https://rag.example.com"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"jwt_issuer": None}, "requires jwt_issuer"),
            (
                {"auth_mode": "simple", "api_auth_token": "test-token"},
                "requires auth_mode='jwt' and mcp_transport='streamable-http'",
            ),
            (
                {"mcp_resource_server_url": "http://rag.example.com/mcp"},
                "must use HTTPS except on loopback",
            ),
            (
                {"jwt_issuer": "https://user:secret@auth.example.com"},
                "must not include credentials",
            ),
            (
                {"mcp_resource_server_url": "https://rag.example.com/mcp?tenant=x"},
                "must not include query or fragment",
            ),
        ],
    )
    def test_mcp_oauth_rejects_incomplete_mode(
        self,
        overrides: dict[str, Any],
        message: str,
    ) -> None:
        values = {
            "auth_mode": "jwt",
            "jwt_verification_key": "test-jwt-verification-key",
            "jwt_issuer": "https://auth.example.com",
            "jwt_audience": "https://rag.example.com/mcp",
            "mcp_transport": "streamable-http",
            "mcp_resource_server_url": "https://rag.example.com/mcp",
        }
        values.update(overrides)
        with pytest.raises(ValidationError, match=message):
            _settings_config(
                cors_allow_origins=["https://app.example.com"],
                embedding=_default_test_config().embedding,
                **values,
            )

    def test_jwt_mode_requires_verification_key_or_jwks_url(self) -> None:
        with pytest.raises(ValidationError, match="jwt_verification_key or jwt_jwks_url"):
            DlightragConfig(
                auth_mode="jwt",
                cors_allow_origins=["http://localhost:3000"],
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
            )

    def test_jwt_jwks_url_requires_issuer_and_audience(self) -> None:
        with pytest.raises(ValidationError, match="jwt_issuer and jwt_audience"):
            DlightragConfig(
                auth_mode="jwt",
                jwt_jwks_url="https://login.example.com/discovery/keys",
                jwt_algorithm="RS256",
                cors_allow_origins=["http://localhost:3000"],
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
            )

    @pytest.mark.parametrize(
        "prompt_file",
        [
            "/prompts/domain-entities.yaml",
            "../domain-entities.yaml",
            "entity_type/domain-entities.yaml",
            "domain-entities.md",
            "domain-entities.txt",
        ],
    )
    def test_entity_type_prompt_file_must_match_lightrag_15_contract(
        self, prompt_file: str
    ) -> None:
        """LightRAG loads YAML file names from PROMPT_DIR/entity_type only."""
        with pytest.raises(ValidationError, match="entity_type_prompt_file"):
            _settings_config(
                extraction={"entity_type_prompt_file": prompt_file},
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
            )

    def test_entity_type_prompt_file_accepts_yaml_file_name(self) -> None:
        cfg = _settings_config(
            extraction={"entity_type_prompt_file": "domain-entities.yaml"},
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )

        assert cfg.extraction.entity_type_prompt_file == "domain-entities.yaml"


def test_storage_backends_are_postgres_only() -> None:
    cfg = _default_test_config()

    assert cfg.vector_storage == "PGVectorStorage"
    assert cfg.graph_storage == "PGTableGraphStorage"
    assert cfg.kv_storage == "PGKVStorage"
    assert cfg.doc_status_storage == "PGDocStatusStorage"
    assert cfg.embedding.asymmetric == "auto"


def test_parser_defaults_export_lightrag_env() -> None:
    # Explicit sidecar model, so the curated config.yaml cannot decide the outcome.
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        parser_sidecars=ParserSidecarsConfig(
            mineru=MinerUSidecarConfig(backend="pipeline", language="korean"),
        ),
    )
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(cfg)

    assert cfg.parser_rules == "*:mineru-iteP"
    assert cfg.parser_sidecars.docling is None
    assert os.environ["LIGHTRAG_PARSER"] == "*:mineru-iteP"
    assert os.environ["MINERU_LOCAL_BACKEND"] == "pipeline"
    assert os.environ["MINERU_LANGUAGE"] == "korean"
    assert "DOCLING_ENDPOINT" not in os.environ
    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert cfg.input_dir_path == cfg.working_dir_path / "inputs"
    assert os.environ["INPUT_DIR"] == str(cfg.input_dir_path)


def test_postgres_vector_and_pool_defaults_export_lightrag_env() -> None:
    cfg = _default_test_config()
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(cfg)

    assert cfg.pg_vector_index_type == "HNSW_HALFVEC"
    assert cfg.pg_hnsw_m == 32
    assert cfg.pg_hnsw_ef_construction == 256
    assert cfg.pg_hnsw_ef_search == 256
    assert cfg.postgres_lightrag_pool_max_size == 16
    assert cfg.postgres_pool_min_size == 2
    assert cfg.postgres_pool_max_size == 16
    assert os.environ["POSTGRES_VECTOR_INDEX_TYPE"] == "HNSW_HALFVEC"
    assert os.environ["POSTGRES_MAX_CONNECTIONS"] == "16"
    assert cfg.domain_pool_server_settings() == {"hnsw.ef_search": "256"}


def test_bm25_defaults_cover_supported_language_profiles() -> None:
    cfg = _default_test_config()

    assert [
        (profile.name, profile.text_config, profile.languages, profile.fallback)
        for profile in cfg.bm25_profiles
    ] == [
        ("zh", "public.jiebacfg", ["zh"], False),
        ("en", "english", ["en"], False),
        ("de", "german", ["de"], False),
        ("sv", "swedish", ["sv"], False),
        ("es", "spanish", ["es"], False),
        ("fr", "french", ["fr"], False),
        ("it", "italian", ["it"], False),
        ("pt", "portuguese", ["pt"], False),
        ("nl", "dutch", ["nl"], False),
        ("ru", "russian", ["ru"], False),
        ("da", "danish", ["da"], False),
        ("fi", "finnish", ["fi"], False),
        ("simple", "simple", [], True),
    ]
    assert cfg.bm25_k1 == 1.2
    assert cfg.bm25_b == 0.75


def test_bm25_profiles_accept_safe_pg_textsearch_config_names() -> None:
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        bm25_profiles=[
            {"name": "zh", "text_config": "public.jiebacfg", "languages": ["zh"]},
            {"name": "simple", "text_config": "simple", "fallback": True},
        ],
        bm25_k1=1.4,
        bm25_b=0.65,
    )

    assert cfg.bm25_profiles[0].text_config == "public.jiebacfg"
    assert cfg.bm25_k1 == 1.4
    assert cfg.bm25_b == 0.65


def test_bm25_profiles_reject_multi_language_profile() -> None:
    with pytest.raises(ValidationError, match="exactly one language"):
        _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            bm25_profiles=[
                {"name": "mixed", "text_config": "simple", "languages": ["de", "sv"]},
                {"name": "simple", "text_config": "simple", "fallback": True},
            ],
        )


def test_bm25_profiles_reject_fallback_languages() -> None:
    with pytest.raises(ValidationError, match="fallback profiles must not declare languages"):
        _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            bm25_profiles=[
                {"name": "simple", "text_config": "simple", "languages": ["en"], "fallback": True}
            ],
        )


def test_bm25_profiles_reject_unsafe_text_config_names() -> None:
    with pytest.raises(ValidationError, match="text_config"):
        _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                startup_probe=False,
            ),
            bm25_profiles=[
                {
                    "name": "bad",
                    "text_config": "english'; DROP TABLE x; --",
                    "fallback": True,
                }
            ],
        )


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
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
    )

    assert cfg.pg_vector_index_type == "HNSW_HALFVEC"


def test_vector_index_type_can_fall_back_to_plain_hnsw() -> None:
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=3072,
            startup_probe=False,
        ),
        pg_vector_index_type="HNSW",
    )

    assert cfg.pg_vector_index_type == "HNSW"


def test_role_config_uses_lightrag_role_names() -> None:
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        llm=LLMConfig(
            default=ModelConfig(provider="openai", model="gpt-5.4-mini"),
            roles=LLMRolesConfig(
                extract=ModelConfig(provider="openai", model="gpt-5.4-mini"),
                keyword=ModelConfig(provider="openai", model="gpt-5.4-mini"),
                query=ModelConfig(provider="openai", model="gpt-5.4-mini"),
                vlm=ModelConfig(provider="gemini", model="gemini-2.5-flash"),
            ),
        ),
    )

    assert cfg.llm.roles.keyword is not None
    assert cfg.llm.roles.keyword.model == "gpt-5.4-mini"


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


def test_pg_connection_kwargs_includes_ssl_require_and_exports_lightrag_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "POSTGRES_SSL_MODE",
        "POSTGRES_SSL_CERT",
        "POSTGRES_SSL_KEY",
        "POSTGRES_SSL_ROOT_CERT",
        "POSTGRES_SSL_CRL",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        postgres_ssl_mode="require",
        postgres_ssl_cert="/certs/client.crt",
        postgres_ssl_key="/certs/client.key",
        postgres_ssl_root_cert="/certs/root.crt",
        postgres_ssl_crl="/certs/root.crl",
    )
    _bridge_lightrag_env(cfg)

    assert cfg.pg_connection_kwargs()["ssl"] is True
    assert os.environ["POSTGRES_SSL_MODE"] == "require"
    assert os.environ["POSTGRES_SSL_CERT"] == "/certs/client.crt"
    assert os.environ["POSTGRES_SSL_KEY"] == "/certs/client.key"
    assert os.environ["POSTGRES_SSL_ROOT_CERT"] == "/certs/root.crt"
    assert os.environ["POSTGRES_SSL_CRL"] == "/certs/root.crl"


def test_pg_connection_kwargs_disables_ssl() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        postgres_ssl_mode="disable",
    )

    assert cfg.pg_connection_kwargs()["ssl"] is False


def test_pg_connection_kwargs_builds_verify_ssl_context() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        postgres_ssl_mode="verify-ca",
    )

    ssl_value = cfg.pg_connection_kwargs()["ssl"]
    assert isinstance(ssl_value, ssl.SSLContext)
    assert ssl_value.check_hostname is False


def test_dotenv_ignores_raw_upstream_parser_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "DLIGHTRAG_LLM__DEFAULT__MODEL=gpt-x",
                "VLM_PROCESS_ENABLE=false",
                "VLM_MIN_IMAGE_PIXEL=32",
                "MINERU_API_MODE=official",
                "MINERU_LOCAL_ENDPOINT=http://stale-mineru:8210",
                "MINERU_LOCAL_BACKEND=pipeline",
                "MINERU_LOCAL_PARSE_METHOD=auto",
                "MINERU_LOCAL_IMAGE_ANALYSIS=true",
                "MINERU_LANGUAGE=arabic",
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
        "MINERU_LANGUAGE",
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
        parser_sidecars=ParserSidecarsConfig(mineru=MinerUSidecarConfig()),
    )
    mineru = cfg.parser_sidecars.mineru
    assert mineru is not None
    _bridge_lightrag_env(cfg)

    assert cfg.llm.default.api_key == "sk-env"
    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert os.environ["SURROUNDING_LEADING_MAX_TOKENS"] == "256"
    assert os.environ["SURROUNDING_TRAILING_MAX_TOKENS"] == "256"
    assert os.environ["MINERU_API_MODE"] == mineru.api_mode == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == mineru.local_endpoint
    assert os.environ["MINERU_LOCAL_ENDPOINT"] != "http://stale-mineru:8210"
    assert os.environ["MINERU_LANGUAGE"] == mineru.language != "arabic"
    assert os.environ["MINERU_LOCAL_BACKEND"] == "hybrid-engine"
    assert "MINERU_LOCAL_IMAGE_ANALYSIS" not in os.environ


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

    cfg = _settings_config(
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
                "language": "cyrillic",
            },
        },
    )
    _bridge_lightrag_env(cfg)

    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["VLM_MAX_IMAGE_BYTES"] == "7000000"
    assert os.environ["SURROUNDING_LEADING_MAX_TOKENS"] == "123"
    assert os.environ["SURROUNDING_TRAILING_MAX_TOKENS"] == "456"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://shared-mineru.local:8210"
    assert os.environ["MINERU_LANGUAGE"] == "cyrillic"
    assert os.environ["MINERU_POLL_INTERVAL_SECONDS"] == "5"
    assert os.environ["MINERU_MAX_POLLS"] == "1440"


def test_docling_parser_exports_only_docling_and_shared_vlm_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://stale-mineru:8210")

    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars={
            "mineru": None,
            "docling": {
                "endpoint": "http://docling.internal:5001",
            },
        },
    )
    _bridge_lightrag_env(cfg)

    assert cfg.parser_rules == "*:docling-iteP"
    assert cfg.parser_sidecars.docling is not None
    assert cfg.parser_sidecars.docling.endpoint == "http://docling.internal:5001"
    assert cfg.parser_sidecars.docling.max_polls == 1440
    assert os.environ["DOCLING_ENDPOINT"] == "http://docling.internal:5001"
    assert os.environ["DOCLING_POLL_INTERVAL_SECONDS"] == "5"
    assert os.environ["DOCLING_MAX_POLLS"] == "1440"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert "MINERU_API_MODE" not in os.environ
    assert "MINERU_LOCAL_ENDPOINT" not in os.environ


def test_mineru_parser_clears_stale_docling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://stale-docling:5001")

    cfg = _default_test_config()
    _bridge_lightrag_env(cfg)

    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert "DOCLING_ENDPOINT" not in os.environ


def test_vlm_min_image_pixel_rejects_none() -> None:
    with pytest.raises(ValidationError):
        _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
            parser_sidecars={"vlm": {"min_image_pixel": None}},
        )


def test_mineru_takes_priority_when_both_parser_sidecars_are_configured() -> None:
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars={
            "mineru": {"local_endpoint": "http://mineru.internal:8210"},
            "docling": {"endpoint": "http://docling.internal:5001"},
        },
    )
    _bridge_lightrag_env(cfg)

    assert cfg.parser_rules == "*:mineru-iteP"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://mineru.internal:8210"
    assert "DOCLING_ENDPOINT" not in os.environ


def test_mineru_backend_maps_to_env_and_uses_canonical_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINERU_LOCAL_BACKEND", raising=False)

    # Explicit backend is exported to LightRAG's MINERU_LOCAL_BACKEND env.
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars={"mineru": {"backend": "pipeline"}},
    )
    _bridge_lightrag_env(cfg)
    assert os.environ["MINERU_LOCAL_BACKEND"] == "pipeline"

    # Unset backend exports DlightRAG's canonical default instead of inheriting upstream.
    monkeypatch.delenv("MINERU_LOCAL_BACKEND", raising=False)
    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars=ParserSidecarsConfig(mineru=MinerUSidecarConfig(language="ch")),
    )
    _bridge_lightrag_env(cfg)
    assert os.environ["MINERU_LOCAL_BACKEND"] == "hybrid-engine"


def test_mineru_backend_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        cast(Any, MinerUSidecarConfig)(backend="paddle")


@pytest.mark.parametrize("legacy_backend", ["vlm-auto-engine", "hybrid-auto-engine"])
def test_mineru_backend_rejects_legacy_aliases(legacy_backend: str) -> None:
    with pytest.raises(ValidationError):
        cast(Any, MinerUSidecarConfig)(backend=legacy_backend)


def test_sidecar_env_loader_does_not_export_service_helper_keys(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "DLIGHTRAG_LLM__DEFAULT__MODEL=gpt-x",
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
    monkeypatch.delenv("POSTGRES_CONNECTION_RETRIES", raising=False)
    monkeypatch.delenv("POSTGRES_CONNECTION_RETRY_BACKOFF", raising=False)
    monkeypatch.delenv("POSTGRES_CONNECTION_RETRY_BACKOFF_MAX", raising=False)
    monkeypatch.delenv("POSTGRES_POOL_CLOSE_TIMEOUT", raising=False)
    monkeypatch.delenv("POSTGRES_MAX_CONNECTIONS", raising=False)

    cfg = _settings_config(
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
        postgres_connection_retries=12,
        postgres_connection_retry_backoff=1.5,
        postgres_connection_retry_backoff_max=9.0,
        postgres_pool_close_timeout=2.5,
        postgres_lightrag_pool_max_size=18,
    )
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(cfg)

    assert cfg.domain_pool_server_settings() == {
        "hnsw.ef_search": "384",
        "application_name": "dlightrag api",
        "statement_timeout": "60000",
    }
    assert os.environ["POSTGRES_SERVER_SETTINGS"] == (
        "hnsw.ef_search=384&application_name=dlightrag+api&statement_timeout=60000"
    )
    assert os.environ["POSTGRES_STATEMENT_CACHE_SIZE"] == "256"
    assert os.environ["POSTGRES_CONNECTION_RETRIES"] == "12"
    assert os.environ["POSTGRES_CONNECTION_RETRY_BACKOFF"] == "1.5"
    assert os.environ["POSTGRES_CONNECTION_RETRY_BACKOFF_MAX"] == "9.0"
    assert os.environ["POSTGRES_POOL_CLOSE_TIMEOUT"] == "2.5"
    assert os.environ["POSTGRES_MAX_CONNECTIONS"] == "18"


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
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(cfg)

    assert "POSTGRES_WORKSPACE" not in os.environ


def test_url_ingest_private_host_allowlist_defaults_empty() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )

    assert cfg.url_ingest_private_host_allowlist == []


def test_lightrag_parser_env_follows_active_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LightRAG parser routing must share DlightRAG's product-level policy."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "pdf:stale-route")

    cfg = _settings_config(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
        parser_sidecars={
            "mineru": None,
            "docling": {"endpoint": "http://docling.internal:5001"},
        },
    )
    from dlightrag.adapters.postgres.corpus import apply_lightrag_environment

    apply_lightrag_environment(cfg)

    assert os.environ["LIGHTRAG_PARSER"] == cfg.parser_rules == "*:docling-iteP"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_request_timeout": 0},
        {"max_upload_bytes": 0},
        {"max_upload_size_mb": 0},
        {"request_timeout": 0},
        {"ingest_timeout": -1},
        {"parser_sidecars": {"vlm": {"max_image_bytes": 0}}},
        {"parser_sidecars": {"vlm": {"surrounding_leading_max_tokens": -1}}},
        {"parser_sidecars": {"mineru": {"max_polls": 0}}},
        {"visual_assets": {"thumb_max_px": 0}},
    ],
)
def test_config_parsing_rejects_invalid_numeric_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _settings_config(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
            **kwargs,
        )


def test_dotenv_rejects_unknown_dlightrag_keys(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    unknown_key = "DLIGHTRAG_UNKNOWN__API_KEY"
    env_file.write_text(f"{unknown_key}=sk-old\n", encoding="utf-8")
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    with pytest.raises(ValidationError, match="unknown"):
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
                "DLIGHTRAG_LLM__DEFAULT__MODEL=gpt-x",
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
    _bridge_lightrag_env(cfg)

    assert cfg.api_port == 9900
    assert cfg.llm.default.api_key == "sk-explicit"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == MinerUSidecarConfig().local_endpoint


def test_load_config_rejection_never_quotes_the_api_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role with a key but no model is invalid; the key must not ride along."""
    monkeypatch.chdir(tmp_path)  # the repo config.yaml would supply the missing model
    secret = "sk-or-v1-must-never-be-logged"
    env_file = tmp_path / ".env"
    env_file.write_text(f"DLIGHTRAG_LLM__ROLES__QUERY__API_KEY={secret}\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_config(env_file)

    assert "llm.roles.query.model: Field required" in str(excinfo.value)
    assert secret not in str(excinfo.value)
    # A chained pydantic error would put the key back into the traceback.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_config_repr_redacts_api_keys():
    """repr(config) must not expose plaintext API keys."""
    cfg = _settings_config(
        embedding={
            "provider": "openai_compatible",
            "model": "text-embed-v4",
            "api_key": "sk-secret-key-12345678",
        },
        llm={
            "default": {
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "api_key": "sk-llm-secret-abcd1234",
            },
        },
    )
    rendered = repr(cfg)
    assert "sk-secret-key-12345678" not in rendered
    assert "sk-llm-secret-abcd1234" not in rendered
    assert "***" in rendered


def test_blank_sidecar_values_do_not_override_typed_defaults(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DLIGHTRAG_LLM__DEFAULT__API_KEY=sk-env",
                "DLIGHTRAG_LLM__DEFAULT__MODEL=gpt-x",
                "MINERU_API_MODE=local",
                "MINERU_LOCAL_ENDPOINT=",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MINERU_API_MODE", raising=False)
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", env_file)

    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )
    _bridge_lightrag_env(cfg)

    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == MinerUSidecarConfig().local_endpoint
