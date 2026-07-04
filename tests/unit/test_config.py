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
    MetadataConfig,
    ModelConfig,
    RerankConfig,
    load_config,
)


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


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model="gpt-5.4-mini")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.4-mini"
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
            ModelConfig(provider=cast(Any, "invalid"), model="test")

    def test_model_kwargs(self):
        cfg = ModelConfig(model="gpt-5.4-mini", model_kwargs={"top_p": 0.9})
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
        assert cfg.max_concurrency == 8
        assert cfg.batch_size == 8
        assert cfg.image_max_bytes == 1_500_000
        assert cfg.image_max_total_bytes == 8_000_000
        assert cfg.image_max_px == 1280
        assert cfg.image_min_px == 768
        assert cfg.image_quality == 86
        assert cfg.image_min_quality == 76

    def test_jina_strategy(self):
        cfg = RerankConfig(strategy="jina_reranker", model="jina-reranker-m0", api_key="key")
        assert cfg.strategy == "jina_reranker"


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
        assert cfg.context_top_k == 30
        assert cfg.image_quality == 89
        assert cfg.image_min_quality == 79


class TestMetadataConfig:
    def test_defaults_expose_only_live_metadata_policy_controls(self) -> None:
        cfg = MetadataConfig()

        assert cfg.allow_ad_hoc_json is True
        assert cfg.default_ingest_policy == "validate"
        assert cfg.fields == {}


class TestDlightragConfigNested:
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

    def test_public_api_host_without_auth_warns(self):
        with pytest.warns(UserWarning, match="api_host"):
            DlightragConfig(
                embedding=EmbeddingConfig(
                    provider="voyage",
                    model="voyage-multimodal-3.5",
                    api_key="sk-test",
                    startup_probe=False,
                ),
                api_host="0.0.0.0",
                auth_mode="none",
            )

    def test_minimal_config(self, tmp_path, monkeypatch):
        """Only embedding required; llm defaults are nested under llm.default."""
        monkeypatch.chdir(tmp_path)
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )
        assert cfg.llm.default.model == "gpt-5.4-mini"
        assert cfg.embedding.model == "voyage-multimodal-3.5"

    def test_chat_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                startup_probe=False,
            ),
        )
        assert cfg.llm.default.model == "gpt-5.4-mini"
        assert cfg.llm.default.temperature == 0.5

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
    assert cfg.graph_storage == "PGGraphStorage"
    assert cfg.kv_storage == "PGKVStorage"
    assert cfg.doc_status_storage == "PGDocStatusStorage"
    assert cfg.embedding.asymmetric == "auto"


def test_parser_defaults_export_lightrag_env() -> None:
    cfg = _default_test_config()

    assert cfg.parser.rules == "docx:native-iteP,md:native-iteP,textpack:native-iteP,*:mineru-iteP"
    assert cfg.extraction.use_json is True
    assert cfg.extraction.language == "English"
    assert cfg.parser_sidecars.vlm.enabled is True
    assert cfg.parser_sidecars.vlm.max_image_bytes == 5_242_880
    assert cfg.parser_sidecars.vlm.surrounding_leading_max_tokens == 128
    assert cfg.parser_sidecars.vlm.surrounding_trailing_max_tokens == 128
    assert cfg.parser_sidecars.mineru.api_mode == "local"
    assert cfg.parser_sidecars.mineru.local_endpoint == "http://127.0.0.1:8210"
    assert cfg.parser_sidecars.mineru.language == "ch"
    assert cfg.parser_sidecars.mineru.auxiliary_block_policy == "conservative"
    assert (
        os.environ["LIGHTRAG_PARSER"]
        == "docx:native-iteP,md:native-iteP,textpack:native-iteP,*:mineru-iteP"
    )
    assert os.environ["DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY"] == "conservative"
    assert cfg.input_dir_path == cfg.working_dir_path / "inputs"
    assert os.environ["INPUT_DIR"] == str(cfg.input_dir_path)


def test_metadata_and_remote_source_defaults() -> None:
    cfg = _default_test_config()

    assert cfg.metadata.default_ingest_policy == "validate"
    assert cfg.metadata.allow_ad_hoc_json is True
    assert cfg.retain_remote_source_files is False


def test_postgres_vector_and_pool_defaults_export_lightrag_env() -> None:
    cfg = _default_test_config()

    assert cfg.pg_vector_index_type == "HNSW_HALFVEC"
    assert cfg.pg_hnsw_m == 32
    assert cfg.pg_hnsw_ef_construction == 256
    assert cfg.pg_hnsw_ef_search == 256
    assert cfg.postgres_lightrag_pool_max_size == 16
    assert cfg.postgres_pool_min_size == 2
    assert cfg.postgres_pool_max_size == 10
    assert os.environ["POSTGRES_VECTOR_INDEX_TYPE"] == "HNSW_HALFVEC"
    assert os.environ["POSTGRES_MAX_CONNECTIONS"] == "16"
    assert cfg.postgres_server_settings_dict() == {"hnsw.ef_search": "256"}
    assert cfg.lightrag_pipeline_kwargs() == {
        "max_parallel_insert": 3,
        "max_parallel_parse_native": 5,
        "max_parallel_parse_mineru": 2,
        "max_parallel_analyze": 5,
        "queue_size_parse": 20,
        "queue_size_analyze": 32,
        "queue_size_insert": 4,
    }


def test_multimodal_retrieval_defaults() -> None:
    cfg = _default_test_config()

    assert cfg.rerank.image_max_bytes == 1_500_000
    assert cfg.rerank.image_max_total_bytes == 8_000_000
    assert cfg.rerank.image_max_px == 1280
    assert cfg.rerank.image_min_px == 768
    assert cfg.rerank.image_quality == 86
    assert cfg.rerank.image_min_quality == 76
    assert cfg.citations.highlights.enabled is True
    assert cfg.answer.max_images == 6
    assert cfg.answer.image_max_bytes == 3_000_000
    assert cfg.answer.image_max_total_bytes == 24_000_000
    assert cfg.answer.image_max_px == 1536
    assert cfg.answer.image_min_px == 1024
    assert cfg.answer.image_quality == 89
    assert cfg.answer.image_min_quality == 79
    assert cfg.checkpoint_session_ttl_seconds == 30 * 24 * 60 * 60
    assert cfg.query_images.max_described_images == 3
    assert cfg.visual_assets.thumb_max_px == 300


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
    )

    assert cfg.llm.default.api_key == "sk-env"
    assert "vlm_process_enable" not in cfg.model_fields_set
    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["SURROUNDING_LEADING_MAX_TOKENS"] == "128"
    assert os.environ["SURROUNDING_TRAILING_MAX_TOKENS"] == "128"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://127.0.0.1:8210"
    assert os.environ["MINERU_LANGUAGE"] == "arabic"
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
        "DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)

    _settings_config(
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
                "auxiliary_block_policy": "extended",
            },
        },
    )

    assert os.environ["VLM_PROCESS_ENABLE"] == "true"
    assert os.environ["VLM_MAX_IMAGE_BYTES"] == "7000000"
    assert os.environ["SURROUNDING_LEADING_MAX_TOKENS"] == "123"
    assert os.environ["SURROUNDING_TRAILING_MAX_TOKENS"] == "456"
    assert os.environ["MINERU_API_MODE"] == "local"
    assert os.environ["MINERU_LOCAL_ENDPOINT"] == "http://shared-mineru.local:8210"
    assert os.environ["MINERU_LANGUAGE"] == "cyrillic"
    assert os.environ["DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY"] == "extended"


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

    assert cfg.postgres_server_settings_dict() == {
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
    cfg.apply_lightrag_backend_env(force=True)

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


def test_lightrag_parser_env_follows_dlightrag_parser_rules(
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
        parser={"rules": "docx:native-iteP,pdf:mineru-iteP,*:mineru-iteP"},
    )

    assert os.environ["LIGHTRAG_PARSER"] == cfg.parser.rules


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
