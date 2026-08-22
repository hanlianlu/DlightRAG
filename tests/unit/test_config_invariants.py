# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral invariants retained by the canonical 2.0 configuration."""

from __future__ import annotations

import os
import ssl
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.ai.settings import (
    EmbeddingSettings,
    ModelCapacityOverrideSettings,
    ModelSettings,
    ModelsSettings,
    RerankSettings,
)
from dlightrag.config import (
    AccessSectionSettings,
    AnswerConfig,
    AnswerSectionSettings,
    ApiInterfaceSettings,
    CitationHighlightConfig,
    DeploymentSettings,
    DlightragConfig,
    InterfacesSettings,
    LightRAGStorageSettings,
    McpInterfaceSettings,
    PostgresSettings,
    StorageSettings,
    load_config,
)
from dlightrag.rag.settings import (
    BM25ProfileSettings,
    CorpusSettings,
    DoclingSidecarSettings,
    ExtractionSettings,
    MinerUSidecarSettings,
    ParserSidecarsSettings,
    RetrievalSettings,
    VisualAssetSettings,
    VLMSidecarSettings,
)


@pytest.fixture(autouse=True)
def _clean_config_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("DLIGHTRAG_") or key in {
            "LIGHTRAG_PARSER",
            "POSTGRES_SERVER_SETTINGS",
            "POSTGRES_WORKSPACE",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api://dlightrag", "api://dlightrag"),
        (["api://dlightrag", "proxy-id"], ("api://dlightrag", "proxy-id")),
        ('["a", "b"]', ("a", "b")),
        ("   ", None),
    ],
)
def test_jwt_audience_normalization(raw: Any, expected: Any) -> None:
    assert AccessSectionSettings(jwt_audience=raw).jwt_audience == expected


def test_model_defaults_and_case_folding() -> None:
    settings = ModelSettings(provider=cast(Any, " GEMINI "), model="gemini-model")

    assert settings.provider == "gemini"
    assert settings.temperature is None
    assert settings.timeout == 240.0
    assert settings.max_retries == 3
    assert settings.structured_output == "auto"


@pytest.mark.parametrize(
    "values",
    [
        {"model": "x", "temperature": -0.1},
        {"model": "x", "timeout": 0},
        {"model": "x", "max_retries": -1},
        {"model": "x", "structured_output": "json_yaml"},
        {"provider": "anthropic", "model": "x", "structured_output": "json_object"},
        {"provider": "invalid", "model": "x"},
    ],
)
def test_invalid_model_settings_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelSettings(**values)  # type: ignore[arg-type]


def test_capacity_override_accepts_complete_facts_and_rejects_input_overflow() -> None:
    settings = ModelCapacityOverrideSettings(
        provider="openai",
        model="private-model",
        context_window_tokens=262_144,
        max_input_tokens=200_000,
        max_output_tokens=32_768,
        supports_images=True,
        supports_tools=True,
    )
    assert settings.supports_images and settings.supports_tools
    with pytest.raises(ValidationError, match="max_input_tokens"):
        ModelCapacityOverrideSettings(
            model="invalid",
            context_window_tokens=100,
            max_input_tokens=101,
        )


@pytest.mark.parametrize("values", [{"dim": 0}, {"max_token_size": 0}, {"batch_size": 0}])
def test_invalid_embedding_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingSettings(**values)


def test_embedding_defaults_preserve_shipped_pipeline_contract() -> None:
    settings = EmbeddingSettings()
    assert settings.provider == "voyage"
    assert settings.model == "voyage-multimodal-3.5"
    assert settings.base_url == "https://api.voyageai.com/v1"
    assert settings.dim == 1024
    assert settings.max_token_size == 8192
    assert settings.batch_size == 64
    assert settings.max_concurrency == 16
    assert settings.timeout == 120


@pytest.mark.parametrize(
    "values",
    [
        {"max_concurrency": 0},
        {"batch_size": 0},
        {"score_threshold": -0.1},
        {"temperature": -0.1},
    ],
)
def test_invalid_rerank_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RerankSettings(**values)


def test_rerank_defaults_and_provider_case_folding() -> None:
    settings = RerankSettings(provider=cast(Any, " OpenAI "))
    assert settings.enabled is True
    assert settings.strategy == "chat_llm_reranker"
    assert settings.provider == "openai"
    assert settings.max_concurrency == 8
    assert settings.batch_size == 8


def test_answer_and_citation_defaults_are_preserved() -> None:
    answer = AnswerConfig()
    highlights = CitationHighlightConfig()
    assert answer.max_attachments == 6
    assert answer.max_attachment_bytes == 100 * 1024 * 1024
    assert answer.max_total_attachment_bytes == 128 * 1024 * 1024
    assert answer.max_images == 12
    assert answer.image_max_pixels == 40_000_000
    assert highlights.enabled is True
    assert highlights.timeout == 10.0


@pytest.mark.parametrize(
    "values",
    [
        {"max_images": -1},
        {"image_max_total_bytes": 0},
        {"image_min_quality": 96},
        {"max_attachments": -1},
        {"max_attachment_bytes": 0},
    ],
)
def test_invalid_answer_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnswerConfig(**values)


@pytest.mark.parametrize("values", [{"thumb_max_px": 0}, {"thumb_cache_size": 0}])
def test_invalid_visual_asset_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VisualAssetSettings(**values)


def test_postgres_only_storage_literals_and_vector_value_type() -> None:
    storage = LightRAGStorageSettings(vector_index_type="HNSW_HALFVEC")
    assert storage.vector_storage == "PGVectorStorage"
    assert storage.graph_storage == "PGTableGraphStorage"
    assert storage.kv_storage == "PGKVStorage"
    with pytest.raises(ValidationError):
        LightRAGStorageSettings(vector_storage="QdrantStorage")  # type: ignore[arg-type]


def test_vector_and_pool_defaults_export_lightrag_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DlightragConfig()
    config.apply_lightrag_backend_env(force=True)

    assert config.storage.lightrag.hnsw_ef_construction == 256
    assert config.storage.lightrag.hnsw_ef_search == 256
    assert config.domain_pool_server_settings()["hnsw.ef_search"] == "256"
    assert os.environ["POSTGRES_HNSW_EF"] == "256"
    assert os.environ["POSTGRES_VECTOR_INDEX_TYPE"] == "HNSW_HALFVEC"
    assert os.environ["POSTGRES_MAX_CONNECTIONS"] == "16"


def test_postgres_ssl_modes_project_to_asyncpg() -> None:
    required = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="require"))
    )
    disabled = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="disable"))
    )
    verified = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(ssl_mode="verify-full"))
    )
    assert required.pg_connection_kwargs()["ssl"] is True
    assert disabled.pg_connection_kwargs()["ssl"] is False
    context = verified.pg_connection_kwargs()["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True


def test_postgres_session_settings_merge_hnsw_and_reader_policy() -> None:
    config = DlightragConfig(
        deployment=DeploymentSettings(service_role="reader"),
        storage=StorageSettings(
            postgres=PostgresSettings(
                session_settings={"application_name": "test", "hnsw.ef_search": 999}
            ),
            lightrag=LightRAGStorageSettings(hnsw_ef_search=256),
        ),
    )
    assert config.domain_pool_server_settings() == {
        "hnsw.ef_search": "999",
        "application_name": "test",
    }
    assert config.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"


def test_bm25_defaults_cover_languages_and_one_fallback() -> None:
    profiles = RetrievalSettings().bm25_profiles
    assert {profile.languages[0] for profile in profiles if profile.languages} >= {"zh", "en"}
    assert sum(profile.fallback for profile in profiles) == 1


@pytest.mark.parametrize(
    "profile",
    [
        {"name": "bad-name", "text_config": "english", "languages": ("en",)},
        {"name": "x", "text_config": "unsafe;drop", "languages": ("en",)},
        {"name": "x", "text_config": "english", "languages": ("en", "de")},
        {"name": "x", "text_config": "simple", "languages": ("en",), "fallback": True},
    ],
)
def test_invalid_bm25_profiles_are_rejected(profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BM25ProfileSettings(**profile)


def test_parser_defaults_to_external_docling_and_explicit_mineru_still_works() -> None:
    defaults = ParserSidecarsSettings()
    assert defaults.active_parser == "docling"
    assert defaults.mineru is None
    assert defaults.docling == DoclingSidecarSettings(
        endpoint="http://127.0.0.1:5001",
        code_formula_preset="granite_docling",
    )

    mineru_only = ParserSidecarsSettings(mineru=MinerUSidecarSettings())
    assert mineru_only.active_parser == "mineru"
    assert mineru_only.docling is None

    both = ParserSidecarsSettings(
        mineru=MinerUSidecarSettings(),
        docling=DoclingSidecarSettings(),
    )
    assert both.active_parser == "mineru"


def test_docling_only_selection_and_sidecar_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "stale")
    config = DlightragConfig(
        corpus=CorpusSettings(
            sidecars=ParserSidecarsSettings(
                docling=DoclingSidecarSettings(endpoint="http://docling:5001")
            )
        )
    )
    config.apply_lightrag_sidecar_env()
    assert config.parser_rules == "*:docling-iteP"
    assert os.environ["DOCLING_ENDPOINT"] == "http://docling:5001"
    assert "MINERU_LOCAL_ENDPOINT" not in os.environ


def test_mineru_backend_and_vlm_environment_are_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DlightragConfig(
        corpus=CorpusSettings(
            sidecars=ParserSidecarsSettings(
                vlm=VLMSidecarSettings(min_image_pixel=80),
                mineru=MinerUSidecarSettings(backend="hybrid-engine"),
            )
        )
    )
    config.apply_lightrag_sidecar_env()
    assert os.environ["MINERU_LOCAL_BACKEND"] == "hybrid-engine"
    assert os.environ["VLM_MIN_IMAGE_PIXEL"] == "80"
    assert "LIGHTRAG_PARSER" not in os.environ
    config.apply_lightrag_runtime_env()
    assert os.environ["LIGHTRAG_PARSER"] == "*:mineru-iteP"


def test_entity_type_prompt_file_is_one_yaml_filename() -> None:
    assert ExtractionSettings(entity_type_prompt_file="finance.yaml").entity_type_prompt_file == (
        "finance.yaml"
    )
    for value in ("../finance.yaml", "/tmp/finance.yaml", "finance.txt"):
        with pytest.raises(ValidationError):
            ExtractionSettings(entity_type_prompt_file=value)


def test_public_listener_without_auth_is_refused_and_override_is_explicit() -> None:
    public = InterfacesSettings(api=ApiInterfaceSettings(host="0.0.0.0"))
    with pytest.raises(ValidationError, match="non-loopback"):
        DlightragConfig(interfaces=public)
    config = DlightragConfig(
        interfaces=public,
        access=AccessSectionSettings(allow_insecure_no_auth=True),
    )
    assert config.interfaces.api.host == "0.0.0.0"


def test_jwt_jwks_and_mcp_oauth_validation() -> None:
    access = AccessSectionSettings(
        auth_mode="jwt",
        jwt_jwks_url="https://issuer.example/jwks.json",
        jwt_issuer="https://issuer.example",
        jwt_audience="api://dlightrag",
        jwt_algorithm="RS256",
    )
    interfaces = InterfacesSettings(
        mcp=McpInterfaceSettings(
            transport="streamable-http",
            resource_server_url="https://rag.example/mcp",
        )
    )
    assert DlightragConfig(access=access, interfaces=interfaces).access.jwt_algorithm == "RS256"
    with pytest.raises(ValidationError, match="requires jwt_issuer and jwt_audience"):
        DlightragConfig(
            access=AccessSectionSettings(auth_mode="jwt", jwt_jwks_url="https://x/jwks")
        )


def test_explicit_env_file_and_error_redaction(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=super-secret-value\n",
        encoding="utf-8",
    )
    config = load_config(env_file)
    assert config.models.chat.default.api_key == "super-secret-value"
    assert "super-secret-value" not in repr(config)

    bad = tmp_path / "bad.env"
    bad.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=never-echo-me\n"
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__TIMEOUT=0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        load_config(bad)
    assert "never-echo-me" not in str(caught.value)


def test_incomplete_secret_only_role_error_never_echoes_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=never-echo-role-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.chat.roles.extract.model") as caught:
        load_config(env_file)
    assert "never-echo-role-secret" not in str(caught.value)


def test_legacy_dotenv_key_is_rejected(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("DLIGHTRAG_POSTGRES_HOST=legacy-db\n", encoding="utf-8")

    with pytest.raises(ValueError, match="postgres_host: Extra inputs"):
        load_config(env_file)


def test_config_composes_canonical_models_without_snapshot_copy() -> None:
    from dlightrag.model_settings import rag_settings

    models = ModelsSettings(embedding=EmbeddingSettings(startup_probe=False))
    answer = AnswerSectionSettings()
    config = DlightragConfig(models=models, answer=answer)
    runtime = rag_settings(config)
    assert config.models is models
    assert config.answer is answer
    assert runtime.models is config.models
    assert runtime.corpus is config.corpus
