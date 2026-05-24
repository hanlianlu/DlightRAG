# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Self-contained configuration for dlightrag.

Configuration sources (highest → lowest precedence):
    1. Constructor arguments (when used as library)
    2. Environment variables (DLIGHTRAG_ prefix)
    3. .env file (secrets + deployment)
    4. config.yaml (structured app settings)
    5. Default values

LightRAG reads backend-specific env vars directly — model_post_init bridges
DLIGHTRAG_* → backend env vars so both modes work seamlessly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_YAML_FILE = "config.yaml"
_ENV_FILE = ".env"


def _find_env_file() -> Path | None:
    """Locate .env in the current working directory only."""
    candidate = Path(_ENV_FILE)
    return candidate if candidate.is_file() else None


def _find_yaml_config() -> Path | None:
    """Locate config.yaml in the current working directory."""
    cwd = Path(_YAML_FILE)
    if cwd.is_file():
        return cwd
    return None


class ModelConfig(BaseModel):
    """Reusable model configuration block."""

    provider: Literal["openai", "anthropic", "gemini"] = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    timeout: float = 120.0
    max_retries: int = 3
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding-specific configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal[
        "voyage",
        "dashscope_qwen",
        "qwen_openai_compatible",
        "gemini",
        "jina",
        "openai_compatible",
        "ollama",
    ]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024
    max_token_size: int = 8192
    asymmetric: Literal["auto", "require", "disable"] = "auto"
    startup_probe: bool = True
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class LLMRolesConfig(BaseModel):
    """Role-specific LLM overrides aligned with LightRAG role names."""

    model_config = ConfigDict(extra="forbid")

    extract: ModelConfig | None = None
    keyword: ModelConfig | None = None
    query: ModelConfig | None = None
    vlm: ModelConfig | None = None


class LLMConfig(BaseModel):
    """Default model plus LightRAG-compatible role overrides."""

    model_config = ConfigDict(extra="forbid")

    default: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider="openai",
            model="gpt-4.1",
            temperature=0.5,
        )
    )
    roles: LLMRolesConfig = Field(default_factory=LLMRolesConfig)


class ParserConfig(BaseModel):
    """LightRAG parser routing rules and optional chunker snapshot overrides."""

    model_config = ConfigDict(extra="forbid")

    rules: str = "*:native-iteP,*:mineru-iteP,*:legacy-R"
    native_images_bypass_lightrag: bool = True
    chunk_options: dict[str, Any] = Field(default_factory=dict)


class ExtractionConfig(BaseModel):
    """Entity/relation extraction stability controls."""

    model_config = ConfigDict(extra="forbid")

    use_json: bool = True
    entity_type_prompt_file: str | None = None


class MetadataConfig(BaseModel):
    """Metadata registry and ingest policy controls."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_ad_hoc_json: bool = True
    default_ingest_policy: Literal["validate", "reject_unknown", "store_only"] = "validate"
    unknown_filter_policy: Literal[
        "reject",
        "ignore_with_warning",
        "allow_declared_json_contains",
    ] = "ignore_with_warning"
    fields: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RerankConfig(BaseModel):
    """Reranking configuration."""

    enabled: bool = True
    strategy: Literal[
        "chat_llm_reranker",
        "jina_reranker",
        "aliyun_reranker",
        "local_reranker",
        "azure_cohere",
    ] = "chat_llm_reranker"
    provider: Literal["openai", "anthropic", "gemini"] | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    score_threshold: float = 0.5
    max_concurrency: int = 4
    batch_size: int = 7
    temperature: float | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class DlightragConfig(BaseSettings):
    """DlightRAG configuration.

    Configuration sources (highest → lowest precedence):
        1. Constructor arguments (when used as library)
        2. Environment variables (DLIGHTRAG_ prefix)
        3. .env file (secrets + deployment)
        4. config.yaml (structured app settings)
        5. Default values

    Supports both pure-.env and hybrid config.yaml + .env setups.
    """

    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Reject any unknown DLIGHTRAG_* env var, .env entry, or
        # config.yaml key. Catches typos AND old flat-schema fields
        # (openai_api_key, llm_provider, etc.) without hardcoding a
        # finite legacy list.
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """Priority: init > env > .env > config.yaml > defaults."""
        from pydantic_settings import YamlConfigSettingsSource

        yaml_path = _find_yaml_config()
        sources = [init_settings, env_settings, dotenv_settings]
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    # ===== PostgreSQL (Default Storage Backend) =====
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="dlightrag")
    postgres_password: str = Field(default="dlightrag")
    postgres_database: str = Field(default="dlightrag")
    postgres_pool_min_size: int = Field(
        default=2, description="DlightRAG domain store pool min connections."
    )
    postgres_pool_max_size: int = Field(
        default=10, description="DlightRAG domain store pool max connections."
    )
    workspace: str = Field(default="default")

    postgres_required_major: int = Field(default=18)
    postgres_min_minor: str = Field(default="18.4")

    # pgvector index configuration. LightRAG derives VECTOR/HALFVEC from this.
    pg_vector_index_type: Literal["HNSW", "HNSW_HALFVEC", "IVFFLAT", "VCHORDRQ"] = Field(
        default="HNSW",
        description="pgvector index type — case-sensitive (HNSW, HNSW_HALFVEC, IVFFLAT, VCHORDRQ)",
    )
    pg_hnsw_m: int = Field(default=32, description="HNSW M parameter (connections per node)")
    pg_hnsw_ef_construction: int = Field(
        default=256, description="HNSW ef_construction (index build quality)"
    )
    pg_hnsw_ef_search: int = Field(
        default=256,
        description="HNSW ef_search (query-time exploration depth, pgvector default is 40). "
        "Bridged via POSTGRES_SERVER_SETTINGS because LightRAG's POSTGRES_HNSW_EF "
        "is actually ef_construction despite the misleading name.",
    )

    # ===== Storage Backends (PostgreSQL-only core path) =====
    vector_storage: Literal["PGVectorStorage"] = Field(
        default="PGVectorStorage", description="LightRAG vector storage backend."
    )
    graph_storage: Literal["PGGraphStorage"] = Field(
        default="PGGraphStorage", description="LightRAG graph storage backend."
    )
    kv_storage: Literal["PGKVStorage"] = Field(
        default="PGKVStorage", description="LightRAG KV storage backend."
    )
    doc_status_storage: Literal["PGDocStatusStorage"] = Field(
        default="PGDocStatusStorage", description="LightRAG doc status backend."
    )

    # ===== Vector DB Backend Kwargs (passthrough to LightRAG) =====
    vector_db_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra parameters forwarded to LightRAG vector_db_storage_cls_kwargs. "
        "Backend-specific — see .env.example for per-backend options. "
        'Example: DLIGHTRAG_VECTOR_DB_KWARGS=\'{"index_type": "HNSW_SQ", "sq_type": "SQ8"}\'',
    )

    # ===== Model config =====
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig  # required, no default
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)

    # ===== RAG Processing =====
    working_dir: str = Field(default="./dlightrag_storage")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=52)
    context_window: int = Field(default=2)
    context_filter_types: str = Field(default="text,table")
    max_context_tokens: int = Field(default=3000)

    # ===== Ingestion Performance =====
    max_concurrent_ingestion: int = Field(default=4)
    max_parallel_insert: int = Field(default=2)
    max_async: int = Field(default=4)
    embedding_func_max_async: int = Field(default=8)
    embedding_batch_num: int = Field(default=10)
    embedding_request_timeout: int = Field(
        default=120,
        description="Per-request timeout for embedding calls (seconds). "
        "LightRAG worker timeout is 2x this value. "
        "Increase for slow local models (CPU inference).",
    )
    ingestion_replace_default: bool = Field(default=False)
    ingestion_batch_pages: int = Field(
        default=20, description="Pages per batch during streaming ingestion."
    )

    # ===== Query Configuration =====
    top_k: int = Field(default=60)
    chunk_top_k: int = Field(default=30)
    bm25_enabled: bool = Field(default=True)
    bm25_top_k: int = Field(default=40)
    bm25_text_config: Literal["simple", "english"] = Field(default="simple")
    rrf_k: int = Field(default=60)
    direct_visual_top_k: int = Field(default=20)
    metadata_filter_exact_vector_threshold: int = Field(
        default=8192,
        description="Use exact vector scoring inside metadata candidates at or below this size.",
    )
    max_entity_tokens: int = Field(default=6000)
    max_relation_tokens: int = Field(default=8000)
    max_total_tokens: int = Field(default=40000)
    default_mode: Literal["mix"] = Field(default="mix")

    max_conversation_turns: int = Field(default=50)
    max_conversation_tokens: int = Field(default=150000)

    # ===== Knowledge Graph =====
    kg_chunk_pick_method: Literal["VECTOR", "WEIGHT"] = Field(
        default="VECTOR",
        description="How LightRAG selects chunks linked to KG entities/relations. "
        "VECTOR: rank by similarity to query embedding (recommended for most workloads). "
        "WEIGHT: rank by entity/relation importance scores. "
        "Maps to LightRAG's KG_CHUNK_PICK_METHOD (1.4.7+).",
    )
    kg_entity_types: list[str] = Field(
        default=[
            "Product",
            "Component",
            "Technology",
            "Design",
            "Genre",
            "Organization",
            "Standard",
            "Biography",
            "Location",
            "Event",
        ],
    )

    # ===== LibreOffice Conversion =====
    excel_auto_convert_to_pdf: bool = Field(default=True)
    excel_pdf_delete_original: bool = Field(default=True)
    libreoffice_timeout: int = Field(default=120)
    libreoffice_pdf_quality: int = Field(default=90)

    # ===== Sourcing (Optional) =====
    blob_connection_string: str | None = Field(default=None)
    azure_sas_expiry: int = Field(default=3600, description="Azure SAS URL expiry in seconds")

    # ===== MCP Server =====
    mcp_transport: Literal["stdio", "streamable-http"] = Field(default="stdio")
    mcp_host: str = Field(
        default="127.0.0.1",
        description="MCP streamable-http bind address. Default 127.0.0.1 (loopback only) "
        "because MCP exposes ingest/answer/delete_files with no native auth — exposing "
        "to a network without auth would let any reachable client wipe data. Set to "
        "0.0.0.0 only when api_auth_token is set with an explicit auth_mode; "
        "the bearer-token middleware then guards the transport.",
    )
    mcp_port: int = Field(default=8101)

    # ===== REST API Server =====
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8100)
    api_auth_token: str | None = Field(
        default=None,
        description=(
            "Bearer token for auth_mode='simple' and MCP streamable-http. "
            "Setting this while auth_mode='none' is invalid."
        ),
    )
    auth_mode: Literal["none", "simple", "jwt"] = Field(
        default="none",
        description="API auth strategy: 'none', 'simple' (bearer token), 'jwt'.",
    )
    jwt_secret: str | None = Field(default=None)
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256"] = Field(
        default="HS256",
        description="JWT signature algorithm — restricted to safe HMAC/RSA/ECDSA variants. "
        "Notably 'none' is rejected to prevent unsigned-token forgery.",
    )
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allow_origins list. Default '*' is convenient for dev; set "
        "explicit origins (e.g. ['https://app.example.com']) when auth_mode is "
        "not 'none'. Browsers reject the '*' + credentials combo.",
    )
    max_upload_size_mb: int = Field(
        default=512,
        description="Per-request upload cap for /web/files/upload (MB). Reject the "
        "whole request when Content-Length exceeds this so we don't fill the temp "
        "directory before noticing.",
    )

    # ===== Operational =====
    log_level: str = Field(default="info")
    request_timeout: int = Field(
        default=600,
        description="Overall request timeout in seconds for ingest/retrieve/answer operations.",
    )

    # ===== Observability =====
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    langfuse_export_external_spans: bool = Field(
        default=False,
        description="Export third-party GenAI/LLM OpenTelemetry spans to Langfuse. "
        "Disabled by default because DlightRAG records model calls manually.",
    )

    # ===== Validators =====

    @model_validator(mode="after")
    def _validate_auth_mode(self):
        """Validate explicit auth configuration."""
        if self.auth_mode == "none" and self.api_auth_token:
            raise ValueError("api_auth_token is set; configure auth_mode='simple' explicitly")
        if self.auth_mode == "simple" and not self.api_auth_token:
            raise ValueError("auth_mode='simple' requires api_auth_token to be set")
        # Fail-fast: jwt mode requires a secret; otherwise every request 500s.
        if self.auth_mode == "jwt" and not self.jwt_secret:
            raise ValueError("auth_mode='jwt' requires jwt_secret to be set")
        # Browsers reject allow_origins=['*'] with credentials. When auth is
        # on, an explicit origin list MUST replace the wildcard.
        if self.auth_mode != "none" and self.cors_allow_origins == ["*"]:
            import warnings

            warnings.warn(
                "auth_mode is enabled but cors_allow_origins=['*']; browsers will "
                "reject credentialed cross-origin requests. Set DLIGHTRAG_CORS_ALLOW_ORIGINS "
                "to explicit origins (e.g. ['https://your-frontend.example.com']).",
                stacklevel=2,
            )
        return self

    # ===== Computed Properties =====

    @property
    def chat(self) -> ModelConfig:
        """Compatibility accessor for old callers; config input must use llm.default."""
        return self.llm.default

    @property
    def extract(self) -> ModelConfig | None:
        """Compatibility accessor for old callers; config input must use llm.roles.extract."""
        return self.llm.roles.extract

    @property
    def keywords(self) -> ModelConfig | None:
        """Compatibility accessor for old callers; config input must use llm.roles.keyword."""
        return self.llm.roles.keyword

    @property
    def query(self) -> ModelConfig | None:
        """Compatibility accessor for old callers; config input must use llm.roles.query."""
        return self.llm.roles.query

    @property
    def vlm(self) -> ModelConfig | None:
        """Compatibility accessor for old callers; config input must use llm.roles.vlm."""
        return self.llm.roles.vlm

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir).resolve()

    @property
    def temp_dir(self) -> Path:
        return self.working_dir_path / ".tmp"

    @property
    def artifacts_dir(self) -> Path:
        return self.working_dir_path / "artifacts"

    def model_post_init(self, __context) -> None:
        """Pydantic lifecycle hook: bridge DLIGHTRAG_* → backend env vars."""
        pg_env_map = {
            "POSTGRES_HOST": self.postgres_host,
            "POSTGRES_PORT": str(self.postgres_port),
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_PASSWORD": self.postgres_password,
            "POSTGRES_DATABASE": self.postgres_database,
            "POSTGRES_WORKSPACE": self.workspace,
            "POSTGRES_VECTOR_INDEX_TYPE": self.pg_vector_index_type,
            "POSTGRES_HNSW_M": str(self.pg_hnsw_m),
            # LightRAG's POSTGRES_HNSW_EF is used as ef_construction.
            "POSTGRES_HNSW_EF": str(self.pg_hnsw_ef_construction),
        }
        for key, value in pg_env_map.items():
            if key not in os.environ:
                os.environ[key] = value

        if "POSTGRES_SERVER_SETTINGS" not in os.environ:
            os.environ["POSTGRES_SERVER_SETTINGS"] = f"hnsw.ef_search={self.pg_hnsw_ef_search}"

        # Resolve working_dir to absolute path
        path = Path(self.working_dir)
        if not path.is_absolute():
            self.working_dir = str(path.resolve())


# Singleton for standalone mode (MCP/API server)
_config: DlightragConfig | None = None


def get_config() -> DlightragConfig:
    """Get global dlightrag configuration (singleton).

    For standalone use (MCP/API server). When used as a library,
    construct DlightragConfig directly and pass it to RAGService.
    """
    global _config
    if _config is None:
        _config = DlightragConfig()  # type: ignore[call-arg]
    return _config


def set_config(config: DlightragConfig) -> None:
    """Set the global config singleton. Useful for testing."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
