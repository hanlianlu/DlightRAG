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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_YAML_FILE = "config.yaml"
_ENV_FILE = ".env"
PostgresTarget = Literal["primary", "replica"]
_LIGHTRAG_SIDECAR_ENV_KEYS = frozenset(
    {
        "VLM_PROCESS_ENABLE",
        "VLM_MAX_IMAGE_BYTES",
        "SURROUNDING_LEADING_MAX_TOKENS",
        "SURROUNDING_TRAILING_MAX_TOKENS",
        "LIGHTRAG_FORCE_REPARSE_MINERU",
        "MINERU_API_MODE",
        "MINERU_API_TOKEN",
        "MINERU_OFFICIAL_ENDPOINT",
        "MINERU_MODEL_VERSION",
        "MINERU_IS_OCR",
        "MINERU_POLL_INTERVAL_SECONDS",
        "MINERU_MAX_POLLS",
        "MINERU_LANGUAGE",
        "MINERU_ENABLE_TABLE",
        "MINERU_ENABLE_FORMULA",
        "MINERU_PAGE_RANGES",
        "MINERU_ENGINE_VERSION",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_LOCAL_BACKEND",
        "MINERU_LOCAL_PARSE_METHOD",
        "MINERU_LOCAL_IMAGE_ANALYSIS",
        "MINERU_LOCAL_START_PAGE_ID",
        "MINERU_LOCAL_END_PAGE_ID",
    }
)


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


def _iter_env_files(env_file: Any) -> Iterable[Path]:
    if env_file is None:
        return ()
    if isinstance(env_file, (str, Path)):
        return (Path(env_file),)
    return tuple(Path(path) for path in env_file if path is not None)


def _is_lightrag_sidecar_env_key(key: str) -> bool:
    return key in _LIGHTRAG_SIDECAR_ENV_KEYS


def _load_lightrag_sidecar_env(env_file: Any, *, force: bool = False) -> None:
    """Load declared upstream LightRAG sidecar env vars from .env."""
    for path in _iter_env_files(env_file):
        if not path.is_file():
            continue
        for key, value in dotenv_values(path).items():
            if not value or not _is_lightrag_sidecar_env_key(key):
                continue
            if force or key not in os.environ:
                os.environ[key] = value


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

    rules: str = "docx:native-iteP,*:mineru-iteP"
    chunk_options: dict[str, Any] = Field(default_factory=dict)


class ExtractionConfig(BaseModel):
    """Entity/relation extraction stability controls."""

    model_config = ConfigDict(extra="forbid")

    use_json: bool = True
    language: str = "English"
    entity_type_prompt_file: str | None = None


class VLMSidecarConfig(BaseModel):
    """LightRAG visual sidecar analysis settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_image_bytes: int = 5_242_880
    surrounding_leading_max_tokens: int | None = None
    surrounding_trailing_max_tokens: int | None = None


class MinerUSidecarConfig(BaseModel):
    """LightRAG MinerU parser sidecar settings."""

    model_config = ConfigDict(extra="forbid")

    api_mode: Literal["local", "official"] = "local"
    api_token: str | None = None
    official_endpoint: str = "https://mineru.net"
    model_version: str = "vlm"
    is_ocr: bool = False
    poll_interval_seconds: int = 2
    max_polls: int = 180
    language: str = "ch"
    enable_table: bool = True
    enable_formula: bool = True
    page_ranges: str | None = None
    engine_version: str | None = None
    force_reparse: bool = False
    local_endpoint: str = "http://127.0.0.1:8210"
    local_backend: str = "hybrid-auto-engine"
    local_parse_method: Literal["auto", "txt", "ocr"] = "auto"
    local_image_analysis: bool = True
    local_start_page_id: int | None = None
    local_end_page_id: int | None = None


class ParserSidecarsConfig(BaseModel):
    """Typed config that DlightRAG bridges into LightRAG parser env vars."""

    model_config = ConfigDict(extra="forbid")

    vlm: VLMSidecarConfig = Field(default_factory=VLMSidecarConfig)
    mineru: MinerUSidecarConfig = Field(default_factory=MinerUSidecarConfig)


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
    image_max_bytes: int = Field(
        default=1_500_000,
        description="Maximum compressed binary bytes per image sent to rerank model calls.",
    )
    image_max_total_bytes: int = Field(
        default=8_000_000,
        description="Maximum total compressed binary image bytes per rerank model request.",
    )
    image_max_px: int = Field(
        default=1280,
        description="Maximum image long edge sent to rerank model calls.",
    )
    image_min_px: int = Field(
        default=768,
        description="Minimum long edge preserved before skipping oversized rerank images.",
    )
    image_quality: int = Field(
        default=86,
        description="Initial JPEG quality for rerank model image previews.",
    )
    image_min_quality: int = Field(
        default=76,
        description="Minimum JPEG quality before skipping oversized rerank images.",
    )
    temperature: float | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class CitationHighlightConfig(BaseModel):
    """Optional semantic highlighting for cited source snippets."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timeout: float = 5.0
    max_concurrency: int = 4
    max_input_chars: int = 4096
    cache_size: int = 500


class CitationsConfig(BaseModel):
    """Citation validation and UI enrichment configuration."""

    model_config = ConfigDict(extra="forbid")

    highlights: CitationHighlightConfig = Field(default_factory=CitationHighlightConfig)


class AnswerConfig(BaseModel):
    """Final answer generation controls."""

    model_config = ConfigDict(extra="forbid")

    candidate_top_k: int = Field(
        default=60,
        ge=1,
        description=(
            "Retrieval candidate chunks fetched for answer generation before answer-stage "
            "packing. Keep this at or above context_top_k to allow visual backfill."
        ),
    )
    context_top_k: int = Field(
        default=30,
        ge=1,
        description="Maximum chunks included in the final answer LLM prompt.",
    )
    max_images: int = Field(
        default=6,
        description="Maximum total images sent to the answer LLM.",
    )
    image_max_bytes: int = Field(
        default=3_000_000,
        description="Maximum compressed binary bytes per answer image.",
    )
    image_max_total_bytes: int = Field(
        default=24_000_000,
        description="Maximum total compressed binary image bytes per answer request.",
    )
    image_max_px: int = Field(
        default=1536,
        description="Maximum image long edge sent to the answer LLM.",
    )
    image_min_px: int = Field(
        default=1024,
        description="Minimum long edge preserved before skipping oversized answer images.",
    )
    image_quality: int = Field(
        default=88,
        description="Initial JPEG quality for answer LLM image previews.",
    )
    image_min_quality: int = Field(
        default=72,
        description="Minimum JPEG quality before skipping oversized answer images.",
    )


class QueryImagesConfig(BaseModel):
    """Query-time image memory and semantic enhancement controls."""

    model_config = ConfigDict(extra="forbid")

    semantic_enhancement: bool = True
    max_described_images: int = 3
    session_max_images: int = 50
    session_max_sessions: int = 100
    session_ttl_seconds: int = 3600


class VisualAssetsConfig(BaseModel):
    """Browser-facing visual asset serving controls."""

    model_config = ConfigDict(extra="forbid")

    thumb_max_px: int = 300
    thumb_cache_size: int = 256


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
        dotenv_filtering="match_prefix",
        case_sensitive=False,
        # Reject any unknown DLIGHTRAG_* env var, .env entry, or config.yaml key.
        # Catches typos and removed flat-schema fields without hardcoding a finite list.
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
    runtime_role: Literal["ingest", "admin", "query"] = Field(
        default="ingest",
        description="Process role. ingest/admin use primary PostgreSQL; query uses replica.",
    )
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="dlightrag")
    postgres_password: str = Field(default="dlightrag")
    postgres_database: str = Field(default="dlightrag")
    postgres_replica_host: str | None = Field(default=None)
    postgres_replica_port: int | None = Field(default=None)
    postgres_replica_user: str | None = Field(default=None)
    postgres_replica_password: str | None = Field(default=None)
    postgres_replica_database: str | None = Field(default=None)
    read_after_write_mode: Literal["eventual", "wait_for_replay"] = Field(
        default="eventual",
        description="Replica read-after-write policy for write acknowledgements.",
    )
    read_after_write_timeout: float = Field(
        default=15.0,
        description="Seconds to wait for replica WAL replay when read_after_write_mode='wait_for_replay'.",
    )
    postgres_pool_min_size: int = Field(
        default=2, description="DlightRAG domain store pool min connections."
    )
    postgres_pool_max_size: int = Field(
        default=10, description="DlightRAG domain store pool max connections."
    )
    postgres_session_settings: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Per-connection PostgreSQL GUCs applied to both LightRAG and DlightRAG pools.",
    )
    postgres_statement_cache_size: int | None = Field(
        default=None,
        description="Optional asyncpg statement cache size for LightRAG and DlightRAG pools.",
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
    parser_sidecars: ParserSidecarsConfig = Field(default_factory=ParserSidecarsConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    citations: CitationsConfig = Field(default_factory=CitationsConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    query_images: QueryImagesConfig = Field(default_factory=QueryImagesConfig)
    visual_assets: VisualAssetsConfig = Field(default_factory=VisualAssetsConfig)

    # ===== RAG Processing =====
    working_dir: str = Field(default="./dlightrag_storage")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=52)
    context_window: int = Field(default=2)
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

    # ===== Sourcing (Optional) =====
    blob_connection_string: str | None = Field(default=None)
    azure_sas_expiry: int = Field(default=3600, description="Azure SAS URL expiry in seconds")

    # ===== MCP Server =====
    mcp_transport: Literal["stdio", "streamable-http"] = Field(default="stdio")
    mcp_host: str = Field(
        default="127.0.0.1",
        description="MCP streamable-http bind address. Default 127.0.0.1 (loopback only) "
        "because MCP exposes ingest/answer/delete_files with no native auth — exposing "
        "to a network without auth would let any reachable client wipe data. Use "
        "0.0.0.0 only behind a loopback host-port mapping/trusted network, or when "
        "api_auth_token is set with an explicit auth_mode so bearer-token middleware "
        "guards the transport.",
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
    def working_dir_path(self) -> Path:
        return Path(self.working_dir).resolve()

    @property
    def temp_dir(self) -> Path:
        return self.working_dir_path / ".tmp"

    @property
    def artifacts_dir(self) -> Path:
        return self.working_dir_path / "artifacts"

    @property
    def is_query_role(self) -> bool:
        return self.runtime_role == "query"

    @property
    def is_writable_role(self) -> bool:
        return self.runtime_role in {"ingest", "admin"}

    def pg_target_for_runtime(self) -> PostgresTarget:
        return "replica" if self.is_query_role else "primary"

    def _pg_endpoint_value(self, target: PostgresTarget, field: str) -> Any:
        if target == "primary":
            return getattr(self, f"postgres_{field}")

        value = getattr(self, f"postgres_replica_{field}")
        if value is not None:
            return value
        return self._pg_endpoint_value("primary", field)

    def pg_connection_kwargs(self, target: PostgresTarget | None = None) -> dict[str, Any]:
        """Return asyncpg connection kwargs for primary, replica, or current runtime role."""
        resolved = target or self.pg_target_for_runtime()
        return {
            "host": self._pg_endpoint_value(resolved, "host"),
            "port": self._pg_endpoint_value(resolved, "port"),
            "user": self._pg_endpoint_value(resolved, "user"),
            "password": self._pg_endpoint_value(resolved, "password"),
            "database": self._pg_endpoint_value(resolved, "database"),
        }

    @staticmethod
    def _env_value(value: str | int | float | bool | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip()
        return text or None

    def postgres_server_settings_dict(self) -> dict[str, str]:
        """Return asyncpg server_settings for both LightRAG and DlightRAG pools."""
        settings: dict[str, str] = {"hnsw.ef_search": str(self.pg_hnsw_ef_search)}
        for key, value in self.postgres_session_settings.items():
            rendered = self._env_value(value)
            if rendered is not None:
                settings[str(key)] = rendered
        return settings

    def postgres_server_settings_env_value(self) -> str:
        """Return LightRAG's POSTGRES_SERVER_SETTINGS query-string format."""
        return urlencode(self.postgres_server_settings_dict())

    def _lightrag_sidecar_env_map(self) -> dict[str, str]:
        vlm = self.parser_sidecars.vlm
        mineru = self.parser_sidecars.mineru
        raw: dict[str, str | int | float | bool | None] = {
            "VLM_PROCESS_ENABLE": vlm.enabled,
            "VLM_MAX_IMAGE_BYTES": vlm.max_image_bytes,
            "SURROUNDING_LEADING_MAX_TOKENS": vlm.surrounding_leading_max_tokens,
            "SURROUNDING_TRAILING_MAX_TOKENS": vlm.surrounding_trailing_max_tokens,
            "MINERU_API_MODE": mineru.api_mode,
            "MINERU_API_TOKEN": mineru.api_token,
            "MINERU_OFFICIAL_ENDPOINT": mineru.official_endpoint,
            "MINERU_MODEL_VERSION": mineru.model_version,
            "MINERU_IS_OCR": mineru.is_ocr,
            "MINERU_POLL_INTERVAL_SECONDS": mineru.poll_interval_seconds,
            "MINERU_MAX_POLLS": mineru.max_polls,
            "MINERU_LANGUAGE": mineru.language,
            "MINERU_ENABLE_TABLE": mineru.enable_table,
            "MINERU_ENABLE_FORMULA": mineru.enable_formula,
            "MINERU_PAGE_RANGES": mineru.page_ranges,
            "MINERU_ENGINE_VERSION": mineru.engine_version,
            "MINERU_LOCAL_ENDPOINT": mineru.local_endpoint,
            "MINERU_LOCAL_BACKEND": mineru.local_backend,
            "MINERU_LOCAL_PARSE_METHOD": mineru.local_parse_method,
            "MINERU_LOCAL_IMAGE_ANALYSIS": mineru.local_image_analysis,
            "MINERU_LOCAL_START_PAGE_ID": mineru.local_start_page_id,
            "MINERU_LOCAL_END_PAGE_ID": mineru.local_end_page_id,
            "LIGHTRAG_FORCE_REPARSE_MINERU": True if mineru.force_reparse else None,
        }
        rendered: dict[str, str] = {}
        for key, value in raw.items():
            text = self._env_value(value)
            if text is not None:
                rendered[key] = text
        return rendered

    def apply_lightrag_sidecar_env(self, *, force: bool = False) -> None:
        """Bridge typed parser sidecar config into LightRAG's upstream env API."""
        for key, value in self._lightrag_sidecar_env_map().items():
            if force or key not in os.environ:
                os.environ[key] = value

    def apply_lightrag_backend_env(self, *, force: bool = False) -> None:
        """Bridge this config's active PostgreSQL endpoint into LightRAG env vars."""
        active_pg = self.pg_connection_kwargs()
        # LightRAG's PostgreSQL ClientManager is process-wide. A global
        # POSTGRES_WORKSPACE would pin every LightRAG instance to whichever
        # workspace initialized first, so DlightRAG always passes workspace via
        # the LightRAG constructor instead.
        os.environ.pop("POSTGRES_WORKSPACE", None)
        pg_env_map = {
            "POSTGRES_HOST": active_pg["host"],
            "POSTGRES_PORT": str(active_pg["port"]),
            "POSTGRES_USER": active_pg["user"],
            "POSTGRES_PASSWORD": active_pg["password"],
            "POSTGRES_DATABASE": active_pg["database"],
            "POSTGRES_VECTOR_INDEX_TYPE": self.pg_vector_index_type,
            "POSTGRES_HNSW_M": str(self.pg_hnsw_m),
            # LightRAG's POSTGRES_HNSW_EF is used as ef_construction.
            "POSTGRES_HNSW_EF": str(self.pg_hnsw_ef_construction),
        }
        if self.postgres_statement_cache_size is not None:
            pg_env_map["POSTGRES_STATEMENT_CACHE_SIZE"] = str(self.postgres_statement_cache_size)
        for key, value in pg_env_map.items():
            if force or key not in os.environ:
                os.environ[key] = value

        if force or "POSTGRES_SERVER_SETTINGS" not in os.environ:
            os.environ["POSTGRES_SERVER_SETTINGS"] = self.postgres_server_settings_env_value()

    def model_post_init(self, __context) -> None:
        """Pydantic lifecycle hook: bridge DLIGHTRAG_* → backend env vars."""
        _load_lightrag_sidecar_env(self.__class__.model_config.get("env_file"), force=False)
        self.apply_lightrag_sidecar_env(force=False)
        self.apply_lightrag_backend_env(force=False)

        # Resolve working_dir to absolute path
        path = Path(self.working_dir)
        if not path.is_absolute():
            self.working_dir = str(path.resolve())


# Singleton for standalone mode (MCP/API server)
_config: DlightragConfig | None = None


def load_config(env_file: str | Path | None = None, **overrides: Any) -> DlightragConfig:
    """Build config from an optional .env file without globally loading dotenv."""
    if env_file is not None:
        _load_lightrag_sidecar_env(env_file, force=False)
        return DlightragConfig(_env_file=env_file, **overrides)  # type: ignore[call-arg]
    return DlightragConfig(**overrides)


def get_config() -> DlightragConfig:
    """Get global dlightrag configuration (singleton).

    For standalone use (MCP/API server). When used as a library,
    construct DlightragConfig directly and pass it to RAGService.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: DlightragConfig) -> None:
    """Set the global config singleton. Useful for testing."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
