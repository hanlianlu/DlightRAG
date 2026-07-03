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

import json
import os
import re
import ssl
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict
from urllib.parse import urlencode

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_YAML_FILE = "config.yaml"
_ENV_FILE = ".env"
_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PG_QUALIFIED_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_LOCAL_MCP_ALLOWED_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_MCP_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]
_LOCAL_API_HOSTS = {"127.0.0.1", "localhost", "::1"}
PostgresSSLMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
MinerULanguage = Literal[
    "ch",
    "ch_server",
    "korean",
    "ta",
    "te",
    "ka",
    "th",
    "el",
    "arabic",
    "east_slavic",
    "cyrillic",
    "devanagari",
]


class LightRAGPipelineKwargs(TypedDict):
    max_parallel_insert: int
    max_parallel_parse_native: int
    max_parallel_parse_mineru: int
    max_parallel_analyze: int
    queue_size_parse: int
    queue_size_analyze: int
    queue_size_insert: int


# Auto-derived from VLMSidecarConfig._ENV_MAP and MinerUSidecarConfig._ENV_MAP.
# No manual maintenance — add a field + env mapping to the model and this picks it up.
_LIGHTRAG_SIDECAR_ENV_KEYS: frozenset[str] = frozenset()  # populated after class definitions


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

    model_config = ConfigDict(extra="forbid")

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
            model="gpt-5.4-mini",
            temperature=0.5,
        )
    )
    roles: LLMRolesConfig = Field(default_factory=LLMRolesConfig)


class ParserConfig(BaseModel):
    """LightRAG parser routing rules and optional chunker snapshot overrides."""

    model_config = ConfigDict(extra="forbid")

    rules: str = "docx:native-iteP,md:native-iteP,textpack:native-iteP,*:mineru-iteP"
    chunk_options: dict[str, Any] = Field(default_factory=dict)


class ExtractionConfig(BaseModel):
    """Entity/relation extraction stability controls."""

    model_config = ConfigDict(extra="forbid")

    use_json: bool = True
    language: str = "English"
    entity_type_prompt_file: str | None = Field(
        default=None,
        description=(
            "LightRAG entity-type prompt profile YAML file name. Loaded from "
            "PROMPT_DIR/entity_type; must be a .yml/.yaml file name, not a path."
        ),
    )

    @field_validator("entity_type_prompt_file")
    @classmethod
    def _validate_entity_type_prompt_file(cls, value: str | None) -> str | None:
        """Mirror LightRAG's ENTITY_TYPE_PROMPT_FILE contract."""
        if value is None:
            return None
        file_name = value.strip()
        if not file_name:
            return None
        candidate = Path(file_name)
        if (
            "\\" in file_name
            or candidate.is_absolute()
            or candidate.name != file_name
            or ".." in candidate.parts
        ):
            raise ValueError(
                "entity_type_prompt_file must be a file name under PROMPT_DIR/entity_type"
            )
        if candidate.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError("entity_type_prompt_file must use a .yml or .yaml extension")
        return file_name


class VLMSidecarConfig(BaseModel):
    """LightRAG visual sidecar analysis settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_image_bytes: int = 5_242_880
    min_image_pixel: int = 100
    surrounding_leading_max_tokens: int | None = 128
    surrounding_trailing_max_tokens: int | None = 128

    # Pydantic field → LightRAG env var. Single source of truth;
    # _lightrag_sidecar_env_map() and _LIGHTRAG_SIDECAR_ENV_KEYS derive from this.
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "enabled": "VLM_PROCESS_ENABLE",
        "max_image_bytes": "VLM_MAX_IMAGE_BYTES",
        "min_image_pixel": "VLM_MIN_IMAGE_PIXEL",
        "surrounding_leading_max_tokens": "SURROUNDING_LEADING_MAX_TOKENS",
        "surrounding_trailing_max_tokens": "SURROUNDING_TRAILING_MAX_TOKENS",
    }


class MinerUSidecarConfig(BaseModel):
    """LightRAG MinerU parser sidecar settings.

    Only fields that are *necessary to route or override* are declared here.
    Everything else — backend, parse method, table/formula, OCR, page ranges,
    and MinerU-side image analysis — is left to LightRAG's built-in defaults.
    """

    model_config = ConfigDict(extra="forbid")

    api_mode: Literal["local", "official"] = "local"
    api_token: str | None = None
    official_endpoint: str = "https://mineru.net"
    local_endpoint: str = "http://127.0.0.1:8210"
    language: MinerULanguage = "ch"
    force_reparse: bool = False

    # macOS MinerU hard-codes max_concurrent_requests=1 (see mineru/cli/fast_api.py).
    # Two large PDFs submitted in parallel will serialize, and the second one's poll
    # clock starts counting from submit time — not from when it actually starts
    # processing. 1200 polls × 2s = 40 min covers two large PDFs in serial.
    max_polls: int = 1200

    # DlightRAG-only: filter header/footer blocks that pollute chunk text.
    auxiliary_block_policy: Literal["conservative", "extended"] = "conservative"

    # Pydantic field → LightRAG/MinerU env var. auxiliary_block_policy is
    # intentionally absent — it's a DlightRAG post-processing step.
    _ENV_MAP: ClassVar[dict[str, str]] = {
        "api_mode": "MINERU_API_MODE",
        "api_token": "MINERU_API_TOKEN",
        "official_endpoint": "MINERU_OFFICIAL_ENDPOINT",
        "local_endpoint": "MINERU_LOCAL_ENDPOINT",
        "language": "MINERU_LANGUAGE",
        "force_reparse": "LIGHTRAG_FORCE_REPARSE_MINERU",
        "max_polls": "MINERU_MAX_POLLS",
    }


class ParserSidecarsConfig(BaseModel):
    """Typed config that DlightRAG bridges into LightRAG parser env vars."""

    model_config = ConfigDict(extra="forbid")

    vlm: VLMSidecarConfig = Field(default_factory=VLMSidecarConfig)
    mineru: MinerUSidecarConfig = Field(default_factory=MinerUSidecarConfig)


# Populate the sidecar env keys frozenset from the model _ENV_MAP declarations.
_SIDECAR_ENV_KEYS: set[str] = set()
for _cls in (VLMSidecarConfig, MinerUSidecarConfig):
    _SIDECAR_ENV_KEYS.update(_cls._ENV_MAP.values())
_LIGHTRAG_SIDECAR_ENV_KEYS = frozenset(_SIDECAR_ENV_KEYS)


class MetadataFieldConfig(BaseModel):
    """User-declared metadata field behavior."""

    model_config = ConfigDict(extra="forbid")

    type: str = "string"
    normalizer: str | None = None
    filter_ops: list[str] = Field(default_factory=list)


class MetadataConfig(BaseModel):
    """Metadata registry and ingest policy controls."""

    model_config = ConfigDict(extra="forbid")

    allow_ad_hoc_json: bool = True
    default_ingest_policy: Literal["validate", "reject_unknown", "store_only"] = "validate"
    fields: dict[str, MetadataFieldConfig] = Field(default_factory=dict)


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
    multimodal: bool = Field(
        default=True,
        description=(
            "When True, image_data in chunks is sent to the reranker for multimodal scoring. "
            "When False, image_data is ignored and only text content is used — safe for "
            "text-only reranker models/APIs. Multimodal chunks always carry VLM-generated "
            "text descriptions, so text-only fallback retains substantial signal."
        ),
    )
    score_threshold: float = 0.5
    max_concurrency: int = 8
    batch_size: int = 8
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

    enabled: bool = True
    timeout: float = Field(default=10.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_input_chars: int = Field(default=4096, ge=1)
    cache_size: int = Field(default=500, ge=1)


class CitationsConfig(BaseModel):
    """Citation validation and UI enrichment configuration."""

    model_config = ConfigDict(extra="forbid")

    highlights: CitationHighlightConfig = Field(default_factory=CitationHighlightConfig)


class AnswerConfig(BaseModel):
    """Final answer generation controls."""

    model_config = ConfigDict(extra="forbid")

    context_top_k: int = Field(
        default=30,
        ge=1,
        description="Maximum chunks included in the final answer LLM prompt.",
    )
    max_images: int = Field(
        default=6,
        description="Maximum RAG context images sent to the answer LLM.",
    )
    max_user_images: int = Field(
        default=3,
        ge=1,
        description="Maximum user-attached images (query_images + history) sent to the answer LLM.",
    )

    # Vision support is runtime manager state, not config. Users do not set it
    # in config.yaml; the startup probe records it on RAGServiceManager.
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
        default=89,
        description="Initial JPEG quality for answer LLM image previews.",
    )
    image_min_quality: int = Field(
        default=79,
        description="Minimum JPEG quality before skipping oversized answer images.",
    )


class QueryImagesConfig(BaseModel):
    """Query-time image memory and semantic enhancement controls."""

    model_config = ConfigDict(extra="forbid")

    max_described_images: int = 3
    session_max_images: int = 50
    session_max_sessions: int = 100


class VisualAssetsConfig(BaseModel):
    """Browser-facing visual asset serving controls."""

    model_config = ConfigDict(extra="forbid")

    thumb_max_px: int = 300
    thumb_cache_size: int = 256


class BM25ProfileConfig(BaseModel):
    """Query-language-routed pg_textsearch BM25 profile."""

    model_config = ConfigDict(extra="forbid")

    name: str
    text_config: str
    languages: list[str] = Field(default_factory=list)
    fallback: bool = False

    @field_validator("name")
    @classmethod
    def _validate_pg_identifier(cls, value: str) -> str:
        text = value.strip()
        if not _PG_IDENTIFIER_RE.fullmatch(text):
            raise ValueError("BM25 profile name must be a safe PostgreSQL identifier")
        return text

    @field_validator("text_config")
    @classmethod
    def _validate_pg_text_config(cls, value: str) -> str:
        text = value.strip()
        if not _PG_QUALIFIED_IDENTIFIER_RE.fullmatch(text):
            raise ValueError("BM25 text_config must be a safe PostgreSQL identifier")
        return text

    @field_validator("languages")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        return [language.strip().lower() for language in value if language.strip()]


def _redact_dict(data: dict[str, Any], patterns: tuple[str, ...]) -> dict[str, Any]:
    """Recursively redact values whose keys match sensitive patterns."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if any(pattern in key.lower() for pattern in patterns):
            if isinstance(value, str) and len(value) > 8:
                result[key] = value[:4] + "***" + value[-4:]
            elif isinstance(value, str):
                result[key] = "***"
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _redact_dict(value, patterns)
        elif isinstance(value, list):
            result[key] = [
                _redact_dict(item, patterns) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


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

    _SECRET_PATTERNS: tuple[str, ...] = (
        "api_key",
        "api_secret",
        "secret",
        "verification_key",
        "password",
        "connection_string",
        "account_key",
        "sas_token",
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize config with sensitive fields redacted."""
        data = super().model_dump(**kwargs)
        return _redact_dict(data, self._SECRET_PATTERNS)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize config to JSON with sensitive fields redacted."""
        # Extract json.dumps-specific kwargs that model_dump doesn't accept
        indent = kwargs.pop("indent", None)
        data = self.model_dump(**kwargs)
        dump_kwargs: dict[str, Any] = {"default": str}
        if indent is not None:
            dump_kwargs["indent"] = indent
        return json.dumps(data, **dump_kwargs)

    def __repr__(self) -> str:
        """Redact secrets in repr for safe logging."""
        data = self.model_dump()
        inner = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"{self.__class__.__name__}({inner})"

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
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="dlightrag")
    postgres_password: str = Field(default="dlightrag")
    postgres_database: str = Field(default="dlightrag")
    postgres_ssl_mode: PostgresSSLMode | None = Field(
        default=None,
        description=(
            "PostgreSQL SSL mode bridged to LightRAG and DlightRAG asyncpg connections. "
            "Matches LightRAG's POSTGRES_SSL_MODE contract."
        ),
    )
    postgres_ssl_cert: str | None = Field(default=None)
    postgres_ssl_key: str | None = Field(default=None)
    postgres_ssl_root_cert: str | None = Field(default=None)
    postgres_ssl_crl: str | None = Field(default=None)
    postgres_pool_min_size: int = Field(
        default=2, description="DlightRAG domain store pool min connections."
    )
    postgres_pool_max_size: int = Field(
        default=10, description="DlightRAG domain store pool max connections."
    )
    postgres_lightrag_pool_max_size: int = Field(
        default=16,
        ge=1,
        description=(
            "LightRAG PostgreSQL backend pool max connections per process. "
            "Bridged to LightRAG's POSTGRES_MAX_CONNECTIONS."
        ),
    )
    postgres_session_settings: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Per-connection PostgreSQL GUCs applied to both LightRAG and DlightRAG pools.",
    )
    postgres_statement_cache_size: int | None = Field(
        default=None,
        description="Optional asyncpg statement cache size for LightRAG and DlightRAG pools.",
    )
    postgres_connection_retries: int = Field(
        default=10,
        ge=1,
        le=100,
        description="PostgreSQL transient connection retry attempts for LightRAG and DlightRAG pools.",
    )
    postgres_connection_retry_backoff: float = Field(
        default=3.0,
        ge=0.0,
        le=300.0,
        description="Initial PostgreSQL transient retry backoff in seconds.",
    )
    postgres_connection_retry_backoff_max: float = Field(
        default=30.0,
        ge=0.0,
        le=600.0,
        description="Maximum PostgreSQL transient retry backoff in seconds.",
    )
    postgres_pool_close_timeout: float = Field(
        default=5.0,
        ge=0.0,
        le=30.0,
        description="Seconds to wait when closing a stale PostgreSQL pool after transient failure.",
    )
    workspace: str = Field(default="default")

    postgres_required_major: int = Field(default=18)

    # pgvector index configuration. LightRAG derives VECTOR/HALFVEC from this.
    pg_vector_index_type: Literal["HNSW", "HNSW_HALFVEC", "IVFFLAT", "VCHORDRQ"] = Field(
        default="HNSW_HALFVEC",
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
    chunk_p_token_size: int = Field(default=1024)

    # ===== Ingestion Performance =====
    max_parallel_insert: int = Field(
        default=3,
        ge=1,
        description="LightRAG staged pipeline insert-worker concurrency.",
    )
    max_parallel_parse_native: int = Field(
        default=5,
        ge=1,
        description="LightRAG native-parser worker concurrency.",
    )
    max_parallel_parse_mineru: int = Field(
        default=2,
        ge=1,
        description="LightRAG external parser worker concurrency for the MinerU-compatible route.",
    )
    max_parallel_analyze: int = Field(
        default=5,
        ge=1,
        description=("LightRAG multimodal analysis worker concurrency."),
    )
    queue_size_parse: int = Field(
        default=20,
        ge=1,
        description="LightRAG staged pipeline parse queue size.",
    )
    queue_size_analyze: int = Field(
        default=32,
        ge=1,
        description="LightRAG staged pipeline multimodal analysis queue size.",
    )
    queue_size_insert: int = Field(
        default=4,
        ge=1,
        description="LightRAG staged pipeline insert queue size.",
    )
    max_async: int = Field(default=8, ge=1)
    embedding_func_max_async: int = Field(default=16, ge=1)
    embedding_batch_num: int = Field(default=10, ge=1)
    embedding_request_timeout: int = Field(
        default=120,
        description="Per-request timeout for embedding calls (seconds). "
        "LightRAG worker timeout is 2x this value. "
        "Increase for slow local models (CPU inference).",
    )
    ingestion_replace_default: bool = Field(default=False)
    retain_remote_source_files: bool = Field(
        default=False,
        description=(
            "Keep S3/Azure/URL/SDK remote source files under the workspace input root. "
            "When false, only parser artifacts and remote source URI metadata are retained."
        ),
    )
    url_ingest_max_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=1,
        description="Maximum bytes to download for one public HTTPS URL ingest.",
    )
    url_ingest_private_host_allowlist: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit host/IP patterns allowed to resolve to private addresses for URL ingest. "
            "Keep empty unless ingesting trusted enterprise intranet URLs."
        ),
    )
    max_upload_bytes: int = Field(
        default=100 * 1024 * 1024,
        description="Maximum file size in bytes for /api/ingest/blob uploads (default 100MB).",
    )

    # ===== Query Configuration =====
    top_k: int = Field(default=60)
    chunk_top_k: int = Field(default=30)
    bm25_enabled: bool = Field(default=True)
    bm25_profiles: list[BM25ProfileConfig] = Field(
        default_factory=lambda: [
            BM25ProfileConfig(name="zh", text_config="public.jiebacfg", languages=["zh"]),
            BM25ProfileConfig(name="en", text_config="english", languages=["en"]),
            BM25ProfileConfig(name="de", text_config="german", languages=["de"]),
            BM25ProfileConfig(name="sv", text_config="swedish", languages=["sv"]),
            BM25ProfileConfig(name="es", text_config="spanish", languages=["es"]),
            BM25ProfileConfig(name="fr", text_config="french", languages=["fr"]),
            BM25ProfileConfig(name="it", text_config="italian", languages=["it"]),
            BM25ProfileConfig(name="pt", text_config="portuguese", languages=["pt"]),
            BM25ProfileConfig(name="nl", text_config="dutch", languages=["nl"]),
            BM25ProfileConfig(name="ru", text_config="russian", languages=["ru"]),
            BM25ProfileConfig(name="da", text_config="danish", languages=["da"]),
            BM25ProfileConfig(name="fi", text_config="finnish", languages=["fi"]),
            BM25ProfileConfig(name="simple", text_config="simple", fallback=True),
        ],
        description="Query-language-routed pg_textsearch BM25 profiles.",
    )
    bm25_k1: float = Field(
        default=1.2,
        gt=0,
        description="BM25 term-frequency saturation parameter passed to pg_textsearch.",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description="BM25 document-length normalization parameter passed to pg_textsearch.",
    )
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
    s3_presign_expiry: int = Field(default=3600, description="S3 presigned URL expiry in seconds")
    s3_region: str | None = Field(default=None, description="S3 client region")

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
    mcp_allowed_hosts: list[str] = Field(
        default_factory=lambda: list(_LOCAL_MCP_ALLOWED_HOSTS),
        description=(
            "Allowed Host header values for MCP streamable-http DNS rebinding protection. "
            "Use explicit deployment hostnames before exposing MCP beyond loopback."
        ),
    )
    mcp_allowed_origins: list[str] = Field(
        default_factory=lambda: list(_LOCAL_MCP_ALLOWED_ORIGINS),
        description=(
            "Allowed Origin header values for MCP streamable-http DNS rebinding protection. "
            "Origin may be absent for non-browser clients."
        ),
    )

    # ===== REST API Server =====
    api_host: str = Field(default="127.0.0.1")
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
    jwt_verification_key: str | None = Field(
        default=None,
        description=(
            "JWT signature verification key. For HS* algorithms this is the shared "
            "HMAC key; for RS*/ES* algorithms this is the public key PEM. DlightRAG "
            "verifies externally issued JWTs and does not sign or renew tokens."
        ),
    )
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
    ingest_timeout: int | None = Field(
        default=None,
        description="Seconds to wait for synchronous ingest calls. None waits until completion; timeouts return the running job instead of cancelling ingest.",
    )
    request_timeout: int = Field(
        default=300,
        description="Timeout in seconds for retrieve/answer/query operations.",
    )
    stalled_doc_timeout_seconds: int = Field(
        default=3600,
        ge=300,
        description=(
            "Seconds before a document stuck in PARSING/ANALYZING/PROCESSING is "
            "considered stalled and reset to PENDING at startup. Individual document "
            "processing typically completes within minutes; 1 hour is conservative. "
            "Set to 0 to disable stalled-document recovery."
        ),
    )
    checkpoint_session_ttl_days: int = Field(
        default=30,
        ge=1,
        description=(
            "Delete checkpoint sessions and their referenced web session image memory "
            "older than this many days at startup. Session data grows unboundedly "
            "without this — conversation history, retrieval provenance, citation anchors, "
            "and checkpoint-restorable images accumulate per session indefinitely."
        ),
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
    langfuse_environment: str | None = Field(
        default=None,
        description="Optional Langfuse tracing environment label, e.g. production or staging.",
    )
    langfuse_release: str | None = Field(
        default=None,
        description="Optional Langfuse release/version label for trace grouping.",
    )
    langfuse_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Langfuse trace sample rate.",
    )
    langfuse_timeout: int | None = Field(
        default=None,
        ge=1,
        le=300,
        description="Langfuse SDK request timeout in seconds.",
    )
    langfuse_flush_at: int | None = Field(
        default=None,
        ge=1,
        description="Langfuse SDK batch size before flush.",
    )
    langfuse_flush_interval: float | None = Field(
        default=None,
        ge=0.1,
        le=300.0,
        description="Langfuse SDK automatic flush interval in seconds.",
    )

    # ===== Validators =====

    @model_validator(mode="after")
    def _validate_config(self):
        """Validate cross-field configuration."""
        self._validate_bm25_profiles()
        self._validate_auth_mode()
        return self

    def _validate_bm25_profiles(self) -> None:
        """Validate BM25 profile routing configuration."""
        profile_names = [profile.name for profile in self.bm25_profiles]
        if len(profile_names) != len(set(profile_names)):
            raise ValueError("bm25_profiles names must be unique")
        for profile in self.bm25_profiles:
            if profile.fallback and profile.languages:
                raise ValueError("BM25 fallback profiles must not declare languages")
            if not profile.fallback and len(profile.languages) != 1:
                raise ValueError("BM25 language profiles must declare exactly one language")
        if self.bm25_enabled and not any(profile.fallback for profile in self.bm25_profiles):
            raise ValueError("bm25_profiles must include at least one fallback profile")

    def _validate_auth_mode(self) -> None:
        """Validate API/web authentication configuration."""
        if self.auth_mode == "none" and self.api_auth_token:
            raise ValueError("api_auth_token is set; configure auth_mode='simple' explicitly")
        if self.auth_mode == "simple" and not self.api_auth_token:
            raise ValueError("auth_mode='simple' requires api_auth_token to be set")
        # Fail-fast: jwt mode requires verification material; otherwise every request 500s.
        if self.auth_mode == "jwt" and not self.jwt_verification_key:
            raise ValueError("auth_mode='jwt' requires jwt_verification_key to be set")
        if self.auth_mode == "none" and self.api_host not in _LOCAL_API_HOSTS:
            warnings.warn(
                "api_host is non-loopback while auth_mode='none'; set auth_mode or bind "
                "DLIGHTRAG_API_HOST to 127.0.0.1 for local-only use.",
                stacklevel=2,
            )
        # Browsers reject allow_origins=['*'] with credentials. When auth is
        # on, an explicit origin list MUST replace the wildcard.
        if self.auth_mode != "none" and self.cors_allow_origins == ["*"]:
            warnings.warn(
                "auth_mode is enabled but cors_allow_origins=['*']; browsers will "
                "reject credentialed cross-origin requests. Set DLIGHTRAG_CORS_ALLOW_ORIGINS "
                "to explicit origins (e.g. ['https://your-frontend.example.com']).",
                stacklevel=2,
            )

    # ===== Computed Properties =====

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir).resolve()

    @property
    def temp_dir(self) -> Path:
        return self.working_dir_path / ".tmp"

    @property
    def input_dir_path(self) -> Path:
        return self.working_dir_path / "inputs"

    @property
    def checkpoint_session_ttl_seconds(self) -> int:
        return int(self.checkpoint_session_ttl_days) * 24 * 60 * 60

    def _pg_ssl_value(self) -> ssl.SSLContext | bool | None:
        """Return asyncpg's ssl argument matching LightRAG's SSL mode semantics."""
        if self.postgres_ssl_mode is None:
            return None

        if self.postgres_ssl_mode in {"require", "prefer"}:
            return True
        if self.postgres_ssl_mode == "disable":
            return False
        if self.postgres_ssl_mode == "allow":
            return None

        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = self.postgres_ssl_mode == "verify-full"

            if self.postgres_ssl_root_cert and Path(self.postgres_ssl_root_cert).exists():
                context.load_verify_locations(cafile=self.postgres_ssl_root_cert)
            if (
                self.postgres_ssl_cert
                and self.postgres_ssl_key
                and Path(self.postgres_ssl_cert).exists()
                and Path(self.postgres_ssl_key).exists()
            ):
                context.load_cert_chain(self.postgres_ssl_cert, self.postgres_ssl_key)
            if self.postgres_ssl_crl and Path(self.postgres_ssl_crl).exists():
                context.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
                context.load_verify_locations(cafile=self.postgres_ssl_crl)
            return context
        except Exception as exc:
            raise ValueError(f"PostgreSQL SSL configuration error: {exc}") from exc

    def pg_connection_kwargs(self) -> dict[str, Any]:
        """Return asyncpg connection kwargs for DlightRAG's PostgreSQL endpoint."""
        kwargs = {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "database": self.postgres_database,
        }
        ssl_value = self._pg_ssl_value()
        if ssl_value is not None:
            kwargs["ssl"] = ssl_value
        return kwargs

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

    def lightrag_pipeline_kwargs(self) -> LightRAGPipelineKwargs:
        """Return LightRAG staged ingestion pipeline controls.

        These are constructor kwargs, not raw upstream env vars, so DlightRAG
        owns one typed product config surface while still using LightRAG's
        native pipeline.
        """
        return {
            "max_parallel_insert": self.max_parallel_insert,
            "max_parallel_parse_native": self.max_parallel_parse_native,
            "max_parallel_parse_mineru": self.max_parallel_parse_mineru,
            "max_parallel_analyze": self.max_parallel_analyze,
            "queue_size_parse": self.queue_size_parse,
            "queue_size_analyze": self.queue_size_analyze,
            "queue_size_insert": self.queue_size_insert,
        }

    def _lightrag_sidecar_env_map(self) -> dict[str, str]:
        """Derive LightRAG sidecar env vars from typed config models.

        Iterates VLMSidecarConfig._ENV_MAP and MinerUSidecarConfig._ENV_MAP
        so adding a field only requires: (1) Pydantic field, (2) one line in _ENV_MAP.
        """
        raw: dict[str, str | int | float | bool | None] = {}
        for config_obj in (self.parser_sidecars.vlm, self.parser_sidecars.mineru):
            cls = type(config_obj)
            for field_name, env_name in cls._ENV_MAP.items():
                value = getattr(config_obj, field_name)
                # force_reparse=False → don't emit the LightRAG env var
                # (LightRAG defaults to not force-reparsing on its own)
                if field_name == "force_reparse":
                    value = True if value else None
                raw[env_name] = value
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
            "POSTGRES_MAX_CONNECTIONS": str(self.postgres_lightrag_pool_max_size),
        }
        if self.postgres_statement_cache_size is not None:
            pg_env_map["POSTGRES_STATEMENT_CACHE_SIZE"] = str(self.postgres_statement_cache_size)
        postgres_ssl_env = {
            "POSTGRES_SSL_MODE": self.postgres_ssl_mode,
            "POSTGRES_SSL_CERT": self.postgres_ssl_cert,
            "POSTGRES_SSL_KEY": self.postgres_ssl_key,
            "POSTGRES_SSL_ROOT_CERT": self.postgres_ssl_root_cert,
            "POSTGRES_SSL_CRL": self.postgres_ssl_crl,
        }
        for key, value in postgres_ssl_env.items():
            rendered = self._env_value(value)
            if rendered is not None:
                pg_env_map[key] = rendered
        pg_env_map["POSTGRES_CONNECTION_RETRIES"] = str(self.postgres_connection_retries)
        pg_env_map["POSTGRES_CONNECTION_RETRY_BACKOFF"] = str(
            self.postgres_connection_retry_backoff
        )
        pg_env_map["POSTGRES_CONNECTION_RETRY_BACKOFF_MAX"] = str(
            self.postgres_connection_retry_backoff_max
        )
        pg_env_map["POSTGRES_POOL_CLOSE_TIMEOUT"] = str(self.postgres_pool_close_timeout)
        for key, value in pg_env_map.items():
            if force or key not in os.environ:
                os.environ[key] = value

        if force or "POSTGRES_SERVER_SETTINGS" not in os.environ:
            os.environ["POSTGRES_SERVER_SETTINGS"] = self.postgres_server_settings_env_value()

    def apply_lightrag_runtime_env(self, *, force: bool = False) -> None:
        """Bridge LightRAG runtime settings controlled by DlightRAG."""
        if force or "LIGHTRAG_PARSER" not in os.environ:
            os.environ["LIGHTRAG_PARSER"] = self.parser.rules
        if force or "INPUT_DIR" not in os.environ:
            os.environ["INPUT_DIR"] = str(self.input_dir_path)
        if force or "DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY" not in os.environ:
            os.environ["DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY"] = (
                self.parser_sidecars.mineru.auxiliary_block_policy
            )

    def model_post_init(self, _context) -> None:
        """Pydantic lifecycle hook: bridge DLIGHTRAG_* → backend env vars."""
        # Resolve working_dir to absolute path
        path = Path(self.working_dir)
        if not path.is_absolute():
            self.working_dir = str(path.resolve())

        _load_lightrag_sidecar_env(self.__class__.model_config.get("env_file"), force=False)
        self.apply_lightrag_sidecar_env(force=False)
        self.apply_lightrag_backend_env(force=False)
        self.apply_lightrag_runtime_env(force=True)


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
