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

from dotenv import find_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_YAML_FILE = "config.yaml"


def _find_yaml_config() -> Path | None:
    """Locate config.yaml: cwd first, then next to .env, then None."""
    cwd = Path(_YAML_FILE)
    if cwd.is_file():
        return cwd
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        candidate = Path(dotenv_path).parent / _YAML_FILE
        if candidate.is_file():
            return candidate
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

    provider: str | None = None  # None = auto-detect from model/base_url
    model: str = "text-embedding-3-large"
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024
    max_token_size: int = 8192


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
        env_file=find_dotenv(usecwd=True),
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

    # pgvector HNSW index configuration
    pg_vector_index_type: str = Field(
        default="HNSW",
        description="pgvector index type — case-sensitive (HNSW, IVFFLAT, VCHORDRQ)",
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

    # ===== Storage Backends (configurable, default PostgreSQL) =====
    vector_storage: Literal[
        "PGVectorStorage",
        "MilvusVectorDBStorage",
        "NanoVectorDBStorage",
        "ChromaVectorDBStorage",
        "FaissVectorDBStorage",
        "QdrantVectorDBStorage",
    ] = Field(default="PGVectorStorage", description="LightRAG vector storage backend.")
    graph_storage: Literal[
        "PGGraphStorage",
        "Neo4JStorage",
        "NetworkXStorage",
        "MemgraphStorage",
    ] = Field(default="PGGraphStorage", description="LightRAG graph storage backend.")
    kv_storage: Literal[
        "PGKVStorage",
        "JsonKVStorage",
        "RedisKVStorage",
        "MongoKVStorage",
    ] = Field(default="PGKVStorage", description="LightRAG KV storage backend.")
    doc_status_storage: Literal[
        "PGDocStatusStorage",
        "JsonDocStatusStorage",
        "RedisDocStatusStorage",
        "MongoDocStatusStorage",
    ] = Field(default="PGDocStatusStorage", description="LightRAG doc status backend.")

    # ===== Optional: Neo4j (if graph_storage=Neo4JStorage) =====
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="neo4j")

    # ===== Optional: Milvus (if vector_storage=MilvusVectorDBStorage) =====
    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_user: str = Field(default="")
    milvus_password: str = Field(default="")
    milvus_token: str | None = Field(default=None)
    milvus_db_name: str = Field(default="default")

    # ===== Optional: Qdrant (if vector_storage=QdrantVectorDBStorage) =====
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)

    # ===== Vector DB Backend Kwargs (passthrough to LightRAG) =====
    vector_db_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra parameters forwarded to LightRAG vector_db_storage_cls_kwargs. "
        "Backend-specific — see .env.example for per-backend options. "
        'Example: DLIGHTRAG_VECTOR_DB_KWARGS=\'{"index_type": "HNSW_SQ", "sq_type": "SQ8"}\'',
    )

    # ===== New nested config =====
    chat: ModelConfig = Field(default_factory=lambda: ModelConfig(model="gpt-4.1", temperature=0.5))
    ingest: ModelConfig | None = None
    # Per-role LLM configs (optional, fallback to chat). All four are routed via
    # LightRAG 1.5.0+ role_llm_configs registry where applicable; vlm is also
    # used by DlightRAG's own vision_model_func paths (vlm_parser, multimodal
    # query, unified extractor).
    extract: ModelConfig | None = None
    keywords: ModelConfig | None = None
    query: ModelConfig | None = None
    vlm: ModelConfig | None = None
    embedding: EmbeddingConfig  # required, no default
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    # ===== RAG Processing =====
    rag_mode: Literal["caption", "unified"] = Field(
        default="caption",
        description="RAG mode. 'caption': RAGAnything caption-based pipeline. "
        "'unified': visual page embedding pipeline (requires multimodal models).",
    )
    page_render_dpi: int = Field(
        default=250,
        description="Page rendering DPI for unified mode. Higher = better quality but larger images.",
    )
    working_dir: str = Field(default="./dlightrag_storage")
    parser: Literal["mineru", "docling", "vlm"] = Field(default="mineru")
    parse_method: Literal["auto", "ocr", "txt"] = Field(default="auto")
    mineru_backend: str | None = Field(
        default=None,
        description="MinerU parsing backend. If None, auto-detects: CUDA GPU → hybrid-auto-engine, "
        "otherwise → pipeline. "
        "Valid values: pipeline, vlm-auto-engine, vlm-http-client, hybrid-auto-engine, hybrid-http-client.",
    )
    enable_image_processing: bool = Field(default=True)
    enable_table_processing: bool = Field(default=True)
    enable_equation_processing: bool = Field(default=False)
    display_content_stats: bool = Field(default=False)
    use_full_path: bool = Field(default=True)
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
    max_entity_tokens: int = Field(default=6000)
    max_relation_tokens: int = Field(default=8000)
    max_total_tokens: int = Field(default=40000)
    default_mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(default="mix")

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
        "0.0.0.0 only when api_auth_token is also set; the bearer-token middleware "
        "then guards the transport.",
    )
    mcp_port: int = Field(default=8101)

    # ===== REST API Server =====
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8100)
    api_auth_token: str | None = Field(
        default=None,
        description="Bearer token for REST API authentication. If not set, no auth required.",
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

    # ===== Validators =====

    @model_validator(mode="after")
    def _infer_auth_mode(self):
        """Auto-upgrade: if api_auth_token is set but auth_mode is default, use 'simple'."""
        if self.auth_mode == "none" and self.api_auth_token:
            self.auth_mode = "simple"
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

    def model_post_init(self, __context) -> None:
        """Pydantic lifecycle hook: bridge DLIGHTRAG_* → backend env vars."""
        # Bridge PostgreSQL env vars for LightRAG (only when PG backends are used)
        _storage_fields = (
            self.vector_storage,
            self.graph_storage,
            self.kv_storage,
            self.doc_status_storage,
        )
        _uses_pg = any(s.startswith("PG") for s in _storage_fields)

        if _uses_pg:
            pg_env_map = {
                "POSTGRES_HOST": self.postgres_host,
                "POSTGRES_PORT": str(self.postgres_port),
                "POSTGRES_USER": self.postgres_user,
                "POSTGRES_PASSWORD": self.postgres_password,
                "POSTGRES_DATABASE": self.postgres_database,
                "POSTGRES_WORKSPACE": self.workspace,
                "POSTGRES_VECTOR_INDEX_TYPE": self.pg_vector_index_type,
                "POSTGRES_HNSW_M": str(self.pg_hnsw_m),
                # NB: LightRAG's POSTGRES_HNSW_EF is actually ef_construction
                # (used in CREATE INDEX ... WITH (ef_construction = ...)),
                # not ef_search. The name is misleading in LightRAG.
                "POSTGRES_HNSW_EF": str(self.pg_hnsw_ef_construction),
            }
            for key, value in pg_env_map.items():
                if key not in os.environ:
                    os.environ[key] = value

            # ef_search is a session-level GUC, not an index build param.
            # LightRAG has no native support for it, so we inject it via
            # POSTGRES_SERVER_SETTINGS → asyncpg create_pool(server_settings=...).
            if "POSTGRES_SERVER_SETTINGS" not in os.environ:
                os.environ["POSTGRES_SERVER_SETTINGS"] = f"hnsw.ef_search={self.pg_hnsw_ef_search}"

        # Bridge workspace env vars for non-PG backends that override constructor
        _WORKSPACE_ENV_MAP = {
            "Neo4J": "NEO4J_WORKSPACE",
            "Memgraph": "MEMGRAPH_WORKSPACE",
            "Mongo": "MONGODB_WORKSPACE",
            "Redis": "REDIS_WORKSPACE",
        }
        for prefix, env_key in _WORKSPACE_ENV_MAP.items():
            if any(s.startswith(prefix) for s in _storage_fields):
                if env_key not in os.environ:
                    os.environ[env_key] = self.workspace

        # Bridge Neo4j connection env vars if using Neo4JStorage
        if self.graph_storage == "Neo4JStorage":
            neo4j_env_map = {
                "NEO4J_URI": self.neo4j_uri,
                "NEO4J_USERNAME": self.neo4j_username,
                "NEO4J_PASSWORD": self.neo4j_password,
            }
            for key, value in neo4j_env_map.items():
                if key not in os.environ:
                    os.environ[key] = value

        # Bridge Milvus env vars if using MilvusVectorDBStorage
        if self.vector_storage == "MilvusVectorDBStorage":
            milvus_env_map: dict[str, str] = {
                "MILVUS_URI": self.milvus_uri,
                "MILVUS_DB_NAME": self.milvus_db_name,
            }
            if self.milvus_user:
                milvus_env_map["MILVUS_USER"] = self.milvus_user
            if self.milvus_password:
                milvus_env_map["MILVUS_PASSWORD"] = self.milvus_password
            if self.milvus_token:
                milvus_env_map["MILVUS_TOKEN"] = self.milvus_token
            for key, value in milvus_env_map.items():
                if key not in os.environ:
                    os.environ[key] = value

        # Bridge Qdrant env vars if using QdrantVectorDBStorage
        if self.vector_storage == "QdrantVectorDBStorage":
            qdrant_env_map: dict[str, str] = {"QDRANT_URL": self.qdrant_url}
            if self.qdrant_api_key:
                qdrant_env_map["QDRANT_API_KEY"] = self.qdrant_api_key
            for key, value in qdrant_env_map.items():
                if key not in os.environ:
                    os.environ[key] = value

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
