# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Self-contained configuration for dlightrag.

Uses pydantic-settings with DLIGHTRAG_ env prefix. Works standalone (MCP/API server)
and as library (imported by other projects which construct DlightragConfig explicitly).

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


class ModelConfig(BaseModel):
    """Reusable model configuration block."""

    provider: Literal["openai", "litellm"] = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    timeout: float = 120.0
    max_retries: int = 3
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(ModelConfig):
    """Embedding-specific configuration."""

    model: str = "text-embedding-3-large"
    dim: int = 1024
    max_token_size: int = 8192


class RerankConfig(BaseModel):
    """Reranking configuration (independent provider system)."""

    enabled: bool = True
    backend: Literal["llm", "cohere", "jina", "aliyun", "azure_cohere"] = "llm"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    score_threshold: float = 0.5


class DlightragConfig(BaseSettings):
    """DlightRAG configuration.

    Configuration sources (in order of precedence):
        1. Constructor arguments (when used as library)
        2. Environment variables (DLIGHTRAG_ prefix)
        3. .env file (development)
        4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
        env_file=find_dotenv(usecwd=True),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== PostgreSQL (Default Storage Backend) =====
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="dlightrag")
    postgres_password: str = Field(default="dlightrag")
    postgres_database: str = Field(default="dlightrag")
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
    vector_storage: str = Field(
        default="PGVectorStorage",
        description="LightRAG vector storage backend. Options: PGVectorStorage, "
        "MilvusVectorDBStorage, NanoVectorDBStorage, ChromaVectorDBStorage, "
        "FaissVectorDBStorage, QdrantVectorDBStorage",
    )
    graph_storage: str = Field(
        default="PGGraphStorage",
        description="LightRAG graph storage backend. Options: PGGraphStorage, "
        "Neo4JStorage, NetworkXStorage, MemgraphStorage",
    )
    kv_storage: str = Field(
        default="PGKVStorage",
        description="LightRAG KV storage backend. Options: PGKVStorage, "
        "JsonKVStorage, RedisKVStorage, MongoKVStorage",
    )
    doc_status_storage: str = Field(
        default="PGDocStatusStorage",
        description="LightRAG doc status backend. Options: PGDocStatusStorage, "
        "JsonDocStatusStorage, RedisDocStatusStorage, MongoDocStatusStorage",
    )

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
    chat: ModelConfig = Field(
        default_factory=lambda: ModelConfig(model="gpt-4.1-mini", temperature=0.5)
    )
    ingest: ModelConfig | None = None
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

    # ===== Query Configuration =====
    top_k: int = Field(default=60)
    chunk_top_k: int = Field(default=30)
    max_entity_tokens: int = Field(default=8000)
    max_relation_tokens: int = Field(default=10000)
    max_total_tokens: int = Field(default=40000)
    default_mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(default="mix")
    rrf_k: int = Field(default=60, description="RRF fusion parameter k for multi-path retrieval")
    max_conversation_turns: int = Field(default=50)
    max_conversation_tokens: int = Field(default=150000)

    # ===== Knowledge Graph =====
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
    snowflake_account: str | None = Field(default=None)
    snowflake_user: str | None = Field(default=None)
    snowflake_password: str | None = Field(default=None)
    snowflake_database: str | None = Field(default=None)
    snowflake_schema: str | None = Field(default=None)
    snowflake_warehouse: str | None = Field(default=None)

    # ===== Domain Knowledge (injected by caller) =====
    domain_knowledge_hints: str = Field(
        default="",
        description="Domain-specific hints for reranking. Injected by the calling application.",
    )

    # ===== MCP Server =====
    mcp_transport: Literal["stdio", "streamable-http"] = Field(default="stdio")
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=8101)

    # ===== REST API Server =====
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8100)
    api_auth_token: str | None = Field(
        default=None,
        description="Bearer token for REST API authentication. If not set, no auth required.",
    )

    # ===== Operational =====
    log_level: str = Field(default="info")
    request_timeout: int = Field(
        default=600,
        description="Overall request timeout in seconds for ingest/retrieve/answer operations.",
    )

    # ===== Validators =====

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_fields(cls, values):
        if not isinstance(values, dict):
            return values
        legacy = {
            "openai_api_key", "azure_openai_api_key", "anthropic_api_key",
            "qwen_api_key", "minimax_api_key", "ollama_api_key",
            "xinference_api_key", "openrouter_api_key", "voyage_api_key",
            "llm_provider", "chat_model", "vision_model", "vision_provider",
            "embedding_provider", "embedding_model", "embedding_dim",
            "ingestion_model", "llm_temperature", "llm_request_timeout",
        }
        found = [k for k in values if k in legacy]
        if found:
            raise ValueError(
                f"Legacy config fields detected: {found}. "
                "Please migrate to the new nested format. "
                "See .env.example for the new configuration format."
            )
        return values

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
