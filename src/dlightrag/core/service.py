# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG Service - High-level facade for ingestion and retrieval.

Slim RAGService that uses a SINGLE RAGAnything instance, composing
IngestionPipeline and RetrievalEngine. No external backend dependencies.
Uses PostgreSQL advisory locks instead of Redis for distributed
initialization coordination.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol

from dlightrag.captionrag.chunking import docling_hybrid_chunking_func
from dlightrag.config import DlightragConfig, get_config
from dlightrag.utils.tokens import truncate_conversation_history

logger = logging.getLogger(__name__)

# Init lock key for PostgreSQL advisory lock (arbitrary 64-bit int)
_PG_INIT_LOCK_KEY = 0x436F727072616700  # "Dlightrag\0" as int


def _detect_mineru_backend(manual_override: str | None = None) -> str:
    """Detect optimal MinerU parsing backend based on hardware."""
    if manual_override:
        logger.info(f"MinerU backend: {manual_override} (manual override)")
        return manual_override

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"MinerU backend: hybrid-auto-engine (CUDA GPU detected: {device_name})")
            return "hybrid-auto-engine"

        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and torch.backends.mps.is_available()
        ):
            logger.info("MinerU backend: Apple Silicon detected")
            return "pipeline"  # Fallback to pipeline for now

    except ImportError:
        logger.debug("torch not available, skipping GPU detection")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    logger.info("MinerU backend: pipeline (fallback, no GPU acceleration detected)")
    return "pipeline"


def _ensure_venv_in_path() -> None:
    """Add venv bin to PATH for MinerU CLI."""
    venv_bin = Path(sys.executable).parent
    current_path = os.environ.get("PATH", "")
    if str(venv_bin) not in current_path.split(":"):
        os.environ["PATH"] = f"{venv_bin}:{current_path}" if current_path else str(venv_bin)


_ensure_venv_in_path()

# Inject custom VLM prompts before importing RAGAnything
from dlightrag.models.prompts import inject_custom_prompts  # noqa: E402

inject_custom_prompts()

# RAGAnything is only needed for caption mode — make import optional
# so unified mode doesn't require raganything dependencies.
try:
    from raganything import RAGAnything, RAGAnythingConfig  # noqa: E402

    _HAS_RAGANYTHING = True
except ImportError:
    _HAS_RAGANYTHING = False

from lightrag.utils import EmbeddingFunc  # noqa: E402

from dlightrag.captionrag.pipeline import IngestionPipeline  # noqa: E402
from dlightrag.captionrag.retrieval import RetrievalEngine  # noqa: E402
from dlightrag.core.retrieval.path_resolver import PathResolver  # noqa: E402
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult  # noqa: E402
from dlightrag.models.llm import (  # noqa: E402
    get_embedding_func,
    get_llm_model_func,
    get_rerank_func,
    get_vision_model_func,
)
from dlightrag.unifiedrepresent.lifecycle import unified_delete_files, unified_ingest  # noqa: E402


class RAGService:
    """High-level RAG service facade.

    Responsibilities:
    - Expose simple API for agents and external modules
    - Coordinate ingestion and retrieval pipelines
    - Manage RAG instances and configuration

    Usage:
        # Production: use async factory method
        rag = await RAGService.create(config=my_config, enable_vlm=True)

        # With callbacks:
        rag = await RAGService.create(
            config=my_config,
            cancel_checker=my_cancel_fn,
        )
    """

    @classmethod
    async def create(
        cls,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    ) -> RAGService:
        """Async factory method - creates and initializes RAGService."""
        instance = cls(
            config=config,
            enable_vlm=enable_vlm,
            cancel_checker=cancel_checker,
        )
        await instance.initialize()
        return instance

    def __init__(
        self,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Store configuration only. Use RAGService.create() for full initialization."""
        self.config = config or get_config()
        self.enable_vlm = enable_vlm
        self._initialized: bool = False

        # Callbacks for decoupled integration
        self._cancel_checker = cancel_checker

        # Caption mode (Mode 1): RAGAnything + composed pipelines
        self.rag: Any = None  # RAGAnything (caption mode)
        self.ingestion: IngestionPipeline | None = None
        self.retrieval: RetrievalEngine | None = None

        # Unified mode (Mode 2): direct LightRAG + UnifiedRepresentEngine
        self.unified: Any = None  # UnifiedRepresentEngine
        self._lightrag: Any = None  # Direct LightRAG reference (unified mode)
        self._visual_chunks: Any = None  # Visual chunks KV store (unified mode)
        self._hash_index: Any = None  # Content-hash deduplication index

        # Retrieval backend (satisfies RetrievalBackend Protocol).
        # Explicitly wired by _do_initialize / _do_initialize_unified;
        # falls back to self.unified or self.retrieval so tests that set
        # those attributes directly (without going through initialize) still work.
        self._backend: Any = None

    @property
    def _effective_backend(self) -> Any:
        """Return the active retrieval backend, with lazy fallback."""
        return self._backend or self.unified or self.retrieval

    @property
    def lightrag(self) -> Any:
        """Return the underlying LightRAG instance regardless of mode.

        - Unified mode: ``self._lightrag`` (created directly)
        - Caption mode: ``self.rag.lightrag`` (via RAGAnything)
        """
        return self._lightrag or getattr(self.rag, "lightrag", None)

    @staticmethod
    def _build_vector_db_kwargs(config: DlightragConfig) -> dict[str, Any]:
        """Build vector_db_storage_cls_kwargs from config.vector_db_kwargs passthrough."""
        kwargs: dict[str, Any] = {"cosine_better_than_threshold": 0.3}
        kwargs.update(config.vector_db_kwargs)
        return kwargs

    def _unregister_atexit_cleanup(self, rag_obj: Any) -> None:
        """Prevent double-close logging errors by removing raganything atexit hooks."""
        try:
            atexit.unregister(rag_obj.close)
        except Exception:  # noqa: BLE001
            logger.debug("Unable to unregister atexit hook for %s", rag_obj)

    async def initialize(self) -> None:
        """Initialize LightRAG storages and caches (idempotent).

        Uses PostgreSQL advisory lock for distributed coordination when
        using PG storage backends, or proceeds directly otherwise.
        """
        if self._initialized:
            return

        if self.config.kv_storage.startswith("PG"):
            await self._initialize_with_pg_lock()
        else:
            await self._do_initialize()

        self._initialized = True
        logger.debug("RAGService initialized")

    async def _initialize_with_pg_lock(self) -> None:
        """Initialize with PostgreSQL advisory lock for multi-worker safety."""
        try:
            import asyncpg
        except ImportError:
            logger.warning("asyncpg not available, proceeding without distributed lock")
            await self._do_initialize()
            return

        try:
            conn = await asyncpg.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                database=self.config.postgres_database,
            )
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable for init lock, proceeding without: {e}")
            await self._do_initialize()
            return

        try:
            acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY)

            if acquired:
                logger.info("Acquired PG advisory lock, initializing RAG pipelines...")
                try:
                    await self._ensure_pg_schema(conn)
                    await self._do_initialize()
                    logger.info("RAG pipelines initialized successfully")
                finally:
                    await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
            else:
                logger.info("Another worker is initializing, waiting for lock...")
                for _ in range(180):
                    await asyncio.sleep(1)
                    if await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY):
                        await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
                        break
                logger.info("Lock released, connecting to existing storages...")
                await self._do_initialize()
        finally:
            await conn.close()

    async def _ensure_pg_schema(self, conn) -> None:
        """Ensure required PostgreSQL extensions and DlightRAG tables exist.

        Extension creation is delegated to LightRAG (``configure_vector_extension``,
        ``configure_age_extension``). We only verify they are available (fail-fast)
        and create DlightRAG's own tables.
        """
        # Fail-fast: verify extensions are available (LightRAG creates them but
        # only warns on failure — we want a hard error before wasting time).
        installed = {r["extname"] for r in await conn.fetch("SELECT extname FROM pg_extension")}
        missing = {"vector", "age"} - installed
        if missing:
            raise RuntimeError(
                f"Required PostgreSQL extensions not installed: {missing}. "
                "Ask your DBA to run: "
                + "; ".join(f"CREATE EXTENSION {ext}" for ext in sorted(missing))
            )

        # DlightRAG's own table for content-hash deduplication
        await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_file_hashes (
            content_hash TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            workspace TEXT NOT NULL DEFAULT 'default',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id ON dlightrag_file_hashes(doc_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace ON dlightrag_file_hashes(workspace)"
        )
        logger.info("PostgreSQL schema ensured")

    async def _do_initialize(self) -> None:
        """Create RAG backend and compose pipelines based on rag_mode."""
        config = self.config
        logger.info("RAG mode: %s", config.rag_mode)

        if config.rag_mode == "unified":
            await self._do_initialize_unified()
            return

        # --- Caption mode (existing code, unchanged) ---
        if not _HAS_RAGANYTHING:
            raise ImportError(
                "raganything is required for caption mode. Install it with: pip install raganything"
            )

        # Detect optimal MinerU backend based on hardware (only for mineru parser)
        mineru_backend = None
        if config.parser == "mineru":
            mineru_backend = _detect_mineru_backend(config.mineru_backend)
        else:
            logger.info(f"Using {config.parser} parser, MinerU backend detection skipped")

        # Configure RAGAnything
        rag_config = RAGAnythingConfig(
            working_dir=str(config.working_dir_path),
            max_concurrent_files=config.max_concurrent_ingestion,
            parser=config.parser,
            parse_method=config.parse_method,
            enable_image_processing=config.enable_image_processing,
            enable_table_processing=config.enable_table_processing,
            enable_equation_processing=config.enable_equation_processing,
            display_content_stats=config.display_content_stats,
            use_full_path=config.use_full_path,
            context_window=config.context_window,
            context_filter_content_types=config.context_filter_types.split(","),
            max_context_tokens=config.max_context_tokens,
        )

        # Get model functions
        llm_func = get_llm_model_func(config)
        vision_func = get_vision_model_func(config) if self.enable_vlm else None
        embedding_func = get_embedding_func(config)
        rerank_func = get_rerank_func(config)

        # Create VLM OCR parser if configured
        vlm_parser = None
        if config.parser == "vlm":
            if vision_func is None:
                raise ValueError(
                    "parser='vlm' requires a vision model. "
                    "Set vision_model and vision_provider in config."
                )
            from dlightrag.captionrag.vlm_parser import VlmOcrParser

            vlm_parser = VlmOcrParser(
                vision_model_func=vision_func,
                dpi=config.page_render_dpi,
            )

        # LightRAG configuration
        lightrag_kwargs: dict[str, Any] = {
            "workspace": config.workspace,
            "default_llm_timeout": config.llm_request_timeout,
            "default_embedding_timeout": config.embedding_request_timeout,
            "chunk_token_size": config.chunk_size,
            "chunk_overlap_token_size": config.chunk_overlap,
            "max_parallel_insert": config.max_parallel_insert,
            "llm_model_max_async": config.max_async,
            "embedding_func_max_async": config.embedding_func_max_async,
            "embedding_batch_num": config.embedding_batch_num,
            "vector_storage": config.vector_storage,
            "graph_storage": config.graph_storage,
            "kv_storage": config.kv_storage,
            "doc_status_storage": config.doc_status_storage,
            "rerank_model_func": rerank_func,
            "vector_db_storage_cls_kwargs": self._build_vector_db_kwargs(config),
            "addon_params": {
                "entity_types": config.kg_entity_types,
                "language": "English",
            },
        }

        # Only use our custom HybridChunker for parsers that benefit from it
        if config.parser in ("docling", "vlm"):
            lightrag_kwargs["chunking_func"] = docling_hybrid_chunking_func

        # ONE RAGAnything with unified llm_func (chat model)
        logger.info("Creating RAGAnything instance...")
        self.rag = RAGAnything(
            None,
            llm_func,
            vision_func,
            embedding_func,
            rag_config,
            lightrag_kwargs,
        )
        self._unregister_atexit_cleanup(self.rag)

        # Initialize LightRAG storages
        if hasattr(self.rag, "_ensure_lightrag_initialized"):
            init_result = await self.rag._ensure_lightrag_initialized()
            if isinstance(init_result, dict) and not init_result.get("success", True):
                error_msg = init_result.get("error", "LightRAG initialization failed")
                logger.error(f"LightRAG initialization failed: {error_msg}")
                raise RuntimeError(f"LightRAG initialization failed: {error_msg}")

        # Auto-detect hash index backend based on storage config
        hash_index = await self._create_hash_index(config)

        # Compose pipelines
        self.ingestion = IngestionPipeline(
            self.rag,
            config=config,
            max_concurrent=config.max_concurrent_ingestion,
            mineru_backend=mineru_backend,
            cancel_checker=self._cancel_checker,
            hash_index=hash_index,
            vlm_parser=vlm_parser,
        )

        self.retrieval = RetrievalEngine(rag=self.rag, config=config)
        self._path_resolver = PathResolver(
            working_dir=str(config.working_dir_path),
        )
        self.retrieval._path_resolver = self._path_resolver
        self._backend = self.retrieval

        logger.info("RAG pipelines initialized successfully (caption mode)")

    async def _do_initialize_unified(self) -> None:
        """Initialize unified representational RAG mode (Mode 2).

        Creates LightRAG directly (no RAGAnything), sets up visual_chunks
        KV store, and creates UnifiedRepresentEngine.
        """
        import dataclasses

        from lightrag import LightRAG

        from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine

        config = self.config
        logger.info("Initializing unified representational RAG mode...")

        # Get model functions
        llm_func = get_llm_model_func(config)
        vision_func = get_vision_model_func(config) if self.enable_vlm else None

        # Use httpx_text_embed instead of LightRAG's openai_embed for text
        # embedding.  openai_embed uses encoding_format:"base64" and the openai
        # Python client — both fail with Xinference VL models.
        # partial() with plain strings is deepcopy-safe (LightRAG's __post_init__
        # does asdict→deepcopy on embedding_func).
        from functools import partial

        from dlightrag.unifiedrepresent.embedder import (
            OpenAICompatProvider,
            VisualEmbedder,
            VoyageProvider,
            httpx_text_embed,
        )

        emb_provider = config.effective_embedding_provider
        emb_base_url = config._get_url(f"{emb_provider}_base_url") or ""
        emb_api_key = config._get_provider_api_key(emb_provider) or ""

        if emb_provider == "voyage":
            embed_provider = VoyageProvider()
        else:
            embed_provider = OpenAICompatProvider()

        embedding_func = EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=8192,
            func=partial(
                httpx_text_embed,
                model=config.embedding_model,
                base_url=emb_base_url,
                api_key=emb_api_key,
                provider=embed_provider,
            ),
            model_name=config.embedding_model,
        )

        # VisualEmbedder for image embedding (reuses persistent httpx client)
        visual_embedder = VisualEmbedder(
            model=config.embedding_model,
            base_url=emb_base_url,
            api_key=emb_api_key,
            dim=config.embedding_dim,
            batch_size=config.embedding_func_max_async,
            provider=embed_provider,
        )

        # LightRAG configuration (same storage backends as caption mode)
        # Do NOT pass rerank_model_func — we handle reranking ourselves
        lightrag = LightRAG(
            working_dir=str(config.working_dir_path),
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            workspace=config.workspace,
            default_llm_timeout=config.llm_request_timeout,
            default_embedding_timeout=config.embedding_request_timeout,
            chunk_token_size=config.chunk_size,
            chunk_overlap_token_size=config.chunk_overlap,
            max_parallel_insert=config.max_parallel_insert,
            llm_model_max_async=config.max_async,
            embedding_func_max_async=config.embedding_func_max_async,
            embedding_batch_num=config.embedding_batch_num,
            vector_storage=config.vector_storage,
            graph_storage=config.graph_storage,
            kv_storage=config.kv_storage,
            doc_status_storage=config.doc_status_storage,
            vector_db_storage_cls_kwargs=self._build_vector_db_kwargs(config),
            addon_params={
                "entity_types": config.kg_entity_types,
                "language": "English",
            },
        )
        await lightrag.initialize_storages()
        self._lightrag = lightrag

        # Create visual_chunks KV store.
        # LightRAG's PGKVStorage only supports 7 hardcoded namespaces; custom ones
        # (like "visual_chunks") cause KeyError / silent no-op / SQL errors.
        # See pg_jsonb_kv.py module docstring for details.
        kv_cls = lightrag.key_string_value_json_storage_cls
        if config.kv_storage.startswith("PG"):
            from dlightrag.storage.pg_jsonb_kv import PGJsonbKVStorage

            kv_cls = PGJsonbKVStorage
        kv_kwargs: dict[str, Any] = {
            "namespace": "visual_chunks",
            "workspace": config.workspace,
            "global_config": dataclasses.asdict(lightrag),
            "embedding_func": embedding_func,
        }
        if kv_cls is PGJsonbKVStorage:
            kv_kwargs["blob_field"] = "image_data"
        visual_chunks = kv_cls(**kv_kwargs)
        await visual_chunks.initialize()
        self._visual_chunks = visual_chunks

        # Create PathResolver for unified mode
        self._path_resolver = PathResolver(
            working_dir=str(config.working_dir_path),
        )

        # Create engine (pass pre-built embedder to avoid creating a duplicate)
        self.unified = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=vision_func,
            visual_embedder=visual_embedder,
            path_resolver=self._path_resolver,
        )
        self._backend = self.unified

        # Create hash index for deduplication (same backend as caption mode)
        self._hash_index = await self._create_hash_index(config)

        logger.info("Unified representational RAG mode initialized")

    async def _create_hash_index(self, config: DlightragConfig) -> HashIndexProtocol:
        """Create the appropriate hash index backend based on KV storage config.

        Uses the same backend as the configured KV storage for consistency.
        Falls back to JSON file-based HashIndex if the backend package is unavailable.
        """
        kv = config.kv_storage

        if kv.startswith("PG"):
            try:
                from dlightrag.core.ingestion.hash_index import PGHashIndex

                idx = PGHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: PGHashIndex (PostgreSQL via shared pool)")
                return idx
            except ImportError:
                logger.warning("asyncpg not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"PGHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Redis"):
            try:
                from dlightrag.core.ingestion.hash_index import RedisHashIndex

                idx = RedisHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: RedisHashIndex (Redis via shared pool)")
                return idx
            except ImportError:
                logger.warning("redis not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"RedisHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Mongo"):
            try:
                from dlightrag.core.ingestion.hash_index import MongoHashIndex

                idx = MongoHashIndex(workspace=config.workspace)
                await idx.initialize()
                logger.info("Hash index: MongoHashIndex (MongoDB via shared client)")
                return idx
            except ImportError:
                logger.warning("motor not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"MongoHashIndex creation failed, falling back to JSON: {e}")

        from dlightrag.core.ingestion.hash_index import HashIndex

        logger.info("Hash index: HashIndex (JSON file)")
        return HashIndex(config.working_dir_path, workspace=config.workspace)

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RAGService not initialized. Use 'await RAGService.create()' instead."
            )

    async def close(self) -> None:
        """Clean up storages and worker pools (best-effort)."""
        # Shutdown LightRAG worker pools first — they hold background asyncio
        # tasks that block asyncio.run() from exiting.
        await self._shutdown_worker_pools()

        if self.rag is not None:
            try:
                await self.rag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize storages", exc_info=True)

        # Unified mode cleanup
        if self.unified is not None:
            try:
                await self.unified.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close unified engine", exc_info=True)
        if self._visual_chunks is not None:
            try:
                await self._visual_chunks.finalize()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize visual_chunks", exc_info=True)
        if self._lightrag is not None:
            try:
                await self._lightrag.finalize_storages()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to finalize LightRAG storages", exc_info=True)

    async def _shutdown_worker_pools(self) -> None:
        """Shutdown LightRAG's EmbeddingFunc/LLM worker pools.

        LightRAG wraps embedding_func and llm_model_func with
        priority_limit_async_func_call which creates background asyncio
        tasks.  finalize_storages() does NOT shut these down, so they
        block asyncio.run() from exiting.  The wrapped functions expose
        a ``.shutdown()`` coroutine we can call explicitly.
        """
        lr = self.lightrag
        if lr is None:
            return

        for attr in ("embedding_func", "llm_model_func"):
            try:
                obj = getattr(lr, attr, None)
                # EmbeddingFunc stores the wrapped func in .func
                func = getattr(obj, "func", obj)
                shutdown = getattr(func, "shutdown", None)
                if shutdown is not None:
                    await shutdown()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to shutdown %s worker pool", attr, exc_info=True)

    # === INGESTION API ===

    async def aingest(
        self,
        source_type: Literal["local", "azure_blob", "snowflake"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Unified ingestion API.

        Args:
            source_type: "local", "azure_blob", or "snowflake"
            kwargs:
                local: path, replace
                azure_blob: source, container_name, blob_path, prefix, replace
                snowflake: query, table
        """
        self._ensure_initialized()

        # Unified mode
        if self.unified is not None:
            return await unified_ingest(
                engine=self.unified,
                config=self.config,
                hash_index=self._hash_index,
                source_type=source_type,
                **kwargs,
            )

        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")

        ingestion = self.ingestion

        # Get replace default from config if not explicitly set
        def get_replace_value() -> bool:
            replace_arg = kwargs.get("replace")
            if replace_arg is None:
                return self.config.ingestion_replace_default
            return bool(replace_arg)

        if source_type == "local":
            path = Path(kwargs["path"])
            replace = get_replace_value()

            logger.info(f"Ingesting from local path: {path}")

            result = await ingestion.aingest_from_local(
                path=path,
                replace=replace,
            )
            return result.model_dump(exclude_none=True)

        if source_type == "azure_blob":
            container_name = kwargs["container_name"]
            blob_path = kwargs.get("blob_path")
            prefix = kwargs.get("prefix")
            replace = get_replace_value()
            source = kwargs.get("source")

            if source is None:
                from dlightrag.sourcing.azure_blob import AzureBlobDataSource

                source = AzureBlobDataSource(
                    container_name=container_name,
                    connection_string=self.config.blob_connection_string,
                )

            # Default to entire container if neither blob_path nor prefix provided
            if blob_path is None and prefix is None:
                prefix = ""

            try:
                result = await ingestion.aingest_from_azure_blob(
                    source=source,
                    container_name=container_name,
                    blob_path=blob_path,
                    prefix=prefix,
                    replace=replace,
                )
                return result.model_dump(exclude_none=True)
            finally:
                if hasattr(source, "aclose"):
                    await source.aclose()

        # source_type == "snowflake"
        ingestion_result = await ingestion.aingest_from_snowflake(
            query=kwargs["query"],
            table=kwargs.get("table"),
        )
        return ingestion_result.model_dump(exclude_none=True)

    # === RETRIEVAL API ===

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer."""
        self._ensure_initialized()
        backend = self._effective_backend
        if not backend:
            raise RuntimeError("Retrieval backend not initialized")
        return await backend.aretrieve(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            is_reretrieve=is_reretrieve,
            **kwargs,
        )

    def _truncate_history(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Truncate conversation_history in kwargs (single point of truncation)."""
        history = kwargs.get("conversation_history")
        if history:
            kwargs["conversation_history"] = truncate_conversation_history(
                history,
                max_messages=self.config.max_conversation_turns * 2,
                max_tokens=self.config.max_conversation_tokens,
            )
        return kwargs

    async def aanswer(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve contexts and generate an LLM answer."""
        self._ensure_initialized()
        backend = self._effective_backend
        if not backend:
            raise RuntimeError("Retrieval backend not initialized")
        self._truncate_history(kwargs)
        return await backend.aanswer(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            **kwargs,
        )

    async def aanswer_stream(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[RetrievalContexts, AsyncIterator[str]]:
        """Streaming answer: retrieve contexts, then stream LLM tokens."""
        self._ensure_initialized()
        backend = self._effective_backend
        if not backend:
            raise RuntimeError("Retrieval backend not initialized")
        self._truncate_history(kwargs)
        return await backend.aanswer_stream(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            **kwargs,
        )

    # === FILE MANAGEMENT API ===

    async def alist_ingested_files(self) -> list[dict[str, Any]]:
        """List all ingested files."""
        self._ensure_initialized()
        if self.unified is not None:
            return await self._hash_index.list_all()
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.alist_ingested_files()

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        delete_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Unified file deletion."""
        self._ensure_initialized()
        if self.unified is not None:
            return await unified_delete_files(
                engine=self.unified,
                hash_index=self._hash_index,
                lightrag=self._lightrag,
                file_paths=file_paths,
                filenames=filenames,
            )
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            delete_source=delete_source,
        )


__all__ = ["RAGService"]
