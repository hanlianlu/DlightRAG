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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol

from dlightrag.captionrag.chunking import docling_hybrid_chunking_func
from dlightrag.config import DlightragConfig, get_config

logger = logging.getLogger(__name__)

# Init lock key for PostgreSQL advisory lock (arbitrary 64-bit int)
_PG_INIT_LOCK_KEY = 0x436F727072616700  # "Dlightrag\0" as int

# LightRAG storage attribute names — authoritative list for reset/drop.
_STORAGE_ATTRS = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)


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
from dlightrag.core.retrieval.models import MetadataFilter  # noqa: E402
from dlightrag.core.retrieval.path_resolver import PathResolver  # noqa: E402
from dlightrag.core.retrieval.protocols import RetrievalResult  # noqa: E402
from dlightrag.models.llm import (  # noqa: E402
    get_chat_model_func,
    get_chat_model_func_for_lightrag,
    get_embedding_func,
    get_ingest_model_func,
    get_rerank_func,
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
        self._metadata_index: Any = None  # Document metadata index
        self._table_schema: dict[str, Any] | None = None  # Cached metadata table schema

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
        """Initialize with PostgreSQL advisory lock for multi-worker safety.

        The lock only coordinates ``_do_initialize()`` — LightRAG storage
        creation and domain store setup. All DDL is idempotent (CREATE TABLE
        IF NOT EXISTS), so stores self-initialize without needing the lock.
        PG extension management is delegated entirely to LightRAG.
        """
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

    async def _do_initialize(self) -> None:
        """Create RAG backend and compose pipelines based on rag_mode."""
        from dlightrag.core._lightrag_patches import apply as apply_lightrag_patches

        apply_lightrag_patches()

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
        rag_config = RAGAnythingConfig(  # type: ignore[possibly-undefined]
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
        chat_func_lr = get_chat_model_func_for_lightrag(config)
        chat_func = get_chat_model_func(config)
        embedding_func = get_embedding_func(config)
        rerank_func = get_rerank_func(config)

        # LightRAG needs (query, documents: list[str]) — wrap multimodal reranker
        lightrag_rerank_func = None
        if rerank_func is not None:
            from dlightrag.models.rerank import build_lightrag_rerank_adapter

            lightrag_rerank_func = build_lightrag_rerank_adapter(rerank_func)

        # Create VLM OCR parser if configured
        vlm_parser = None
        if config.parser == "vlm":
            from dlightrag.captionrag.vlm_parser import VlmOcrParser

            vlm_parser = VlmOcrParser(
                vision_model_func=chat_func,
                dpi=config.page_render_dpi,
            )

        # LightRAG configuration
        lightrag_kwargs: dict[str, Any] = {
            "workspace": config.workspace,
            "default_llm_timeout": int(config.chat.timeout),
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
            "rerank_model_func": lightrag_rerank_func,
            "vector_db_storage_cls_kwargs": self._build_vector_db_kwargs(config),
            "addon_params": {
                "entity_types": config.kg_entity_types,
                "language": "English",
            },
        }

        # Only use our custom HybridChunker for parsers that benefit from it
        if config.parser in ("docling", "vlm"):
            lightrag_kwargs["chunking_func"] = docling_hybrid_chunking_func

        # ONE RAGAnything with unified chat model (LightRAG-adapted)
        logger.info("Creating RAGAnything instance...")
        self.rag = RAGAnything(  # type: ignore[possibly-undefined]
            None,
            chat_func_lr,
            chat_func_lr,
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

        # Wrap chunks_vdb for metadata in-filtering (caption mode)
        lr = getattr(self.rag, "lightrag", None)
        if lr is not None and getattr(lr, "chunks_vdb", None) is not None:
            from dlightrag.core.retrieval.filtered_vdb import FilteredVectorStorage

            lr.chunks_vdb = FilteredVectorStorage(
                original=lr.chunks_vdb,
                embedding_func=embedding_func,
            )

        # Auto-detect hash index backend based on storage config
        hash_index = await self._create_hash_index(config)

        # Initialize metadata index (best-effort — PG only for now)
        self._metadata_index = await self._create_metadata_index(config)

        # Compose pipelines
        self.ingestion = IngestionPipeline(
            self.rag,
            config=config,
            max_concurrent=config.max_concurrent_ingestion,
            mineru_backend=mineru_backend,
            cancel_checker=self._cancel_checker,
            hash_index=hash_index,
            vlm_parser=vlm_parser,
            metadata_index=self._metadata_index,
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
        chat_func_lr = get_chat_model_func_for_lightrag(config)
        chat_func = get_chat_model_func(config)
        ingest_func = get_ingest_model_func(config)
        rerank_func = get_rerank_func(config)

        # Use httpx_text_embed instead of LightRAG's openai_embed for text
        # embedding.  openai_embed uses encoding_format:"base64" and the openai
        # Python client — both fail with Xinference VL models.
        # partial() with plain strings is deepcopy-safe (LightRAG's __post_init__
        # does asdict→deepcopy on embedding_func).
        from functools import partial

        from dlightrag.models.embedding import httpx_embed
        from dlightrag.models.providers.embed_providers import (
            OpenAICompatEmbedProvider,
            VoyageEmbedProvider,
        )
        from dlightrag.unifiedrepresent.embedder import VisualEmbedder

        emb_base_url = config.embedding.base_url or ""
        emb_api_key = config.embedding.api_key or ""

        if "voyage" in config.embedding.model.lower():
            embed_provider = VoyageEmbedProvider()
        else:
            embed_provider = OpenAICompatEmbedProvider()

        _httpx_embed = partial(
            httpx_embed,
            model=config.embedding.model,
            base_url=emb_base_url,
            api_key=emb_api_key,
            provider=embed_provider,
            timeout=float(config.embedding_request_timeout),
        )

        async def _embed_for_lightrag(texts: list[str]) -> Any:
            result = await _httpx_embed(texts)
            # LightRAG's EmbeddingFunc validates via result.size — requires numpy
            import numpy as np

            return np.array(result)

        embedding_func = EmbeddingFunc(
            embedding_dim=config.embedding.dim,
            max_token_size=8192,
            func=_embed_for_lightrag,
            model_name=config.embedding.model,
        )

        # VisualEmbedder for image embedding (reuses persistent httpx client)
        visual_embedder = VisualEmbedder(
            model=config.embedding.model,
            base_url=emb_base_url,
            api_key=emb_api_key,
            dim=config.embedding.dim,
            batch_size=config.embedding_func_max_async,
            timeout=float(config.embedding_request_timeout),
            provider=embed_provider,
        )

        # LightRAG configuration (same storage backends as caption mode)
        # Do NOT pass rerank_model_func — we handle reranking ourselves
        lightrag = LightRAG(
            working_dir=str(config.working_dir_path),
            llm_model_func=chat_func_lr,
            embedding_func=embedding_func,
            workspace=config.workspace,
            default_llm_timeout=int(config.chat.timeout),
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
        logger.info("LightRAG storages initialized")

        # Wrap chunks_vdb for metadata in-filtering
        if lightrag.chunks_vdb is not None:
            from dlightrag.core.retrieval.filtered_vdb import FilteredVectorStorage

            lightrag.chunks_vdb = FilteredVectorStorage(  # type: ignore[assignment]
                original=lightrag.chunks_vdb,
                embedding_func=embedding_func,
            )

        # Post-init verification: ensure AGE graph labels actually exist.
        # Catches stale/corrupted graphs left by previous failed inits
        # (e.g., schema created but vlabels missing due to uncaught errors).
        await self._verify_graph_labels(lightrag)

        # Create visual_chunks KV store.
        # LightRAG's PGKVStorage only supports 7 hardcoded namespaces; custom ones
        # (like "visual_chunks") cause KeyError / silent no-op / SQL errors.
        # See pg_jsonb_kv.py module docstring for details.
        from dlightrag.storage.pg_jsonb_kv import PGJsonbKVStorage

        kv_cls = lightrag.key_string_value_json_storage_cls
        if config.kv_storage.startswith("PG"):
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
        logger.info("Visual chunks KV store initialized (%s)", kv_cls.__name__)

        # Create PathResolver for unified mode
        self._path_resolver = PathResolver(
            working_dir=str(config.working_dir_path),
        )

        # Create engine (pass pre-built embedder to avoid creating a duplicate)
        self.unified = UnifiedRepresentEngine(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=chat_func,
            visual_embedder=visual_embedder,
            path_resolver=self._path_resolver,
            context_model_func=ingest_func,
            rerank_func=rerank_func,
        )
        self._backend = self.unified

        # Create hash index for deduplication (same backend as caption mode)
        self._hash_index = await self._create_hash_index(config)

        # Initialize metadata index
        self._metadata_index = await self._create_metadata_index(config)

        logger.info("Unified representational RAG mode ready")

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

    async def _create_metadata_index(self, config: DlightragConfig) -> Any:
        """Create PGMetadataIndex if using PostgreSQL backend, else None.

        Metadata index is best-effort: non-PG backends silently skip it.
        """
        if not config.kv_storage.startswith("PG"):
            logger.info("Metadata index: disabled (non-PG backend)")
            return None

        try:
            from dlightrag.storage.metadata_index import PGMetadataIndex

            idx = PGMetadataIndex(workspace=config.workspace)
            await idx.initialize()
            logger.info("Metadata index: PGMetadataIndex (PostgreSQL)")
            return idx
        except ImportError:
            logger.warning("asyncpg not available — metadata index disabled")
            return None
        except Exception as e:
            logger.warning("PGMetadataIndex creation failed — metadata index disabled: %s", e)
            return None

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RAGService not initialized. Use 'await RAGService.create()' instead."
            )

    # -- Graph verification ----------------------------------------------------

    @staticmethod
    async def _verify_graph_labels(lightrag: Any) -> None:
        """Verify AGE graph labels exist after initialize_storages().

        If the graph schema exists but required labels (``base``, ``DIRECTED``)
        are missing — typically from a previous failed init before the
        DuplicateSchemaError patch — drop the corrupted graph and re-run
        initialization to rebuild it cleanly.
        """
        graph_storage = getattr(lightrag, "chunk_entity_relation_graph", None)
        if graph_storage is None:
            return

        graph_name = getattr(graph_storage, "graph_name", None)
        db = getattr(graph_storage, "db", None)
        if not graph_name or db is None:
            return

        pool = getattr(db, "pool", None)
        if pool is None:
            return

        try:
            async with pool.acquire() as conn:
                # Check if the graph schema exists at all
                schema_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = $1)",
                    graph_name,
                )
                if not schema_exists:
                    return  # No schema → init will create everything fresh

                # Check if the 'base' label table exists within the schema
                base_exists = await conn.fetchval(
                    "SELECT EXISTS("
                    "  SELECT 1 FROM pg_tables"
                    "  WHERE schemaname = $1 AND tablename = 'base'"
                    ")",
                    graph_name,
                )
                if base_exists:
                    return  # Healthy graph

                # Corrupted: schema exists but 'base' label missing.
                # Drop the graph and let initialize() rebuild it.
                logger.warning(
                    "Corrupted AGE graph '%s': schema exists but 'base' label missing. "
                    "Dropping and rebuilding.",
                    graph_name,
                )
                import re

                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", graph_name)
                await conn.execute('SET search_path = ag_catalog, "$user", public')
                await conn.execute(f"SELECT drop_graph('{safe_name}', true)")

        except Exception:
            logger.warning(
                "Graph verification failed for '%s', proceeding anyway",
                graph_name,
                exc_info=True,
            )
            return

        # Re-run graph initialization after dropping corrupted graph
        logger.info("Re-initializing graph storage for '%s'", graph_name)
        await graph_storage.initialize()

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

        from dlightrag.storage.pool import pg_pool

        await pg_pool.close()

    async def areset(
        self,
        *,
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Completely remove this workspace -- all data, graph schemas, and files.

        Delegates to the dedicated 6-phase reset module (``dlightrag.core.reset``).
        """
        from dlightrag.core.reset import areset

        return await areset(self, keep_files=keep_files, dry_run=dry_run)

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

    async def _upsert_workspace_meta(self) -> None:
        """Record embedding model for this workspace (PG-only, best-effort).

        Self-initializing: creates the table idempotently if it doesn't exist.
        No advisory lock needed — all DDL is IF NOT EXISTS.
        """
        if not self.config.kv_storage.startswith("PG"):
            return
        try:
            from dlightrag.storage.pool import pg_pool

            pool = await pg_pool.get()
            async with pool.acquire() as conn:
                # Idempotent DDL — safe to call from multiple workers
                await conn.execute(
                    """CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
                        workspace TEXT PRIMARY KEY,
                        embedding_model TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )"""
                )
                await conn.execute(
                    """INSERT INTO dlightrag_workspace_meta (workspace, embedding_model)
                       VALUES ($1, $2)
                       ON CONFLICT (workspace)
                       DO UPDATE SET embedding_model = $2, updated_at = NOW()""",
                    self.config.workspace,
                    self.config.embedding.model,
                )
        except Exception:
            logger.debug("workspace meta upsert failed", exc_info=True)

    # === INGESTION API ===

    async def aingest(
        self,
        source_type: Literal["local", "azure_blob", "s3"],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Unified ingestion API.

        Args:
            source_type: "local", "azure_blob", or "s3"
            kwargs:
                local: path, replace
                azure_blob: source, container_name, blob_path, prefix, replace
                s3: bucket, key, prefix, replace
        """
        self._ensure_initialized()
        await self._upsert_workspace_meta()

        # Unified mode
        if self.unified is not None:
            return await unified_ingest(
                engine=self.unified,
                config=self.config,
                hash_index=self._hash_index,
                source_type=source_type,
                metadata_index=self._metadata_index,
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
            path_str = kwargs.get("path")
            if not path_str:
                raise ValueError("'path' is required for local source_type")
            path = Path(path_str)
            replace = get_replace_value()
            user_metadata = kwargs.get("metadata")

            logger.info(f"Ingesting from local path: {path}")

            result = await ingestion.aingest_from_local(
                path=path,
                replace=replace,
                user_metadata=user_metadata,
            )
            return result.model_dump(exclude_none=True)

        if source_type == "azure_blob":
            container_name = kwargs.get("container_name")
            if not container_name:
                raise ValueError("'container_name' is required for azure_blob source_type")
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

        # source_type == "s3"
        bucket = kwargs.get("bucket")
        key = kwargs.get("key")
        if not bucket or not key:
            raise ValueError("S3 ingestion requires 'bucket' and 'key'")
        from dlightrag.sourcing.aws_s3 import S3DataSource

        source = S3DataSource(bucket=bucket)
        try:
            content = await source.aload_document(key)
            # Write to temp file and ingest
            tmp_dir = self.config.temp_dir / "s3"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            local_path = tmp_dir / Path(key).name
            local_path.write_bytes(content)

            replace = kwargs.get("replace", self.config.ingestion_replace_default)
            result = await ingestion.aingest_from_local(path=local_path, replace=replace)
            return result.model_dump(exclude_none=True)
        finally:
            await source.aclose()

    # === RETRIEVAL API ===

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] | None = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        filters: MetadataFilter | None = None,
        *,
        _plan: Any = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer.

        Architecture: Plan-First In-Filtering (matching ArtRAG pattern).

        1. Resolve metadata filters → candidate chunk_ids (upfront)
        2. Set contextvar via metadata_filter_scope
        3. KG retrieval runs with FilteredVectorStorage (WHERE id = ANY)
        4. Rerank operates only on filtered results
        5. Enrich with document metadata

        Args:
            filters: Optional MetadataFilter for structured metadata queries.
            _plan: Pre-computed QueryPlan from QueryPlanner (via ServiceManager).
        """
        self._ensure_initialized()
        backend = self._effective_backend
        if not backend:
            raise RuntimeError("Retrieval backend not initialized")

        if filters and self._metadata_index is None:
            logger.warning("Metadata filters ignored: metadata index requires PostgreSQL backend")

        # --- Step 1: Resolve metadata filter → candidate chunk_ids ---
        candidate_ids: set[str] | None = None
        effective_filters = filters
        if effective_filters is None and _plan is not None:
            effective_filters = getattr(_plan, "metadata_filter", None)

        if effective_filters and self._metadata_index is not None:
            try:
                from dlightrag.core.retrieval.metadata_path import metadata_retrieve

                lr = self._lightrag or getattr(self.rag, "lightrag", None)
                chunk_ids = await metadata_retrieve(
                    self._metadata_index,
                    effective_filters,
                    lr,
                    self.config.rag_mode,
                )
                if chunk_ids:
                    candidate_ids = set(chunk_ids)
                    logger.info("In-filter: %d candidate chunks from metadata", len(candidate_ids))
                else:
                    logger.info("Metadata filter matched 0 candidates, proceeding unfiltered")
            except Exception as exc:
                logger.warning("Metadata candidate resolution failed (non-fatal): %s", exc)

        # --- Step 2: KG retrieval with in-filtering scope ---
        from dlightrag.core.retrieval.filtered_vdb import metadata_filter_scope

        async with metadata_filter_scope(candidate_ids):
            kg_result = await backend.aretrieve(
                query,
                multimodal_content=multimodal_content,
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                is_reretrieve=is_reretrieve,
                **kwargs,
            )

        # --- Step 2b: Force-inject metadata-resolved chunks if missing ---
        # Unified mode handles this inside VisualRetriever._retrieve (pre-rerank).
        # Caption mode needs it here (post-rerank, since LightRAG controls reranking).
        if candidate_ids:
            returned_cids = {c.get("chunk_id") for c in kg_result.contexts.get("chunks", [])}
            missing = candidate_ids - returned_cids
            if missing:
                await self._inject_candidate_chunks(kg_result, sorted(missing))

        # --- Step 3: Enrich chunks with document metadata ---
        if self._metadata_index is not None:
            await self._enrich_chunks_with_metadata(kg_result)

        return kg_result

    async def _inject_candidate_chunks(self, result: RetrievalResult, chunk_ids: list[str]) -> None:
        """Force-inject metadata-resolved chunks missing from retrieval results.

        Fetches content from text_chunks and (in unified mode) image_data
        from visual_chunks. Appends to the chunks list so the LLM sees the
        user-requested documents.
        """
        lr = self.lightrag
        if lr is None:
            return

        raw_contents = await lr.text_chunks.get_by_ids(chunk_ids)

        # Unified mode: also fetch visual data
        visual_map: dict[str, dict] = {}
        if self._visual_chunks is not None:
            raw_visuals = await self._visual_chunks.get_by_ids(chunk_ids)
            for cid, vd in zip(chunk_ids, raw_visuals, strict=False):
                if isinstance(vd, dict):
                    visual_map[cid] = vd

        injected: list[dict] = []
        for cid, content_raw in zip(chunk_ids, raw_contents, strict=False):
            if content_raw is None:
                continue
            content = (
                content_raw if isinstance(content_raw, str) else content_raw.get("content", "")
            )
            vd = visual_map.get(cid, {})
            injected.append(
                {
                    "chunk_id": cid,
                    "content": content,
                    "reference_id": "",
                    "file_path": vd.get("file_path", ""),
                    "image_data": vd.get("image_data"),
                    "page_idx": (vd.get("page_index", 0) or 0) + 1,
                }
            )

        if injected:
            chunks = result.contexts.get("chunks", [])
            result.contexts["chunks"] = chunks + injected
            logger.info("Force-injected %d metadata-resolved chunks", len(injected))

    async def _enrich_chunks_with_metadata(self, result: RetrievalResult) -> None:
        """Inject document metadata into chunk contexts for LLM consumption.

        Looks up each chunk's file_path in the metadata index and merges
        any non-empty fields into the chunk's metadata dict. Fields are
        dynamic: whatever the metadata index returns is included, minus
        internal/system fields that add no value to the LLM context.
        """
        _SKIP = frozenset(
            {
                "workspace",
                "doc_id",
                "file_path",
                "file_extension",
                "filename",
                "filename_stem",
                "ingested_at",
                "custom_metadata",
                "rag_mode",
                "page_count",
                "original_format",
                # doc_title/doc_author already injected into reranker content
                # from visual_chunks — skip to avoid duplication in answer header.
                "doc_title",
                "doc_author",
            }
        )

        chunks = result.contexts.get("chunks", [])
        if not chunks:
            return

        # Collect unique file_paths for batch lookup
        path_meta: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            if fp and fp not in path_meta:
                path_meta[fp] = {}

        # Lookup metadata by filename (best-effort)
        from pathlib import Path as _Path

        for fp in path_meta:
            try:
                filename = _Path(fp).name
                doc_ids = await self._metadata_index.find_by_filename(filename)
                if doc_ids:
                    meta = await self._metadata_index.get(doc_ids[0])
                    if meta:
                        path_meta[fp] = {
                            k: v
                            for k, v in meta.items()
                            if k not in _SKIP and v is not None and v != ""
                        }
            except Exception:
                pass  # enrichment is best-effort

        # Inject into each chunk's metadata field
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            doc_meta = path_meta.get(fp)
            if doc_meta:
                existing = chunk.get("metadata") or {}
                existing.update(doc_meta)
                chunk["metadata"] = existing

    # === METADATA API ===

    async def aget_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get document metadata by ID."""
        if self._metadata_index is None:
            raise ValueError("Metadata index not available for this workspace")
        result = await self._metadata_index.get(doc_id)
        return result if result else {}

    async def aupdate_metadata(self, doc_id: str, data: dict[str, Any]) -> None:
        """Update (merge) document metadata."""
        if self._metadata_index is None:
            raise ValueError("Metadata index not available for this workspace")
        await self._metadata_index.upsert(doc_id, data)

    async def asearch_metadata(self, filters: dict[str, Any]) -> list[str]:
        """Search metadata by filters, return matching doc_ids."""
        if self._metadata_index is None:
            raise ValueError("Metadata index not available for this workspace")
        return await self._metadata_index.query(filters)

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
                metadata_index=self._metadata_index,
            )
        if not self.ingestion:
            raise RuntimeError("Ingestion pipeline not initialized")
        return await self.ingestion.adelete_files(
            file_paths=file_paths,
            filenames=filenames,
            delete_source=delete_source,
        )


__all__ = ["RAGService"]
