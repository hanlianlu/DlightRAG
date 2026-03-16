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
import shutil
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
from dlightrag.core.retrieval.path_resolver import PathResolver  # noqa: E402
from dlightrag.core.retrieval.protocols import RetrievalResult  # noqa: E402
from dlightrag.models.llm import (  # noqa: E402
    get_chat_model_func,
    get_chat_model_func_for_lightrag,
    get_embedding_func,
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
        self._table_schema_ttl: float = 0.0  # Cache expiry timestamp

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
        # Workspace metadata for compatibility filtering (federated queries)
        await conn.execute("""CREATE TABLE IF NOT EXISTS dlightrag_workspace_meta (
            workspace TEXT PRIMARY KEY,
            embedding_model TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )""")
        # pg_trgm for fuzzy metadata search (best-effort, not required)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        except Exception:
            logger.info("pg_trgm extension not available — fuzzy metadata search disabled")
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
        chat_func_lr = get_chat_model_func_for_lightrag(config)
        chat_func = get_chat_model_func(config)
        embedding_func = get_embedding_func(config)
        rerank_func = get_rerank_func(config)

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

        # ONE RAGAnything with unified chat model (LightRAG-adapted)
        logger.info("Creating RAGAnything instance...")
        self.rag = RAGAnything(
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

        emb_base_url = config.embedding.base_url or ""
        emb_api_key = config.embedding.api_key or ""

        if "voyage" in config.embedding.model.lower():
            embed_provider = VoyageProvider()
        else:
            embed_provider = OpenAICompatProvider()

        embedding_func = EmbeddingFunc(
            embedding_dim=config.embedding.dim,
            max_token_size=8192,
            func=partial(
                httpx_text_embed,
                model=config.embedding.model,
                base_url=emb_base_url,
                api_key=emb_api_key,
                provider=embed_provider,
            ),
            model_name=config.embedding.model,
        )

        # VisualEmbedder for image embedding (reuses persistent httpx client)
        visual_embedder = VisualEmbedder(
            model=config.embedding.model,
            base_url=emb_base_url,
            api_key=emb_api_key,
            dim=config.embedding.dim,
            batch_size=config.embedding_func_max_async,
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
            from dlightrag.core.retrieval.metadata_index import PGMetadataIndex

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

    async def areset(
        self,
        *,
        keep_files: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reset all storage for this workspace.

        5-phase reset:
        1. LightRAG stores (drop)
        2. DlightRAG stores (clear/drop)
        3. PG orphan table cleanup (PG backend only)
        4. Workspace metadata (PG backend only)
        5. Local files (skip if keep_files=True)
        """
        workspace = self.config.workspace
        errors: list[str] = []
        stats: dict[str, Any] = {
            "workspace": workspace,
            "storages_dropped": 0,
            "dlightrag_cleared": [],
            "orphan_tables_cleaned": 0,
            "local_files_removed": 0,
            "errors": errors,
        }

        # Phase 1: LightRAG stores
        lr = self.lightrag  # property handles both modes
        if lr is not None:
            for attr in _STORAGE_ATTRS:
                storage = getattr(lr, attr, None)
                if storage is None:
                    continue
                try:
                    if not dry_run:
                        await storage.drop()
                    stats["storages_dropped"] += 1
                except Exception as exc:
                    errors.append(f"Phase 1 ({attr}): {exc}")
                    logger.warning("areset Phase 1 failed for %s: %s", attr, exc)

        # Phase 2: DlightRAG stores
        for name, store, method in (
            ("hash_index", self._hash_index, "clear"),
            ("metadata_index", self._metadata_index, "clear"),
            ("visual_chunks", self._visual_chunks, "drop"),
        ):
            if store is None:
                continue
            try:
                if not dry_run:
                    await getattr(store, method)()
                stats["dlightrag_cleared"].append(name)
            except Exception as exc:
                errors.append(f"Phase 2 ({name}): {exc}")
                logger.warning("areset Phase 2 failed for %s: %s", name, exc)

        # Phase 3 & 4: PG-specific cleanup
        is_pg = self.config.kv_storage.startswith("PG")
        if is_pg:
            try:
                orphans = await self._clean_pg_orphan_tables(workspace, dry_run=dry_run)
                stats["orphan_tables_cleaned"] = orphans
            except Exception as exc:
                errors.append(f"Phase 3 (orphan tables): {exc}")
                logger.warning("areset Phase 3 failed: %s", exc)

            try:
                if not dry_run:
                    await self._clean_workspace_meta(workspace)
            except Exception as exc:
                errors.append(f"Phase 4 (workspace meta): {exc}")
                logger.warning("areset Phase 4 failed: %s", exc)

        # Phase 5: Local files
        if not keep_files:
            try:
                working_dir = Path(self.config.working_dir)
                stats["local_files_removed"] = self._reset_local_files(working_dir, dry_run=dry_run)
            except Exception as exc:
                errors.append(f"Phase 5 (local files): {exc}")
                logger.warning("areset Phase 5 failed: %s", exc)

        self._initialized = False
        logger.info("areset complete for workspace=%s: %s", workspace, stats)
        return stats

    @staticmethod
    async def _clean_pg_orphan_tables(workspace: str, *, dry_run: bool) -> int:
        """Scan lightrag_*/dlightrag_* PG tables, delete rows for this workspace."""
        try:
            from lightrag.kg.postgres_impl import ClientManager

            db = await ClientManager.get_client()
            pool = db.pool
            if pool is None:
                return 0
        except Exception:
            return 0

        try:
            async with pool.acquire() as conn:
                table_rows = await conn.fetch(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND (tablename LIKE 'lightrag_%' OR tablename LIKE 'dlightrag_%') "
                    "ORDER BY tablename"
                )
                if not table_rows:
                    return 0

                cleaned = 0
                for row in table_rows:
                    table = row["tablename"]
                    col = await conn.fetchrow(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = $1 AND column_name = 'workspace'",
                        table,
                    )
                    if col is None:
                        continue

                    count_row = await conn.fetchrow(
                        f'SELECT COUNT(*) as count FROM "{table}" WHERE workspace = $1',
                        workspace,
                    )
                    count = count_row["count"] if count_row else 0

                    if count > 0:
                        if not dry_run:
                            await conn.execute(
                                f'DELETE FROM "{table}" WHERE workspace = $1',
                                workspace,
                            )
                        cleaned += 1

                    if not dry_run:
                        remaining = await conn.fetchrow(
                            f'SELECT EXISTS (SELECT 1 FROM "{table}") AS has_rows'
                        )
                        if not remaining["has_rows"]:
                            await conn.execute(f'DROP TABLE "{table}"')

                return cleaned
        except Exception as exc:
            logger.warning("PG orphan table cleanup failed: %s", exc)
            return 0

    @staticmethod
    async def _clean_workspace_meta(workspace: str) -> None:
        """Delete workspace record from dlightrag_workspace_meta."""
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        pool = db.pool
        if pool is None:
            return
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'dlightrag_workspace_meta')"
            )
            if exists:
                await conn.execute(
                    "DELETE FROM dlightrag_workspace_meta WHERE workspace = $1",
                    workspace,
                )

    @staticmethod
    def _reset_local_files(working_dir: Path, *, dry_run: bool) -> int:
        """Remove working_dir/* except sources/. Returns file count."""
        if not working_dir.exists():
            return 0

        file_count = 0
        for item in sorted(working_dir.iterdir()):
            if item.name == "sources":
                continue
            if item.is_file():
                file_count += 1
                if not dry_run:
                    item.unlink()
            elif item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        file_count += 1
                if not dry_run:
                    shutil.rmtree(item, ignore_errors=True)
        return file_count

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
        """Record embedding model for this workspace (PG-only, best-effort)."""
        if not self.config.kv_storage.startswith("PG"):
            return
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                database=self.config.postgres_database,
            )
            try:
                await conn.execute(
                    """INSERT INTO dlightrag_workspace_meta (workspace, embedding_model)
                       VALUES ($1, $2)
                       ON CONFLICT (workspace)
                       DO UPDATE SET embedding_model = $2, updated_at = NOW()""",
                    self.config.workspace,
                    self.config.embedding.model,
                )
            finally:
                await conn.close()
        except Exception:
            logger.debug("workspace meta upsert failed", exc_info=True)

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
            path = Path(kwargs["path"])
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
        filters: Any = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve structured data without generating answer.

        Args:
            filters: Optional MetadataFilter for structured metadata queries.
                     Also auto-detected from query text via QueryAnalyzer.
        """
        self._ensure_initialized()
        backend = self._effective_backend
        if not backend:
            raise RuntimeError("Retrieval backend not initialized")

        if filters and self._metadata_index is None:
            logger.warning("Metadata filters ignored: metadata index requires PostgreSQL backend")

        # Phase 1: KG retrieval (existing path)
        kg_result = await backend.aretrieve(
            query,
            multimodal_content=multimodal_content,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            is_reretrieve=is_reretrieve,
            **kwargs,
        )

        # Phase 2: Multi-path retrieval (metadata + vector, supplementary)
        if self._metadata_index is not None or (
            self.config.rag_mode == "unified" and self._lightrag
        ):
            try:
                supplementary = await self._run_multi_path_retrieval(
                    query,
                    filters,
                    top_k=top_k or self.config.top_k,
                )
                if supplementary:
                    self._merge_supplementary_chunks(kg_result, supplementary)
            except Exception as exc:
                logger.warning("Multi-path retrieval failed (non-fatal): %s", exc)

        return kg_result

    async def _run_multi_path_retrieval(
        self,
        query: str,
        explicit_filters: Any,
        top_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Run QueryAnalyzer + Orchestrator for supplementary chunk retrieval."""
        from dlightrag.core.retrieval.orchestrator import RetrievalOrchestrator
        from dlightrag.core.retrieval.query_analyzer import QueryAnalyzer

        # Get LLM func for QueryAnalyzer's NL extraction fallback
        lr = self._lightrag or (getattr(self.rag, "lightrag", None) if self.rag else None)
        llm_func = getattr(lr, "llm_model_func", None) if lr else None

        # Refresh table schema (cached, 5-min TTL)
        import time

        now = time.monotonic()
        if self._metadata_index and now > self._table_schema_ttl:
            try:
                self._table_schema = await self._metadata_index.get_table_schema()
                self._table_schema_ttl = now + 300
            except Exception as exc:
                logger.debug("Table schema refresh failed: %s", exc)

        analyzer = QueryAnalyzer(llm_func=llm_func, schema=self._table_schema)
        plan = await analyzer.analyze(query, explicit_filters=explicit_filters)

        if not plan.metadata_filters and not plan.semantic_query:
            return []

        chunks_vdb = None
        if self._lightrag and self.config.rag_mode == "unified":
            chunks_vdb = getattr(self._lightrag, "chunks_vdb", None)

        orchestrator = RetrievalOrchestrator(
            metadata_index=self._metadata_index,
            lightrag=self._lightrag or getattr(self.rag, "lightrag", None),
            rag_mode=self.config.rag_mode,
            chunks_vdb=chunks_vdb,
        )

        chunk_ids = await orchestrator.orchestrate(plan, top_k=top_k)
        if not chunk_ids:
            return []

        return await self._resolve_chunk_contexts(chunk_ids)

    async def _resolve_chunk_contexts(
        self,
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Resolve chunk_ids to ChunkContext dicts via text_chunks KV store."""
        lr = self._lightrag or (getattr(self.rag, "lightrag", None) if self.rag else None)
        if not lr or not hasattr(lr, "text_chunks"):
            return []

        try:
            chunk_data_list = await lr.text_chunks.get_by_ids(chunk_ids)
        except Exception:
            return []

        contexts: list[dict[str, Any]] = []
        for cid, cdata in zip(chunk_ids, chunk_data_list, strict=True):
            if not cdata:
                continue
            ctx: dict[str, Any] = {
                "chunk_id": cid,
                "reference_id": cdata.get("full_doc_id", ""),
                "file_path": cdata.get("file_path", ""),
                "content": cdata.get("content", ""),
                "metadata": {"source": "multi_path_retrieval"},
            }
            if cdata.get("page_idx") is not None:
                ctx["page_idx"] = cdata["page_idx"] + 1  # 0-based to 1-based
            contexts.append(ctx)

        # Unified mode: enrich with visual data (image_data, page_index)
        if self.config.rag_mode == "unified":
            visual_store = getattr(self.unified, "visual_chunks", None) if self.unified else None
            if visual_store:
                try:
                    resolved_cids = [ctx["chunk_id"] for ctx in contexts]
                    visual_data = await visual_store.get_by_ids(resolved_cids)
                    for ctx, vd in zip(contexts, visual_data, strict=False):
                        if not vd or not isinstance(vd, dict):
                            continue
                        if vd.get("image_data"):
                            ctx["image_data"] = vd["image_data"]
                        if vd.get("page_index") is not None:
                            ctx["page_idx"] = vd["page_index"] + 1
                except Exception as exc:
                    logger.warning("Visual enrichment failed (non-fatal): %s", exc)

        return contexts

    @staticmethod
    def _merge_supplementary_chunks(
        result: RetrievalResult,
        supplementary: list[dict[str, Any]],
    ) -> None:
        """Merge supplementary chunks into the result, avoiding duplicates."""
        existing_ids = {c.get("chunk_id") for c in result.contexts.get("chunks", [])}
        for ctx in supplementary:
            if ctx.get("chunk_id") not in existing_ids:
                result.contexts.setdefault("chunks", []).append(ctx)
                existing_ids.add(ctx.get("chunk_id"))

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
