# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Per-workspace LightRAG service facade for ingestion and retrieval.

DlightRAG owns one LightRAG instance per workspace and adds PostgreSQL
metadata, direct visual embedding, and retrieval orchestration around it.
PostgreSQL advisory locks coordinate first-time storage initialization across
concurrent workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from lightrag.constants import PARSED_DIR_NAME

from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.client_contracts import SourceType
from dlightrag.core.ingestion.engine import PreparedIngestFile
from dlightrag.core.ingestion.paths import (
    iter_ingestable_files,
    remote_ingest_batch_root,
    remote_parser_input_path,
    retained_remote_source_path,
    stage_input_file,
    staged_input_path,
    workspace_input_root,
)
from dlightrag.core.retrieval.metadata_fields import MetadataIngestPolicy
from dlightrag.sourcing.base import AsyncDataSource
from dlightrag.storage.protocols import MetadataIndexProtocol
from dlightrag.utils import normalize_workspace

logger = logging.getLogger(__name__)

# PostgreSQL advisory lock serializing first-time storage init across
# concurrent workers (multi-Gunicorn processes, Docker scale-out, launchd
# respawn). Wraps the multi-step DDL sequence — CREATE EXTENSION pgvector
# / age, CREATE TABLE, CREATE INDEX, AGE create_graph(), seed rows — as
# a critical section. Per-statement IF NOT EXISTS isn't enough: the PG
# catalog has known races on concurrent CREATE TABLE, AGE create_graph()
# is non-idempotent, and the sequence itself can't run in a single
# transaction (CREATE EXTENSION / CREATE INDEX both bar that). First
# acquirer runs _do_initialize(); other workers poll with backoff up to
# 180s, then proceed against ready storage. Same pattern Flyway / Django
# / Rails / golang-migrate all use; PG ships pg_try_advisory_lock for
# exactly this purpose, so we are not working around a missing feature.
#
# Key layout matches ArtRAG: 6-byte project tag + 0x00 + 1-byte
# namespace. 0x01 = init; 0x02..0xFF reserved for future DlightRAG
# locks. Advisory locks are session-scoped and not persisted, so the
# specific number is irrelevant beyond uniqueness within DlightRAG.
_PG_INIT_LOCK_KEY = 0x446C_6967_6874_0001  # "Dlight\x00\x01"

# LightRAG storage attribute names — authoritative list referenced by reset
# tests to assert that areset() iterates over the full set. The reset module
# itself uses dynamic ``vars()`` discovery; this constant exists as the test
# contract / documentation of what should be covered.
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

_REMOTE_INGEST_BATCH_SIZE = 64
_REMOTE_DOWNLOAD_CONCURRENCY = 8
_PROCESS_COUNT_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")

RemoteIngestProgressCallback = Callable[["RemoteIngestWindowProgress"], Awaitable[None]]


@dataclass(frozen=True)
class RemoteIngestWindowProgress:
    """Progress emitted after one remote object ingest window finishes."""

    source_type: str
    batch_index: int
    total_delta: int
    processed_delta: int
    failed_delta: int
    errors: tuple[str, ...] = ()


def _configured_process_count() -> int:
    """Best-effort process count from common server env vars."""
    for name in _PROCESS_COUNT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count > 0:
            return count
    return 1


from dlightrag.core.contract_guard import LightRAGContractGuard  # noqa: E402
from dlightrag.core.retrieval.models import MetadataFilter  # noqa: E402
from dlightrag.core.retrieval.protocols import RetrievalResult  # noqa: E402
from dlightrag.models.llm import (  # noqa: E402
    build_role_llm_configs,
    get_default_model_func_for_lightrag,
    get_embedding_func,
    get_multimodal_embedder,
    get_rerank_func,
)
from dlightrag.storage.postgres_version import (  # noqa: E402
    ensure_pgvector_halfvec,
    ensure_postgres_extensions,
    ensure_postgres_major,
)


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

        # Direct LightRAG runtime and DlightRAG orchestration.
        self._lightrag: Any = None  # Direct LightRAG reference
        self._metadata_index: MetadataIndexProtocol | None = None
        self._table_schema: dict[str, Any] | None = None  # Cached metadata table schema
        self._metadata_registry: Any = None
        self._allow_ad_hoc_metadata = self.config.metadata.allow_ad_hoc_json
        self._default_metadata_policy: MetadataIngestPolicy = (
            self.config.metadata.default_ingest_policy
        )
        self._lightrag_stores: Any = None
        self._ingestion_engine: Any = None
        self._bm25: Any = None
        self._retrieval_orchestrator: Any = None
        self._multimodal_embedder: Any = None
        self._direct_image_embedding_enabled = False
        self._visual_asset_resolver: Any = None

        # Retrieval backend (satisfies RetrievalBackend Protocol).
        # Explicitly wired by the unified LightRAG initialization path.
        self._backend: Any = None

    @property
    def lightrag(self) -> Any:
        """Return the underlying LightRAG instance for the unified runtime."""
        return self._lightrag

    @staticmethod
    def _build_vector_db_kwargs(config: DlightragConfig) -> dict[str, Any]:
        """Build vector_db_storage_cls_kwargs from config.vector_db_kwargs passthrough."""
        kwargs: dict[str, Any] = {"cosine_better_than_threshold": 0.3}
        kwargs.update(config.vector_db_kwargs)
        return kwargs

    @staticmethod
    async def _resolve_direct_image_embedding_enabled(
        embedder: Any,
        *,
        startup_probe: bool,
    ) -> bool:
        """Return whether DlightRAG direct image vectors are safe to use.

        The probe calls only the embedding provider with an in-memory 1x1
        image; it does not touch LightRAG storage, PostgreSQL, or local files.
        """
        if not getattr(embedder, "supports_images", False):
            logger.info(
                "Direct image embedding disabled: %s does not support image inputs",
                embedder.__class__.__name__,
            )
            return False
        if not startup_probe:
            return True
        try:
            await embedder.probe_image_embedding()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Direct image embedding probe failed; using LightRAG semantic visual path only",
                exc_info=True,
            )
            return False
        logger.info("Direct image embedding probe passed — visual chunk vectors enabled")
        return True

    @staticmethod
    def _build_addon_params(config: DlightragConfig) -> dict[str, Any]:
        """Build LightRAG 1.5+ addon_params from the active config."""
        params: dict[str, Any] = {
            "language": config.extraction.language,
            "chunker": {
                "paragraph_semantic": {
                    "chunk_token_size": config.chunk_p_token_size,
                }
            },
        }
        if config.extraction.entity_type_prompt_file:
            params["entity_type_prompt_file"] = config.extraction.entity_type_prompt_file
        if config.kg_entity_types:
            params["entity_types_guidance"] = (
                "Prioritize domain entities in these categories: "
                f"{', '.join(config.kg_entity_types)}."
            )
        return params

    @staticmethod
    def _build_retrieval_backend(
        config: DlightragConfig,
        *,
        lightrag: Any,
        embedder: Any | None = None,
        rerank_func: Any | None = None,
    ) -> Any:
        """Build the LightRAG retrieval backend from typed DlightRAG config."""
        from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend

        return LightRAGMixBackend(
            lightrag=lightrag,
            embedder=embedder,
            rerank_func=rerank_func,
            direct_visual_top_k=config.direct_visual_top_k,
            max_entity_tokens=config.max_entity_tokens,
            max_relation_tokens=config.max_relation_tokens,
            max_total_tokens=config.max_total_tokens,
        )

    @staticmethod
    def _required_postgres_extensions(config: DlightragConfig) -> tuple[str, ...]:
        """Return DlightRAG-owned PostgreSQL extensions required by runtime config."""
        if not config.bm25_enabled:
            return ()

        extensions: list[str] = ["pg_textsearch"]
        if any(profile.text_config == "public.jiebacfg" for profile in config.bm25_profiles):
            extensions.append("pg_jieba")
        return tuple(extensions)

    async def initialize(self) -> None:
        """Initialize LightRAG storages and caches (idempotent).

        Uses PostgreSQL advisory lock for distributed coordination. DlightRAG's
        core runtime is PostgreSQL-only, so unavailable PG is a startup error.
        """
        if self._initialized:
            return

        await self._initialize_with_pg_lock()

        self._initialized = True
        logger.debug("RAGService initialized")

    async def _initialize_with_pg_lock(self) -> None:
        """Initialize with PostgreSQL advisory lock for multi-worker safety.

        The lock only coordinates ``_do_initialize()`` — LightRAG storage
        creation and domain store setup. All DDL is idempotent (CREATE TABLE
        IF NOT EXISTS), so stores self-initialize without needing the lock.
        LightRAG owns its storage bootstrap; DlightRAG only bootstraps extensions
        required by DlightRAG-owned stores such as pg_textsearch BM25.
        """
        try:
            import asyncpg
        except ImportError:
            raise RuntimeError("asyncpg is required for DlightRAG PostgreSQL storage") from None

        try:
            conn = await asyncpg.connect(**self.config.pg_connection_kwargs())
        except Exception as e:
            raise RuntimeError(
                "PostgreSQL is required for DlightRAG startup and could not be reached"
            ) from e

        try:
            await ensure_postgres_major(
                conn,
                required_major=self.config.postgres_required_major,
            )
            if self.config.pg_vector_index_type == "HNSW_HALFVEC":
                await ensure_pgvector_halfvec(conn)
            await self._check_postgres_concurrency_sanity(conn)

            acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY)

            if acquired:
                logger.info("Acquired PG advisory lock, initializing RAG pipelines...")
                try:
                    await ensure_postgres_extensions(
                        conn,
                        self._required_postgres_extensions(self.config),
                    )
                    await self._do_initialize()
                    logger.info("RAG pipelines initialized successfully")
                finally:
                    await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
            else:
                logger.info("Another worker is initializing, waiting for lock...")
                waited = 0.0
                backoff = 0.1
                while waited < 180:
                    # Jitter avoids thundering-herd when many workers
                    # retry on the same beat.
                    jitter = random.uniform(0, backoff * 0.5)
                    await asyncio.sleep(backoff + jitter)
                    waited += backoff + jitter
                    if await conn.fetchval("SELECT pg_try_advisory_lock($1)", _PG_INIT_LOCK_KEY):
                        await conn.execute("SELECT pg_advisory_unlock($1)", _PG_INIT_LOCK_KEY)
                        break
                    backoff = min(backoff * 1.5, 5.0)
                else:
                    logger.warning("Lock acquisition timeout after 180s")
                logger.info("Lock released, connecting to existing storages...")
                await self._do_initialize()
        finally:
            await conn.close()

    async def _check_postgres_concurrency_sanity(self, conn: Any) -> None:
        """Warn when configured pools can exhaust server connections."""
        try:
            max_connections = int(await conn.fetchval("SHOW max_connections"))
        except Exception:
            logger.debug("Could not read PostgreSQL max_connections", exc_info=True)
            return

        config = self.config
        per_process = config.postgres_lightrag_pool_max_size + config.postgres_pool_max_size
        process_count = _configured_process_count()
        estimated = per_process * process_count
        reserved = max(5, max_connections // 10)
        usable = max_connections - reserved

        logger.info(
            "PostgreSQL connection sanity: max_connections=%d usable_after_headroom=%d "
            "configured_pool_connections_per_process=%d "
            "(lightrag=%d, dlightrag=%d) process_count=%d estimated_pool_connections=%d",
            max_connections,
            usable,
            per_process,
            config.postgres_lightrag_pool_max_size,
            config.postgres_pool_max_size,
            process_count,
            estimated,
        )
        if estimated > usable:
            logger.warning(
                "PostgreSQL connection budget is tight: estimated_pool_connections=%d "
                "exceeds usable_after_headroom=%d (max_connections=%d, headroom=%d). "
                "Lower postgres_lightrag_pool_max_size/postgres_pool_max_size/process count "
                "or raise max_connections.",
                estimated,
                usable,
                max_connections,
                reserved,
            )

    async def _do_initialize(self) -> None:
        """Create one LightRAG-backed unified pipeline."""
        from dlightrag.core._lightrag_patches import apply as apply_lightrag_patches

        self.config.apply_lightrag_backend_env(force=True)
        self.config.apply_lightrag_runtime_env(force=True)
        apply_lightrag_patches()
        await self._do_initialize_unified()

    async def _do_initialize_unified(self) -> None:
        """Initialize the direct LightRAG multimodal runtime.

        Creates LightRAG directly and wires DlightRAG's retrieval/metadata
        orchestration around LightRAG's PostgreSQL stores.
        """
        from lightrag import LightRAG

        config = self.config
        logger.info("Initializing unified representational RAG mode...")

        # Get model functions
        default_func_lr = get_default_model_func_for_lightrag(config)
        rerank_func = get_rerank_func(config)
        role_overrides = build_role_llm_configs(config)
        if role_overrides is not None:
            logger.info("LightRAG role overrides: %s", sorted(role_overrides.keys()))

        multimodal_embedder = get_multimodal_embedder(config)
        self._multimodal_embedder = multimodal_embedder
        embedding_func = get_embedding_func(config, embedder=multimodal_embedder)
        self._direct_image_embedding_enabled = await self._resolve_direct_image_embedding_enabled(
            multimodal_embedder,
            startup_probe=config.embedding.startup_probe,
        )

        # Vision probe lives in RAGServiceManager._probe_vision_support()
        # — it runs once at server startup, not per workspace.

        # LightRAG configuration.
        # Do NOT pass rerank_model_func — we handle reranking ourselves
        lightrag = LightRAG(
            working_dir=str(config.working_dir_path),
            llm_model_func=default_func_lr,
            embedding_func=embedding_func,
            workspace=config.workspace,
            default_llm_timeout=int(config.llm.default.timeout),
            default_embedding_timeout=config.embedding_request_timeout,
            **config.lightrag_pipeline_kwargs(),
            llm_model_max_async=config.max_async,
            embedding_func_max_async=config.embedding_func_max_async,
            embedding_batch_num=config.embedding_batch_num,
            vector_storage=config.vector_storage,
            graph_storage=config.graph_storage,
            kv_storage=config.kv_storage,
            doc_status_storage=config.doc_status_storage,
            vector_db_storage_cls_kwargs=self._build_vector_db_kwargs(config),
            role_llm_configs=role_overrides,
            kg_chunk_pick_method=config.kg_chunk_pick_method,
            entity_extraction_use_json=config.extraction.use_json,
            addon_params=self._build_addon_params(config),
        )
        await lightrag.initialize_storages()
        self._lightrag = lightrag
        logger.info("LightRAG storages initialized")

        from dlightrag.core.lightrag_stores import LightRAGStores

        self._lightrag_stores = LightRAGStores(lightrag)

        from dlightrag.core.visual_assets import ThumbnailCache, VisualAssetResolver

        self._visual_asset_resolver = VisualAssetResolver(
            lightrag=lightrag,
            thumb_cache=ThumbnailCache(max_size=config.visual_assets.thumb_cache_size),
        )

        # Wrap chunks_vdb for metadata in-filtering
        if lightrag.chunks_vdb is not None:
            await LightRAGContractGuard(lightrag).verify_all()

            from dlightrag.core.retrieval.filtered_vdb import FilteredVectorStorage

            lightrag.chunks_vdb = FilteredVectorStorage(  # type: ignore[assignment]
                original=lightrag.chunks_vdb,
                embedding_func=embedding_func,
                exact_threshold=config.metadata_filter_exact_vector_threshold,
            )

        # Post-init verification: ensure AGE graph labels actually exist.
        # Catches stale/corrupted graphs left by previous failed inits
        # (e.g., schema created but vlabels missing due to uncaught errors).
        await self._verify_graph_labels(lightrag)

        self._backend = self._build_retrieval_backend(
            config,
            lightrag=lightrag,
            embedder=multimodal_embedder if self._direct_image_embedding_enabled else None,
            rerank_func=rerank_func,
        )

        # Initialize metadata index
        self._metadata_index = await self._create_metadata_index(config)
        from dlightrag.core.ingestion.engine import UnifiedIngestionEngine
        from dlightrag.core.retrieval.bm25 import BM25Profile
        from dlightrag.core.retrieval.bm25_language import BM25LanguageClassifier
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldRegistry

        self._metadata_registry = MetadataFieldRegistry.from_config(config.metadata.fields)
        bm25_profiles = [
            BM25Profile(
                name=profile.name,
                text_config=profile.text_config,
                languages=tuple(profile.languages),
                fallback=profile.fallback,
            )
            for profile in config.bm25_profiles
        ]
        bm25_language_classifier = (
            BM25LanguageClassifier(
                tuple(
                    language
                    for profile in bm25_profiles
                    if not profile.fallback
                    for language in profile.languages
                )
            )
            if config.bm25_enabled
            else None
        )
        self._ingestion_engine = UnifiedIngestionEngine(
            lightrag=lightrag,
            stores=self._lightrag_stores,
            metadata_index=self._metadata_index,
            multimodal_embedder=multimodal_embedder,
            direct_image_embedding_enabled=self._direct_image_embedding_enabled,
            workspace=config.workspace,
            parser_rules=config.parser.rules,
            chunk_options=config.parser.chunk_options,
            metadata_registry=self._metadata_registry,
            allow_ad_hoc_metadata=config.metadata.allow_ad_hoc_json,
            default_metadata_policy=config.metadata.default_ingest_policy,
            min_image_pixel=config.parser_sidecars.vlm.min_image_pixel,
            bm25_language_classifier=bm25_language_classifier,
        )

        if config.bm25_enabled:
            from dlightrag.core.retrieval.bm25 import PostgresBM25
            from dlightrag.storage.pool import pg_pool

            self._bm25 = PostgresBM25(
                pool=pg_pool,
                workspace=config.workspace,
                profiles=bm25_profiles,
            )
            await self._bm25.ensure_indexes(
                k1=config.bm25_k1,
                b=config.bm25_b,
            )

        from dlightrag.core.retrieval.retriever import UnifiedRetriever

        self._retrieval_orchestrator = UnifiedRetriever(
            backend=self._backend,
            bm25=self._bm25,
            metadata_index=self._metadata_index,
            stores=self._lightrag_stores,
            rrf_k=config.rrf_k,
        )

        logger.info("LightRAG main runtime path ready")

        # Pre-warm worker pools in background.  LightRAG lazily initializes
        # LLM keyword (~4 workers) and embedding (~8 workers) pools on first
        # use, which adds ~100 s to the first query after restart.  Trigger
        # init now so real users never pay that cost.
        try:
            asyncio.create_task(self._warmup_lightrag_workers())
        except RuntimeError:
            pass  # no event loop running (e.g. sync test setup)

    async def _warmup_lightrag_workers(self) -> None:
        """Pre-initialize LightRAG worker pools in the background."""
        try:
            await self._lightrag.aquery("__warmup__", mode="naive")
            logger.info("LightRAG worker warm-up complete")
        except Exception:
            logger.debug("LightRAG worker warm-up failed (non-critical)", exc_info=True)

    async def _create_metadata_index(
        self,
        config: DlightragConfig,
    ) -> MetadataIndexProtocol:
        """Create the PostgreSQL metadata index backend."""
        from dlightrag.storage.pg_metadata_index import PGMetadataIndex

        idx = PGMetadataIndex(workspace=config.workspace)
        await idx.initialize()
        logger.info("Metadata index: PGMetadataIndex (PostgreSQL)")
        return idx

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

                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", graph_name):
                    logger.error("Invalid AGE graph name: %r, skipping drop", graph_name)
                    return
                await conn.execute('SET search_path = ag_catalog, "$user", public')
                await conn.execute("SELECT drop_graph($1, true)", graph_name)

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

        if self._multimodal_embedder is not None:
            try:
                await self._multimodal_embedder.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close multimodal embedder", exc_info=True)
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
        """Completely remove this workspace -- all data, graph schemas, and files.

        Delegates to the dedicated 6-phase reset module (``dlightrag.core.reset``).
        """
        from dlightrag.core.reset import areset

        return await areset(self, keep_files=keep_files, dry_run=dry_run)

    async def _shutdown_worker_pools(self) -> None:
        """Shutdown LightRAG priority-queue worker pools."""
        lr = self.lightrag
        if lr is None:
            return

        from dlightrag.utils.concurrency import shutdown_async_callable

        funcs: list[tuple[str, Any]] = []
        for attr in ("embedding_func", "llm_model_func", "rerank_model_func"):
            try:
                obj = getattr(lr, attr, None)
                funcs.append((attr, getattr(obj, "func", obj)))
            except Exception:  # noqa: BLE001
                logger.debug("Failed to collect %s worker pool", attr, exc_info=True)

        role_funcs = getattr(lr, "role_llm_funcs", None) or {}
        items = role_funcs.items() if isinstance(role_funcs, Mapping) else ()
        funcs.extend((f"role_llm_funcs.{role}", func) for role, func in items)

        seen: set[int] = set()
        for label, func in funcs:
            if func is None or id(func) in seen:
                continue
            seen.add(id(func))
            try:
                await shutdown_async_callable(func)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to shutdown %s worker pool", label, exc_info=True)

    async def _upsert_workspace_meta(self, *, display_name: str | None = None) -> None:
        """Persist this workspace in DlightRAG's PostgreSQL registry."""
        from dlightrag.storage.workspaces import PGWorkspaceRegistry

        registry = PGWorkspaceRegistry()
        await registry.initialize()
        await registry.upsert(
            workspace=normalize_workspace(self.config.workspace),
            display_name=display_name or self.config.workspace,
            embedding_model=self.config.embedding.model,
        )

    async def aregister_workspace(self, *, display_name: str | None = None) -> None:
        """Persist this initialized workspace so it is discoverable by managers."""
        self._ensure_initialized()
        await self._upsert_workspace_meta(display_name=display_name)

    # === INGESTION API ===

    def _resolve_replace(self, explicit: Any) -> bool:
        if explicit is None:
            return self.config.ingestion_replace_default
        return bool(explicit)

    async def _purge_existing_for_replace(
        self,
        *,
        file_path: Path,
        stored_file_path: str | None = None,
    ) -> None:
        """Delete an existing ingest target before replacing it."""
        lightrag = self.lightrag
        if lightrag is None:
            return

        from dlightrag.core.ingestion.cleanup import cascade_delete, collect_deletion_context

        identifiers: list[str] = []
        for candidate in (stored_file_path, str(file_path), file_path.name):
            if candidate and candidate not in identifiers:
                identifiers.append(candidate)

        seen_doc_ids: set[str] = set()
        for identifier in identifiers:
            ctx = await collect_deletion_context(
                identifier=identifier,
                lightrag=lightrag,
                metadata_index=self._metadata_index,
            )
            ctx.doc_ids.difference_update(seen_doc_ids)
            if not ctx.doc_ids:
                continue

            stats = await cascade_delete(
                ctx=ctx,
                lightrag=lightrag,
                metadata_index=self._metadata_index,
            )
            seen_doc_ids.update(ctx.doc_ids)

            errors = stats.get("errors") or []
            if errors:
                raise RuntimeError(
                    f"replace cleanup failed for {identifier}: {'; '.join(map(str, errors))}"
                )

    async def _aingest_local_file(
        self,
        file_path: Path,
        *,
        replace: bool,
        stored_file_path: str | None = None,
        source_root: Path | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest one local file through the unified LightRAG path."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        if replace:
            await self._purge_existing_for_replace(
                file_path=staged_input_path(
                    input_root=self._workspace_input_root(),
                    file_path=file_path,
                    relative_to=source_root,
                ),
                stored_file_path=stored_file_path,
            )

        file_path = stage_input_file(
            input_root=self._workspace_input_root(),
            file_path=file_path,
            relative_to=source_root,
        )
        result = await self._ingestion_engine.aingest_file(
            file_path,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return result

    async def _aingest_local_files(
        self,
        file_paths: list[Path],
        *,
        replace: bool,
        source_root: Path | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest local files through one LightRAG staged batch."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        if not file_paths:
            return {"processed": 0, "errors": [], "results": []}

        if replace:
            for file_path in file_paths:
                await self._purge_existing_for_replace(
                    file_path=staged_input_path(
                        input_root=self._workspace_input_root(),
                        file_path=file_path,
                        relative_to=source_root,
                    )
                )

        staged_paths = [
            stage_input_file(
                input_root=self._workspace_input_root(),
                file_path=file_path,
                relative_to=source_root,
            )
            for file_path in file_paths
        ]
        result = await self._ingestion_engine.aingest_files(
            staged_paths,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return result

    def _workspace_input_root(self) -> Path:
        return workspace_input_root(self.config.input_dir_path, self.config.workspace)

    async def _download_remote_to_prepared_item(
        self,
        *,
        source: AsyncDataSource,
        key: str,
        source_type: str,
        source_uri: str,
        batch_root: Path,
        retain_source_file: bool,
    ) -> PreparedIngestFile:
        if retain_source_file:
            parser_path = retained_remote_source_path(
                input_root=self._workspace_input_root(),
                source_type=source_type,
                source_uri=source_uri,
                key=key,
            )
            metadata_path = str(parser_path)
        else:
            parser_path = remote_parser_input_path(
                batch_root=batch_root,
                source_uri=source_uri,
                key=key,
            )
            metadata_path = source_uri
        parser_path.parent.mkdir(parents=True, exist_ok=True)
        await source.amaterialize_document(key, parser_path)
        return PreparedIngestFile(
            parser_path=parser_path,
            metadata_path=metadata_path,
            display_filename=Path(key).name,
        )

    async def _aingest_remote_keys(
        self,
        *,
        source: AsyncDataSource,
        source_type: str,
        keys: Iterable[str] | AsyncIterable[str],
        source_uri_for_key: Callable[[str], str],
        replace: bool,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
        progress_callback: RemoteIngestProgressCallback | None = None,
        resume_from_window: int = 0,
        retain_source_file: bool | None = None,
    ) -> dict[str, Any]:
        """Download remote objects into ephemeral parser batches and ingest them."""
        from dlightrag.utils.concurrency import bounded_map

        resume_from_window = max(0, int(resume_from_window))
        processed = 0
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        saw_keys = False
        retain_source_files = (
            self.config.retain_remote_source_files
            if retain_source_file is None
            else bool(retain_source_file)
        )

        async for batch_index, window in _aiter_chunks(keys, _REMOTE_INGEST_BATCH_SIZE):
            saw_keys = True
            if batch_index < resume_from_window:
                continue

            batch_root = remote_ingest_batch_root(
                input_root=self._workspace_input_root(),
                source_type=source_type,
                batch_id=f"{batch_index:04d}-{uuid.uuid4().hex}",
            )
            source_uris = [source_uri_for_key(key) for key in window]
            prepared_items: list[PreparedIngestFile] = []
            window_errors: list[str] = []

            async def _download(
                item: tuple[str, str],
                *,
                current_batch_root: Path = batch_root,
            ) -> PreparedIngestFile:
                key, source_uri = item
                return await self._download_remote_to_prepared_item(
                    source=source,
                    key=key,
                    source_type=source_type,
                    source_uri=source_uri,
                    batch_root=current_batch_root,
                    retain_source_file=retain_source_files,
                )

            try:
                download_results = await bounded_map(
                    list(zip(window, source_uris, strict=True)),
                    _download,
                    max_concurrent=_REMOTE_DOWNLOAD_CONCURRENCY,
                    task_name=f"{source_type}-download",
                )

                prepared_source_uris: list[str] = []
                prepared_keys: list[str] = []
                for _key, source_uri, downloaded in zip(
                    window, source_uris, download_results, strict=True
                ):
                    if isinstance(downloaded, Exception):
                        error = f"{source_uri}: {downloaded}"
                        errors.append(error)
                        window_errors.append(error)
                        continue
                    prepared_items.append(downloaded)
                    prepared_source_uris.append(source_uri)
                    prepared_keys.append(_key)

                if not prepared_items:
                    if progress_callback is not None:
                        await progress_callback(
                            RemoteIngestWindowProgress(
                                source_type=source_type,
                                batch_index=batch_index,
                                total_delta=len(window),
                                processed_delta=0,
                                failed_delta=len(window),
                                errors=tuple(window_errors),
                            )
                        )
                    continue

                if replace:
                    for key, prepared_item, source_uri in zip(
                        prepared_keys, prepared_items, prepared_source_uris, strict=True
                    ):
                        await self._purge_existing_for_replace(
                            file_path=prepared_item.parser_path,
                            stored_file_path=source_uri,
                        )
                        if not retain_source_files:
                            await self._purge_existing_for_replace(
                                file_path=retained_remote_source_path(
                                    input_root=self._workspace_input_root(),
                                    source_type=source_type,
                                    source_uri=source_uri,
                                    key=key,
                                )
                            )

                batch_result = await self._ingestion_engine.aingest_files(
                    prepared_items,
                    replace=False,
                    title=title,
                    author=author,
                    metadata=metadata,
                    metadata_policy=metadata_policy,
                )
                if hasattr(batch_result, "model_dump"):
                    batch_result = batch_result.model_dump()

                processed += int(batch_result.get("processed") or 0)
                results.extend(batch_result.get("results") or [])
                batch_errors = [str(error) for error in batch_result.get("errors") or []]
                errors.extend(batch_errors)
                if progress_callback is not None:
                    await progress_callback(
                        RemoteIngestWindowProgress(
                            source_type=source_type,
                            batch_index=batch_index,
                            total_delta=len(window),
                            processed_delta=int(batch_result.get("processed") or 0),
                            failed_delta=len(batch_errors)
                            + max(0, len(window) - len(prepared_items)),
                            errors=tuple([*window_errors, *batch_errors]),
                        )
                    )
            finally:
                if not retain_source_files:
                    await asyncio.to_thread(_remove_remote_parser_sources, prepared_items)
                await asyncio.to_thread(
                    _remove_empty_parents,
                    batch_root,
                    self._workspace_input_root(),
                )

        if not saw_keys:
            return {"processed": 0, "errors": [], "results": []}
        return {"processed": processed, "errors": errors, "results": results}

    @staticmethod
    def _single_file_result(batch_result: dict[str, Any]) -> dict[str, Any]:
        results = batch_result.get("results")
        if isinstance(results, list) and results:
            return results[0]
        return batch_result

    @staticmethod
    def _default_source_uri_for_key(source_type: str, key: str) -> str:
        if not source_type or "://" in source_type:
            raise ValueError("source_type must be a non-empty URI scheme name")
        return f"{source_type}://{key.lstrip('/')}"

    async def aingest_source(
        self,
        source: AsyncDataSource,
        *,
        source_type: str = "source",
        keys: Iterable[str] | AsyncIterable[str] | None = None,
        prefix: str | None = None,
        source_uri_for_key: Callable[[str], str] | None = None,
        replace: bool | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
        retain_source_file: bool | None = None,
        _progress_callback: RemoteIngestProgressCallback | None = None,
        _resume_from_window: int = 0,
    ) -> dict[str, Any]:
        """Ingest documents from a caller-provided async data source.

        SDK connectors expose stable document ids and write bytes into the
        destination path DlightRAG provides. DlightRAG handles temporary parser
        files, metadata provenance, replace semantics, and cleanup.
        """
        self._ensure_initialized()
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        resolved_keys = (
            keys
            if keys is not None
            else cast(Iterable[str] | AsyncIterable[str], source.aiter_documents(prefix=prefix))
        )
        uri_for_key = source_uri_for_key or (
            lambda key: self._default_source_uri_for_key(source_type, key)
        )
        close = getattr(source, "aclose", None)
        try:
            return await self._aingest_remote_keys(
                source=source,
                source_type=source_type,
                keys=resolved_keys,
                source_uri_for_key=uri_for_key,
                replace=self._resolve_replace(replace),
                title=title,
                author=author,
                metadata=metadata,
                metadata_policy=metadata_policy,
                progress_callback=_progress_callback,
                resume_from_window=_resume_from_window,
                retain_source_file=retain_source_file,
            )
        finally:
            if close is not None:
                await close()

    async def _aingest_url(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        raw_urls = kwargs.get("urls")
        urls = list(raw_urls) if isinstance(raw_urls, list) else []
        if kwargs.get("url"):
            urls = [str(kwargs["url"])]

        from dlightrag.sourcing.url import URLDataSource

        source_kwargs: dict[str, Any] = {"urls": urls}
        if kwargs.get("filename") is not None:
            source_kwargs["filename"] = kwargs["filename"]
        if kwargs.get("source_uri") is not None:
            source_kwargs["source_uri"] = kwargs["source_uri"]
        if kwargs.get("source_uris") is not None:
            source_kwargs["source_uris"] = kwargs["source_uris"]
        source = URLDataSource(
            **source_kwargs,
            max_download_bytes=self.config.url_ingest_max_bytes,
        )
        result = await self.aingest_source(
            source,
            source_type="url",
            source_uri_for_key=source.source_uri_for_key,
            replace=replace,
            title=kwargs.get("title"),
            author=kwargs.get("author"),
            metadata=kwargs.get("metadata"),
            metadata_policy=kwargs.get("metadata_policy"),
            retain_source_file=kwargs.get("retain_source_file"),
            _progress_callback=kwargs.get("_progress_callback"),
            _resume_from_window=int(kwargs.get("_resume_from_window") or 0),
        )
        if len(urls) == 1:
            return self._single_file_result(result)
        return result

    async def _aingest_azure_blob(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        container_name = kwargs.get("container_name")
        source = kwargs.get("source")
        if source is None:
            if not container_name:
                raise ValueError("'container_name' is required for azure_blob source_type")
            from dlightrag.sourcing.azure_blob import AzureBlobDataSource

            source = AzureBlobDataSource(
                connection_string=self.config.blob_connection_string,
                container_name=container_name,
            )

        common_kwargs = {
            "source_type": "azure_blob",
            "source_uri_for_key": lambda key: f"azure://{container_name}/{key}",
            "replace": replace,
            "title": kwargs.get("title"),
            "author": kwargs.get("author"),
            "metadata": kwargs.get("metadata"),
            "metadata_policy": kwargs.get("metadata_policy"),
            "retain_source_file": kwargs.get("retain_source_file"),
            "_progress_callback": kwargs.get("_progress_callback"),
            "_resume_from_window": int(kwargs.get("_resume_from_window") or 0),
        }
        if kwargs.get("blob_path"):
            return self._single_file_result(
                await self.aingest_source(source, keys=[str(kwargs["blob_path"])], **common_kwargs)
            )

        prefix = "" if kwargs.get("prefix") is None else str(kwargs.get("prefix"))
        return await self.aingest_source(source, prefix=prefix, **common_kwargs)

    async def _aingest_s3(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs.get("bucket")
        source = kwargs.get("source")
        if source is None:
            if not bucket:
                raise ValueError("'bucket' is required for s3 source_type")
            from dlightrag.sourcing.aws_s3 import S3DataSource

            source = S3DataSource(
                bucket=str(bucket), region=kwargs.get("region") or self.config.s3_region
            )

        common_kwargs = {
            "source_type": "s3",
            "source_uri_for_key": lambda key: f"s3://{bucket}/{key}",
            "replace": replace,
            "title": kwargs.get("title"),
            "author": kwargs.get("author"),
            "metadata": kwargs.get("metadata"),
            "metadata_policy": kwargs.get("metadata_policy"),
            "retain_source_file": kwargs.get("retain_source_file"),
            "_progress_callback": kwargs.get("_progress_callback"),
            "_resume_from_window": int(kwargs.get("_resume_from_window") or 0),
        }
        key = kwargs.get("key") or kwargs.get("blob_path")
        if key:
            return self._single_file_result(
                await self.aingest_source(source, keys=[str(key)], **common_kwargs)
            )

        prefix = "" if kwargs.get("prefix") is None else str(kwargs.get("prefix"))
        return await self.aingest_source(source, prefix=prefix, **common_kwargs)

    async def aingest(
        self,
        source_type: SourceType,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Unified ingestion API.

        Args:
            source_type: "local", "azure_blob", "s3", or "url"
            kwargs:
                local: path, replace
                azure_blob: source, container_name, blob_path, prefix, replace
                s3: bucket, key, prefix, replace
                url: url or urls, optional filename, replace
        """
        self._ensure_initialized()
        replace = self._resolve_replace(kwargs.get("replace"))

        if self._ingestion_engine is not None and source_type == "local":
            path_str = kwargs.get("path")
            if not path_str:
                raise ValueError("'path' is required for local source_type")
            local_path = Path(path_str)
            file_paths = iter_ingestable_files(local_path)
            common_kwargs = {
                "replace": replace,
                "title": kwargs.get("title"),
                "author": kwargs.get("author"),
                "metadata": kwargs.get("metadata"),
                "metadata_policy": kwargs.get("metadata_policy"),
            }
            if local_path.is_file():
                return await self._aingest_local_file(local_path, **common_kwargs)

            return await self._aingest_local_files(
                file_paths,
                source_root=local_path,
                **common_kwargs,
            )

        if self._ingestion_engine is not None and source_type == "azure_blob":
            return await self._aingest_azure_blob(replace=replace, **kwargs)

        if self._ingestion_engine is not None and source_type == "s3":
            return await self._aingest_s3(replace=replace, **kwargs)

        if self._ingestion_engine is not None and source_type == "url":
            return await self._aingest_url(replace=replace, **kwargs)

        raise RuntimeError("Ingestion engine not initialized")

    # === RETRIEVAL API ===

    async def aretrieve(
        self,
        query: str,
        multimodal_content: list[dict[str, Any]] | None = None,
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
        if self._retrieval_orchestrator is None:
            raise RuntimeError("Retrieval orchestrator not initialized")

        effective_filters = filters
        filter_source = "explicit" if filters is not None else None
        if effective_filters is None and _plan is not None:
            effective_filters = getattr(_plan, "metadata_filter", None)
            filter_source = getattr(_plan, "metadata_filter_source", None)
        effective_filters = self._normalize_metadata_filter(effective_filters)

        kg_result = await self._retrieval_orchestrator.aretrieve(
            query,
            multimodal_content=multimodal_content,
            metadata_filter=effective_filters,
            metadata_filter_source=filter_source,
            bm25_query=getattr(_plan, "bm25_query", None) if _plan is not None else None,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            is_reretrieve=is_reretrieve,
            **kwargs,
        )

        # --- Step 2.5: Hydrate image data for all retrieved chunks ---
        # BM25 chunks and other post-fusion additions haven't been through
        # hydration yet, so they lack image_data for visual chunks.
        lr = self.lightrag
        if lr is not None:
            from dlightrag.core.retrieval.provenance import hydrate_lightrag_chunk_provenance

            chunks_to_hydrate = kg_result.contexts.get("chunks", [])
            if chunks_to_hydrate:
                await hydrate_lightrag_chunk_provenance(lr, chunks_to_hydrate)

        # --- Step 3: Enrich chunks with document metadata ---
        await self._enrich_chunks_with_metadata(kg_result)
        kg_result.trace["metadata_enriched"] = True

        # --- Step 4: Canonicalize reference_id across all merged chunks ---
        # Required so fused chunks get stable doc-level IDs and become citable
        # as [ref-chunk_idx].
        from dlightrag.core.retrieval import canonicalize_reference_ids

        kg_result.contexts["chunks"] = canonicalize_reference_ids(
            kg_result.contexts.get("chunks", [])
        )
        for chunk in kg_result.contexts.get("chunks", []):
            chunk.setdefault("_workspace", self.config.workspace)
        kg_result.trace["workspace"] = self.config.workspace

        return kg_result

    async def aget_visual_asset(self, chunk_id: str, *, size: str = "full") -> Any | None:
        """Resolve a chunk image asset for API/Web image routes."""
        self._ensure_initialized()
        if self._visual_asset_resolver is None:
            return None
        if size == "thumb":
            return await self._visual_asset_resolver.resolve_thumbnail(
                chunk_id,
                max_px=self.config.visual_assets.thumb_max_px,
            )
        if size == "full":
            return await self._visual_asset_resolver.resolve(chunk_id)
        raise ValueError("size must be 'full' or 'thumb'")

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
                "page_count",
                "original_format",
                "doc_title",
                "doc_author",
            }
        )

        chunks = result.contexts.get("chunks", [])
        if not chunks:
            return

        # Collect unique file_paths for batch lookup.  When available, keep the
        # LightRAG full_doc_id as the strongest metadata key.
        path_meta: dict[str, dict[str, Any]] = {}
        path_doc_ids: dict[str, str] = {}
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            if fp and fp not in path_meta:
                path_meta[fp] = {}
            doc_id = chunk.get("full_doc_id") or chunk.get("_full_doc_id")
            if fp and doc_id and fp not in path_doc_ids:
                path_doc_ids[fp] = str(doc_id)

        idx = self._metadata_index
        if idx is None:
            return
        from dlightrag.utils.concurrency import bounded_map

        async def _fetch(fp: str) -> tuple[str, dict[str, Any]]:
            try:
                meta = None
                doc_id = path_doc_ids.get(fp)
                if doc_id:
                    meta = await idx.get(doc_id)
                if meta is None:
                    doc_ids = await idx.find_by_file_path(fp)
                    if not doc_ids:
                        from pathlib import Path as _Path

                        doc_ids = await idx.find_by_filename(_Path(fp).name)
                    if not doc_ids:
                        return fp, {}
                    meta = await idx.get(doc_ids[0])
                if not meta:
                    return fp, {}
                return fp, {
                    k: v for k, v in meta.items() if k not in _SKIP and v is not None and v != ""
                }
            except Exception:
                return fp, {}  # enrichment is best-effort

        results = await bounded_map(
            list(path_meta),
            _fetch,
            max_concurrent=8,
            task_name="metadata-enrichment",
        )
        for fetch_result in results:
            if isinstance(fetch_result, Exception):
                continue
            fp, doc_meta = fetch_result
            if doc_meta:
                path_meta[fp] = doc_meta

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
        result = await self._metadata_index.get(doc_id)  # type: ignore[union-attr]
        return result if result else {}

    async def aupdate_metadata(
        self,
        doc_id: str,
        data: dict[str, Any],
        *,
        mode: Literal["merge", "replace"] = "merge",
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> None:
        """Update (merge) document metadata."""
        from dlightrag.core.retrieval.metadata_fields import normalize_user_metadata

        if self._metadata_index is None:
            raise RuntimeError("Metadata index not initialized")
        normalized = normalize_user_metadata(
            data,
            self._metadata_registry,
            metadata_policy=metadata_policy or self._default_metadata_policy,
            allow_ad_hoc_json=self._allow_ad_hoc_metadata,
        )
        await self._metadata_index.upsert(
            doc_id,
            {
                "metadata_update_mode": mode,
                "user_metadata": dict(data),
                "metadata_filterable": normalized.filterable,
                "metadata_json": normalized.raw_json,
            },
        )

    async def asearch_metadata(self, filters: MetadataFilter) -> list[str]:
        """Search metadata by filters, return matching doc_ids."""
        if self._metadata_index is None:
            return []
        normalized_filters = self._normalize_metadata_filter(filters)
        assert normalized_filters is not None
        return await self._metadata_index.query(normalized_filters)

    def _normalize_metadata_filter(self, filters: MetadataFilter | None) -> MetadataFilter | None:
        if filters is None or self._metadata_registry is None:
            return filters
        return self._metadata_registry.normalize_filter(filters)

    # === FILE MANAGEMENT API ===

    async def alist_ingested_files(self) -> list[dict[str, Any]]:
        """List all processed files from LightRAG doc_status."""
        self._ensure_initialized()
        if self._lightrag is None:
            return []

        from lightrag.base import DocStatus

        try:
            processed = await self._lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
        except Exception as exc:
            logger.warning("Failed to query PROCESSED docs: %s", exc)
            return []

        return [
            {
                "doc_id": doc_id,
                "file_path": getattr(info, "file_path", "") or "",
                "status": "processed",
                "updated_at": str(getattr(info, "updated_at", "")),
            }
            for doc_id, info in (processed or {}).items()
        ]

    async def alist_failed_docs(self) -> list[dict[str, Any]]:
        """Return all documents currently in DocStatus.FAILED for this workspace."""
        self._ensure_initialized()
        if self._lightrag is None:
            return []

        from lightrag.base import DocStatus

        try:
            failed = await self._lightrag.doc_status.get_docs_by_status(DocStatus.FAILED)
        except Exception as exc:
            logger.warning("Failed to query FAILED docs: %s", exc)
            return []

        return [
            {
                "doc_id": doc_id,
                "file_path": getattr(info, "file_path", "") or "",
                "error": getattr(info, "error_msg", None) or getattr(info, "content_summary", ""),
                "updated_at": str(getattr(info, "updated_at", "")),
            }
            for doc_id, info in (failed or {}).items()
        ]

    async def aretry_failed_docs(self) -> dict[str, Any]:
        """Re-ingest every FAILED document with ``replace=True``.

        Each doc's stored ``file_path`` is parsed via
        :func:`dlightrag.sourcing.uri.parse_remote_uri` to dispatch the right
        ``aingest()`` source type.

        To make this endpoint actually drain the FAILED list, we explicitly
        call ``adelete_by_doc_id`` before re-ingest. That drops the FAILED
        doc_status row and any partial chunks/entities it may have left
        behind.

        Returns a summary dict with counts and per-doc outcomes.
        """
        from dlightrag.sourcing.uri import parse_remote_uri

        failed = await self.alist_failed_docs()
        if not failed:
            return {"retried": 0, "succeeded": 0, "failed": 0, "results": []}

        lr = self.lightrag
        succeeded: list[dict[str, Any]] = []
        still_failed: list[dict[str, Any]] = []

        for entry in failed:
            doc_id = entry["doc_id"]
            file_path = entry["file_path"]
            if not file_path:
                still_failed.append({"doc_id": doc_id, "reason": "no file_path on doc_status"})
                continue

            try:
                source_type, ingest_kwargs = parse_remote_uri(file_path)
            except ValueError as exc:
                still_failed.append({"doc_id": doc_id, "file_path": file_path, "reason": str(exc)})
                continue

            # Drop the FAILED doc_status row before re-ingesting so the next
            # alist_failed_docs() call no longer returns this doc_id.
            # Best-effort: a missing doc_id is a no-op upstream.
            if lr is not None:
                try:
                    await lr.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                except Exception as exc:
                    logger.warning(
                        "Pre-retry adelete_by_doc_id failed for doc_id=%s: %s", doc_id, exc
                    )

            try:
                result = await self.aingest(source_type, replace=True, **ingest_kwargs)
                succeeded.append({"doc_id": doc_id, "file_path": file_path, "result": result})
            except Exception as exc:
                logger.warning(
                    "Retry failed for doc_id=%s file_path=%s: %s", doc_id, file_path, exc
                )
                still_failed.append({"doc_id": doc_id, "file_path": file_path, "reason": str(exc)})

        return {
            "retried": len(failed),
            "succeeded": len(succeeded),
            "failed": len(still_failed),
            "succeeded_docs": succeeded,
            "failed_docs": still_failed,
        }

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Unified file deletion — DB records and physical files."""
        self._ensure_initialized()
        from dlightrag.core.ingestion.cleanup import (
            cascade_delete,
            collect_deletion_context,
            remove_deleted_files,
        )

        identifiers = [*(file_paths or []), *(filenames or [])]
        results: list[dict[str, Any]] = []
        for identifier in identifiers:
            ctx = await collect_deletion_context(
                identifier=identifier,
                lightrag=self._lightrag,
                metadata_index=self._metadata_index,
            )
            if dry_run:
                results.append(
                    {
                        "identifier": identifier,
                        "status": "would_delete" if ctx.doc_ids else "not_found",
                        "dry_run": True,
                        "docs_deleted": 0,
                        "errors": [],
                        "matched_doc_ids": sorted(ctx.doc_ids),
                        "matched_file_paths": sorted(ctx.file_paths),
                        "sources_used": list(ctx.sources_used),
                    }
                )
                continue

            stats = await cascade_delete(
                ctx=ctx,
                lightrag=self._lightrag,
                metadata_index=self._metadata_index,
            )
            if not ctx.doc_ids:
                status = "not_found"
            elif stats.get("errors"):
                status = "deleted_with_errors"
            else:
                status = "deleted"
                # Remove physical files after successful DB cleanup.
                remove_deleted_files(
                    ctx.file_paths,
                    str(self.config.input_dir_path / self.config.workspace),
                )

            results.append({"identifier": identifier, "status": status, **stats})
        return results

    async def aget_pipeline_status(self) -> dict[str, Any]:
        """Return LightRAG pipeline_status for progress reporting."""
        from lightrag.kg.shared_storage import get_namespace_data

        if self._lightrag is None:
            return {"busy": False, "latest_message": "No LightRAG instance"}

        ns = await get_namespace_data("pipeline_status", workspace=self.config.workspace)
        return {
            "busy": bool(ns.get("busy", False)),
            "job_name": ns.get("job_name", ""),
            "latest_message": ns.get("latest_message", ""),
            "docs": ns.get("docs", 0),
            "batchs": ns.get("batchs", 0),
            "cur_batch": ns.get("cur_batch", 0),
            "pending_enqueues": int(ns.get("pending_enqueues", 0) or 0),
            "history_messages": list(ns.get("history_messages", [])[-10:]),
        }


async def _aiter_chunks[T](
    items: Iterable[T] | AsyncIterable[T],
    size: int,
) -> AsyncIterator[tuple[int, list[T]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch_index = 0
    window: list[T] = []
    async for item in _aiter_items(items):
        window.append(item)
        if len(window) >= size:
            yield batch_index, window
            batch_index += 1
            window = []
    if window:
        yield batch_index, window


async def _aiter_items[T](items: Iterable[T] | AsyncIterable[T]) -> AsyncIterator[T]:
    if isinstance(items, AsyncIterable):
        async for item in items:
            yield item
        return
    for item in items:
        yield item


def _remove_empty_parents(path: Path, stop: Path) -> None:
    stop = stop.resolve()
    current = path.resolve()
    while current != stop and current.is_relative_to(stop):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_remote_parser_sources(items: list[PreparedIngestFile]) -> None:
    for item in items:
        parser_path = item.parser_path
        for candidate in (
            parser_path,
            parser_path.parent / PARSED_DIR_NAME / parser_path.name,
        ):
            try:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
            except OSError:
                logger.debug("Failed to remove remote parser source: %s", candidate, exc_info=True)


__all__ = ["RAGService"]
