# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Per-workspace LightRAG service facade for ingestion and retrieval.

DlightRAG owns one LightRAG instance per workspace and adds PostgreSQL
metadata, direct visual embedding, and retrieval orchestration around it.
PostgreSQL advisory locks coordinate first-time storage initialization across
concurrent workers.
"""

import asyncio
import logging
import os
import random
import shutil
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_MM_IMAGE_MIN_PIXEL,
    PARSED_DIR_NAME,
)

from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.client_contracts import IngestDocument, SourceType
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
from dlightrag.sourcing.base import AsyncDataSource, SourceDocument
from dlightrag.sourcing.source_contract import (
    SourceDownloadContractError,
    local_source_uri,
    safe_source_filename,
    validate_download_uri,
    validate_source_uri,
)
from dlightrag.storage.protocols import MetadataIndexProtocol
from dlightrag.utils import normalize_workspace

if TYPE_CHECKING:
    from dlightrag.core.document_embedding import RobustDocumentEmbedder
    from dlightrag.core.ingestion.engine import UnifiedIngestionEngine
    from dlightrag.core.lightrag_stores import LightRAGStores
    from dlightrag.core.retrieval.bm25 import PostgresBM25
    from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend
    from dlightrag.core.retrieval.protocols import RetrievalBackend
    from dlightrag.core.retrieval.retriever import UnifiedRetriever
    from dlightrag.core.visual_assets import VisualAssetResolver
    from dlightrag.models.composer import ComposerModelBundle
    from dlightrag.models.multimodal_embedding import MultimodalEmbedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ComposerProcessingResources:
    """Non-owning resources borrowed for one prepared Web Composer turn."""

    lightrag: Any
    robust_document_embedder: RobustDocumentEmbedder
    direct_image_embedding_enabled: bool
    rerank_func: Any
    model_bundle: ComposerModelBundle
    config: DlightragConfig


def _ingest_documents(value: Any | None) -> list[IngestDocument] | None:
    if value is None:
        return None
    return [
        document
        if isinstance(document, IngestDocument)
        else IngestDocument.model_validate(document)
        for document in value
    ]


def _source_document_from_manifest(document: IngestDocument, *, key: str) -> SourceDocument:
    return SourceDocument(
        key=key,
        source_uri=document.source_uri,
        download_uri=document.download_uri,
        display_filename=document.filename,
        title=document.title,
        author=document.author,
        metadata=document.metadata,
        metadata_policy=document.metadata_policy,
    )


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


@dataclass(frozen=True)
class _RemoteDownloadFailure:
    """Sanitized per-document failure returned without exposing source exceptions."""

    error: str


def _safe_remote_source_id(document: SourceDocument) -> str:
    return safe_source_filename(document.display_filename or document.key)


def _retry_display_filename(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("source filename is invalid")
    raw_basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    if not raw_basename or raw_basename in {".", ".."}:
        raise ValueError("source filename is invalid")
    filename = safe_source_filename(value)
    if filename in {".", ".."}:
        raise ValueError("source filename is invalid")
    return filename


def _retry_parser_filename(display_filename: str) -> str:
    path = Path(display_filename)
    suffix = path.suffix
    stem = path.stem or "document"
    marker = f"__retry_{uuid.uuid4().hex}"
    max_stem_length = max(1, 96 - len(marker))
    return f"{stem[:max_stem_length]}{marker}{suffix}"


def _normalized_operation_status(result: object) -> str:
    raw_status = (
        result.get("status") if isinstance(result, Mapping) else getattr(result, "status", None)
    )
    return str(getattr(raw_status, "value", raw_status) or "").strip().lower()


def _download_locator_kind(locator: str | None, *, retained: bool = False) -> str:
    if retained:
        return "local"
    if locator is None:
        return "none"
    for scheme in ("s3", "azure", "https"):
        if locator.startswith(f"{scheme}://"):
            return scheme
    return "unsupported"


def _log_download_locator_outcome(*, outcome: str, locator_kind: str, source_filename: str) -> None:
    logger.info(
        "source_download_locator_outcome",
        extra={
            "outcome": outcome,
            "locator_kind": locator_kind,
            "source_filename": source_filename,
        },
    )


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
        rag = await RAGService.acreate(config=my_config, enable_vlm=True)

        # With callbacks:
        rag = await RAGService.acreate(
            config=my_config,
            cancel_checker=my_cancel_fn,
        )
    """

    @classmethod
    async def acreate(
        cls,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
        rerank_supports_vision: bool | None = None,
    ) -> RAGService:
        """Async factory method - creates and initializes RAGService."""
        instance = cls(
            config=config,
            enable_vlm=enable_vlm,
            cancel_checker=cancel_checker,
            rerank_supports_vision=rerank_supports_vision,
        )
        await instance.initialize()
        return instance

    def __init__(
        self,
        config: DlightragConfig | None = None,
        enable_vlm: bool = True,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
        rerank_supports_vision: bool | None = None,
    ) -> None:
        """Store configuration only. Use RAGService.acreate() for full initialization."""
        self.config = config or get_config()
        self.enable_vlm = enable_vlm
        self._initialized: bool = False
        self._warmup_task: asyncio.Task[None] | None = None

        # Callbacks for decoupled integration
        self._cancel_checker = cancel_checker
        self._rerank_supports_vision = rerank_supports_vision
        self._rerank_consumes_images: bool = True

        # Direct LightRAG runtime and DlightRAG orchestration.
        self._lightrag: Any = None  # Direct LightRAG reference
        self._metadata_index: MetadataIndexProtocol | None = None
        self._table_schema: dict[str, Any] | None = None  # Cached metadata table schema
        self._metadata_registry: Any = None
        self._allow_ad_hoc_metadata = self.config.metadata.allow_ad_hoc_json
        self._default_metadata_policy: MetadataIngestPolicy = (
            self.config.metadata.default_ingest_policy
        )
        self._lightrag_stores: LightRAGStores | None = None
        self._ingestion_engine: UnifiedIngestionEngine | None = None
        self._bm25: PostgresBM25 | None = None
        self._retrieval_orchestrator: UnifiedRetriever | None = None
        self._multimodal_embedder: MultimodalEmbedder | None = None
        self._document_embedder: RobustDocumentEmbedder | None = None
        self._rerank_func: Any = None
        self._direct_image_embedding_enabled = False
        self._visual_asset_resolver: VisualAssetResolver | None = None

        # Retrieval backend (satisfies RetrievalBackend Protocol).
        # Explicitly wired by the unified LightRAG initialization path.
        self._backend: RetrievalBackend | None = None

    @property
    def lightrag(self) -> Any:
        """Return the underlying LightRAG instance for the unified runtime."""
        return self._lightrag

    @property
    def robust_document_embedder(self) -> RobustDocumentEmbedder:
        """Return the initialized document embedder without transferring ownership."""
        if self._document_embedder is None:
            raise RuntimeError("RAGService document embedder is not initialized")
        return self._document_embedder

    @property
    def direct_image_embedding_enabled(self) -> bool:
        """Return the probed image-embedding capability for this service."""
        return self._direct_image_embedding_enabled

    @property
    def rerank_func(self) -> Any:
        """Return the service-owned reranker callable, if configured."""
        return self._rerank_func

    @staticmethod
    def _build_vector_db_kwargs(config: DlightragConfig) -> dict[str, Any]:
        """Build vector_db_storage_cls_kwargs from config.vector_db_kwargs passthrough."""
        kwargs: dict[str, Any] = {
            "cosine_better_than_threshold": DEFAULT_COSINE_THRESHOLD,
        }
        kwargs.update(config.vector_db_kwargs)
        return kwargs

    @staticmethod
    async def _resolve_direct_image_embedding_enabled(
        embedder: Any,
        *,
        startup_probe: bool,
        require_image_support: bool,
    ) -> bool:
        """Return whether DlightRAG image embedding paths are safe to use.

        One capability drives both DlightRAG image paths -- the image->image
        direct-visual query leg and the fused visual-chunk vector overwrite.
        Every image-capable provider is a unified multimodal model that fuses
        text+image, so ``supports_images`` plus the live probe is the only gate.
        The probe calls only the embedding provider with an in-memory 1x1 image;
        it does not touch LightRAG storage, PostgreSQL, or local files.
        """
        if not getattr(embedder, "supports_images", False):
            if require_image_support:
                raise ValueError(
                    "embedding.input_modality='multimodal' cannot be honored because "
                    f"{embedder.__class__.__name__} does not support image inputs"
                )
            logger.info(
                "Image embedding disabled: %s does not support image inputs",
                embedder.__class__.__name__,
            )
            return False
        if not startup_probe:
            return True
        try:
            await embedder.probe_image_embedding()
        except Exception as exc:  # noqa: BLE001
            if require_image_support:
                raise ValueError(
                    "embedding.input_modality='multimodal' requires working image embeddings, "
                    "but the startup probe failed"
                ) from exc
            logger.warning(
                "Image embedding probe failed; using LightRAG semantic visual path only",
                exc_info=True,
            )
            return False
        logger.info("Image embedding probe passed — direct-visual leg enabled")
        return True

    @staticmethod
    def _build_document_embedder(
        config: DlightragConfig,
        embedder: MultimodalEmbedder,
        *,
        image_enabled: bool,
    ) -> RobustDocumentEmbedder:
        from dlightrag.core.document_embedding import RobustDocumentEmbedder

        return RobustDocumentEmbedder(
            embedder=embedder,
            image_enabled=image_enabled,
            dimension=config.embedding.dim,
            min_image_pixel=(
                config.parser_sidecars.vlm.min_image_pixel or DEFAULT_MM_IMAGE_MIN_PIXEL
            ),
            batch_size=8,
            max_concurrency=config.embedding_func_max_async,
        )

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
        stores: Any,
        embedder: Any | None = None,
    ) -> LightRAGMixBackend:
        """Build the LightRAG retrieval backend from typed DlightRAG config."""
        from dlightrag.core.retrieval.lightrag_backend import LightRAGMixBackend

        return LightRAGMixBackend(
            lightrag=lightrag,
            stores=stores,
            embedder=embedder,
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
                    jitter = random.uniform(0, backoff * 0.5)  # noqa: S311 - non-security jitter
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
        rerank_func = get_rerank_func(config, supports_vision=self._rerank_supports_vision)
        self._rerank_func = rerank_func
        from dlightrag.models.rerank import rerank_consumes_images

        self._rerank_consumes_images = rerank_consumes_images(
            config.rerank, supports_vision=self._rerank_supports_vision
        )
        role_overrides = build_role_llm_configs(config)
        if role_overrides is not None:
            logger.info("LightRAG role overrides: %s", sorted(role_overrides.keys()))

        multimodal_embedder = get_multimodal_embedder(config)
        self._multimodal_embedder = multimodal_embedder
        embedding_func = get_embedding_func(config, embedder=multimodal_embedder)
        self._direct_image_embedding_enabled = await self._resolve_direct_image_embedding_enabled(
            multimodal_embedder,
            startup_probe=config.embedding.startup_probe,
            require_image_support=config.embedding.input_modality == "multimodal",
        )
        document_embedder = self._build_document_embedder(
            config,
            multimodal_embedder,
            image_enabled=self._direct_image_embedding_enabled,
        )
        self._document_embedder = document_embedder

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

        # Wrap chunks_vdb for metadata in-filtering
        if lightrag.chunks_vdb is not None:
            await LightRAGContractGuard(lightrag).verify_all()

            from dlightrag.core.retrieval.filtered_vdb import FilteredVectorStorage

            lightrag.chunks_vdb = FilteredVectorStorage(  # type: ignore[assignment]
                original=lightrag.chunks_vdb,
                embedding_func=embedding_func,
                exact_threshold=config.metadata_filter_exact_vector_threshold,
            )

        from dlightrag.core.lightrag_stores import LightRAGStores

        self._lightrag_stores = LightRAGStores(lightrag)

        from dlightrag.core.visual_assets import ThumbnailCache, VisualAssetResolver

        self._visual_asset_resolver = VisualAssetResolver(
            stores=self._lightrag_stores,
            thumb_cache=ThumbnailCache(max_size=config.visual_assets.thumb_cache_size),
        )

        # Post-init verification: ensure AGE graph labels actually exist.
        # Catches stale/corrupted graphs left by previous failed inits
        # (e.g., schema created but vlabels missing due to uncaught errors).
        await self._verify_graph_labels(lightrag)

        self._backend = self._build_retrieval_backend(
            config,
            lightrag=lightrag,
            stores=self._lightrag_stores,
            embedder=multimodal_embedder if self._direct_image_embedding_enabled else None,
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
            document_embedder=document_embedder,
            workspace=config.workspace,
            parser_rules=config.parser.rules,
            chunk_options=config.parser.chunk_options,
            metadata_registry=self._metadata_registry,
            allow_ad_hoc_metadata=config.metadata.allow_ad_hoc_json,
            default_metadata_policy=config.metadata.default_ingest_policy,
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
            self._warmup_task = asyncio.create_task(self._warmup_lightrag_workers())
        except RuntimeError:
            pass  # no event loop running (e.g. sync test setup)

    async def _warmup_lightrag_workers(self) -> None:
        """Pre-initialize LightRAG worker pools in the background."""
        try:
            from lightrag import QueryParam

            await self._lightrag.aquery(
                "__warmup__", param=QueryParam(mode="naive", enable_rerank=False)
            )
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
                "RAGService not initialized. Use 'await RAGService.acreate()' instead."
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

    async def aclose(self) -> None:
        """Clean up storages and worker pools (best-effort)."""
        # Cancel the background warmup task so it cannot run against
        # half-closed resources during shutdown.
        if self._warmup_task is not None:
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                # Expected: we cancelled the task above and await it only to
                # observe completion during shutdown.
                pass
            except Exception:
                logger.warning("Warmup task raised during shutdown", exc_info=True)
            self._warmup_task = None
        # Shutdown LightRAG worker pools first — they hold background asyncio
        # tasks that block asyncio.run() from exiting.
        await self._shutdown_worker_pools()

        if self._rerank_func is not None and hasattr(self._rerank_func, "aclose"):
            try:
                await self._rerank_func.aclose()
            except Exception:
                logger.warning("Failed to close rerank function", exc_info=True)

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

    async def _purge_existing_download_locator(self, download_locator: str) -> None:
        """Delete only documents owning one exact locator in this workspace."""
        lightrag = self.lightrag
        if lightrag is None or self._metadata_index is None:
            return

        doc_ids = await self._metadata_index.find_by_download_locator(download_locator)
        for doc_id in doc_ids:
            result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            if _normalized_operation_status(result) != "success":
                raise RuntimeError("remote replace cleanup failed")
            await self._metadata_index.delete(doc_id)

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

        file_path = await asyncio.to_thread(
            stage_input_file,
            input_root=self._workspace_input_root(),
            file_path=file_path,
            relative_to=source_root,
        )
        source_uri = local_source_uri(
            self.config.workspace,
            file_path.relative_to(self._workspace_input_root()),
        )
        result = await self._ingestion_engine.aingest_file(
            file_path,
            source_uri=source_uri,
            download_locator=str(file_path),
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
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
            await asyncio.to_thread(
                stage_input_file,
                input_root=self._workspace_input_root(),
                file_path=file_path,
                relative_to=source_root,
            )
            for file_path in file_paths
        ]
        prepared_items = [
            PreparedIngestFile(
                parser_path=staged,
                source_uri=local_source_uri(
                    self.config.workspace,
                    staged.relative_to(self._workspace_input_root()),
                ),
                download_locator=str(staged),
            )
            for staged in staged_paths
        ]
        result = await self._ingestion_engine.aingest_files(
            prepared_items,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        return result

    async def _aingest_local_manifest(
        self,
        documents: list[IngestDocument],
        *,
        replace: bool,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest explicitly listed local files with per-document metadata."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        prepared_items: list[PreparedIngestFile] = []
        for document in documents:
            if document.path is None:
                raise ValueError("local manifest documents require a path")
            file_path = Path(document.path)
            relative_to = self._local_manifest_relative_root(file_path)
            if replace:
                await self._purge_existing_for_replace(
                    file_path=staged_input_path(
                        input_root=self._workspace_input_root(),
                        file_path=file_path,
                        relative_to=relative_to,
                    )
                )
            staged = await asyncio.to_thread(
                stage_input_file,
                input_root=self._workspace_input_root(),
                file_path=file_path,
                relative_to=relative_to,
            )
            prepared_items.append(
                PreparedIngestFile(
                    parser_path=staged,
                    source_uri=local_source_uri(
                        self.config.workspace,
                        staged.relative_to(self._workspace_input_root()),
                    ),
                    download_locator=str(staged),
                    display_filename=document.filename,
                    title=document.title,
                    author=document.author,
                    metadata=document.metadata,
                    metadata_policy=document.metadata_policy,
                )
            )
        result = await self._ingestion_engine.aingest_files(
            prepared_items,
            replace=replace,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        return result

    def _workspace_input_root(self) -> Path:
        return workspace_input_root(self.config.input_dir_path, self.config.workspace)

    def _local_manifest_relative_root(self, file_path: Path) -> Path | None:
        input_root = self._workspace_input_root()
        try:
            file_path.resolve().relative_to(input_root.resolve())
        except ValueError:
            return None
        return input_root

    async def _download_remote_to_prepared_item(
        self,
        *,
        source: AsyncDataSource,
        document: SourceDocument,
        source_uri: str,
        download_locator: str,
        batch_root: Path,
        retain_source_file: bool,
        parser_filename_override: str | None = None,
    ) -> PreparedIngestFile:
        key = parser_filename_override or document.display_filename or document.key
        if retain_source_file:
            parser_path = Path(download_locator)
        else:
            parser_path = remote_parser_input_path(
                batch_root=batch_root,
                source_uri=source_uri,
                key=key,
            )
        parser_path.parent.mkdir(parents=True, exist_ok=True)
        await source.amaterialize_document(document, parser_path)
        return PreparedIngestFile(
            parser_path=parser_path,
            source_uri=source_uri,
            download_locator=download_locator,
            display_filename=document.display_filename or Path(key).name,
            title=document.title,
            author=document.author,
            metadata=document.metadata,
            metadata_policy=document.metadata_policy,
        )

    async def _aingest_remote_documents(
        self,
        *,
        source: AsyncDataSource,
        source_type: str,
        documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument],
        source_uri_for_key: Callable[[str], str],
        download_uri_for_key: Callable[[str], str | None] | None,
        replace: bool,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
        progress_callback: RemoteIngestProgressCallback | None = None,
        resume_from_window: int = 0,
        retain_source_file: bool | None = None,
        parser_filename_override: str | None = None,
    ) -> dict[str, Any]:
        """Download remote objects into ephemeral parser batches and ingest them."""
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")
        from dlightrag.utils.concurrency import bounded_map

        resume_from_window = max(0, int(resume_from_window))
        processed = 0
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        saw_documents = False
        retain_source_files = (
            self.config.retain_remote_source_files
            if retain_source_file is None
            else bool(retain_source_file)
        )

        async for batch_index, window in _aiter_chunks(documents, _REMOTE_INGEST_BATCH_SIZE):
            saw_documents = True
            if batch_index < resume_from_window:
                continue

            batch_root = remote_ingest_batch_root(
                input_root=self._workspace_input_root(),
                source_type=source_type,
                batch_id=f"{batch_index:04d}-{uuid.uuid4().hex}",
            )
            prepared_items: list[PreparedIngestFile] = []
            window_errors: list[str] = []

            async def _download(
                document: SourceDocument,
                *,
                current_batch_root: Path = batch_root,
            ) -> PreparedIngestFile | _RemoteDownloadFailure:
                safe_source_id = _safe_remote_source_id(document)
                try:
                    source_uri = validate_source_uri(
                        document.source_uri or source_uri_for_key(document.key)
                    )
                except asyncio.CancelledError:
                    raise
                except TypeError, ValueError:
                    return _RemoteDownloadFailure(f"{safe_source_id}: source_uri is invalid")
                except Exception:  # noqa: BLE001
                    return _RemoteDownloadFailure(f"{safe_source_id}: source_uri resolution failed")

                try:
                    download_uri = document.download_uri
                    if download_uri is None and download_uri_for_key is not None:
                        download_uri = download_uri_for_key(document.key)

                    if retain_source_files:
                        download_locator = str(
                            retained_remote_source_path(
                                input_root=self._workspace_input_root(),
                                source_type=source_type,
                                source_uri=source_uri,
                                key=document.display_filename or document.key,
                            )
                        )
                        _log_download_locator_outcome(
                            outcome="accepted",
                            locator_kind=_download_locator_kind(None, retained=True),
                            source_filename=safe_source_id,
                        )
                    else:
                        if download_uri is None:
                            _log_download_locator_outcome(
                                outcome="missing",
                                locator_kind=_download_locator_kind(None),
                                source_filename=safe_source_id,
                            )
                            raise SourceDownloadContractError(
                                "retain_source_file=false requires a durable download_uri "
                                f"for source {safe_source_id}; "
                                "provide download_uri/download_uri_for_key or enable "
                                "retain_source_file"
                            )
                        try:
                            download_locator = validate_download_uri(download_uri)
                        except ValueError:
                            _log_download_locator_outcome(
                                outcome="unsupported",
                                locator_kind=_download_locator_kind(download_uri),
                                source_filename=safe_source_id,
                            )
                            raise SourceDownloadContractError(
                                f"invalid durable download_uri for source {safe_source_id}; "
                                "provide a supported durable URI or enable retain_source_file"
                            ) from None
                        _log_download_locator_outcome(
                            outcome="accepted",
                            locator_kind=_download_locator_kind(download_locator),
                            source_filename=safe_source_id,
                        )

                    return await self._download_remote_to_prepared_item(
                        source=source,
                        document=document,
                        source_uri=source_uri,
                        download_locator=download_locator,
                        batch_root=current_batch_root,
                        retain_source_file=retain_source_files,
                        parser_filename_override=parser_filename_override,
                    )
                except SourceDownloadContractError as exc:
                    return _RemoteDownloadFailure(str(exc))
                except Exception:  # noqa: BLE001
                    return _RemoteDownloadFailure(
                        f"{safe_source_id}: remote materialization failed"
                    )

            try:
                download_results = await bounded_map(
                    window,
                    _download,
                    max_concurrent=_REMOTE_DOWNLOAD_CONCURRENCY,
                    task_name=f"{source_type}-download",
                )

                prepared_documents: list[SourceDocument] = []
                for document, downloaded in zip(window, download_results, strict=True):
                    if isinstance(downloaded, _RemoteDownloadFailure):
                        error = downloaded.error
                        errors.append(error)
                        window_errors.append(error)
                        continue
                    if isinstance(downloaded, Exception):
                        error = f"{_safe_remote_source_id(document)}: remote materialization failed"
                        errors.append(error)
                        window_errors.append(error)
                        continue
                    prepared_items.append(downloaded)
                    prepared_documents.append(document)

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
                    for document, prepared_item in zip(
                        prepared_documents, prepared_items, strict=True
                    ):
                        await self._purge_existing_download_locator(prepared_item.download_locator)
                        if not retain_source_files:
                            await self._purge_existing_download_locator(
                                str(
                                    retained_remote_source_path(
                                        input_root=self._workspace_input_root(),
                                        source_type=source_type,
                                        source_uri=prepared_item.source_uri,
                                        key=document.display_filename or document.key,
                                    )
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

        if not saw_documents:
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
        documents: Iterable[SourceDocument] | AsyncIterable[SourceDocument] | None = None,
        prefix: str | None = None,
        source_uri_for_key: Callable[[str], str] | None = None,
        download_uri_for_key: Callable[[str], str | None] | None = None,
        replace: bool | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
        retain_source_file: bool | None = None,
        _progress_callback: RemoteIngestProgressCallback | None = None,
        _resume_from_window: int = 0,
        _parser_filename_override: str | None = None,
    ) -> dict[str, Any]:
        """Ingest documents from a caller-provided async data source.

        SDK connectors expose stable document ids and write bytes into the
        destination path DlightRAG provides. DlightRAG handles temporary parser
        files, metadata provenance, replace semantics, and cleanup.
        """
        self._ensure_initialized()
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        resolved_documents = (
            documents
            if documents is not None
            else cast(
                Iterable[SourceDocument] | AsyncIterable[SourceDocument],
                source.aiter_documents(prefix=prefix),
            )
        )
        uri_for_key = source_uri_for_key or (
            lambda key: self._default_source_uri_for_key(source_type, key)
        )
        close = getattr(source, "aclose", None)
        try:
            return await self._aingest_remote_documents(
                source=source,
                source_type=source_type,
                documents=resolved_documents,
                source_uri_for_key=uri_for_key,
                download_uri_for_key=download_uri_for_key,
                replace=self._resolve_replace(replace),
                title=title,
                author=author,
                metadata=metadata,
                metadata_policy=metadata_policy,
                progress_callback=_progress_callback,
                resume_from_window=_resume_from_window,
                retain_source_file=retain_source_file,
                parser_filename_override=_parser_filename_override,
            )
        finally:
            if close is not None:
                result = close()
                if isawaitable(result):
                    _ = await result

    async def _aingest_url(self, *, replace: bool, **kwargs: Any) -> dict[str, Any]:
        documents = _ingest_documents(kwargs.get("documents"))
        if documents is not None:
            from dlightrag.sourcing.url import URLDataSource

            source = URLDataSource(
                documents=[
                    _source_document_from_manifest(document, key=cast(str, document.url))
                    for document in documents
                ],
                max_download_bytes=self.config.url_ingest_max_bytes,
                allow_private_hosts=self.config.url_ingest_private_host_allowlist,
            )
            result = await self.aingest_source(
                source,
                source_type="url",
                source_uri_for_key=source.source_uri_for_key,
                download_uri_for_key=source.download_uri_for_key,
                replace=replace,
                title=kwargs.get("title"),
                author=kwargs.get("author"),
                metadata=kwargs.get("metadata"),
                metadata_policy=kwargs.get("metadata_policy"),
                retain_source_file=kwargs.get("retain_source_file"),
                _progress_callback=kwargs.get("_progress_callback"),
                _resume_from_window=int(kwargs.get("_resume_from_window") or 0),
                _parser_filename_override=kwargs.get("_parser_filename_override"),
            )
            if len(documents) == 1:
                return self._single_file_result(result)
            return result

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
        if kwargs.get("download_uri") is not None:
            source_kwargs["download_uri"] = kwargs["download_uri"]
        if kwargs.get("download_uris") is not None:
            source_kwargs["download_uris"] = kwargs["download_uris"]
        source = URLDataSource(
            **source_kwargs,
            max_download_bytes=self.config.url_ingest_max_bytes,
            allow_private_hosts=self.config.url_ingest_private_host_allowlist,
        )
        result = await self.aingest_source(
            source,
            source_type="url",
            source_uri_for_key=source.source_uri_for_key,
            download_uri_for_key=source.download_uri_for_key,
            replace=replace,
            title=kwargs.get("title"),
            author=kwargs.get("author"),
            metadata=kwargs.get("metadata"),
            metadata_policy=kwargs.get("metadata_policy"),
            retain_source_file=kwargs.get("retain_source_file"),
            _progress_callback=kwargs.get("_progress_callback"),
            _resume_from_window=int(kwargs.get("_resume_from_window") or 0),
            _parser_filename_override=kwargs.get("_parser_filename_override"),
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

        locator_for_key = lambda key: f"azure://{container_name}/{key}"  # noqa: E731
        common_kwargs = {
            "source_type": "azure_blob",
            "source_uri_for_key": locator_for_key,
            "download_uri_for_key": locator_for_key,
            "replace": replace,
            "title": kwargs.get("title"),
            "author": kwargs.get("author"),
            "metadata": kwargs.get("metadata"),
            "metadata_policy": kwargs.get("metadata_policy"),
            "retain_source_file": kwargs.get("retain_source_file"),
            "_progress_callback": kwargs.get("_progress_callback"),
            "_resume_from_window": int(kwargs.get("_resume_from_window") or 0),
            "_parser_filename_override": kwargs.get("_parser_filename_override"),
        }
        documents = _ingest_documents(kwargs.get("documents"))
        if documents is not None:
            return await self.aingest_source(
                source,
                documents=[
                    _source_document_from_manifest(document, key=cast(str, document.key))
                    for document in documents
                ],
                **common_kwargs,
            )
        if kwargs.get("blob_path"):
            return self._single_file_result(
                await self.aingest_source(
                    source,
                    documents=[SourceDocument(key=str(kwargs["blob_path"]))],
                    **common_kwargs,
                )
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
                bucket=str(bucket), region=kwargs.get("s3_region") or self.config.s3_region
            )

        locator_for_key = lambda key: f"s3://{bucket}/{key}"  # noqa: E731
        common_kwargs = {
            "source_type": "s3",
            "source_uri_for_key": locator_for_key,
            "download_uri_for_key": locator_for_key,
            "replace": replace,
            "title": kwargs.get("title"),
            "author": kwargs.get("author"),
            "metadata": kwargs.get("metadata"),
            "metadata_policy": kwargs.get("metadata_policy"),
            "retain_source_file": kwargs.get("retain_source_file"),
            "_progress_callback": kwargs.get("_progress_callback"),
            "_resume_from_window": int(kwargs.get("_resume_from_window") or 0),
            "_parser_filename_override": kwargs.get("_parser_filename_override"),
        }
        documents = _ingest_documents(kwargs.get("documents"))
        if documents is not None:
            return await self.aingest_source(
                source,
                documents=[
                    _source_document_from_manifest(document, key=cast(str, document.key))
                    for document in documents
                ],
                **common_kwargs,
            )
        key = kwargs.get("s3_key") or kwargs.get("blob_path")
        if key:
            return self._single_file_result(
                await self.aingest_source(
                    source,
                    documents=[SourceDocument(key=str(key))],
                    **common_kwargs,
                )
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
        replace = self._resolve_replace(kwargs.pop("replace", None))

        if self._ingestion_engine is not None and source_type == "local":
            documents = _ingest_documents(kwargs.get("documents"))
            if documents is not None:
                return await self._aingest_local_manifest(
                    documents,
                    replace=replace,
                    title=kwargs.get("title"),
                    author=kwargs.get("author"),
                    metadata=kwargs.get("metadata"),
                    metadata_policy=kwargs.get("metadata_policy"),
                )
            path_str = kwargs.get("path")
            if not path_str:
                raise ValueError("'path' is required for local source_type")
            local_path = Path(path_str)
            file_paths = await asyncio.to_thread(iter_ingestable_files, local_path)
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
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        is_reretrieve: bool = False,
        filters: MetadataFilter | None = None,
        *,
        _plan: Any = None,
        bm25_query: str | None = None,
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

        effective_bm25_query = (bm25_query or "").strip() or None
        if effective_bm25_query is None and _plan is not None:
            effective_bm25_query = getattr(_plan, "bm25_query", None)

        kg_result = await self._retrieval_orchestrator.aretrieve(
            query,
            metadata_filter=effective_filters,
            metadata_filter_source=filter_source,
            bm25_query=effective_bm25_query,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            is_reretrieve=is_reretrieve,
            **kwargs,
        )

        # --- Step 2.5: Hydrate image data for all retrieved chunks ---
        # BM25 chunks and other post-fusion additions haven't been through
        # hydration yet, so they lack image_data for visual chunks.
        stores = self._lightrag_stores
        if stores is not None:
            from dlightrag.core.retrieval.provenance import hydrate_lightrag_chunk_provenance

            chunks_to_hydrate = kg_result.contexts.get("chunks", [])
            if chunks_to_hydrate:
                hydrated_ids = {
                    str(chunk_id)
                    for chunk_id in kg_result.trace.get("provenance_hydrated_chunk_ids", [])
                }
                pending_chunks = [
                    chunk
                    for chunk in chunks_to_hydrate
                    if str(chunk.get("chunk_id") or "") not in hydrated_ids
                ]
                if pending_chunks:
                    # Defer the expensive image base64 read past rerank truncation
                    # for a text-only reranker, so chunks that rerank drops never
                    # read their image bytes.
                    await hydrate_lightrag_chunk_provenance(
                        stores,
                        pending_chunks,
                        include_image_data=self._rerank_consumes_images,
                    )

        await self._rerank_retrieval_chunks(
            query,
            kg_result,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
        )

        # Text-reranker path: image bytes were deferred above, so hydrate them for
        # the surviving chunks now (already-hydrated chunks skip the read).
        if stores is not None and not self._rerank_consumes_images:
            survivors = kg_result.contexts.get("chunks", [])
            if survivors:
                from dlightrag.core.retrieval.provenance import (
                    hydrate_lightrag_chunk_provenance,
                )

                await hydrate_lightrag_chunk_provenance(stores, survivors)

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

    async def _rerank_retrieval_chunks(
        self,
        query: str,
        result: RetrievalResult,
        *,
        top_k: int | None,
        chunk_top_k: int | None,
    ) -> None:
        chunks = result.contexts.get("chunks", [])
        result.trace.setdefault("reranked_chunk_count", 0)
        if not chunks:
            logger.info(
                "[Rerank] skipped: strategy=%s enabled=%s chunks=0 reason=no_chunks",
                self.config.rerank.strategy,
                self.config.rerank.enabled,
            )
            return

        limit = chunk_top_k or top_k or len(chunks)
        from dlightrag.core.retrieval.rerank import rerank_with_fallback

        outcome = await rerank_with_fallback(
            query=query,
            chunks=chunks,
            top_k=limit,
            rerank_func=self._rerank_func,
        )
        result.contexts["chunks"] = outcome.chunks
        result.trace["reranked_chunk_count"] = len(outcome.chunks)
        if self._rerank_func is None:
            logger.info(
                "[Rerank] skipped: strategy=%s enabled=%s chunks=%d limit=%d reason=no_function",
                self.config.rerank.strategy,
                self.config.rerank.enabled,
                len(chunks),
                limit,
            )
            return
        if outcome.error_type:
            result.trace["rerank_error"] = outcome.error_type
            logger.warning(
                "Rerank failed; returning top fused chunks: error_type=%s",
                outcome.error_type,
            )
            return
        top_score = outcome.chunks[0].get("rerank_score") if outcome.chunks else None
        logger.info(
            "[Rerank] strategy=%s model=%s input_chunks=%d limit=%d output_chunks=%d top_score=%s",
            self.config.rerank.strategy,
            self.config.rerank.model,
            len(chunks),
            limit,
            result.trace["reranked_chunk_count"],
            top_score,
        )

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

        Looks up each chunk's LightRAG full_doc_id in the metadata index and merges
        any non-empty fields into the chunk's metadata dict. Fields are
        dynamic: whatever the metadata index returns is included, minus
        internal/system fields that add no value to the LLM context.
        """
        _SKIP = frozenset(
            {
                "workspace",
                "doc_id",
                "file_path",
                "source_uri",
                "download_locator",
                "file_extension",
                "filename",
                "filename_stem",
                "ingested_at",
                "custom_metadata",
                "ingest_strategy",
                "parse_engine",
                "metadata_json",
                "process_options",
                "page_count",
                "original_format",
                "doc_title",
                "doc_author",
            }
        )

        chunks = result.contexts.get("chunks", [])
        if not chunks:
            return

        # Collect unique LightRAG full_doc_id values. Retrieval adapters must
        # propagate this identity; file_path lookup is for deletion/cleanup, not
        # answer-time enrichment.
        doc_meta: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = chunk.get("full_doc_id") or chunk.get("_full_doc_id")
            if doc_id:
                doc_meta.setdefault(str(doc_id), {})

        idx = self._metadata_index
        if idx is None or not doc_meta:
            return

        try:
            fetched = await idx.get_many(list(doc_meta))
        except Exception:
            fetched = {}  # enrichment is best-effort

        for doc_id, meta in fetched.items():
            fetched_meta = {
                k: v for k, v in meta.items() if k not in _SKIP and v is not None and v != ""
            }
            source_uri = meta.get("source_uri")
            download_locator = meta.get("download_locator")
            if isinstance(source_uri, str) and source_uri:
                fetched_meta["source_uri"] = source_uri
            if isinstance(download_locator, str) and download_locator:
                fetched_meta["source_download_locator"] = download_locator
            display_name = meta.get("filename")
            if isinstance(display_name, str) and display_name:
                fetched_meta["source_file_name"] = display_name
            if fetched_meta:
                doc_meta[doc_id] = fetched_meta

        # Inject into each chunk's metadata field
        for chunk in chunks:
            doc_id = chunk.get("full_doc_id") or chunk.get("_full_doc_id")
            fetched_meta = doc_meta.get(str(doc_id)) if doc_id else None
            if fetched_meta:
                existing = chunk.get("metadata") or {}
                existing.update(fetched_meta)
                chunk["metadata"] = existing

    # === METADATA API ===

    async def aget_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get document metadata by ID."""
        result = await self._metadata_index.get(doc_id)  # type: ignore[union-attr]
        if not result:
            return {}
        internal_fields = frozenset({"workspace", "doc_id", "file_path", "download_locator"})
        return {key: value for key, value in result.items() if key not in internal_fields}

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
        if normalized_filters is None:
            return []
        return await self._metadata_index.query(normalized_filters)

    def _normalize_metadata_filter(self, filters: MetadataFilter | None) -> MetadataFilter | None:
        if filters is None or self._metadata_registry is None:
            return filters
        return self._metadata_registry.normalize_filter(filters)

    # === FILE MANAGEMENT API ===

    async def alist_ingested_files(self) -> list[dict[str, Any]]:
        """List all processed files from LightRAG doc_status."""
        self._ensure_initialized()
        if self._lightrag_stores is None:
            return []

        from lightrag.base import DocStatus

        try:
            processed = await self._lightrag_stores.docs_by_status(DocStatus.PROCESSED)
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
        if self._lightrag_stores is None:
            return []

        from lightrag.base import DocStatus

        try:
            failed = await self._lightrag_stores.docs_by_status(DocStatus.FAILED)
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
        """Re-ingest every FAILED document from its durable metadata contract."""

        failed = await self.alist_failed_docs()
        if not failed:
            return {"retried": 0, "succeeded": 0, "failed": 0, "results": []}

        lr = self.lightrag
        succeeded: list[dict[str, Any]] = []
        still_failed: list[dict[str, Any]] = []

        for entry in failed:
            doc_id = entry["doc_id"]
            if self._metadata_index is None:
                still_failed.append({"doc_id": doc_id, "reason": "source metadata unavailable"})
                continue

            try:
                metadata = await self._metadata_index.get(doc_id)
            except Exception:
                logger.warning("Failed to load retry metadata for doc_id=%s", doc_id)
                still_failed.append({"doc_id": doc_id, "reason": "source metadata unavailable"})
                continue

            source_uri = metadata.get("source_uri") if metadata else None
            download_locator = metadata.get("download_locator") if metadata else None
            stored_filename = metadata.get("filename") if metadata else None
            if not isinstance(source_uri, str) or not source_uri:
                still_failed.append({"doc_id": doc_id, "reason": "source metadata incomplete"})
                continue
            if not isinstance(download_locator, str) or not download_locator:
                still_failed.append({"doc_id": doc_id, "reason": "source metadata incomplete"})
                continue
            if not isinstance(stored_filename, str) or not stored_filename:
                still_failed.append({"doc_id": doc_id, "reason": "source metadata incomplete"})
                continue

            try:
                display_filename = _retry_display_filename(stored_filename)
                self._validate_retry_source_contract(source_uri, download_locator)
            except OSError, TypeError, ValueError:
                still_failed.append({"doc_id": doc_id, "reason": "source metadata invalid"})
                continue

            try:
                result = await self._aingest_download_locator(
                    source_uri, download_locator, display_filename
                )
                processed = result.get("processed")
                if result.get("errors") or (isinstance(processed, int | float) and processed < 1):
                    still_failed.append({"doc_id": doc_id, "reason": "retry ingestion failed"})
                    continue
                replacement_doc_ids = self._retry_result_doc_ids(result)
                if not replacement_doc_ids:
                    still_failed.append({"doc_id": doc_id, "reason": "retry ingestion failed"})
                    continue
                if doc_id not in replacement_doc_ids:
                    if lr is None:
                        await self._rollback_retry_replacements(
                            replacement_doc_ids, original_doc_id=doc_id
                        )
                        raise RuntimeError("old LightRAG document cleanup unavailable")
                    try:
                        deletion_result = await lr.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                    except Exception:
                        await self._rollback_retry_replacements(
                            replacement_doc_ids, original_doc_id=doc_id
                        )
                        raise
                    if _normalized_operation_status(deletion_result) != "success":
                        await self._rollback_retry_replacements(
                            replacement_doc_ids, original_doc_id=doc_id
                        )
                        raise RuntimeError("old LightRAG document cleanup failed")
                    try:
                        await self._metadata_index.delete(doc_id)
                    except Exception:
                        logger.warning(
                            "Old retry metadata cleanup incomplete for doc_id=%s", doc_id
                        )
                succeeded.append(
                    {"doc_id": doc_id, "file_path": entry.get("file_path", ""), "result": result}
                )
            except Exception:
                logger.warning("Retry failed for doc_id=%s", doc_id)
                still_failed.append(
                    {
                        "doc_id": doc_id,
                        "file_path": entry.get("file_path", ""),
                        "reason": "retry ingestion failed",
                    }
                )

        return {
            "retried": len(failed),
            "succeeded": len(succeeded),
            "failed": len(still_failed),
            "succeeded_docs": succeeded,
            "failed_docs": still_failed,
        }

    @staticmethod
    def _retry_result_doc_ids(result: Mapping[str, Any]) -> set[str]:
        doc_ids: set[str] = set()
        direct_doc_id = result.get("doc_id")
        if isinstance(direct_doc_id, str) and direct_doc_id:
            doc_ids.add(direct_doc_id)

        nested_results = result.get("results")
        if isinstance(nested_results, list):
            for item in nested_results:
                if not isinstance(item, Mapping):
                    continue
                nested_doc_id = item.get("doc_id")
                if isinstance(nested_doc_id, str) and nested_doc_id:
                    doc_ids.add(nested_doc_id)
        return doc_ids

    async def _rollback_retry_replacements(
        self,
        replacement_doc_ids: set[str],
        *,
        original_doc_id: str,
    ) -> None:
        """Best-effort compensation when the original FAILED row still exists."""
        lr = self.lightrag
        if lr is None or self._metadata_index is None:
            return
        for replacement_doc_id in sorted(replacement_doc_ids - {original_doc_id}):
            try:
                result = await lr.adelete_by_doc_id(replacement_doc_id, delete_llm_cache=True)
            except Exception:
                logger.warning(
                    "Retry replacement rollback failed for doc_id=%s", replacement_doc_id
                )
                continue
            if _normalized_operation_status(result) != "success":
                logger.warning(
                    "Retry replacement rollback was not acknowledged for doc_id=%s",
                    replacement_doc_id,
                )
                continue
            try:
                await self._metadata_index.delete(replacement_doc_id)
            except Exception:
                logger.warning(
                    "Retry replacement metadata rollback failed for doc_id=%s",
                    replacement_doc_id,
                )

    @staticmethod
    def _validate_retry_source_contract(
        source_uri: str,
        download_locator: str,
    ) -> tuple[SourceType, dict[str, Any]]:
        from dlightrag.sourcing.uri import parse_remote_uri

        stable_source_uri = validate_source_uri(source_uri)
        if not download_locator or "\x00" in download_locator:
            raise ValueError("download locator is invalid")
        source_type, parts = parse_remote_uri(download_locator)
        if source_type == "local":
            if not Path(download_locator).is_file():
                raise FileNotFoundError("download locator is unavailable")
        else:
            validate_download_uri(download_locator)
        if not stable_source_uri:
            raise ValueError("source identity is invalid")
        return source_type, parts

    async def _aingest_download_locator(
        self,
        source_uri: str,
        download_locator: str,
        display_filename: str,
    ) -> dict[str, Any]:
        """Materialize one validated locator while preserving source provenance."""
        source_type, parts = self._validate_retry_source_contract(source_uri, download_locator)
        stable_source_uri = validate_source_uri(source_uri)
        parser_filename = _retry_parser_filename(display_filename)

        if source_type == "local":
            return await self._aingest_local_retry_locator(
                source_uri=stable_source_uri,
                download_locator=download_locator,
                display_filename=display_filename,
                parser_filename=parser_filename,
            )

        if source_type == "url":
            document = IngestDocument(
                url=download_locator,
                filename=display_filename,
                source_uri=stable_source_uri,
                download_uri=download_locator,
            )
            return await self.aingest(
                "url",
                documents=[document],
                replace=False,
                retain_source_file=False,
                _parser_filename_override=parser_filename,
            )

        if source_type == "s3":
            document = IngestDocument(
                key=str(parts["key"]),
                filename=display_filename,
                source_uri=stable_source_uri,
            )
            return await self.aingest(
                "s3",
                bucket=str(parts["bucket"]),
                documents=[document],
                replace=False,
                retain_source_file=False,
                _parser_filename_override=parser_filename,
            )

        document = IngestDocument(
            key=str(parts["blob_path"]),
            filename=display_filename,
            source_uri=stable_source_uri,
        )
        return await self.aingest(
            "azure_blob",
            container_name=str(parts["container_name"]),
            documents=[document],
            replace=False,
            retain_source_file=False,
            _parser_filename_override=parser_filename,
        )

    async def _aingest_local_retry_locator(
        self,
        *,
        source_uri: str,
        download_locator: str,
        display_filename: str,
        parser_filename: str,
    ) -> dict[str, Any]:
        if self._ingestion_engine is None:
            raise RuntimeError("Ingestion engine not initialized")

        input_root = self._workspace_input_root()
        batch_root = remote_ingest_batch_root(
            input_root=input_root,
            source_type="retry",
            batch_id=uuid.uuid4().hex,
        )
        parser_path = remote_parser_input_path(
            batch_root=batch_root,
            source_uri=source_uri,
            key=parser_filename,
        )
        item = PreparedIngestFile(
            parser_path=parser_path,
            source_uri=source_uri,
            download_locator=download_locator,
            display_filename=display_filename,
        )
        parser_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(shutil.copy2, Path(download_locator), parser_path)
            result = await self._ingestion_engine.aingest_files([item], replace=False)
            return self._single_file_result(result)
        finally:
            await asyncio.to_thread(_remove_remote_parser_sources, [item])
            await asyncio.to_thread(_remove_empty_parents, batch_root, input_root)

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


__all__ = ["ComposerProcessingResources", "RAGService"]
