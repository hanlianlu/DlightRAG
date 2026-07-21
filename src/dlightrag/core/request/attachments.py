# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web-only query attachment parsing, chunking, budgeting and retrieval.

This module is CORE-layer: it must never import from ``dlightrag.web`` /
``api`` / ``mcp``. Its inputs are primitives or core-local dataclasses so the
Web transport layer adapts its own ``ValidatedWebDocument`` into these types.

The parse path REUSES the real LightRAG/MinerU parser stack via an ephemeral
"parse-owner shim" (:class:`_ParseOwnerShim`) that neutralises the single
workspace-write hook (``_persist_parsed_full_docs``) and confines all scratch
files to a temporary directory. Nothing is written to any workspace store; the
resulting chunks are materialised into a separate content-addressed PG parse
cache owned by the injected store.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from dlightrag.core.document_embedding import (
    DocumentEmbeddingFallbackReason,
    DocumentEmbeddingInput,
    DocumentEmbeddingTrace,
    DocumentEmbeddingVector,
)
from dlightrag.core.request.composer_analysis import (
    ComposerAnalysisOutcome,
    ComposerAnalysisResult,
    aanalyze_composer_sidecars,
    build_composer_analysis_signature,
)
from dlightrag.core.request.composer_evidence import COMPOSER_FULL_PASS_TOKENS
from dlightrag.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)

_ATTACHMENT_WORKSPACE = "__web_attachment__"
_COMPOSER_EMBEDDING_CONTRACT_VERSION = 2
COMPOSER_DENSE_SCORE_THRESHOLD = 0.5
COMPOSER_DENSE_BLOCK_SIZE = 256
COMPOSER_DENSE_PAGE_SIZE = 256


class DocumentEmbedderProtocol(Protocol):
    """Borrowed robust document embedder surface used by Composer."""

    @property
    def image_enabled(self) -> bool: ...

    @property
    def dimension(self) -> int: ...

    @property
    def asymmetric(self) -> bool: ...

    @property
    def min_image_pixel(self) -> int: ...

    async def aembed_documents(
        self,
        items: list[DocumentEmbeddingInput],
    ) -> tuple[list[DocumentEmbeddingVector], DocumentEmbeddingTrace]: ...

    async def aembed_query(self, query: str) -> list[float] | None: ...


class _TemporaryDirectoryProtocol(Protocol):
    name: str

    def cleanup(self) -> None: ...


async def _await_thread_call[Result](operation: Callable[[], Result]) -> Result:
    """Finish a resource operation in a worker before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and not task.cancelled():
            try:
                task.result()
            except BaseException:
                logger.debug(
                    "Threaded attachment resource operation failed during cancellation",
                    exc_info=True,
                )
        raise cancelled


@dataclass(frozen=True, slots=True)
class AttachmentCacheKey:
    """Stable conversation-cache identity, distinct from citation identity."""

    content_sha256: str
    parser_signature: str
    chunk_signature: str
    cache_chunk_id: str


@dataclass(frozen=True, slots=True)
class AttachmentVectorPageRow:
    """One TTL-scoped cached vector row before core validation."""

    global_order: int
    cache_key: AttachmentCacheKey
    embedding_signature: str | None
    embedding_vector: Sequence[object] | None


def validate_attachment_vector(
    row: AttachmentVectorPageRow,
    *,
    expected_signature: str,
    expected_dimension: int,
) -> NDArray[np.float32] | None:
    """Return a valid float32 document vector, or treat the row as a cache miss."""
    if row.embedding_signature != expected_signature or expected_dimension <= 0:
        return None
    try:
        vector = np.asarray(row.embedding_vector, dtype=np.float32)
    except TypeError, ValueError, OverflowError:
        return None
    if vector.ndim != 1 or vector.shape[0] != expected_dimension:
        return None
    if not bool(np.isfinite(vector).all()):
        return None
    norm = np.linalg.norm(vector)
    if not bool(np.isfinite(norm)) or float(norm) == 0.0:
        return None
    return vector


def _validated_attachment_vector_page_indices(
    page: Sequence[AttachmentVectorPageRow],
    *,
    expected_by_order: dict[int, tuple[AttachmentCacheKey, frozenset[str]]],
    expected_dimension: int,
) -> dict[int, str]:
    """Validate one decoded page and retain only cache-hit ordinals/signatures."""
    valid_signatures: dict[int, str] = {}
    for row in page:
        expected = expected_by_order.get(row.global_order)
        if expected is None or row.cache_key != expected[0]:
            continue
        signature = row.embedding_signature
        if signature is None or signature not in expected[1]:
            continue
        if (
            validate_attachment_vector(
                row,
                expected_signature=signature,
                expected_dimension=expected_dimension,
            )
            is not None
        ):
            valid_signatures[row.global_order] = signature
    return valid_signatures


@dataclass(frozen=True, slots=True)
class AttachmentContextChunk:
    """One parsed, retrievable unit of a query-time document attachment.

    Text and visual chunks share this shape; visual chunks additionally carry
    ``image_bytes`` / ``image_mime_type``. This is the only chunk type the
    downstream Web answer path sees.
    """

    chunk_id: str
    attachment_id: str
    filename: str
    chunk_index: int
    content: str
    token_estimate: int = 0
    page_idx: int | None = None
    bbox: Any = None
    sidecar_type: str | None = None
    image_bytes: bytes | None = None
    image_mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: AttachmentCacheKey | None = None
    embedding_signature: str | None = None
    embedding_vector: list[float] | None = None

    def to_context_row(self) -> dict[str, Any]:
        """Project into the shared answer-context row shape.

        Emits image bytes as-is; the single ``AnswerContextPacker`` +
        ``AnswerImageBudget`` is the only owner of image budgeting across
        workspace and attachment chunks, so no pre-budget flag is set here.
        """
        source_uri = f"web-attachment://{self.attachment_id}"
        metadata = {
            **self.metadata,
            "source_type": "web_attachment",
            "source_uri": source_uri,
            "source_download_locator": source_uri,
        }
        row: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "reference_id": self.attachment_id,
            # full_doc_id lets build_sources set document_id = attachment_id so
            # the Web layer can project a conversation-scoped download URL.
            "full_doc_id": self.attachment_id,
            "file_path": self.filename,
            "content": self.content,
            "page_idx": self.page_idx,
            "bbox": self.bbox,
            "_workspace": _ATTACHMENT_WORKSPACE,
            "metadata": metadata,
        }
        if self.image_bytes is not None:
            row["image_data"] = base64.b64encode(self.image_bytes).decode("ascii")
            row["image_mime_type"] = self.image_mime_type or "image/png"
        if self.cache_key is not None:
            row["_cache_key"] = self.cache_key
        return row


@dataclass(frozen=True, slots=True)
class ParsedAttachmentBundle:
    """An ordered set of parsed chunks tied to the parser that produced them."""

    chunks: list[AttachmentContextChunk]
    parser_signature: str = ""
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedAttachmentDocument:
    """Raw parser output for one attachment, before chunking."""

    content: str
    blocks_path: str
    parser_signature: str


AttachmentWriteStatus = Literal["not_attempted", "written", "skipped", "error"]


@dataclass(slots=True)
class AttachmentProcessingTrace:
    """Mutable per-attachment trace builder with stable transport projection."""

    parse_cache_hit: bool = False
    analysis_outcome: str = ComposerAnalysisOutcome.DEGRADED.value
    analysis_error: str | None = None
    mm_chunk_count: int = 0
    vector_cache_hits: int = 0
    vector_cache_misses: int = 0
    cache_materialized: bool = False
    cache_error: str | None = None
    cache_write_status: AttachmentWriteStatus = "not_attempted"
    vector_update_status: AttachmentWriteStatus = "not_attempted"
    embedding_fused: int = 0
    embedding_text: int = 0
    embedding_fallback: int = 0
    embedding_failed: int = 0
    embedding_error: str | None = None
    parse_error: str | None = None
    directive_error: str | None = None
    parser_error: str | None = None
    rendering_error: str | None = None
    cache_read_error: str | None = None
    vector_read_error: str | None = None
    cache_write_error: str | None = None
    vector_update_error: str | None = None

    def preserve_cache_error(self, error_type: str) -> None:
        if self.cache_error is None:
            self.cache_error = error_type

    def record_embedding(self, trace: DocumentEmbeddingTrace) -> None:
        self.embedding_fused = trace.fused
        self.embedding_text = trace.text
        self.embedding_fallback = trace.fused_to_text_fallback
        self.embedding_failed = trace.failed
        self.embedding_error = trace.error_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "attachment_parse_cache_hit": self.parse_cache_hit,
            "attachment_analysis_outcome": self.analysis_outcome,
            "attachment_analysis_error": self.analysis_error,
            "attachment_mm_chunk_count": self.mm_chunk_count,
            "attachment_vector_cache_hits": self.vector_cache_hits,
            "attachment_vector_cache_misses": self.vector_cache_misses,
            "attachment_cache_materialized": self.cache_materialized,
            "attachment_cache_error": self.cache_error,
            "attachment_cache_write_status": self.cache_write_status,
            "attachment_vector_update_status": self.vector_update_status,
            "attachment_embedding_fused": self.embedding_fused,
            "attachment_embedding_text": self.embedding_text,
            "attachment_embedding_fallback": self.embedding_fallback,
            "attachment_embedding_failed": self.embedding_failed,
            "attachment_embedding_error": self.embedding_error,
            "attachment_parse_error": self.parse_error,
            "attachment_directive_error": self.directive_error,
            "attachment_parser_error": self.parser_error,
            "attachment_rendering_error": self.rendering_error,
            "attachment_cache_read_error": self.cache_read_error,
            "attachment_vector_read_error": self.vector_read_error,
            "attachment_cache_write_error": self.cache_write_error,
            "attachment_vector_update_error": self.vector_update_error,
        }


@dataclass(frozen=True, slots=True)
class _AttachmentResolution:
    process_options: str
    parser_signature: str
    chunk_signature: str


class AttachmentCacheStoreProtocol(Protocol):
    """Conversation-scoped cache operations required by the core service."""

    async def load_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
        ttl_days: int,
    ) -> ParsedAttachmentBundle | None: ...

    async def materialize_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
        bundle: ParsedAttachmentBundle,
        ttl_days: int,
    ) -> bool: ...

    def aiter_attachment_vectors(
        self,
        principal_id: str,
        conversation_id: str,
        references: list[tuple[int, AttachmentCacheKey]],
        *,
        ttl_days: int,
        page_size: int,
    ) -> AsyncIterator[list[AttachmentVectorPageRow]]: ...

    async def aupdate_attachment_chunk_vectors(
        self,
        principal_id: str,
        conversation_id: str,
        chunks: list[AttachmentContextChunk],
        *,
        ttl_days: int,
    ) -> bool: ...


def build_text_attachment_chunk(
    *,
    attachment_id: str,
    filename: str,
    chunk_id: str,
    chunk_index: int,
    content: str,
    page_idx: int | None = None,
    bbox: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AttachmentContextChunk:
    return AttachmentContextChunk(
        chunk_id=chunk_id,
        attachment_id=attachment_id,
        filename=filename,
        chunk_index=chunk_index,
        content=content,
        token_estimate=estimate_tokens(content),
        page_idx=page_idx,
        bbox=bbox,
        metadata=dict(metadata or {}),
    )


class _ParseOwnerShim:
    """Minimal ``ParseContext.rag`` owner: reuse parsers, persist nothing.

    Deliberately holds no storage handles. The two hooks the LightRAG parser
    layer calls on its owner during ``parse`` are:

    * ``_resolve_source_file_for_parser`` — returns the temp source path.
    * ``_persist_parsed_full_docs`` — the single workspace-write call, here a
      no-op that only records the sidecar location for later chunk building.

    Signatures MUST stay call-compatible with LightRAG's owner contract; the
    parser-coupling contract test pins them so an upstream rename fails loudly.
    """

    def __init__(self) -> None:
        self.sidecar_location: str | None = None
        self.workspace = ""

    def _resolve_source_file_for_parser(
        self,
        file_path: str,
        *,
        source_file: str | None = None,
        parser_engine: str | None = None,
    ) -> str:
        return file_path

    async def _persist_parsed_full_docs(self, doc_id: str, record: dict[str, Any]) -> None:
        # No workspace write: only remember where the parser stashed sidecars so
        # the caller can materialise visual chunks before the temp dir is gone.
        self.sidecar_location = record.get("sidecar_location")


def resolve_attachment_parser_signature(filename: str, parser_rules: str) -> str:
    """Stable identity of the parser engine + process options for ``filename``."""
    from lightrag.parser.routing import (
        encode_parse_engine,
        resolve_parser_directives,
    )

    directives = resolve_parser_directives(
        Path(filename), parser_rules=parser_rules, require_external_endpoint=False
    )
    engine = encode_parse_engine(directives.engine, directives.engine_params)
    return f"{engine}:{directives.process_options}"


def resolve_attachment_chunk_signature(
    lightrag: Any,
    filename: str,
    parser_rules: str,
    *,
    analysis_signature: str = "",
) -> str:
    """Stable identity of the chunking configuration for ``filename``."""
    from lightrag.parser.routing import (
        resolve_chunk_options,
        resolve_parser_directives,
    )

    directives = resolve_parser_directives(
        Path(filename), parser_rules=parser_rules, require_external_endpoint=False
    )
    chunk_opts = resolve_chunk_options(
        getattr(lightrag, "addon_params", None),
        process_options=directives.process_options,
    )
    tokenizer_name = type(getattr(lightrag, "tokenizer", object())).__name__
    return json.dumps(
        {
            "process_options": directives.process_options,
            "chunk_opts": chunk_opts,
            "tokenizer": tokenizer_name,
            "analysis_signature": analysis_signature,
        },
        sort_keys=True,
        default=str,
    )


def build_composer_embedding_signature(
    *,
    config: DlightragConfig,
    embedder: DocumentEmbedderProtocol,
    mode: Literal["fused", "text"],
    fallback_reason: DocumentEmbeddingFallbackReason | None = None,
) -> str:
    """Return a deterministic non-secret signature for one effective vector mode."""
    from dlightrag.models.composer import normalized_endpoint_fingerprint

    payload = {
        "contract_version": _COMPOSER_EMBEDDING_CONTRACT_VERSION,
        "provider": config.embedding.provider,
        "model": config.embedding.model,
        "endpoint": normalized_endpoint_fingerprint(config.embedding.base_url),
        "dimension": embedder.dimension,
        "asymmetric": getattr(embedder, "asymmetric", config.embedding.asymmetric),
        "mode": mode,
        "fallback_reason": fallback_reason,
        "image_normalization": {
            "min_image_pixel": getattr(
                embedder,
                "min_image_pixel",
                config.parser_sidecars.vlm.min_image_pixel,
            ),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@asynccontextmanager
async def parse_attachment_document(
    *,
    attachment_id: str,
    filename: str,
    document_bytes: bytes,
    parser_rules: str,
) -> AsyncIterator[tuple[ParsedAttachmentDocument, str, str]]:
    """Yield parser output while its temporary source and sidecars remain alive."""
    from lightrag.parser.base import ParseContext
    from lightrag.parser.registry import get_parser
    from lightrag.parser.routing import (
        encode_parse_engine,
        resolve_parser_directives,
    )

    directives = resolve_parser_directives(
        Path(filename), parser_rules=parser_rules, require_external_endpoint=False
    )
    engine = directives.engine
    parse_engine = encode_parse_engine(engine, directives.engine_params)
    parser_signature = f"{parse_engine}:{directives.process_options}"

    parser = get_parser(engine)
    if parser is None:
        raise ValueError(f"no parser registered for engine {engine!r}")

    temporary_directory: _TemporaryDirectoryProtocol = tempfile.TemporaryDirectory(
        prefix="dlightrag-attach-"
    )
    try:
        source = Path(temporary_directory.name) / Path(filename).name
        await _await_thread_call(lambda: source.write_bytes(document_bytes))
        shim = _ParseOwnerShim()
        ctx = ParseContext(
            rag=shim,
            doc_id=f"att-{attachment_id}",
            file_path=str(source),
            content_data={
                "source_file": Path(filename).name,
                "parse_engine": parse_engine,
                "process_options": directives.process_options,
            },
        )
        result = await parser.parse(ctx)
        parsed = ParsedAttachmentDocument(
            content=result.content,
            blocks_path=result.blocks_path,
            parser_signature=parser_signature,
        )
        yield parsed, directives.process_options, parser_signature
    finally:
        await _await_thread_call(temporary_directory.cleanup)


async def parse_attachment_to_bundle(
    *,
    lightrag: Any,
    attachment_id: str,
    filename: str,
    document_bytes: bytes,
    parser_rules: str,
) -> ParsedAttachmentBundle:
    """Parse and chunk one attachment while all parser sidecars remain alive."""
    async with parse_attachment_document(
        attachment_id=attachment_id,
        filename=filename,
        document_bytes=document_bytes,
        parser_rules=parser_rules,
    ) as (parsed, process_options, _parser_signature):
        return await build_attachment_bundle_from_parse_result(
            lightrag=lightrag,
            attachment_id=attachment_id,
            filename=filename,
            parsed=parsed,
            process_options=process_options,
        )


async def build_attachment_bundle_from_parse_result(
    *,
    lightrag: Any,
    attachment_id: str,
    filename: str,
    parsed: ParsedAttachmentDocument,
    process_options: str,
    include_multimodal: bool = True,
) -> ParsedAttachmentBundle:
    """Chunk parser output with the SAME LightRAG chunkers durable ingest uses."""
    from lightrag.chunker import (
        chunking_by_fixed_token,
        chunking_by_paragraph_semantic,
        chunking_by_recursive_character,
        chunking_by_semantic_vector,
    )
    from lightrag.parser.routing import parse_process_options, resolve_chunk_options
    from lightrag.sidecar import backfill_chunk_sidecars
    from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

    doc_id = f"att-{attachment_id}"
    opts = parse_process_options(process_options)
    chunk_opts = resolve_chunk_options(
        getattr(lightrag, "addon_params", None), process_options=process_options
    )
    chunk_size = int(chunk_opts.get("chunk_token_size") or lightrag.chunk_token_size)

    if opts.chunking == "P":
        p_opts = dict(chunk_opts.get("paragraph_semantic") or {})
        p_size = int(p_opts.pop("chunk_token_size", chunk_size))
        chunking_result = chunking_by_paragraph_semantic(
            lightrag.tokenizer,
            parsed.content,
            p_size,
            blocks_path=parsed.blocks_path or None,
            doc_id=doc_id,
            **p_opts,
        )
    elif opts.chunking == "R":
        r_opts = dict(chunk_opts.get("recursive_character") or {})
        r_size = int(r_opts.pop("chunk_token_size", chunk_size))
        chunking_result = chunking_by_recursive_character(
            lightrag.tokenizer, parsed.content, r_size, **r_opts
        )
    elif opts.chunking == "V":
        v_opts = dict(chunk_opts.get("semantic_vector") or {})
        v_size = int(v_opts.pop("chunk_token_size", chunk_size))
        chunking_result = await chunking_by_semantic_vector(
            lightrag.tokenizer,
            parsed.content,
            v_size,
            embedding_func=getattr(lightrag, "embedding_func", None),
            **v_opts,
        )
    else:
        f_opts = dict(chunk_opts.get("fixed_token") or {})
        f_size = int(f_opts.pop("chunk_token_size", chunk_size))
        chunking_result = chunking_by_fixed_token(
            lightrag.tokenizer,
            parsed.content,
            f_size,
            _emit_source_span=True,
            **f_opts,
        )

    if parsed.blocks_path and opts.chunking in {"F", "R", "V"}:
        backfill_chunk_sidecars(chunking_result, parsed.blocks_path)

    max_order = max(
        (int(item.get("chunk_order_index", -1)) for item in chunking_result),
        default=-1,
    )
    mm_chunks: list[dict[str, Any]] = []
    if parsed.blocks_path and include_multimodal:
        mm_chunks = lightrag._build_mm_chunks_from_sidecars(
            doc_id=doc_id,
            file_path=filename,
            blocks_path=parsed.blocks_path,
            base_order_index=max_order + 1,
            process_options=process_options,
        )

    chunk_dict = build_chunks_dict_from_chunking_result(
        list(chunking_result) + list(mm_chunks),
        doc_id=doc_id,
        file_path=filename,
    )

    chunks: list[AttachmentContextChunk] = []
    for index, (chunk_id, payload) in enumerate(chunk_dict.items(), start=1):
        metadata = dict(payload.get("metadata") or {})
        raw_sidecar = payload.get("sidecar")
        sidecar = raw_sidecar if isinstance(raw_sidecar, dict) else None
        image_bytes, image_mime = _materialize_sidecar_image(parsed.blocks_path, sidecar)
        content = str(payload.get("content") or "")
        chunks.append(
            AttachmentContextChunk(
                chunk_id=chunk_id,
                attachment_id=attachment_id,
                filename=filename,
                chunk_index=index,
                content=content,
                token_estimate=estimate_tokens(content),
                page_idx=payload.get("page_idx"),
                bbox=payload.get("bbox"),
                sidecar_type=str(sidecar.get("type")) if sidecar else None,
                image_bytes=image_bytes,
                image_mime_type=image_mime,
                metadata=metadata,
            )
        )
    return ParsedAttachmentBundle(chunks=chunks, parser_signature=parsed.parser_signature)


def _materialize_sidecar_image(
    blocks_path: str, sidecar: dict[str, Any] | None
) -> tuple[bytes | None, str | None]:
    """Read the sidecar-referenced image bytes for a visual chunk, if present."""
    if not blocks_path or not sidecar:
        return None, None
    from dlightrag.core.sidecar_provenance import resolve_sidecar_asset_path

    kind = str(sidecar.get("type") or "")
    sidecar_id = str(sidecar.get("id") or "")
    suffix_by_kind = {"drawing": "drawings", "table": "tables", "equation": "equations"}
    if kind not in suffix_by_kind or not sidecar_id:
        return None, None
    suffix = suffix_by_kind[kind]
    base = str(Path(blocks_path))
    if base.endswith(".blocks.jsonl"):
        base = base[: -len(".blocks.jsonl")]
    payload_path = Path(base + f".{suffix}.json")
    if not payload_path.exists():
        return None, None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        item = payload.get(suffix, {}).get(sidecar_id)
        if not isinstance(item, dict):
            return None, None
        raw_path = item.get("path") or item.get("img_path") or item.get("image_path")
        if not isinstance(raw_path, str):
            return None, None
        image_path = resolve_sidecar_asset_path(Path(blocks_path).parent, raw_path)
        if image_path is None or not image_path.exists():
            return None, None
        return image_path.read_bytes(), "image/png"
    except Exception:
        return None, None


class QueryAttachmentService:
    """Parse-cache-first query-attachment retrieval, wired by dependency injection.

    ``store`` (the Web-conversation store) and ``lightrag`` (a LightRAG accessor)
    are INJECTED to avoid a storage<->core import cycle; this module never
    imports the storage module at top level.
    """

    def __init__(
        self,
        *,
        lightrag: Any,
        store: AttachmentCacheStoreProtocol,
        parser_rules: str,
        ttl_days: int,
        robust_document_embedder: DocumentEmbedderProtocol,
        direct_image_embedding_enabled: bool,
        model_bundle: Any,
        config: DlightragConfig,
        principal_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._store = store
        self._parser_rules = parser_rules
        self._ttl_days = ttl_days
        self._document_embedder = robust_document_embedder
        self._direct_image_embedding_enabled = direct_image_embedding_enabled
        self._model_bundle = model_bundle
        self._config = config
        self._principal_id = principal_id
        self._conversation_id = conversation_id

    async def adense_rankings(
        self,
        query: str,
        current_rows: list[dict[str, Any]],
        history_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Rank both Composer lanes from request-scoped cached vectors."""

        def dense_trace(
            status: str,
            *,
            current_chunks: int = 0,
            history_chunks: int = 0,
            error: str | None = None,
        ) -> dict[str, Any]:
            trace: dict[str, Any] = {
                "composer_dense_status": status,
                "composer_dense_current_chunks": current_chunks,
                "composer_dense_history_chunks": history_chunks,
                "composer_dense_chunks": current_chunks + history_chunks,
            }
            if error is not None:
                trace["composer_dense_error"] = error
            return trace

        if not current_rows and not history_rows:
            return [], [], dense_trace("no_rows")
        if (
            sum(
                estimate_tokens(str(row.get("content") or ""))
                for row in [*current_rows, *history_rows]
            )
            <= COMPOSER_FULL_PASS_TOKENS
        ):
            return [], [], dense_trace("full_pass_not_needed")
        try:
            query_vector = await self._document_embedder.aembed_query(query)
            if query_vector is None:
                return [], [], dense_trace("no_query_vector")
            query_array = np.asarray(query_vector, dtype=np.float32)
            if (
                query_array.ndim != 1
                or query_array.shape[0] != self._document_embedder.dimension
                or not bool(np.isfinite(query_array).all())
            ):
                return [], [], dense_trace("no_query_vector")
            query_norm = np.float32(np.linalg.norm(query_array))
            if not bool(np.isfinite(query_norm)) or float(query_norm) == 0.0:
                return [], [], dense_trace("no_query_vector")
            principal_id = self._principal_id
            conversation_id = self._conversation_id
            if principal_id is None or conversation_id is None:
                raise RuntimeError("dense ranking requires a bound conversation scope")
            current = await self._adense_rank_lane(
                current_rows,
                query_array,
                query_norm,
                principal_id=principal_id,
                conversation_id=conversation_id,
            )
            history = await self._adense_rank_lane(
                history_rows,
                query_array,
                query_norm,
                principal_id=principal_id,
                conversation_id=conversation_id,
            )
            if not current and not history:
                return [], [], dense_trace("no_rows")
            return (
                current,
                history,
                dense_trace(
                    "ranked",
                    current_chunks=len(current),
                    history_chunks=len(history),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Composer dense ranking failed; using local fusion", exc_info=True)
            return [], [], dense_trace("failed", error=type(exc).__name__)

    async def _adense_rank_lane(
        self,
        rows: list[dict[str, Any]],
        query_vector: NDArray[np.float32],
        query_norm: np.float32,
        *,
        principal_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        references: list[tuple[int, AttachmentCacheKey]] = []
        expected_by_order: dict[int, tuple[AttachmentCacheKey, frozenset[str]]] = {}
        for global_order, row in enumerate(rows):
            cache_key = row.get("_cache_key")
            if not isinstance(cache_key, AttachmentCacheKey):
                continue
            prefers_fused = (
                row.get("image_data") is not None
                and self._direct_image_embedding_enabled
                and self._document_embedder.image_enabled
            )
            signatures = {
                build_composer_embedding_signature(
                    config=self._config,
                    embedder=self._document_embedder,
                    mode="fused" if prefers_fused else "text",
                )
            }
            if prefers_fused:
                signatures.update(
                    build_composer_embedding_signature(
                        config=self._config,
                        embedder=self._document_embedder,
                        mode="text",
                        fallback_reason=fallback_reason,
                    )
                    for fallback_reason in ("image_rejected", "fused_provider_failed")
                )
            references.append((global_order, cache_key))
            expected_by_order[global_order] = (cache_key, frozenset(signatures))

        scored: list[tuple[np.float32, int]] = []
        async for page in self._store.aiter_attachment_vectors(
            principal_id,
            conversation_id,
            references,
            ttl_days=self._ttl_days,
            page_size=COMPOSER_DENSE_PAGE_SIZE,
        ):
            for start in range(0, len(page), COMPOSER_DENSE_BLOCK_SIZE):
                vectors: list[NDArray[np.float32]] = []
                global_orders: list[int] = []
                for vector_row in page[start : start + COMPOSER_DENSE_BLOCK_SIZE]:
                    expected = expected_by_order.get(vector_row.global_order)
                    signature = vector_row.embedding_signature
                    if (
                        expected is None
                        or vector_row.cache_key != expected[0]
                        or signature is None
                        or signature not in expected[1]
                    ):
                        continue
                    vector = validate_attachment_vector(
                        vector_row,
                        expected_signature=signature,
                        expected_dimension=self._document_embedder.dimension,
                    )
                    if vector is None:
                        continue
                    vectors.append(vector)
                    global_orders.append(vector_row.global_order)
                if not vectors:
                    continue
                matrix = np.stack(vectors).astype(np.float32, copy=False)
                norms = np.linalg.norm(matrix, axis=1).astype(np.float32, copy=False)
                denominator = norms * query_norm
                scores = np.divide(
                    matrix @ query_vector,
                    denominator,
                    out=np.zeros(len(vectors), dtype=np.float32),
                    where=denominator != 0,
                )
                for score, global_order in zip(scores, global_orders, strict=True):
                    if score >= np.float32(COMPOSER_DENSE_SCORE_THRESHOLD):
                        scored.append((score, global_order))
            del page
        scored.sort(key=lambda item: (-float(item[0]), item[1]))
        ranked: list[dict[str, Any]] = []
        for _score, global_order in scored:
            row = dict(rows[global_order])
            row.pop("embedding_signature", None)
            row.pop("embedding_vector", None)
            ranked.append(row)
        return ranked

    def _with_cache_keys(
        self,
        bundle: ParsedAttachmentBundle,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
    ) -> ParsedAttachmentBundle:
        return replace(
            bundle,
            chunks=[
                replace(
                    chunk,
                    cache_key=AttachmentCacheKey(
                        content_sha256,
                        parser_signature,
                        chunk_signature,
                        chunk.cache_key.cache_chunk_id
                        if chunk.cache_key is not None
                        else chunk.chunk_id,
                    ),
                )
                for chunk in bundle.chunks
            ],
            parser_signature=parser_signature,
        )

    async def _aembed_bundle(
        self,
        bundle: ParsedAttachmentBundle,
        *,
        allow_images: bool = True,
    ) -> tuple[ParsedAttachmentBundle, DocumentEmbeddingTrace]:
        return await self._aembed_chunk_indices(
            bundle,
            list(range(len(bundle.chunks))),
            allow_images=allow_images,
        )

    async def _aembed_chunk_indices(
        self,
        bundle: ParsedAttachmentBundle,
        indices: list[int],
        *,
        allow_images: bool = True,
    ) -> tuple[ParsedAttachmentBundle, DocumentEmbeddingTrace]:
        if not indices:
            return bundle, DocumentEmbeddingTrace(0, 0, 0, 0)
        inputs = [
            DocumentEmbeddingInput(
                key=(
                    chunk.cache_key.cache_chunk_id
                    if chunk.cache_key is not None
                    else chunk.chunk_id
                ),
                text=chunk.content,
                image_bytes=(
                    chunk.image_bytes
                    if allow_images
                    and self._direct_image_embedding_enabled
                    and self._document_embedder.image_enabled
                    else None
                ),
            )
            for chunk in (bundle.chunks[index] for index in indices)
        ]
        vectors, trace = await self._document_embedder.aembed_documents(inputs)
        vectors_by_key = {vector.key: vector for vector in vectors}
        chunks = list(bundle.chunks)
        for index, item in zip(indices, inputs, strict=True):
            chunk = chunks[index]
            vector = vectors_by_key.get(item.key)
            if vector is None:
                continue
            chunks[index] = replace(
                chunk,
                embedding_signature=build_composer_embedding_signature(
                    config=self._config,
                    embedder=self._document_embedder,
                    mode=vector.mode,
                    fallback_reason=vector.fallback_reason,
                ),
                embedding_vector=vector.vector,
            )
        return replace(bundle, chunks=chunks), trace

    def _resolve_attachment(
        self,
        filename: str,
        trace: AttachmentProcessingTrace,
    ) -> _AttachmentResolution | None:
        from lightrag.parser.routing import resolve_parser_directives

        try:
            directives = resolve_parser_directives(
                Path(filename),
                parser_rules=self._parser_rules,
                require_external_endpoint=False,
            )
            parser_signature = resolve_attachment_parser_signature(filename, self._parser_rules)
            analysis_signature = build_composer_analysis_signature(
                lightrag=self._lightrag,
                model_bundle=self._model_bundle,
                config=self._config,
                process_options=directives.process_options,
            )
            chunk_signature = resolve_attachment_chunk_signature(
                self._lightrag,
                filename,
                self._parser_rules,
                analysis_signature=analysis_signature,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            trace.parse_error = error_type
            trace.directive_error = error_type
            logger.warning(
                "Attachment parser directive resolution failed for %r",
                filename,
                exc_info=True,
            )
            return None
        return _AttachmentResolution(
            process_options=directives.process_options,
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
        )

    async def _aload_cached_bundle(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        content_sha256: str,
        resolution: _AttachmentResolution,
        trace: AttachmentProcessingTrace,
    ) -> ParsedAttachmentBundle | None:
        try:
            return await self._store.load_attachment_chunks(
                principal_id,
                conversation_id,
                attachment_id,
                filename,
                content_sha256=content_sha256,
                parser_signature=resolution.parser_signature,
                chunk_signature=resolution.chunk_signature,
                ttl_days=self._ttl_days,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            trace.cache_read_error = error_type
            trace.preserve_cache_error(error_type)
            logger.warning(
                "Attachment cache lookup failed; reparsing request-locally",
                exc_info=True,
            )
            return None

    def _expected_vector_signatures(
        self,
        chunks: Sequence[AttachmentContextChunk],
    ) -> list[frozenset[str]]:
        expected: list[frozenset[str]] = []
        for chunk in chunks:
            prefers_fused = (
                chunk.image_bytes is not None
                and self._direct_image_embedding_enabled
                and self._document_embedder.image_enabled
            )
            signatures = {
                build_composer_embedding_signature(
                    config=self._config,
                    embedder=self._document_embedder,
                    mode="fused" if prefers_fused else "text",
                )
            }
            if prefers_fused:
                signatures.add(
                    build_composer_embedding_signature(
                        config=self._config,
                        embedder=self._document_embedder,
                        mode="text",
                        fallback_reason="image_rejected",
                    )
                )
            expected.append(frozenset(signatures))
        return expected

    async def _arefresh_cached_bundle(
        self,
        cached: ParsedAttachmentBundle,
        *,
        principal_id: str,
        conversation_id: str,
        resolution: _AttachmentResolution,
        trace: AttachmentProcessingTrace,
    ) -> ParsedAttachmentBundle:
        from lightrag.parser.routing import parse_process_options

        trace.parse_cache_hit = True
        chunks = list(cached.chunks)
        options = parse_process_options(resolution.process_options)
        trace.mm_chunk_count = sum(chunk.sidecar_type is not None for chunk in chunks)
        trace.analysis_outcome = (
            ComposerAnalysisOutcome.SUCCESS.value
            if trace.mm_chunk_count > 0
            or (
                self._config.parser_sidecars.vlm.enabled
                and (options.images or options.tables or options.equations)
            )
            else ComposerAnalysisOutcome.INTENTIONALLY_DISABLED.value
        )
        references = [
            (index, chunk.cache_key)
            for index, chunk in enumerate(chunks)
            if chunk.cache_key is not None
        ]
        try:
            expected_signatures = self._expected_vector_signatures(chunks)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace.embedding_error = type(exc).__name__
            trace.embedding_failed = len(chunks)
            trace.vector_cache_misses = len(chunks)
            trace.vector_update_status = "skipped"
            logger.warning(
                "Attachment embedding signature resolution failed; keeping cached chunks",
                exc_info=True,
            )
            return cached
        expected_by_order = {
            index: (cache_key, expected_signatures[index]) for index, cache_key in references
        }
        valid_signatures: dict[int, str] = {}
        try:
            async for page in self._store.aiter_attachment_vectors(
                principal_id,
                conversation_id,
                references,
                ttl_days=self._ttl_days,
                page_size=1000,
            ):
                valid_signatures.update(
                    _validated_attachment_vector_page_indices(
                        page,
                        expected_by_order=expected_by_order,
                        expected_dimension=self._document_embedder.dimension,
                    )
                )
                del page
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            trace.vector_read_error = error_type
            trace.preserve_cache_error(error_type)
            logger.warning(
                "Attachment vector cache read failed; refreshing request-locally",
                exc_info=True,
            )

        missing_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            if index in valid_signatures:
                chunks[index] = replace(
                    chunk,
                    embedding_signature=valid_signatures[index],
                    embedding_vector=None,
                )
            else:
                missing_indices.append(index)
        trace.vector_cache_hits = len(valid_signatures)
        trace.vector_cache_misses = len(missing_indices)
        validated = replace(cached, chunks=chunks)
        if not missing_indices:
            return validated

        try:
            refreshed, embedding_trace = await self._aembed_chunk_indices(
                validated,
                missing_indices,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace.embedding_error = type(exc).__name__
            trace.embedding_failed = len(missing_indices)
            trace.vector_update_status = "skipped"
            logger.warning(
                "Attachment vector refresh failed; keeping cached chunks request-locally",
                exc_info=True,
            )
            return validated
        trace.record_embedding(embedding_trace)
        updated_chunks = [
            refreshed.chunks[index]
            for index in missing_indices
            if refreshed.chunks[index].embedding_signature is not None
            and refreshed.chunks[index].embedding_vector is not None
        ]
        if not updated_chunks:
            trace.vector_update_status = "skipped"
            return refreshed
        try:
            updated = await self._store.aupdate_attachment_chunk_vectors(
                principal_id,
                conversation_id,
                updated_chunks,
                ttl_days=self._ttl_days,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            trace.vector_update_error = error_type
            trace.preserve_cache_error(error_type)
            trace.vector_update_status = "error"
            logger.warning(
                "Attachment vector cache update failed; keeping request-local vectors",
                exc_info=True,
            )
        else:
            trace.vector_update_status = "written" if updated else "skipped"
        return refreshed

    async def _aparse_enrich_materialize(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        document_bytes: bytes,
        content_sha256: str,
        resolution: _AttachmentResolution,
        trace: AttachmentProcessingTrace,
    ) -> ParsedAttachmentBundle:
        analysis = ComposerAnalysisResult(
            ComposerAnalysisOutcome.DEGRADED,
            0,
        )
        try:
            async with parse_attachment_document(
                attachment_id=attachment_id,
                filename=filename,
                document_bytes=document_bytes,
                parser_rules=self._parser_rules,
            ) as (parsed, process_options, actual_parser_signature):
                try:
                    analysis = await aanalyze_composer_sidecars(
                        lightrag=self._lightrag,
                        model_bundle=self._model_bundle,
                        config=self._config,
                        doc_id=f"att-{attachment_id}",
                        file_path=filename,
                        parsed_data={
                            "content": parsed.content,
                            "blocks_path": parsed.blocks_path,
                        },
                        process_options=process_options,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_type = type(exc).__name__
                    trace.analysis_error = error_type
                    analysis = ComposerAnalysisResult(
                        ComposerAnalysisOutcome.DEGRADED,
                        0,
                        error_type=error_type,
                    )
                    logger.warning(
                        "Attachment analysis failed; rendering text-only chunks",
                        exc_info=True,
                    )
                trace.analysis_outcome = analysis.outcome.value
                trace.analysis_error = trace.analysis_error or analysis.error_type
                trace.mm_chunk_count = analysis.mm_chunk_count
                try:
                    bundle = await build_attachment_bundle_from_parse_result(
                        lightrag=self._lightrag,
                        attachment_id=attachment_id,
                        filename=filename,
                        parsed=parsed,
                        process_options=process_options,
                        include_multimodal=(analysis.outcome is ComposerAnalysisOutcome.SUCCESS),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_type = type(exc).__name__
                    trace.parse_error = error_type
                    trace.rendering_error = error_type
                    logger.warning(
                        "Attachment rendering failed for %r",
                        filename,
                        exc_info=True,
                    )
                    return ParsedAttachmentBundle(chunks=[])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            trace.parse_error = trace.parse_error or error_type
            trace.parser_error = error_type
            logger.warning(
                "Attachment parse failed for %r (attachment_id=%s)",
                filename,
                attachment_id,
                exc_info=True,
            )
            return ParsedAttachmentBundle(chunks=[])

        bundle = self._with_cache_keys(
            bundle,
            content_sha256=content_sha256,
            parser_signature=actual_parser_signature,
            chunk_signature=resolution.chunk_signature,
        )
        trace.vector_cache_misses = len(bundle.chunks)
        try:
            bundle, embedding_trace = await self._aembed_bundle(
                bundle,
                allow_images=analysis.outcome is not ComposerAnalysisOutcome.DEGRADED,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace.embedding_error = type(exc).__name__
            trace.embedding_failed = len(bundle.chunks)
            logger.warning(
                "Attachment embedding failed; caching chunks for a later retry",
                exc_info=True,
            )
        else:
            trace.record_embedding(embedding_trace)

        if not analysis.cacheable:
            trace.cache_write_status = "skipped"
            return bundle
        try:
            materialized = await self._store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256=content_sha256,
                parser_signature=resolution.parser_signature,
                chunk_signature=resolution.chunk_signature,
                bundle=bundle,
                ttl_days=self._ttl_days,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            trace.cache_write_error = error_type
            trace.preserve_cache_error(error_type)
            trace.cache_write_status = "error"
            logger.warning(
                "Attachment cache materialization failed; keeping request-local bundle",
                exc_info=True,
            )
        else:
            trace.cache_materialized = materialized
            trace.cache_write_status = "written" if materialized else "skipped"
        return bundle

    async def achunks_for_attachment(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        document_bytes: bytes,
        content_sha256: str,
    ) -> tuple[ParsedAttachmentBundle, dict[str, Any]]:
        """Return cached chunks or parse-and-cache them; failures stay scoped."""
        trace = AttachmentProcessingTrace()
        resolution = self._resolve_attachment(filename, trace)
        if resolution is None:
            return ParsedAttachmentBundle(chunks=[]), trace.as_dict()
        cached = await self._aload_cached_bundle(
            principal_id=principal_id,
            conversation_id=conversation_id,
            attachment_id=attachment_id,
            filename=filename,
            content_sha256=content_sha256,
            resolution=resolution,
            trace=trace,
        )
        if cached is not None:
            bundle = await self._arefresh_cached_bundle(
                cached,
                principal_id=principal_id,
                conversation_id=conversation_id,
                resolution=resolution,
                trace=trace,
            )
        else:
            bundle = await self._aparse_enrich_materialize(
                principal_id=principal_id,
                conversation_id=conversation_id,
                attachment_id=attachment_id,
                filename=filename,
                document_bytes=document_bytes,
                content_sha256=content_sha256,
                resolution=resolution,
                trace=trace,
            )
        return bundle, trace.as_dict()


__all__ = [
    "AttachmentCacheKey",
    "AttachmentContextChunk",
    "AttachmentVectorPageRow",
    "ParsedAttachmentBundle",
    "ParsedAttachmentDocument",
    "QueryAttachmentService",
    "build_composer_embedding_signature",
    "build_attachment_bundle_from_parse_result",
    "build_text_attachment_chunk",
    "parse_attachment_to_bundle",
    "parse_attachment_document",
    "resolve_attachment_chunk_signature",
    "resolve_attachment_parser_signature",
    "validate_attachment_vector",
]
