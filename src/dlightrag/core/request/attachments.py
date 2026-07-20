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

import base64
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dlightrag.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

ATTACHMENT_TEXT_TOKEN_BUDGET = 210_000

_ATTACHMENT_WORKSPACE = "__web_attachment__"


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
        return row


@dataclass(frozen=True, slots=True)
class ParsedAttachmentBundle:
    """An ordered set of parsed chunks tied to the parser that produced them."""

    chunks: list[AttachmentContextChunk]
    parser_signature: str = ""


@dataclass(frozen=True, slots=True)
class ParsedAttachmentDocument:
    """Raw parser output for one attachment, before chunking."""

    content: str
    blocks_path: str
    parser_signature: str


def select_attachment_context(
    bundle: ParsedAttachmentBundle,
    *,
    text_token_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce attachment chunks to the text-token budget only.

    Image budgeting is intentionally NOT done here. Attachment rows keep their
    ``image_data`` so the single ``AnswerContextPacker`` + ``AnswerImageBudget``
    can decide which images are actually sent, shared with workspace and
    current images. Visual chunks (rows carrying image bytes) therefore always
    pass through regardless of the text budget; only pure-text chunks are
    subject to the ``text_token_budget`` cutoff.

    v1 scope: this is a deterministic in-order reducer (keep all visual chunks;
    keep text chunks in document order until the token budget is reached). A
    relevance-driven reducer (embedding/BM25/RRF/rerank or an LLM chunk
    selector) is an OPTIONAL future enhancement per the design's "Open
    Implementation Choices" and is out of scope for this slice.
    """
    rows: list[dict[str, Any]] = []
    text_tokens = 0
    strategy: Literal["full", "budgeted"] = "full"
    for chunk in bundle.chunks:
        if chunk.image_bytes is not None:
            # Visual chunk: never dropped for text-budget reasons; the packer
            # owns all image budgeting.
            rows.append(chunk.to_context_row())
            continue
        if text_tokens + chunk.token_estimate > text_token_budget:
            strategy = "budgeted"
            continue
        text_tokens += chunk.token_estimate
        rows.append(chunk.to_context_row())
    return rows, {
        "attachment_context_strategy": strategy,
        "attachment_context_chunks": len(rows),
        "attachment_context_tokens": text_tokens,
    }


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


def resolve_attachment_chunk_signature(lightrag: Any, filename: str, parser_rules: str) -> str:
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
        },
        sort_keys=True,
        default=str,
    )


async def parse_attachment_to_bundle(
    *,
    lightrag: Any,
    attachment_id: str,
    filename: str,
    document_bytes: bytes,
    parser_rules: str,
) -> ParsedAttachmentBundle:
    """Parse one attachment via the real parser stack, writing no workspace row.

    The bundle (including any visual chunk image bytes) is fully built before
    the temporary directory exits, so no returned object points at deleted
    scratch files.
    """
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

    with tempfile.TemporaryDirectory(prefix="dlightrag-attach-") as tmp:
        source = Path(tmp) / Path(filename).name
        source.write_bytes(document_bytes)
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
        # Build text + visual chunks (and read sidecar images) before cleanup.
        return await build_attachment_bundle_from_parse_result(
            lightrag=lightrag,
            attachment_id=attachment_id,
            filename=filename,
            parsed=parsed,
            process_options=directives.process_options,
        )


async def build_attachment_bundle_from_parse_result(
    *,
    lightrag: Any,
    attachment_id: str,
    filename: str,
    parsed: ParsedAttachmentDocument,
    process_options: str,
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
    if parsed.blocks_path:
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

    def __init__(self, *, lightrag: Any, store: Any, parser_rules: str) -> None:
        self._lightrag = lightrag
        self._store = store
        self._parser_rules = parser_rules

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
        parser_signature = resolve_attachment_parser_signature(filename, self._parser_rules)
        chunk_signature = resolve_attachment_chunk_signature(
            self._lightrag, filename, self._parser_rules
        )
        cached = await self._store.load_attachment_chunks(
            principal_id,
            conversation_id,
            attachment_id,
            filename,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
        )
        if cached is not None:
            return cached, {"attachment_parse_cache_hit": True}
        try:
            bundle = await parse_attachment_to_bundle(
                lightrag=self._lightrag,
                attachment_id=attachment_id,
                filename=filename,
                document_bytes=document_bytes,
                parser_rules=self._parser_rules,
            )
        except Exception as exc:
            # Parser failures are attachment-scoped: the rest of the turn
            # continues with an attachment warning instead of a hard fail. Log
            # loudly so a swallowed parse error (e.g. a missing parser dependency
            # or an unsupported document) is visible in server logs instead of
            # silently dropping the attachment.
            logger.warning(
                "Attachment parse failed for %r (attachment_id=%s): %s",
                filename,
                attachment_id,
                exc,
                exc_info=exc,
            )
            return ParsedAttachmentBundle(chunks=[]), {
                "attachment_parse_error": type(exc).__name__,
                "attachment_parse_cache_hit": False,
            }
        await self._store.materialize_attachment_chunks(
            principal_id,
            conversation_id,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
            bundle=bundle,
        )
        return bundle, {"attachment_parse_cache_hit": False}


__all__ = [
    "ATTACHMENT_TEXT_TOKEN_BUDGET",
    "AttachmentContextChunk",
    "ParsedAttachmentBundle",
    "ParsedAttachmentDocument",
    "QueryAttachmentService",
    "build_attachment_bundle_from_parse_result",
    "build_text_attachment_chunk",
    "parse_attachment_to_bundle",
    "resolve_attachment_chunk_signature",
    "resolve_attachment_parser_signature",
    "select_attachment_context",
]
