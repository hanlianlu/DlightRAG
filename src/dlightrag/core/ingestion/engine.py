# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified ingestion over LightRAG parser sidecars and direct image embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.parser.routing import resolve_file_parser_directives
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path

from dlightrag.core.ingestion.direct_image import (
    build_direct_image_chunk,
    native_image_ref,
)
from dlightrag.core.ingestion.lightrag_sidecar import LightRAGSidecarRef, collect_sidecar_refs
from dlightrag.core.ingestion.visual_semantics import build_visual_semantic_projection
from dlightrag.core.retrieval.metadata_fields import (
    MetadataFieldRegistry,
    MetadataIngestPolicy,
    extract_system_metadata,
    normalize_user_metadata,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class UnifiedIngestionEngine:
    """One ingestion path for LightRAG documents and direct image chunks."""

    def __init__(
        self,
        *,
        lightrag: Any,
        stores: Any,
        metadata_index: Any,
        multimodal_embedder: Any,
        workspace: str,
        parser_rules: str,
        chunk_options: dict[str, Any] | None,
        metadata_registry: MetadataFieldRegistry | None = None,
        allow_ad_hoc_metadata: bool = True,
        default_metadata_policy: MetadataIngestPolicy = "validate",
        vlm_func: Any | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._stores = stores
        self._metadata_index = metadata_index
        self._multimodal_embedder = multimodal_embedder
        self._workspace = workspace
        self._parser_rules = parser_rules
        self._chunk_options = chunk_options or {}
        self._metadata_registry = metadata_registry or MetadataFieldRegistry.from_config({})
        self._allow_ad_hoc_metadata = allow_ad_hoc_metadata
        self._default_metadata_policy: MetadataIngestPolicy = default_metadata_policy
        self._vlm_func = vlm_func
        self._ingest_locks: dict[str, asyncio.Lock] = {}

    async def aingest_file(
        self,
        path: str | Path,
        *,
        replace: bool = False,  # noqa: ARG002
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest a local file through the unified path."""
        file_path = Path(path)
        doc_id = _canonical_file_doc_id(file_path)
        metadata_record = self._prepare_metadata_record(
            file_path,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        if file_path.suffix.lower() in _IMAGE_SUFFIXES:
            return await self._ingest_native_image(
                file_path,
                doc_id=doc_id,
                metadata_record=metadata_record,
            )
        return await self._ingest_document(
            file_path,
            doc_id=doc_id,
            metadata_record=metadata_record,
        )

    def _prepare_metadata_record(
        self,
        file_path: Path,
        *,
        title: str | None,
        author: str | None,
        metadata: Mapping[str, Any] | None,
        metadata_policy: MetadataIngestPolicy | None,
    ) -> dict[str, Any]:
        if title is not None and metadata and "title" in metadata:
            raise ValueError("title supplied both as top-level field and metadata")
        if author is not None and metadata and "author" in metadata:
            raise ValueError("author supplied both as top-level field and metadata")

        normalized_metadata = normalize_user_metadata(
            metadata,
            self._metadata_registry,
            metadata_policy=metadata_policy or self._default_metadata_policy,
            allow_ad_hoc_json=self._allow_ad_hoc_metadata,
        )
        system_metadata = extract_system_metadata(
            file_path,
            ingest_strategy="lightrag_sidecar_unified",
        )
        if title is not None:
            system_metadata["doc_title"] = title
        if author is not None:
            system_metadata["doc_author"] = author
        return {
            **system_metadata,
            "user_metadata": dict(metadata or {}),
            "metadata_filterable": normalized_metadata.filterable,
            "metadata_json": normalized_metadata.raw_json,
        }

    def _get_ingest_lock(self, doc_id: str) -> asyncio.Lock:
        """Return a per-doc async lock to serialize concurrent ingests of the same file."""
        lock = self._ingest_locks.get(doc_id)
        if lock is None:
            lock = asyncio.Lock()
            self._ingest_locks[doc_id] = lock
        return lock

    async def _cleanup_partial_doc(self, doc_id: str) -> None:
        """Remove all traces of a partial/failed document before re-ingest."""
        # 1. LightRAG tables: doc_status, doc_full, doc_chunks.
        await self._stores.cleanup_doc(doc_id)

        # 2. DlightRAG metadata index.
        await self._metadata_index.delete(doc_id)

        # 3. Sidecar directory (parsed artifacts).
        full_doc = await self._stores.get_full_doc(doc_id)
        sidecar_uri = (full_doc or {}).get("sidecar_location")
        artifact_dir = _artifact_dir_from_uri(sidecar_uri)
        if artifact_dir is not None and artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

    async def _ingest_document(
        self,
        file_path: Path,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._get_ingest_lock(doc_id):
            # Self-healing: if a previous ingest was interrupted, clean up first.
            existing_status = await self._stores.get_doc_status(doc_id)
            if existing_status is not None and existing_status.get("status") != "processed":
                await self._cleanup_partial_doc(doc_id)

            parse_engine, process_options = resolve_file_parser_directives(
                file_path,
                parser_rules=self._parser_rules,
                require_external_endpoint=False,
            )
            if parse_engine not in {
                PARSER_ENGINE_DOCLING,
                PARSER_ENGINE_MINERU,
                PARSER_ENGINE_NATIVE,
            }:
                raise ValueError(
                    f"No explicit parser route for {file_path.name!r}. "
                    f"The file resolved to parser engine {parse_engine!r}, which is the "
                    f"built-in fallback. Configure parser.rules to route this file type "
                    f"to a supported parser (mineru, native, or docling). "
                    f"Current parser.rules: {self._parser_rules!r}"
                )
            await self._lightrag.apipeline_enqueue_documents(
                input="",
                file_paths=[str(file_path)],
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                lightrag_document_paths=[str(file_path)],
                parse_engine=parse_engine,
                process_options=process_options,
                chunk_options=self._chunk_options or None,
            )
            await self._lightrag.apipeline_process_enqueue_documents()

            doc_status = await self._stores.get_doc_status(doc_id)
            full_doc = await self._stores.get_full_doc(doc_id)
            light_chunks = list((doc_status or {}).get("chunks_list") or [])
            lightrag_record = self._lightrag_metadata(full_doc, doc_status)
            await self._metadata_index.upsert(doc_id, {**metadata_record, **lightrag_record})

            direct_chunks = await self._ingest_sidecar_direct_images(
                doc_id=doc_id,
                sidecar_location=lightrag_record.get("lightrag.sidecar_location"),
            )

            return {
                "doc_id": doc_id,
                "source_kind": "document",
                "chunks": light_chunks + direct_chunks["chunk_ids"],
                "ingest_strategy": "lightrag_sidecar_unified",
                "parse_engine": parse_engine,
                "process_options": process_options,
            }

    async def _ingest_sidecar_direct_images(
        self,
        *,
        doc_id: str,
        sidecar_location: str | None,
    ) -> dict[str, Any]:
        artifact_dir = _artifact_dir_from_uri(sidecar_location)
        if artifact_dir is None or not artifact_dir.exists():
            return {"chunk_ids": []}

        rows: dict[str, dict[str, Any]] = {}
        vectors: dict[str, list[float]] = {}
        for ref in collect_sidecar_refs(artifact_dir):
            if ref.asset_path is None or not ref.asset_path.exists():
                continue
            content = _sidecar_text_content(ref)
            chunk_id, row, vector = await build_direct_image_chunk(
                workspace=self._workspace,
                full_doc_id=doc_id,
                ref=ref,
                embedder=self._multimodal_embedder,
                text_content=content,
            )
            rows[chunk_id] = row
            vectors[chunk_id] = vector
        if rows:
            await self._stores.upsert_chunks_with_vectors(
                rows,
                vectors,
                embedding_dim=getattr(
                    self._multimodal_embedder, "dim", len(next(iter(vectors.values())))
                ),
                max_token_size=8192,
            )
        return {"chunk_ids": list(rows)}

    async def _ingest_native_image(
        self,
        file_path: Path,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._get_ingest_lock(doc_id):
            # Self-healing: if a previous ingest was interrupted, clean up first.
            existing_status = await self._stores.get_doc_status(doc_id)
            if existing_status is not None and existing_status.get("status") != "processed":
                await self._cleanup_partial_doc(doc_id)

            ref = native_image_ref(file_path)
            chunk_id, row, vector = await build_direct_image_chunk(
                workspace=self._workspace,
                full_doc_id=doc_id,
                ref=ref,
                embedder=self._multimodal_embedder,
                text_content=f"Native image: {file_path.name}",
            )
            content_hash = _file_sha256(file_path)
            await self._stores.upsert_document_record(
                doc_id=doc_id,
                content=f"Native image: {file_path.name}",
                file_path=str(file_path),
                chunks=[chunk_id],
                parse_engine="native_image",
                parse_format=FULL_DOCS_FORMAT_RAW,
                content_hash=content_hash,
                metadata={"source_kind": "image"},
                sidecar_location=file_path.as_uri(),
                process_options="P",
                chunk_options=self._chunk_options,
            )
            await self._stores.upsert_chunks_with_vectors(
                {chunk_id: row},
                {chunk_id: vector},
                embedding_dim=getattr(self._multimodal_embedder, "dim", len(vector)),
                max_token_size=8192,
            )
            if self._vlm_func is not None:
                semantic_doc_id, semantic_text = await build_visual_semantic_projection(
                    workspace=self._workspace,
                    source_doc_id=doc_id,
                    ref=ref,
                    vlm_func=self._vlm_func,
                )
                await self._lightrag.apipeline_enqueue_documents(
                    input=semantic_text,
                    ids=[semantic_doc_id],
                    file_paths=[str(file_path)],
                    docs_format=FULL_DOCS_FORMAT_RAW,
                    process_options="P",
                )
                await self._lightrag.apipeline_process_enqueue_documents()
            await self._metadata_index.upsert(doc_id, metadata_record)
            return {
                "doc_id": doc_id,
                "source_kind": "image",
                "chunks": [chunk_id],
                "ingest_strategy": "direct_image_with_visual_semantics",
            }

    @staticmethod
    def _lightrag_metadata(
        full_doc: Mapping[str, Any] | None,
        doc_status: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        full_doc = full_doc or {}
        doc_status = doc_status or {}
        return {
            "lightrag.parse_engine": full_doc.get("parse_engine"),
            "lightrag.process_options": full_doc.get("process_options"),
            "lightrag.chunk_options": full_doc.get("chunk_options") or {},
            "lightrag.content_hash": doc_status.get("content_hash") or full_doc.get("content_hash"),
            "lightrag.sidecar_location": full_doc.get("sidecar_location"),
        }


def _artifact_dir_from_uri(uri: str | None) -> Path | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(unquote(parsed.path))
    return Path(uri)


def _sidecar_text_content(ref: LightRAGSidecarRef) -> str:
    payload = ref.payload or {}
    analysis = payload.get("llm_analyze_result")
    if isinstance(analysis, dict):
        text = analysis.get("text") or analysis.get("description") or analysis.get("summary")
        if text:
            return str(text)
    if caption := payload.get("caption"):
        return str(caption)
    return f"{ref.sidecar_type} {ref.sidecar_id}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_file_doc_id(path: Path) -> str:
    """Match LightRAG's file-backed document id derivation."""
    return compute_mdhash_id(normalize_document_file_path(path), prefix="doc-")
