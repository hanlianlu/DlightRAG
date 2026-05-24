# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified ingestion over LightRAG parser sidecars and direct image embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE, FULL_DOCS_FORMAT_RAW
from lightrag.parser.routing import resolve_file_parser_directives
from lightrag.utils import compute_mdhash_id

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
        document_artifacts: Any,
        chunk_provenance: Any,
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
        self._document_artifacts = document_artifacts
        self._chunk_provenance = chunk_provenance
        self._multimodal_embedder = multimodal_embedder
        self._workspace = workspace
        self._parser_rules = parser_rules
        self._chunk_options = chunk_options or {}
        self._metadata_registry = metadata_registry or MetadataFieldRegistry.from_config({})
        self._allow_ad_hoc_metadata = allow_ad_hoc_metadata
        self._default_metadata_policy: MetadataIngestPolicy = default_metadata_policy
        self._vlm_func = vlm_func

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
        doc_id = compute_mdhash_id(str(file_path), prefix="doc-")
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

    async def _ingest_document(
        self,
        file_path: Path,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
    ) -> dict[str, Any]:
        parse_engine, process_options = resolve_file_parser_directives(
            file_path,
            parser_rules=self._parser_rules,
            require_external_endpoint=False,
        )
        await self._lightrag.apipeline_enqueue_documents(
            input="",
            ids=[doc_id],
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
        await self._document_artifacts.upsert(
            {
                "full_doc_id": doc_id,
                "source_uri": file_path.as_uri(),
                "parser": "lightrag",
                "parse_engine": lightrag_record.get("lightrag.parse_engine"),
                "process_options": lightrag_record.get("lightrag.process_options"),
                "chunk_options": lightrag_record.get("lightrag.chunk_options") or {},
                "content_hash": lightrag_record.get("lightrag.content_hash"),
                "sidecar_location": lightrag_record.get("lightrag.sidecar_location"),
                "metadata": metadata_record,
            }
        )

        provenance = [
            {
                "chunk_id": chunk_id,
                "full_doc_id": doc_id,
                "embedding_input_kind": "text",
                "sidecar_type": "block",
            }
            for chunk_id in light_chunks
        ]
        direct_chunks = await self._ingest_sidecar_direct_images(
            doc_id=doc_id,
            sidecar_location=lightrag_record.get("lightrag.sidecar_location"),
        )
        provenance.extend(direct_chunks["provenance"])
        await self._chunk_provenance.upsert_many(provenance)

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
            return {"chunk_ids": [], "provenance": []}

        rows: dict[str, dict[str, Any]] = {}
        vectors: dict[str, list[float]] = {}
        provenance = []
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
            provenance.append(
                {
                    "chunk_id": chunk_id,
                    "full_doc_id": doc_id,
                    "embedding_input_kind": "image",
                    "sidecar_type": ref.sidecar_type,
                    "sidecar_id": ref.sidecar_id,
                    "sidecar_path": str(ref.asset_path),
                    "page_index": ref.page_number,
                }
            )
        if rows:
            await self._stores.upsert_chunks_with_vectors(
                rows,
                vectors,
                embedding_dim=getattr(self._multimodal_embedder, "dim", len(next(iter(vectors.values())))),
                max_token_size=8192,
            )
        return {"chunk_ids": list(rows), "provenance": provenance}

    async def _ingest_native_image(
        self,
        file_path: Path,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
    ) -> dict[str, Any]:
        ref = native_image_ref(file_path)
        chunk_id, row, vector = await build_direct_image_chunk(
            workspace=self._workspace,
            full_doc_id=doc_id,
            ref=ref,
            embedder=self._multimodal_embedder,
            text_content=f"Native image: {file_path.name}",
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
        await self._document_artifacts.upsert(
            {
                "full_doc_id": doc_id,
                "source_uri": file_path.as_uri(),
                "parser": "native_image",
                "metadata": metadata_record,
            }
        )
        await self._chunk_provenance.upsert_many(
            [
                {
                    "chunk_id": chunk_id,
                    "full_doc_id": doc_id,
                    "embedding_input_kind": "image",
                    "sidecar_type": "native_image",
                    "sidecar_id": ref.sidecar_id,
                    "sidecar_path": str(file_path),
                }
            ]
        )
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
