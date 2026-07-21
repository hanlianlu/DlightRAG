# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified ingestion over LightRAG parser sidecars and image vector overrides."""

import asyncio
import hashlib
import logging
import shutil
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.parser.routing import (
    chunk_strategy_key,
    encode_parse_engine,
    resolve_chunk_options,
    resolve_parser_directives,
)
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path

from dlightrag.core.document_embedding import DocumentEmbeddingInput, RobustDocumentEmbedder
from dlightrag.core.ingestion.lightrag_sidecar import collect_lightrag_drawing_assets
from dlightrag.core.ingestion.paths import lightrag_archived_source_path
from dlightrag.core.retrieval.metadata_fields import (
    MetadataFieldRegistry,
    MetadataIngestPolicy,
    extract_system_metadata,
    normalize_user_metadata,
)
from dlightrag.core.sidecar_provenance import sidecar_dir_from_location
from dlightrag.sourcing.source_contract import local_source_uri, safe_source_filename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedIngestFile:
    """Prepared parser input with explicit provenance and download identity.

    ``parser_path`` must be a local file because LightRAG pending-parse
    ingestion requires local parser input. ``source_uri`` is stable provenance;
    ``download_locator`` is the durable location used to reacquire the bytes.
    """

    parser_path: Path
    source_uri: str
    download_locator: str
    display_filename: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] | None = None
    metadata_policy: MetadataIngestPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parser_path", Path(self.parser_path))


@dataclass(frozen=True)
class _PendingDocumentIngest:
    index: int
    parser_path: Path
    doc_id: str
    metadata_record: dict[str, Any]
    parse_engine: str
    process_options: str
    chunk_options: dict[str, Any] | None


class UnifiedIngestionEngine:
    """One ingestion path over LightRAG parser/routing."""

    def __init__(
        self,
        *,
        lightrag: Any,
        stores: Any,
        metadata_index: Any,
        document_embedder: RobustDocumentEmbedder,
        workspace: str,
        parser_rules: str,
        chunk_options: dict[str, Any] | None,
        metadata_registry: MetadataFieldRegistry | None = None,
        allow_ad_hoc_metadata: bool = True,
        default_metadata_policy: MetadataIngestPolicy = "validate",
        bm25_language_classifier: Any | None = None,
    ) -> None:
        self._lightrag = lightrag
        self._stores = stores
        self._metadata_index = metadata_index
        self._document_embedder = document_embedder
        self._workspace = workspace
        self._parser_rules = parser_rules
        self._chunk_options = chunk_options or {}
        self._metadata_registry = metadata_registry or MetadataFieldRegistry.from_config({})
        self._allow_ad_hoc_metadata = allow_ad_hoc_metadata
        self._default_metadata_policy: MetadataIngestPolicy = default_metadata_policy
        self._bm25_language_classifier = bm25_language_classifier
        self._ingest_locks: dict[str, asyncio.Lock] = {}

    async def aingest_file(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        download_locator: str | None = None,
        display_filename: str | None = None,
        replace: bool = False,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest a local file through the unified path."""
        file_path = Path(path)
        resolved_source_uri = source_uri or _raw_path_source_uri(
            file_path, workspace=self._workspace
        )
        resolved_download_locator = download_locator or str(file_path.resolve())
        doc_id = _canonical_file_doc_id(file_path)

        # Idempotency: skip re-ingest if file content hasn't changed.
        existing_status = await self._stores.get_doc_status(doc_id)
        if existing_status is not None and replace:
            await self._cleanup_partial_doc(doc_id)
        elif existing_status is not None and existing_status.get("status") == "processed":
            current_hash = await asyncio.to_thread(_file_sha256, file_path)
            stored_hash = existing_status.get("content_hash")
            if stored_hash and current_hash == stored_hash:
                return await self._maybe_update_metadata(
                    file_path,
                    doc_id,
                    source_uri=resolved_source_uri,
                    download_locator=resolved_download_locator,
                    display_filename=display_filename,
                    title=title,
                    author=author,
                    metadata=metadata,
                    metadata_policy=metadata_policy,
                    chunks=existing_status.get("chunks_list", []),
                )

        metadata_record = self._prepare_metadata_record(
            file_path,
            source_uri=resolved_source_uri,
            download_locator=resolved_download_locator,
            display_filename=display_filename,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        return await self._ingest_document(
            file_path,
            doc_id=doc_id,
            metadata_record=metadata_record,
        )

    async def _maybe_update_metadata(
        self,
        file_path: str | Path,
        doc_id: str,
        *,
        source_uri: str,
        download_locator: str,
        display_filename: str | None,
        title: str | None,
        author: str | None,
        metadata: Mapping[str, Any] | None,
        metadata_policy: MetadataIngestPolicy | None,
        chunks: list[str],
    ) -> dict[str, Any]:
        """Apply metadata-only update when content hash matches."""
        if title is None and author is None and metadata is None:
            return {
                "doc_id": doc_id,
                "source_kind": "skipped",
                "reason": "content_hash_match",
                "chunks": chunks,
            }
        metadata_record = self._prepare_metadata_record(
            Path(file_path),
            source_uri=source_uri,
            download_locator=download_locator,
            display_filename=display_filename,
            title=title,
            author=author,
            metadata=metadata,
            metadata_policy=metadata_policy,
        )
        await self._metadata_index.upsert(doc_id, metadata_record)
        return {
            "doc_id": doc_id,
            "source_kind": "metadata_updated",
            "reason": "content_hash_match",
            "chunks": chunks,
        }

    async def aingest_files(
        self,
        paths: Sequence[str | Path | PreparedIngestFile],
        *,
        replace: bool = False,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        metadata_policy: MetadataIngestPolicy | None = None,
    ) -> dict[str, Any]:
        """Ingest local files as one LightRAG staged batch.

        LightRAG 1.5 accepts per-document parser directives, so a single
        enqueue can mix native DOCX parsing and MinerU PDF/image parsing.
        DlightRAG keeps only the product-layer metadata and sidecar vector
        overrides around that native batch pipeline.
        """
        if not paths:
            return {"processed": 0, "errors": [], "results": []}

        entries: list[_PendingDocumentIngest] = []
        for index, item in enumerate(
            _prepare_ingest_item(path, workspace=self._workspace) for path in paths
        ):
            parser_path = item.parser_path
            parse_engine, process_options, chunk_options = self._parser_directives_for(parser_path)
            entries.append(
                _PendingDocumentIngest(
                    index=index,
                    parser_path=parser_path,
                    doc_id=_canonical_file_doc_id(parser_path),
                    metadata_record=self._prepare_metadata_record(
                        parser_path,
                        source_uri=item.source_uri,
                        download_locator=item.download_locator,
                        display_filename=item.display_filename,
                        title=item.title if item.title is not None else title,
                        author=item.author if item.author is not None else author,
                        metadata=_overlay_metadata(metadata, item.metadata),
                        metadata_policy=(
                            item.metadata_policy
                            if item.metadata_policy is not None
                            else metadata_policy
                        ),
                    ),
                    parse_engine=parse_engine,
                    process_options=process_options,
                    chunk_options=chunk_options,
                )
            )

        results_by_index: dict[int, dict[str, Any]] = {}
        errors: list[str] = []
        to_enqueue: list[_PendingDocumentIngest] = []

        async with AsyncExitStack() as stack:
            for doc_id in sorted({entry.doc_id for entry in entries}):
                await stack.enter_async_context(self._get_ingest_lock(doc_id))

            for entry in entries:
                existing_status = await self._stores.get_doc_status(entry.doc_id)
                if existing_status is not None and replace:
                    await self._cleanup_partial_doc(entry.doc_id)
                elif existing_status is not None and existing_status.get("status") == "processed":
                    current_hash = await asyncio.to_thread(_file_sha256, entry.parser_path)
                    stored_hash = existing_status.get("content_hash")
                    if stored_hash and current_hash == stored_hash:
                        results_by_index[entry.index] = {
                            "doc_id": entry.doc_id,
                            "source_kind": "skipped",
                            "reason": "content_hash_match",
                            "chunks": existing_status.get("chunks_list", []),
                        }
                        continue

                if existing_status is not None and existing_status.get("status") != "processed":
                    await self._cleanup_partial_doc(entry.doc_id)

                to_enqueue.append(entry)

            if to_enqueue:
                chunk_options = self._batch_chunk_options(to_enqueue)
                for entry in to_enqueue:
                    await self._metadata_index.upsert(entry.doc_id, entry.metadata_record)
                await self._lightrag.apipeline_enqueue_documents(
                    input=[""] * len(to_enqueue),
                    file_paths=[str(entry.parser_path) for entry in to_enqueue],
                    docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                    parse_engine=[entry.parse_engine for entry in to_enqueue],
                    process_options=[entry.process_options for entry in to_enqueue],
                    chunk_options=chunk_options,
                )
                await self._lightrag.apipeline_process_enqueue_documents()

                for entry in to_enqueue:
                    try:
                        results_by_index[entry.index] = await self._finalize_ingested_document(
                            doc_id=entry.doc_id,
                            metadata_record=entry.metadata_record,
                            parse_engine=entry.parse_engine,
                            process_options=entry.process_options,
                        )
                    except Exception:  # noqa: BLE001
                        filename = safe_source_filename(
                            str(entry.metadata_record.get("filename") or entry.parser_path.name)
                        )
                        logger.warning(
                            "Document finalization failed for %s", filename, exc_info=True
                        )
                        errors.append(f"{filename}: document processing failed")

        return {
            "processed": len(results_by_index),
            "errors": errors,
            "results": [results_by_index[index] for index in sorted(results_by_index)],
        }

    def _prepare_metadata_record(
        self,
        file_path: Path,
        *,
        source_uri: str,
        download_locator: str,
        display_filename: str | None = None,
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
            download_locator,
            ingest_strategy="lightrag_sidecar_unified",
            display_filename=display_filename,
            source_uri=source_uri,
            download_locator=download_locator,
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
        full_doc = await self._stores.get_full_doc(doc_id)
        sidecar_uri = _mapping_get(full_doc, "sidecar_location")

        result = await self._lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        status = _mapping_get(result, "status")
        if status in {"fail", "not_allowed"}:
            message = _mapping_get(result, "message") or "LightRAG document deletion failed"
            raise RuntimeError(str(message))

        await self._metadata_index.delete(doc_id)

        artifact_dir = sidecar_dir_from_location(sidecar_uri)
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

            parse_engine, process_options, chunk_options = self._parser_directives_for(file_path)
            await self._metadata_index.upsert(doc_id, metadata_record)
            await self._lightrag.apipeline_enqueue_documents(
                input="",
                file_paths=[str(file_path)],
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=parse_engine,
                process_options=process_options,
                chunk_options=chunk_options
                if chunk_options is not None
                else self._chunk_options or None,
            )
            await self._lightrag.apipeline_process_enqueue_documents()

            return await self._finalize_ingested_document(
                doc_id=doc_id,
                metadata_record=metadata_record,
                parse_engine=parse_engine,
                process_options=process_options,
            )

    async def _finalize_ingested_document(
        self,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
        parse_engine: str,
        process_options: str,
    ) -> dict[str, Any]:
        doc_status = await self._stores.get_doc_status(doc_id)
        raw_status = (doc_status or {}).get("status")
        normalized_status = str(getattr(raw_status, "value", raw_status) or "").lower()
        if normalized_status != "processed":
            error_summary = (doc_status or {}).get("error_msg") or (doc_status or {}).get(
                "content_summary"
            )
            raise RuntimeError(str(error_summary or "LightRAG document processing failed"))

        full_doc = await self._stores.get_full_doc(doc_id)
        light_chunks = list((doc_status or {}).get("chunks_list") or [])
        lightrag_record = self._lightrag_metadata(full_doc, doc_status)
        finalized_metadata = _with_finalized_local_download_locator(metadata_record)
        await self._metadata_index.upsert(doc_id, {**finalized_metadata, **lightrag_record})

        await self._overwrite_sidecar_image_vectors(
            doc_id=doc_id,
            sidecar_location=lightrag_record.get("lightrag.sidecar_location"),
            chunk_ids=set(light_chunks),
        )
        await self._label_bm25_languages(light_chunks)

        return {
            "doc_id": doc_id,
            "source_kind": "document",
            "chunks": light_chunks,
            "ingest_strategy": "lightrag_sidecar_unified",
            "parse_engine": parse_engine,
            "process_options": process_options,
        }

    def _parser_directives_for(self, file_path: Path) -> tuple[str, str, dict[str, Any] | None]:
        directives = resolve_parser_directives(
            file_path,
            parser_rules=self._parser_rules,
            require_external_endpoint=False,
        )
        parse_engine = encode_parse_engine(directives.engine, directives.engine_params)
        chunk_options = self._chunk_options_for_directives(
            directives.process_options,
            directives.chunk_params,
        )
        return parse_engine, directives.process_options, chunk_options

    def _chunk_options_for_directives(
        self,
        process_options: str,
        chunk_params: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not chunk_params:
            return None

        merged = self._base_chunk_options(process_options)
        for selector, params in chunk_params.items():
            key = chunk_strategy_key(selector)
            current = merged.get(key)
            merged[key] = {
                **(current if isinstance(current, dict) else {}),
                **params,
            }
        return merged

    def _base_chunk_options(self, process_options: str) -> dict[str, Any]:
        if self._chunk_options:
            return deepcopy(self._chunk_options)
        return resolve_chunk_options(
            getattr(self._lightrag, "addon_params", None),
            process_options=process_options,
        )

    def _batch_chunk_options(
        self,
        entries: Sequence[_PendingDocumentIngest],
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        if not any(entry.chunk_options is not None for entry in entries):
            return deepcopy(self._chunk_options) if self._chunk_options else None
        return [
            entry.chunk_options
            if entry.chunk_options is not None
            else self._base_chunk_options(entry.process_options)
            for entry in entries
        ]

    async def _overwrite_sidecar_image_vectors(
        self,
        *,
        doc_id: str,
        sidecar_location: str | None,
        chunk_ids: set[str],
    ) -> None:
        if not self._document_embedder.image_enabled:
            return
        # A unified multimodal embedder (image support probed at startup) fuses the
        # VLM description with the image into one vector, keeping the visual chunk
        # reachable by text queries.
        artifact_dir = sidecar_dir_from_location(sidecar_location)
        if artifact_dir is None or not artifact_dir.exists():
            return

        assets = [
            asset
            for asset in collect_lightrag_drawing_assets(artifact_dir, doc_id=doc_id)
            if asset.chunk_id in chunk_ids and asset.image_path.exists()
        ]
        if not assets:
            return

        descriptions = await self._fetch_chunk_descriptions([a.chunk_id for a in assets])
        inputs = [
            DocumentEmbeddingInput(
                key=asset.chunk_id,
                text=descriptions.get(asset.chunk_id, ""),
                image_path=asset.image_path,
            )
            for asset in assets
        ]
        try:
            embedded, trace = await self._document_embedder.aembed_documents(inputs)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning(
                "Sidecar document embedding failed; preserving existing LightRAG vectors",
                exc_info=True,
            )
            return
        logger.debug(
            "Sidecar document embedding outcomes: fused=%d text=%d fallback=%d failed=%d",
            trace.fused,
            trace.text,
            trace.fused_to_text_fallback,
            trace.failed,
        )
        vectors = {item.key: item.vector for item in embedded}
        if vectors:
            await self._stores.overwrite_chunk_vectors(
                vectors,
                embedding_dim=self._document_embedder.dimension,
            )

    async def _fetch_chunk_descriptions(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: VLM description} to fuse into visual-chunk vectors."""
        rows = await self._stores.fetch_chunk_contents(chunk_ids)
        return {str(row["id"]): str(row.get("content") or "") for row in rows if row.get("id")}

    async def _label_bm25_languages(self, chunk_ids: list[str]) -> None:
        if self._bm25_language_classifier is None or not chunk_ids:
            return
        rows = await self._stores.fetch_chunk_contents(chunk_ids)
        labels = {
            str(row["id"]): self._bm25_language_classifier.detect(str(row.get("content") or ""))
            for row in rows
            if row.get("id")
        }
        if labels:
            await self._stores.update_chunk_bm25_languages(labels)

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _prepare_ingest_item(
    path: str | Path | PreparedIngestFile, *, workspace: str
) -> PreparedIngestFile:
    if isinstance(path, PreparedIngestFile):
        return path
    parser_path = Path(path)
    return PreparedIngestFile(
        parser_path=parser_path,
        source_uri=_raw_path_source_uri(parser_path, workspace=workspace),
        download_locator=str(parser_path.resolve()),
    )


def _with_finalized_local_download_locator(
    metadata_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the exact source location after LightRAG archives parser input."""
    finalized = dict(metadata_record)
    locator = finalized.get("download_locator")
    if not isinstance(locator, str) or "://" in locator:
        return finalized

    source_path = Path(locator)
    archived_path = lightrag_archived_source_path(source_path)
    if source_path.is_file() or not archived_path.is_file():
        return finalized

    resolved = str(archived_path.resolve())
    finalized["download_locator"] = resolved
    if finalized.get("file_path") == locator:
        finalized["file_path"] = resolved
    return finalized


def _raw_path_source_uri(path: Path, *, workspace: str) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return local_source_uri(workspace, Path(digest) / path.name)


def _overlay_metadata(
    defaults: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if defaults is None:
        return overlay
    if overlay is None:
        return defaults
    return {**defaults, **overlay}


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _canonical_file_doc_id(path: Path) -> str:
    """Match LightRAG's file-backed document id derivation."""
    return compute_mdhash_id(normalize_document_file_path(path), prefix="doc-")


__all__ = ["PreparedIngestFile", "UnifiedIngestionEngine"]
