# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ingestion pipeline for RAG document processing.

Provides document ingestion with content filtering to prevent
pollution from MinerU's discarded blocks (headers, footers, page numbers, etc.).

Flow: parse_document() -> policy filter -> insert_content_list()
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from raganything import RAGAnything

from dlightrag.converters.office import create_converter
from dlightrag.core.ingestion.cleanup import collect_deletion_context
from dlightrag.core.ingestion.docling_postprocess import (
    find_docling_json_export,
    rebuild_text_items_from_docling_json,
)
from dlightrag.core.ingestion.hash_index import HashIndex
from dlightrag.core.ingestion.policy import IngestionPolicy, PolicyStats

if TYPE_CHECKING:
    from dlightrag.captionrag.vlm_parser import VlmOcrParser
    from dlightrag.config import DlightragConfig
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol
    from dlightrag.sourcing.base import AsyncDataSource

logger = logging.getLogger(__name__)


class IngestionCancelledError(Exception):
    """Raised when ingestion task is cancelled by the caller."""


class IngestionResult(BaseModel):
    """Ingestion result model."""

    status: Literal["success", "error"] = "success"
    processed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    total_files: int | None = Field(default=None, ge=0)
    source_type: str | None = None
    doc_id: str | None = None
    source_path: str | None = None
    stats: PolicyStats | None = None
    error: str | None = None
    skipped_files: list[str] | None = None

    # Optional metadata
    folder: str | None = None
    container: str | None = None
    prefix: str | None = None
    blob_path: str | None = None


class IngestionPipeline:
    """Caption-mode document ingestion with content filtering.

    Uses parse_document -> policy filter -> insert_content_list flow
    to filter out MinerU discarded blocks before indexing. End users
    don't instantiate this directly — RAGService owns one and dispatches
    to it via ``aingest(source_type=...)``.

    Public API:
      - aingest(source_type, **kwargs)         — dispatcher
      - aingest_content_list(content_list, ...) — pre-parsed input
      - adelete_files(...)
      - alist_failed_docs() / aretry_failed_docs()
    """

    def __init__(
        self,
        rag_instance: RAGAnything,
        config: DlightragConfig,
        max_concurrent: int = 4,
        policy: IngestionPolicy | None = None,
        mineru_backend: str | None = None,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
        hash_index: HashIndexProtocol | None = None,
        vlm_parser: VlmOcrParser | None = None,
        metadata_index: Any = None,
    ) -> None:
        self.rag = rag_instance
        self.config = config
        self.max_concurrent = max_concurrent

        # MinerU backend for parse_document kwargs (None if using docling)
        self.mineru_backend = mineru_backend

        # Content filtering policy
        self.policy = policy or IngestionPolicy()

        # LibreOffice converter for Excel-to-PDF auto-conversion
        self.converter = create_converter(config)

        # Hash index for content deduplication (auto-detected by caller)
        self._hash_index = hash_index or HashIndex(
            self.config.working_dir_path,
        )

        # VLM OCR parser (used when config.parser == "vlm")
        self.vlm_parser = vlm_parser

        # Document metadata index (PGMetadataIndex or None)
        self._metadata_index = metadata_index

        # Callback for cancellation checking (replaces Redis task registry)
        self._cancel_checker = cancel_checker
        self._cancel_check_interval = 5  # Check every N files

    async def _check_cancelled(self) -> None:
        """Check if this ingestion task has been cancelled.

        Uses the cancel_checker callback provided at construction time.

        Raises:
            IngestionCancelledError: If task was cancelled
            asyncio.CancelledError: Re-raised for asyncio cancellation
        """
        # Check asyncio cancellation first
        if asyncio.current_task() and asyncio.current_task().cancelled():  # type: ignore[union-attr]
            raise asyncio.CancelledError()

        # Check via caller-provided callback
        if self._cancel_checker and await self._cancel_checker():
            logger.info("Ingestion task cancelled by caller")
            raise IngestionCancelledError("Task cancelled by caller")

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _get_storage_dir(
        self,
        base_dir: Path,
        source_type: str,
        *path_parts: str,
    ) -> Path:
        """Get storage directory for a specific source type."""
        storage_dir = base_dir / source_type / Path(*path_parts)
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir

    def _get_artifacts_dir(self, source_type: str, *path_parts: str) -> Path:
        """Get artifacts directory for a specific source type."""
        return self._get_storage_dir(
            self.config.artifacts_dir,
            source_type,
            *path_parts,
        )

    def _create_temp_dir(self) -> Path:
        """Create a unique temp directory under working_dir/.tmp/."""
        import uuid

        tmpdir = self.config.temp_dir / uuid.uuid4().hex[:12]
        tmpdir.mkdir(parents=True, exist_ok=True)
        return tmpdir

    async def _prepare_for_parsing(self, file_path: Path, tmpdir: Path) -> Path:
        """Prepare file for parsing. Converts Excel to PDF if needed, otherwise returns original."""
        return await self._maybe_convert_excel_to_pdf(file_path, tmpdir)

    async def _maybe_convert_excel_to_pdf(
        self,
        file_path: Path,
        output_dir: Path,
    ) -> Path:
        """Auto-convert Excel to PDF if configured, otherwise return original path."""
        if not self.converter.should_convert(file_path):
            return file_path

        try:
            pdf_path = await asyncio.to_thread(
                self.converter.convert_to_pdf,
                source_path=file_path,
                output_dir=output_dir,
            )
            logger.info(f"Auto-converted Excel to PDF: {file_path.name} -> {pdf_path.name}")
            return pdf_path
        except Exception as e:
            logger.warning(
                f"Excel-to-PDF conversion failed for {file_path.name}, using original: {e}"
            )
            return file_path

    async def _download_blob_to_temp(
        self,
        source: AsyncDataSource,
        blob_path: str,
        tmpdir: Path,
    ) -> Path:
        """Download Azure Blob to temp directory, auto-converting Excel to PDF."""
        target_path = tmpdir / Path(blob_path).name
        content = await source.aload_document(blob_path)
        await asyncio.to_thread(target_path.write_bytes, content)
        logger.info("Downloaded blob to temp: %s", target_path)
        return await self._prepare_for_parsing(target_path, tmpdir)

    # ─────────────────────────────────────────────────────────────────
    # Core ingestion with policy filtering
    # ─────────────────────────────────────────────────────────────────

    async def _ingest_single_file_with_policy(
        self,
        file_path: Path,
        artifacts_dir: Path,
        content_hash: str | None = None,
        source_uri: str | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a single file with content filtering.

        Flow: parse_document() -> policy.apply() -> insert_content_list()
        """
        try:
            # Check for cancellation before heavy processing
            await self._check_cancelled()

            # Step 1: Parse document (returns raw content_list)
            if self.vlm_parser:
                content_list, doc_id = await self.vlm_parser.parse(
                    file_path=str(file_path),
                    output_dir=str(artifacts_dir),
                )
            else:
                parse_method = self.config.parse_method
                parse_kwargs: dict[str, Any] = {}
                if self.config.parser == "mineru" and self.mineru_backend:
                    parse_kwargs["backend"] = self.mineru_backend
                if self.config.parser == "mineru":
                    if self.config.mineru_timeout is not None:
                        parse_kwargs["timeout"] = self.config.mineru_timeout
                    if self.config.mineru_vlm_url:
                        parse_kwargs["vlm_url"] = self.config.mineru_vlm_url
                elif self.config.parser == "docling":
                    parse_kwargs["table_mode"] = self.config.docling_table_mode
                    parse_kwargs["tables"] = self.config.docling_tables
                    parse_kwargs["allow_ocr"] = self.config.docling_allow_ocr
                    if self.config.docling_artifacts_path:
                        parse_kwargs["artifacts_path"] = self.config.docling_artifacts_path
                content_list, doc_id = await self.rag.parse_document(
                    file_path=str(file_path),
                    output_dir=str(artifacts_dir),
                    parse_method=parse_method,
                    **parse_kwargs,
                )

            # Step 2: Apply policy to filter discarded/noise content
            result = self.policy.apply(content_list)

            # Log stats with drop rate
            logger.info(
                f"Policy filter [{file_path.name}]: "
                f"total={result.stats.total}, indexed={result.stats.indexed}, "
                f"dropped={result.stats.dropped_by_type} ({result.stats.drop_rate:.1f}%)"
            )

            # Step 2.5: Docling JSON post-processing
            # Rebuild text items with heading markers and correct page_idx.
            # The helper does sync file I/O + json.loads — push to a thread.
            if self.config.parser == "docling" and result.index_stream:
                json_path = find_docling_json_export(artifacts_dir, file_path)
                if json_path is not None:
                    improved_texts = await asyncio.to_thread(
                        rebuild_text_items_from_docling_json, json_path
                    )
                    if improved_texts is not None:
                        non_text = [i for i in result.index_stream if i.get("type") != "text"]
                        result.index_stream = improved_texts + non_text

            # Step 3: Insert filtered content
            if result.index_stream:
                await self.rag.insert_content_list(  # type: ignore[misc]
                    content_list=result.index_stream,
                    file_path=source_uri or str(file_path),
                    doc_id=doc_id,
                )

            # Step 3.5: Inject page metadata into chunks
            if result.index_stream and doc_id:
                try:
                    from dlightrag.core.ingestion.page_metadata import (
                        inject_page_idx_to_chunks,
                    )

                    updated = await inject_page_idx_to_chunks(
                        self.rag.lightrag,
                        doc_id,
                        result.index_stream,
                    )
                    if updated:
                        logger.info(f"Injected page_idx for {updated} chunks [{file_path.name}]")
                except Exception as e:
                    logger.warning(f"Page metadata injection failed for {file_path.name}: {e}")

            # Step 4: Register hash for deduplication (if hash was provided)
            if content_hash and doc_id:
                await self._hash_index.register(content_hash, doc_id, source_uri or str(file_path))

            # Step 5: Upsert document metadata (best-effort)
            if doc_id and self._metadata_index is not None:
                try:
                    from dlightrag.core.ingestion.metadata_extract import extract_system_metadata

                    meta = extract_system_metadata(
                        file_path,
                        rag_mode="caption",
                        page_count=getattr(result, "page_count", None),
                    )
                    # Extract PDF title/author if available
                    if file_path.suffix.lower() == ".pdf":
                        try:
                            import pypdfium2 as pdfium

                            doc = pdfium.PdfDocument(str(file_path))
                            try:
                                pdf_meta = doc.get_metadata_dict()
                                if pdf_meta.get("Title"):
                                    meta["doc_title"] = pdf_meta["Title"]
                                if pdf_meta.get("Author"):
                                    meta["doc_author"] = pdf_meta["Author"]
                                if pdf_meta.get("CreationDate"):
                                    meta["creation_date"] = pdf_meta["CreationDate"]
                            finally:
                                doc.close()
                        except Exception:
                            pass  # PDF metadata extraction is best-effort
                    if user_metadata:
                        meta.update(user_metadata)
                    await self._metadata_index.upsert(doc_id, meta)
                except Exception as e:
                    logger.warning("Metadata upsert failed for %s: %s", file_path.name, e)

            return IngestionResult(
                status="success",
                processed=1,
                doc_id=doc_id,
                source_path=source_uri or str(file_path),
                stats=result.stats,
            )

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return IngestionResult(
                status="error",
                processed=0,
                source_path=str(file_path),
                error=str(e),
            )

    # ─────────────────────────────────────────────────────────────────
    # Internal: per-source ingestion (called by RAGService.aingest dispatcher)
    # ─────────────────────────────────────────────────────────────────

    async def _aingest_from_local(
        self,
        path: Path,
        replace: bool = False,
        recursive: bool = True,
        user_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest from local filesystem (file or directory).

        RAGAnything's parser automatically handles all supported file types
        (PDF, images, Office documents, etc.) based on file extension.

        Pre-parse deduplication: Files with identical content (by SHA256 hash)
        are skipped unless replace=True.

        Args:
            path: Path to file or directory to ingest
            replace: If True, delete existing docs with same basename before ingesting
            recursive: If True, recursively process directories
        """
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        artifacts_dir = self._get_artifacts_dir("local")

        # Single file
        if path.is_file():
            # Check for duplicate BEFORE copying/parsing
            should_skip, content_hash, reason = await self._hash_index.should_skip_file(
                path, replace
            )

            if should_skip:
                logger.info(f"Skipped (dedup): {path.name} - {reason}")
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=1,
                    total_files=1,
                    source_type="local",
                    source_path=str(path),
                    skipped_files=[path.name],
                )

            if replace:
                await self.adelete_files(filenames=[path.name], delete_source=True)

            source_uri = str(path.resolve())
            tmpdir = self._create_temp_dir()
            try:
                working_path = await self._prepare_for_parsing(path, tmpdir)

                logger.info(
                    "Ingesting local file: %s",
                    path.resolve(),
                )

                result = await self._ingest_single_file_with_policy(
                    working_path,
                    artifacts_dir,
                    content_hash=content_hash,
                    source_uri=source_uri,
                    user_metadata=user_metadata,
                )
                result.source_type = "local"
                result.total_files = 1
                return result
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Directory
        if path.is_dir():
            # Collect files to process with dedup check
            pattern = "**/*" if recursive else "*"
            files_to_process: list[tuple[Path, str, str]] = []
            skipped_count = 0
            skipped_files: list[str] = []

            for file_path in path.glob(pattern):
                if not file_path.is_file():
                    continue

                # Check for duplicate BEFORE copying
                (
                    should_skip,
                    content_hash,
                    reason,
                ) = await self._hash_index.should_skip_file(file_path, replace)

                if should_skip:
                    skipped_count += 1
                    skipped_files.append(file_path.name)
                    logger.info(f"Skipped (dedup): {file_path.name}")
                    continue

                if replace:
                    await self.adelete_files(filenames=[file_path.name], delete_source=True)

                files_to_process.append((file_path, content_hash or "", str(file_path.resolve())))

            total_files = len(files_to_process) + skipped_count

            if not files_to_process:
                if skipped_count > 0:
                    logger.info(
                        f"No new files to ingest from {path} "
                        f"({skipped_count} skipped as duplicates)"
                    )
                else:
                    logger.warning(f"No matching files found in {path}")
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=skipped_count,
                    total_files=total_files,
                    source_type="local",
                    folder=str(path),
                    skipped_files=skipped_files if skipped_files else None,
                )

            logger.info(
                "Ingesting %d local files from %s (%d skipped, max %d concurrent)",
                len(files_to_process),
                path,
                skipped_count,
                self.max_concurrent,
            )

            # Process files concurrently with bounded worker pool
            from dlightrag.utils.concurrency import bounded_gather

            async def _ingest_local_single(fp: Path, ch: str, uri: str) -> IngestionResult:
                tmpdir = self._create_temp_dir()
                try:
                    working = await self._prepare_for_parsing(fp, tmpdir)
                    return await self._ingest_single_file_with_policy(
                        working,
                        artifacts_dir,
                        content_hash=ch if ch else None,
                        source_uri=uri,
                        user_metadata=user_metadata,
                    )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            coros = [_ingest_local_single(fp, ch, uri) for fp, ch, uri in files_to_process]
            results = await bounded_gather(
                coros, max_concurrent=self.max_concurrent, task_name="ingestion"
            )

            # Count successes
            success_count = sum(
                1 for r in results if isinstance(r, IngestionResult) and r.status == "success"
            )

            # Invalidate hash index cache after batch ingestion
            self._hash_index.invalidate()

            return IngestionResult(
                status="success",
                processed=success_count,
                skipped=skipped_count,
                total_files=total_files,
                source_type="local",
                folder=str(path),
                skipped_files=skipped_files if skipped_files else None,
            )

        raise ValueError(f"Invalid path (neither file nor directory): {path}")

    # ─────────────────────────────────────────────────────────────────
    # Internal: Azure Blob ingestion (called by RAGService.aingest dispatcher)
    # ─────────────────────────────────────────────────────────────────

    async def _aingest_from_azure_blob(
        self,
        source: AsyncDataSource,
        container_name: str,
        blob_path: str | None = None,
        prefix: str | None = None,
        replace: bool = False,
    ) -> IngestionResult:
        """Ingest from Azure Blob Storage.

        Pre-parse deduplication: Files with identical content (by SHA256 hash)
        are skipped unless replace=True.

        Args:
            source: Azure Blob data source (from dlightrag.sourcing.azure_blob)
            container_name: Name of the Azure Blob container
            blob_path: Specific blob path to ingest (mutually exclusive with prefix)
            prefix: Prefix to filter blobs (mutually exclusive with blob_path)
            replace: If True, delete existing docs with same basename before ingesting
        """
        if blob_path and prefix:
            raise ValueError("blob_path and prefix are mutually exclusive")

        artifacts_dir = self._get_artifacts_dir("azure_blobs", container_name)

        # Single blob
        if blob_path:
            basename = Path(blob_path).name
            source_uri = f"azure://{container_name}/{blob_path}"
            tmpdir = self._create_temp_dir()
            try:
                temp_file = await self._download_blob_to_temp(source, blob_path, tmpdir)

                # Dedup check after download (need local file for hash)
                should_skip, content_hash, reason = await self._hash_index.should_skip_file(
                    temp_file, replace
                )

                if should_skip:
                    logger.info(f"Skipped (dedup): {blob_path} - {reason}")
                    return IngestionResult(
                        status="success",
                        processed=0,
                        skipped=1,
                        total_files=1,
                        source_type="azure_blobs",
                        container=container_name,
                        blob_path=blob_path,
                        skipped_files=[basename],
                    )

                if replace:
                    await self.adelete_files(filenames=[basename], delete_source=True)

                logger.info("Ingesting Azure blob: %s", blob_path)

                result = await self._ingest_single_file_with_policy(
                    temp_file, artifacts_dir, content_hash=content_hash, source_uri=source_uri
                )
                result.source_type = "azure_blobs"
                result.container = container_name
                result.blob_path = blob_path
                result.total_files = 1
                return result
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Prefix batch
        if prefix is not None:
            blob_ids = await source.alist_documents(prefix=prefix)
            if not blob_ids:
                logger.warning("No blobs found in %s with prefix %s", container_name, prefix)
                return IngestionResult(
                    status="success",
                    processed=0,
                    total_files=0,
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                )

            logger.info(
                "Preparing to ingest %d blobs from %s/%s (max %d concurrent)",
                len(blob_ids),
                container_name,
                prefix,
                self.max_concurrent,
            )

            # Phase 1: Download all blobs to temp dirs concurrently
            from dlightrag.utils.concurrency import bounded_gather as _bounded_gather

            async def download_one(blob_id: str) -> tuple[Path, Path] | None:
                """Download blob to temp, return (temp_file, tmpdir) or None."""
                tmpdir = self._create_temp_dir()
                try:
                    temp_file = await self._download_blob_to_temp(source, blob_id, tmpdir)
                    return (temp_file, tmpdir)
                except Exception as exc:
                    logger.error("Failed to download blob %s: %s", blob_id, exc)
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return None

            download_coros = [download_one(bid) for bid in blob_ids]
            downloaded = await _bounded_gather(
                download_coros, max_concurrent=self.max_concurrent, task_name="blob-download"
            )
            # Pair each result with its blob_id for source_uri construction
            valid_downloads: list[tuple[str, Path, Path]] = []
            for blob_id, dl in zip(blob_ids, downloaded, strict=True):
                if dl is not None:
                    valid_downloads.append((blob_id, dl[0], dl[1]))

            if not valid_downloads:
                logger.warning(
                    "All downloads failed for prefix %s in container %s",
                    prefix,
                    container_name,
                )
                return IngestionResult(
                    status="success",
                    processed=0,
                    total_files=len(blob_ids),
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                )

            # Phase 2: Check for duplicates and collect files to process
            files_to_process: list[tuple[Path, str, str, Path]] = []  # (file, hash, uri, tmpdir)
            skipped_count = 0
            skipped_files: list[str] = []

            for i, (blob_id, temp_file, tmpdir) in enumerate(valid_downloads):
                # Check for cancellation periodically
                if i % self._cancel_check_interval == 0:
                    await self._check_cancelled()

                (
                    should_skip,
                    content_hash,
                    reason,
                ) = await self._hash_index.should_skip_file(temp_file, replace)

                if should_skip:
                    skipped_count += 1
                    skipped_files.append(temp_file.name)
                    logger.info(f"Skipped (dedup): {temp_file.name}")
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    continue

                if replace:
                    await self.adelete_files(filenames=[temp_file.name], delete_source=True)

                source_uri = f"azure://{container_name}/{blob_id}"
                files_to_process.append((temp_file, content_hash or "", source_uri, tmpdir))

            total_files = len(files_to_process) + skipped_count

            if not files_to_process:
                if skipped_count > 0:
                    logger.info(
                        f"No new blobs to ingest from {container_name}/{prefix} "
                        f"({skipped_count} skipped as duplicates)"
                    )
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=skipped_count,
                    total_files=total_files,
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                    skipped_files=skipped_files if skipped_files else None,
                )

            # Phase 3: Process downloaded files concurrently
            await self._check_cancelled()

            logger.info(
                "Processing %d blobs from %s/%s (%d skipped)",
                len(files_to_process),
                container_name,
                prefix,
                skipped_count,
            )

            from dlightrag.utils.concurrency import bounded_gather

            async def ingest_one(fp: Path, ch: str, uri: str, tmpdir: Path) -> IngestionResult:
                try:
                    return await self._ingest_single_file_with_policy(
                        fp, artifacts_dir, content_hash=ch if ch else None, source_uri=uri
                    )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            coros = [ingest_one(fp, ch, uri, td) for fp, ch, uri, td in files_to_process]
            results = await bounded_gather(
                coros, max_concurrent=self.max_concurrent, task_name="blob-ingestion"
            )

            success_count = sum(
                1 for r in results if isinstance(r, IngestionResult) and r.status == "success"
            )

            # Invalidate hash index cache after batch ingestion
            self._hash_index.invalidate()

            return IngestionResult(
                status="success",
                processed=success_count,
                skipped=skipped_count,
                total_files=total_files,
                source_type="azure_blobs",
                container=container_name,
                prefix=prefix,
                skipped_files=skipped_files if skipped_files else None,
            )

        raise ValueError("Must provide either blob_path or prefix for Azure Blob ingestion")

    # ─────────────────────────────────────────────────────────────────
    # Public API: Content list ingestion
    # ─────────────────────────────────────────────────────────────────

    async def aingest_content_list(
        self,
        content_list: list[dict[str, Any]],
        file_path: str = "content_list",
        display_stats: bool = True,
    ) -> IngestionResult:
        """Ingest an in-memory list of structured content with policy filtering."""
        if not content_list:
            logger.warning("Empty content list provided for ingestion")
            return IngestionResult(status="success", processed=0)

        # Generate doc_id from content — sample head + tail + length for uniqueness
        from lightrag.utils import compute_mdhash_id

        content_repr = f"{file_path}|n={len(content_list)}|{content_list[:5]}|{content_list[-1:]}"
        doc_id = compute_mdhash_id(content_repr, prefix="doc-")

        # Apply policy
        result = self.policy.apply(content_list)

        if display_stats:
            logger.info(
                f"Policy filter [{file_path}]: "
                f"total={result.stats.total}, indexed={result.stats.indexed}, "
                f"dropped={result.stats.dropped_by_type} ({result.stats.drop_rate:.1f}%)"
            )

        if result.index_stream:
            await self.rag.insert_content_list(  # type: ignore[misc]
                content_list=result.index_stream,
                file_path=file_path,
                doc_id=doc_id,
                display_stats=display_stats,
            )

        return IngestionResult(
            status="success",
            processed=len(result.index_stream),
            doc_id=doc_id,
            source_path=file_path,
            stats=result.stats,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API: File management
    # ─────────────────────────────────────────────────────────────────

    async def alist_ingested_files(self) -> list[dict[str, Any]]:
        """List all ingested files from hash index."""
        return await self._hash_index.list_all()

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        delete_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Delete files from RAG storage.

        Finds doc_id(s) via hash index / KV doc_status, then delegates
        to LightRAG's adelete_by_doc_id for comprehensive cleanup across
        all storage layers.

        Args:
            file_paths: List of full file paths to delete
            filenames: List of filenames (basenames) to delete
            delete_source: Whether to delete source files (default True)

        Returns:
            List of deletion results with detailed cleanup status
        """
        if not file_paths and not filenames:
            return []

        results: list[dict[str, Any]] = []

        # Ensure LightRAG is initialized before deletion
        if hasattr(self.rag, "_ensure_lightrag_initialized"):
            await self.rag._ensure_lightrag_initialized()

        lightrag = getattr(self.rag, "lightrag", None)
        if not lightrag:
            logger.warning("LightRAG not initialized - some cleanup may be incomplete")

        # Collect all identifiers to process
        identifiers: list[str] = []
        if file_paths:
            identifiers.extend(file_paths)
        if filenames:
            identifiers.extend(filenames)

        for identifier in identifiers:
            # Phase 1: Multi-strategy context collection
            ctx = await collect_deletion_context(
                identifier=identifier,
                hash_index=self._hash_index,
                lightrag=lightrag,
                metadata_index=self._metadata_index,
            )

            # Initialize result with context info
            deletion_result: dict[str, Any] = {
                "identifier": identifier,
                "doc_ids_found": list(ctx.doc_ids),
                "sources_used": ctx.sources_used,
                "cleanup_results": {},
                "status": "deleted",
            }

            # Check if we have any data to delete
            if not ctx.doc_ids:
                deletion_result["status"] = "not_found"
                deletion_result["file_path"] = identifier
                deletion_result["doc_id"] = None
                results.append(deletion_result)
                continue

            # Phase 2: Delete via LightRAG adelete_by_doc_id for each doc_id
            for doc_id in ctx.doc_ids:
                if lightrag and hasattr(lightrag, "adelete_by_doc_id"):
                    try:
                        result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = (
                            result.status if hasattr(result, "status") else "completed"
                        )
                        logger.info(f"LightRAG deletion for {doc_id}: completed")
                    except Exception as exc:
                        logger.warning(f"LightRAG deletion failed for {doc_id}: {exc}")
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = f"error: {exc}"

            # Phase 3a: Remove from hash index and metadata index
            for content_hash in ctx.content_hashes:
                if await self._hash_index.remove(content_hash):
                    deletion_result["cleanup_results"]["hash_index"] = "removed"
                    logger.info(f"Removed hash index entry: {content_hash[:30]}...")

            if self._metadata_index is not None:
                for doc_id in ctx.doc_ids:
                    try:
                        await self._metadata_index.delete(doc_id)
                        deletion_result["cleanup_results"]["metadata_index"] = "removed"
                    except Exception as exc:
                        logger.warning("Metadata index delete failed for %s: %s", doc_id, exc)

            # Phase 3b: Delete artifacts (best-effort glob search by filename stem)
            if delete_source:
                artifacts_dir = self.config.artifacts_dir
                for fp in ctx.file_paths:
                    stem = Path(fp).stem
                    for match in artifacts_dir.rglob(f"{stem}*"):
                        if match.is_dir():
                            shutil.rmtree(match, ignore_errors=True)
                            logger.info(f"Deleted artifacts directory: {match}")
                deletion_result["cleanup_results"]["artifacts"] = "cleaned"

            # Top-level file_path / doc_id are part of the public delete
            # response shape (REST /files DELETE + MCP delete_files JSON-
            # serialise this dict directly). When a single delete request
            # resolves to multiple doc_ids, we surface the first; the full
            # set is in ``doc_ids_found``.
            deletion_result["file_path"] = list(ctx.file_paths)[0] if ctx.file_paths else identifier
            deletion_result["doc_id"] = list(ctx.doc_ids)[0] if ctx.doc_ids else None

            results.append(deletion_result)

        return results


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "IngestionCancelledError",
]
