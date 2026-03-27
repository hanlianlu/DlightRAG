# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified-mode document lifecycle — ingestion and deletion.

Entry points: ``unified_ingest()`` and ``unified_delete_files()``.
Extracted from service.py to keep RAGService focused on dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol
    from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine

logger = logging.getLogger(__name__)


def _normalize_ingest_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize engine result to match caption-mode IngestionResult shape.

    Ensures unified mode returns the same top-level keys as caption mode:
    ``status``, ``doc_id``, ``page_count``, ``file_path``, plus any extras
    the engine already provides (e.g. ``render_metadata``).
    """
    result = dict(raw)
    result.setdefault("status", "success")
    return result


# ── Ingestion ──────────────────────────────────────────────────────


async def unified_ingest(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    source_type: Literal["local", "azure_blob", "s3"],
    metadata_index: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified ingestion entry point — dispatches by source_type."""
    if source_type == "local":
        path_str = kwargs.get("path")
        if not path_str:
            raise ValueError("'path' is required for local source_type")
        path = Path(path_str)
        replace = kwargs.get("replace", config.ingestion_replace_default)
        user_metadata = kwargs.get("metadata")

        if path.is_file():
            return await _ingest_single_local(
                engine, hash_index, path, replace, metadata_index, user_metadata
            )
        if path.is_dir():
            return await _ingest_local_dir(
                engine, config, hash_index, path, replace, metadata_index, user_metadata
            )
        raise FileNotFoundError(f"Path not found: {path}")

    if source_type == "azure_blob":
        return await _ingest_azure_blob(
            engine, config, hash_index, metadata_index=metadata_index, **kwargs
        )

    # source_type == "s3"
    return await _ingest_s3(engine, config, hash_index, metadata_index=metadata_index, **kwargs)


async def _ingest_single_local(
    engine: UnifiedRepresentEngine,
    hash_index: HashIndexProtocol,
    path: Path,
    replace: bool,
    metadata_index: Any = None,
    user_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a single local file with dedup."""
    should_skip, content_hash, reason = await hash_index.should_skip_file(path, replace)
    if should_skip:
        logger.info("Skipped (dedup): %s - %s", path.name, reason)
        return {"status": "skipped", "reason": reason, "file_path": str(path)}

    raw = await engine.aingest(file_path=str(path))
    result = _normalize_ingest_result(raw)

    if content_hash and result.get("doc_id"):
        await hash_index.register(content_hash, result["doc_id"], str(path))

    # Upsert document metadata (best-effort)
    if result.get("doc_id") and metadata_index is not None:
        try:
            from dlightrag.storage.metadata_index import extract_system_metadata

            meta = extract_system_metadata(
                path,
                rag_mode="unified",
                page_count=result.get("page_count"),
            )
            # Enrich with PDF-extracted title/author/date from renderer
            render_meta = result.get("render_metadata") or {}
            if render_meta.get("title"):
                meta["doc_title"] = render_meta["title"]
            if render_meta.get("author"):
                meta["doc_author"] = render_meta["author"]
            if render_meta.get("creation_date"):
                meta["creation_date"] = render_meta["creation_date"]
            if user_metadata:
                meta.update(user_metadata)
            await metadata_index.upsert(result["doc_id"], meta)
        except Exception as e:
            logger.warning("Metadata upsert failed for %s: %s", path.name, e)

    return result


async def _ingest_local_dir(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    path: Path,
    replace: bool,
    metadata_index: Any = None,
    user_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recursively ingest all supported files in a directory."""
    from dlightrag.unifiedrepresent.renderer import (
        _IMAGE_EXTENSIONS,
        _OFFICE_EXTENSIONS,
    )

    supported = {".pdf"} | _IMAGE_EXTENSIONS | _OFFICE_EXTENSIONS
    files = sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in supported)

    if not files:
        logger.warning("No supported files found in %s", path)
        return {
            "status": "success",
            "source_type": "local",
            "folder": str(path),
            "processed": 0,
            "skipped": 0,
            "total_files": 0,
        }

    from dlightrag.utils.concurrency import bounded_gather

    coros = [
        _ingest_single_local(engine, hash_index, file_path, replace, metadata_index, user_metadata)
        for file_path in files
    ]
    results = await bounded_gather(
        coros,
        max_concurrent=config.max_concurrent_ingestion,
        task_name="unified-ingestion",
    )

    # Separate successes from failures
    completed: list[dict[str, Any]] = []
    skipped = 0
    for r in results:
        if isinstance(r, Exception):
            continue  # already logged by bounded_gather
        completed.append(r)
        if r.get("status") == "skipped":
            skipped += 1

    processed = len(completed) - skipped
    logger.info(
        "Directory ingestion complete: %s (%d processed, %d skipped)",
        path,
        processed,
        skipped,
    )
    return {
        "status": "success",
        "source_type": "local",
        "folder": str(path),
        "results": completed,
        "processed": processed,
        "skipped": skipped,
        "total_files": len(files),
    }


async def _ingest_azure_blob(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    metadata_index: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Download blob(s) to temp -> visual pipeline."""
    from dlightrag.sourcing.azure_blob import AzureBlobDataSource

    container_name = kwargs.get("container_name")
    if not container_name:
        raise ValueError("'container_name' is required for azure_blob source_type")
    blob_path: str | None = kwargs.get("blob_path")
    prefix: str | None = kwargs.get("prefix")
    source = kwargs.get("source")

    if source is None:
        source = AzureBlobDataSource(
            container_name=container_name,
            connection_string=config.blob_connection_string,
        )

    replace: bool = kwargs.get("replace", config.ingestion_replace_default)
    if blob_path is None and prefix is None:
        prefix = ""

    try:
        if blob_path:
            return await _ingest_single_blob(
                engine, config, hash_index, source, blob_path, replace, metadata_index
            )

        from dlightrag.utils.concurrency import bounded_gather

        blob_ids = await source.alist_documents(prefix=prefix)
        coros = [
            _ingest_single_blob(engine, config, hash_index, source, bid, replace, metadata_index)
            for bid in blob_ids
        ]
        raw_results = await bounded_gather(
            coros,
            max_concurrent=config.max_concurrent_ingestion,
            task_name="unified-blob-ingestion",
        )
        completed = [r for r in raw_results if not isinstance(r, Exception)]

        return {
            "status": "success",
            "source_type": "azure_blob",
            "container": container_name,
            "prefix": prefix,
            "results": completed,
            "processed": len(completed),
        }
    finally:
        if hasattr(source, "aclose"):
            await source.aclose()


async def _ingest_single_blob(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    source: Any,
    blob_path: str,
    replace: bool = False,
    metadata_index: Any = None,
) -> dict[str, Any]:
    """Download one blob to temp dir, ingest via unified engine, cleanup."""
    tmpdir = config.temp_dir / uuid.uuid4().hex[:12]
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        target = tmpdir / Path(blob_path).name
        content = await source.aload_document(blob_path)
        await asyncio.to_thread(target.write_bytes, content)

        should_skip, content_hash, reason = await hash_index.should_skip_file(target, replace)
        if should_skip:
            logger.info("Skipped (dedup): %s - %s", blob_path, reason)
            return {"status": "skipped", "reason": reason, "blob_path": blob_path}

        logger.info("Downloaded blob to temp: %s", target)
        raw = await engine.aingest(file_path=str(target))
        result = _normalize_ingest_result(raw)

        if content_hash and result.get("doc_id"):
            await hash_index.register(content_hash, result["doc_id"], f"azure://{blob_path}")

        # Upsert document metadata (best-effort)
        if result.get("doc_id") and metadata_index is not None:
            try:
                from dlightrag.storage.metadata_index import extract_system_metadata

                meta = extract_system_metadata(
                    target,
                    rag_mode="unified",
                    page_count=result.get("page_count"),
                )
                render_meta = result.get("render_metadata") or {}
                if render_meta.get("title"):
                    meta["doc_title"] = render_meta["title"]
                if render_meta.get("author"):
                    meta["doc_author"] = render_meta["author"]
                if render_meta.get("creation_date"):
                    meta["creation_date"] = render_meta["creation_date"]
                await metadata_index.upsert(result["doc_id"], meta)
            except Exception as e:
                logger.warning("Metadata upsert failed for %s: %s", target.name, e)

        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _ingest_s3(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    metadata_index: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Download S3 object(s) to temp -> unified visual pipeline."""
    from dlightrag.sourcing.aws_s3 import S3DataSource

    bucket = kwargs.get("bucket")
    if not bucket:
        raise ValueError("'bucket' is required for s3 source_type")
    key: str | None = kwargs.get("key")
    prefix: str | None = kwargs.get("prefix")
    replace: bool = kwargs.get("replace", config.ingestion_replace_default)

    source = S3DataSource(bucket=bucket)
    try:
        if key:
            return await _ingest_single_s3(
                engine, config, hash_index, source, bucket, key, replace, metadata_index
            )

        from dlightrag.utils.concurrency import bounded_gather

        keys = await source.alist_documents(prefix=prefix or "")
        coros = [
            _ingest_single_s3(
                engine, config, hash_index, source, bucket, k, replace, metadata_index
            )
            for k in keys
        ]
        raw_results = await bounded_gather(
            coros,
            max_concurrent=config.max_concurrent_ingestion,
            task_name="unified-s3-ingestion",
        )
        completed = [r for r in raw_results if not isinstance(r, Exception)]

        return {
            "status": "success",
            "source_type": "s3",
            "bucket": bucket,
            "prefix": prefix,
            "results": completed,
            "processed": len(completed),
        }
    finally:
        await source.aclose()


async def _ingest_single_s3(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    source: Any,
    bucket: str,
    key: str,
    replace: bool = False,
    metadata_index: Any = None,
) -> dict[str, Any]:
    """Download one S3 object to temp dir, ingest via unified engine, cleanup."""
    tmpdir = config.temp_dir / uuid.uuid4().hex[:12]
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        target = tmpdir / Path(key).name
        content = await source.aload_document(key)
        await asyncio.to_thread(target.write_bytes, content)

        should_skip, content_hash, reason = await hash_index.should_skip_file(target, replace)
        if should_skip:
            logger.info("Skipped (dedup): %s - %s", key, reason)
            return {"status": "skipped", "reason": reason, "key": key}

        logger.info("Downloaded S3 object to temp: %s", target)
        raw = await engine.aingest(file_path=str(target))
        result = _normalize_ingest_result(raw)

        if content_hash and result.get("doc_id"):
            await hash_index.register(content_hash, result["doc_id"], f"s3://{bucket}/{key}")

        # Upsert document metadata (best-effort)
        if result.get("doc_id") and metadata_index is not None:
            try:
                from dlightrag.storage.metadata_index import extract_system_metadata

                meta = extract_system_metadata(
                    target,
                    rag_mode="unified",
                    page_count=result.get("page_count"),
                )
                render_meta = result.get("render_metadata") or {}
                if render_meta.get("title"):
                    meta["doc_title"] = render_meta["title"]
                if render_meta.get("author"):
                    meta["doc_author"] = render_meta["author"]
                if render_meta.get("creation_date"):
                    meta["creation_date"] = render_meta["creation_date"]
                await metadata_index.upsert(result["doc_id"], meta)
            except Exception as e:
                logger.warning("Metadata upsert failed for %s: %s", target.name, e)

        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Deletion ───────────────────────────────────────────────────────


async def unified_delete_files(
    engine: UnifiedRepresentEngine,
    hash_index: HashIndexProtocol,
    lightrag: Any,
    *,
    file_paths: list[str] | None = None,
    filenames: list[str] | None = None,
    metadata_index: Any = None,
) -> list[dict[str, Any]]:
    """Delete files in unified mode — delegates to cascade_delete.

    Collects deletion context per identifier, then calls
    ``cascade_delete()`` which handles all 5 layers with per-layer fault
    isolation.
    """
    from dlightrag.core.ingestion.cleanup import cascade_delete, collect_deletion_context

    if not file_paths and not filenames:
        return []

    results: list[dict[str, Any]] = []
    identifiers: list[str] = []
    if file_paths:
        identifiers.extend(file_paths)
    if filenames:
        identifiers.extend(filenames)

    for identifier in identifiers:
        ctx = await collect_deletion_context(
            identifier=identifier,
            hash_index=hash_index,
            lightrag=lightrag,
            metadata_index=metadata_index,
        )

        deletion_result: dict[str, Any] = {
            "identifier": identifier,
            "doc_ids_found": list(ctx.doc_ids),
            "sources_used": ctx.sources_used,
            "cleanup_results": {},
            "status": "deleted",
        }

        if not ctx.doc_ids:
            deletion_result["status"] = "not_found"
            deletion_result["file_path"] = identifier
            deletion_result["doc_id"] = None
            results.append(deletion_result)
            continue

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lightrag,
            visual_chunks=engine.visual_chunks,
            hash_index=hash_index,
            metadata_index=metadata_index,
        )

        deletion_result["cleanup_results"] = stats
        deletion_result["file_path"] = list(ctx.file_paths)[0] if ctx.file_paths else identifier
        deletion_result["doc_id"] = list(ctx.doc_ids)[0] if ctx.doc_ids else None
        results.append(deletion_result)

    return results


__all__ = ["unified_delete_files", "unified_ingest"]
