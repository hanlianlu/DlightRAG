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


# ── Ingestion ──────────────────────────────────────────────────────


async def unified_ingest(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    hash_index: HashIndexProtocol,
    source_type: Literal["local", "azure_blob", "snowflake"],
    metadata_index: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Unified ingestion entry point — dispatches by source_type."""
    if source_type == "local":
        path = Path(kwargs["path"])
        replace = kwargs.get("replace", config.ingestion_replace_default)

        if path.is_file():
            return await _ingest_single_local(engine, hash_index, path, replace, metadata_index)
        if path.is_dir():
            return await _ingest_local_dir(engine, config, hash_index, path, replace, metadata_index)
        raise FileNotFoundError(f"Path not found: {path}")

    if source_type == "azure_blob":
        return await _ingest_azure_blob(engine, config, hash_index, metadata_index=metadata_index, **kwargs)

    # source_type == "snowflake"
    return await _ingest_snowflake(engine, config, **kwargs)


async def _ingest_single_local(
    engine: UnifiedRepresentEngine,
    hash_index: HashIndexProtocol,
    path: Path,
    replace: bool,
    metadata_index: Any = None,
) -> dict[str, Any]:
    """Ingest a single local file with dedup."""
    should_skip, content_hash, reason = await hash_index.should_skip_file(path, replace)
    if should_skip:
        logger.info("Skipped (dedup): %s - %s", path.name, reason)
        return {"status": "skipped", "reason": reason, "file_path": str(path)}

    result = await engine.aingest(file_path=str(path))

    if content_hash and result.get("doc_id"):
        await hash_index.register(content_hash, result["doc_id"], str(path))

    # Upsert document metadata (best-effort)
    if result.get("doc_id") and metadata_index is not None:
        try:
            from dlightrag.core.retrieval.metadata_index import extract_system_metadata

            meta = extract_system_metadata(
                path, rag_mode="unified",
                page_count=result.get("page_count"),
            )
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

    results: list[dict[str, Any]] = []
    skipped = 0
    for file_path in files:
        result = await _ingest_single_local(engine, hash_index, file_path, replace, metadata_index)
        if result.get("status") == "skipped":
            skipped += 1
        results.append(result)

    processed = len(results) - skipped
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
        "results": results,
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

    container_name: str = kwargs["container_name"]
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
            return await _ingest_single_blob(engine, config, hash_index, source, blob_path, replace, metadata_index)

        blob_ids = await source.alist_documents(prefix=prefix)
        results: list[dict[str, Any]] = []
        for bid in blob_ids:
            result = await _ingest_single_blob(engine, config, hash_index, source, bid, replace, metadata_index)
            results.append(result)

        return {
            "status": "success",
            "source_type": "azure_blob",
            "container": container_name,
            "prefix": prefix,
            "results": results,
            "processed": len(results),
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
        result = await engine.aingest(file_path=str(target))

        if content_hash and result.get("doc_id"):
            await hash_index.register(content_hash, result["doc_id"], f"azure://{blob_path}")

        # Upsert document metadata (best-effort)
        if result.get("doc_id") and metadata_index is not None:
            try:
                from dlightrag.core.retrieval.metadata_index import extract_system_metadata

                meta = extract_system_metadata(
                    target, rag_mode="unified",
                    page_count=result.get("page_count"),
                )
                await metadata_index.upsert(result["doc_id"], meta)
            except Exception as e:
                logger.warning("Metadata upsert failed for %s: %s", target.name, e)

        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def _ingest_snowflake(
    engine: UnifiedRepresentEngine,
    config: DlightragConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Snowflake text -> LightRAG ainsert (no visual pipeline)."""
    from dlightrag.sourcing.snowflake import SnowflakeDataSource

    query: str = kwargs["query"]
    table: str | None = kwargs.get("table")
    source_label = table or "query"

    source = await asyncio.to_thread(
        SnowflakeDataSource,
        account=config.snowflake_account or "",
        user=config.snowflake_user or "",
        password=config.snowflake_password or "",
        warehouse=config.snowflake_warehouse,
        database=config.snowflake_database,
        schema=config.snowflake_schema,
    )

    try:
        await asyncio.to_thread(source.execute_query, query, source_label)

        texts: list[str] = []
        for doc_id in source.list_documents():
            raw = source.load_document(doc_id)
            texts.append(raw.decode("utf-8", errors="ignore"))

        if not texts:
            logger.warning("Snowflake query returned no results")
            return {"status": "success", "source_type": "snowflake", "processed": 0}

        combined = "\n\n".join(texts)
        await engine.lightrag.ainsert(combined)
        logger.info("Inserted %d Snowflake rows via LightRAG ainsert", len(texts))

        return {
            "status": "success",
            "source_type": "snowflake",
            "processed": len(texts),
            "source_label": source_label,
        }
    finally:
        await asyncio.to_thread(source.close)


# ── Deletion ───────────────────────────────────────────────────────


async def unified_delete_files(
    engine: UnifiedRepresentEngine,
    hash_index: HashIndexProtocol,
    lightrag: Any,
    *,
    file_paths: list[str] | None = None,
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Delete files in unified mode: hash lookup -> adelete_by_doc_id -> visual_chunks."""
    from dlightrag.core.ingestion.cleanup import collect_deletion_context

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

        for doc_id in ctx.doc_ids:
            # Phase 1: Clean visual_chunks (unified-specific)
            try:
                vc_result = await engine.adelete_doc(doc_id)
                deletion_result["cleanup_results"]["visual_chunks"] = vc_result
            except Exception as exc:
                logger.warning("visual_chunks cleanup failed for %s: %s", doc_id, exc)
                deletion_result["cleanup_results"]["visual_chunks"] = f"error: {exc}"

            # Phase 2: LightRAG cleanup (chunks_vdb, text_chunks, full_docs, KG)
            if lightrag and hasattr(lightrag, "adelete_by_doc_id"):
                try:
                    result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                    deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = (
                        result.status if hasattr(result, "status") else "completed"
                    )
                except Exception as exc:
                    logger.warning("LightRAG deletion failed for %s: %s", doc_id, exc)
                    deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = f"error: {exc}"

        # Phase 3: Remove from hash index
        for content_hash in ctx.content_hashes:
            if await hash_index.remove(content_hash):
                deletion_result["cleanup_results"]["hash_index"] = "removed"

        deletion_result["file_path"] = list(ctx.file_paths)[0] if ctx.file_paths else identifier
        deletion_result["doc_id"] = list(ctx.doc_ids)[0] if ctx.doc_ids else None
        results.append(deletion_result)

    return results


__all__ = ["unified_delete_files", "unified_ingest"]
