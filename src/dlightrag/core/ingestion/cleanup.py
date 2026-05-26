# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# _compute_mdhash_id is imported lazily in cascade_delete to avoid a hard
# dependency on lightrag at module import time.


@dataclass
class DeletionContext:
    """Aggregated deletion context from available data sources."""

    identifier: str  # Original filename/path requested for deletion
    doc_ids: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    sources_used: list[str] = field(default_factory=list)  # For audit trail


async def collect_deletion_context(
    identifier: str,
    lightrag: Any = None,
    metadata_index: Any = None,
) -> DeletionContext:
    """Find doc_id(s) for a file using LightRAG doc_status and metadata index.

    Strategy order:
    1. LightRAG doc_status - storage-agnostic lookup by file_path
    2. Metadata index - filename-based lookup (PGMetadataIndex)
    """
    ctx = DeletionContext(identifier=identifier)
    basename = Path(identifier).name
    stem = Path(identifier).stem

    # Strategy 1: LightRAG doc_status lookup (storage-agnostic)
    if lightrag and hasattr(lightrag, "doc_status"):
        doc_status = lightrag.doc_status
        try:
            # Try exact path match first
            result = await doc_status.get_doc_by_file_path(identifier)

            # If identifier is a basename, also try matching stored paths
            if result is None and not Path(identifier).is_absolute():
                # Query all processed docs and match by basename/stem
                from lightrag.base import DocStatus

                all_docs = await doc_status.get_docs_by_status(DocStatus.PROCESSED)
                for d_id, doc_info in all_docs.items():
                    fp = getattr(doc_info, "file_path", "") or ""
                    stored_name = Path(fp).name
                    stored_stem = Path(fp).stem

                    if stored_name == basename or stored_stem == stem:
                        result = {"id": d_id, "file_path": fp}
                        ctx.doc_ids.add(d_id)
                        ctx.file_paths.add(fp)

            if result and "doc_status" not in ctx.sources_used:
                # get_doc_by_file_path returns dict with id as key or 'id' field
                if isinstance(result, dict):
                    d_id = result.get("id") or result.get("doc_id")
                    fp = result.get("file_path", "")
                    if d_id:
                        ctx.doc_ids.add(d_id)
                    if fp:
                        ctx.file_paths.add(fp)
                    ctx.sources_used.append("doc_status")

        except Exception as e:
            logger.warning(f"LightRAG doc_status lookup failed for {identifier}: {e}")

    # Strategy 2: Metadata index lookup (PGMetadataIndex by filename)
    if metadata_index and not ctx.doc_ids:
        try:
            doc_ids = await metadata_index.find_by_filename(basename)
            for d_id in doc_ids:
                ctx.doc_ids.add(d_id)
            if doc_ids:
                ctx.sources_used.append("metadata_index")
        except Exception as e:
            logger.warning(f"Metadata index lookup failed for {identifier}: {e}")

    logger.info(
        f"Deletion context for {identifier}: "
        f"doc_ids={len(ctx.doc_ids)}, "
        f"file_paths={len(ctx.file_paths)}, sources={ctx.sources_used}"
    )

    return ctx


async def cascade_delete(
    ctx: DeletionContext,
    lightrag: Any,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """Cascade deletion with per-layer fault isolation.

    Each layer is wrapped in try/except so failures in one layer don't
    prevent cleanup in subsequent layers.

    Layers:
        1. LightRAG cross-backend cleanup (adelete_by_doc_id)
        2. DlightRAG metadata index entries
    """
    stats: dict[str, Any] = {"docs_deleted": 0, "errors": []}

    for doc_id in ctx.doc_ids:
        # Layer 1: LightRAG (full_docs, doc_status, text_chunks, chunks_vdb, KG)
        try:
            await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            stats["docs_deleted"] += 1
        except Exception as exc:
            stats["errors"].append(f"Layer 1 LightRAG ({doc_id}): {exc}")
            logger.warning("cascade_delete Layer 1 failed for %s: %s", doc_id, exc)

        # Layer 2: DlightRAG metadata index
        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Layer 2 metadata ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 2 failed for %s: %s", doc_id, exc)

    return stats


def remove_deleted_files(file_paths: set[str], working_dir: str) -> int:
    """Delete physical files and parsed artifact directories from disk.

    Best-effort — failures are logged but never raised, so a missing file
    on disk does not block the DB-level deletion from succeeding.

    Args:
        file_paths: Absolute paths to ingested files (from LightRAG doc_status).
        working_dir: The workspace's working_dir (for locating parsed artifacts).

    Returns:
        Number of files/directories removed.
    """
    import shutil

    removed = 0
    for fp in file_paths:
        path = Path(fp)
        # Remove the source file itself
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed += 1
        except OSError:
            logger.debug("Failed to remove source file: %s", fp, exc_info=True)

        # Remove associated parsed artifact directory (e.g., file.pdf.parsed/)
        parsed_dir = path.parent / (path.name + ".parsed")
        try:
            if parsed_dir.exists() and parsed_dir.is_dir():
                shutil.rmtree(parsed_dir, ignore_errors=True)
                removed += 1
        except OSError:
            logger.debug("Failed to remove parsed dir: %s", parsed_dir, exc_info=True)

    return removed


__all__ = [
    "DeletionContext",
    "cascade_delete",
    "collect_deletion_context",
    "remove_deleted_files",
]
