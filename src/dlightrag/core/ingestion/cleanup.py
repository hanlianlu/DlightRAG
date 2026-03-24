# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG deletion helpers.

Provides doc_id lookup for the deletion pipeline. The actual data cleanup
across all storage layers is handled by LightRAG's adelete_by_doc_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.ingestion.hash_index import HashIndexProtocol

logger = logging.getLogger(__name__)

# _compute_mdhash_id is imported lazily in cascade_delete to avoid a hard
# dependency on lightrag at module import time.


@dataclass
class DeletionContext:
    """Aggregated deletion context from available data sources."""

    identifier: str  # Original filename/path requested for deletion
    doc_ids: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    file_paths: set[str] = field(default_factory=set)
    sources_used: list[str] = field(default_factory=list)  # For audit trail


async def collect_deletion_context(
    identifier: str,
    hash_index: HashIndexProtocol | None,
    lightrag: Any = None,
    metadata_index: Any = None,
) -> DeletionContext:
    """Find doc_id(s) for a file using hash index, doc_status, and metadata index.

    Strategy order:
    1. Hash index - authoritative source for content_hash -> doc_id
    2. LightRAG doc_status - storage-agnostic lookup by file_path
    3. Metadata index - filename-based lookup (PGMetadataIndex)
    """
    ctx = DeletionContext(identifier=identifier)
    basename = Path(identifier).name
    stem = Path(identifier).stem

    # Strategy 1: Hash index lookup (fastest, authoritative)
    if hash_index:
        doc_id, content_hash, file_path = await hash_index.find_by_name(basename)
        if not doc_id:
            doc_id, content_hash, file_path = await hash_index.find_by_path(identifier)

        if doc_id:
            ctx.doc_ids.add(doc_id)
            ctx.sources_used.append("hash_index")
        if content_hash:
            ctx.content_hashes.add(content_hash)
        if file_path:
            ctx.file_paths.add(file_path)

    # Strategy 2: LightRAG doc_status lookup (storage-agnostic)
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

    # Strategy 3: Metadata index lookup (PGMetadataIndex by filename)
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
    visual_chunks: Any | None = None,
    hash_index: HashIndexProtocol | None = None,
    metadata_index: Any | None = None,
) -> dict[str, Any]:
    """5-layer cascade deletion with per-layer fault isolation.

    Each layer is wrapped in try/except so failures in one layer don't
    prevent cleanup in subsequent layers.

    Layers:
        1. LightRAG cross-backend cleanup (adelete_by_doc_id)
        2. Visual chunks (page images in PGJsonbKV, keyed by chunk_id)
        3. Metadata index entries
        4. Hash index entries (by content_hash)
        5. (Reserved for file source cleanup)

    Notes
    -----
    ``visual_chunks`` (PGJsonbKVStorage) exposes ``delete(ids: list[str])``
    but not ``delete_by_doc_id``.  Layer 2 therefore reads ``page_count``
    from ``metadata_index`` to reconstruct the chunk IDs before calling
    ``visual_chunks.delete()``.  This mirrors the logic that previously
    lived inline in ``unified_delete_files()``.
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

        # Layer 2: Visual chunks (reconstruct chunk_ids from metadata page_count)
        if visual_chunks is not None and metadata_index is not None:
            try:
                from lightrag.utils import compute_mdhash_id

                doc_meta = await metadata_index.get(doc_id)
                page_count = (doc_meta or {}).get("page_count", 0)
                if isinstance(page_count, int) and page_count > 0:
                    chunk_ids = [
                        compute_mdhash_id(f"{doc_id}:page:{i}", prefix="chunk-")
                        for i in range(page_count)
                    ]
                    await visual_chunks.delete(chunk_ids)
                    logger.info(
                        "cascade_delete Layer 2: deleted %d visual_chunks for %s",
                        len(chunk_ids),
                        doc_id,
                    )
            except Exception as exc:
                stats["errors"].append(f"Layer 2 visual_chunks ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 2 failed for %s: %s", doc_id, exc)

        # Layer 3: Metadata index
        if metadata_index is not None:
            try:
                await metadata_index.delete(doc_id)
            except Exception as exc:
                stats["errors"].append(f"Layer 3 metadata ({doc_id}): {exc}")
                logger.warning("cascade_delete Layer 3 failed for %s: %s", doc_id, exc)

    # Layer 4: Hash index (by content_hash, not doc_id)
    if hash_index is not None:
        for h in ctx.content_hashes:
            try:
                await hash_index.remove(h)
            except Exception as exc:
                stats["errors"].append(f"Layer 4 hash ({h}): {exc}")
                logger.warning("cascade_delete Layer 4 failed for hash %s: %s", h, exc)

    return stats


__all__ = ["DeletionContext", "cascade_delete", "collect_deletion_context"]
