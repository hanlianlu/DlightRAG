# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata retrieval path — resolves metadata filters to chunk IDs."""

from __future__ import annotations

import logging
from typing import Any

from dlightrag.core.retrieval.metadata_index import PGMetadataIndex
from dlightrag.core.retrieval.models import MetadataFilter

logger = logging.getLogger(__name__)


async def metadata_retrieve(
    metadata_index: PGMetadataIndex,
    filters: MetadataFilter,
    lightrag: Any,
    rag_mode: str,
) -> list[str]:
    """Retrieve chunk_ids matching the given metadata filters.

    Flow: MetadataIndex.query(filters) → doc_ids → resolve to chunk_ids.

    Resolution strategy depends on rag_mode:
    - caption: doc_status stores chunks_list per doc
    - unified: chunk_ids are computed from doc_id + page_count
    """
    doc_ids = await metadata_index.query(filters)
    if not doc_ids:
        return []

    logger.info(
        "[MetadataPath] filters matched %d doc(s): %s",
        len(doc_ids),
        doc_ids[:5],
    )

    chunk_ids: list[str] = []
    for doc_id in doc_ids:
        resolved = await _resolve_chunks_for_doc(doc_id, lightrag, rag_mode)
        chunk_ids.extend(resolved)

    logger.info("[MetadataPath] resolved %d chunk(s) total", len(chunk_ids))
    return chunk_ids


async def _resolve_chunks_for_doc(
    doc_id: str,
    lightrag: Any,
    rag_mode: str,
) -> list[str]:
    """Resolve a doc_id to its chunk_ids."""
    if rag_mode == "unified":
        return await _resolve_unified(doc_id, lightrag)
    return await _resolve_caption(doc_id, lightrag)


async def _resolve_caption(doc_id: str, lightrag: Any) -> list[str]:
    """Caption mode: read chunks_list from doc_status."""
    try:
        doc_info = await lightrag.doc_status.get_by_id(doc_id)
        if doc_info:
            return doc_info.get("chunks_list", [])
    except Exception as exc:
        logger.warning("[MetadataPath] caption chunk resolution failed for %s: %s", doc_id, exc)
    return []


async def _resolve_unified(doc_id: str, lightrag: Any) -> list[str]:
    """Unified mode: compute chunk_ids from doc_id + page_count."""
    try:
        from lightrag.utils import compute_mdhash_id

        doc_data = await lightrag.full_docs.get_by_id(doc_id)
        if not doc_data:
            return []
        page_count = doc_data.get("page_count", 0)
        return [
            compute_mdhash_id(f"{doc_id}:page:{i}", prefix="chunk-")
            for i in range(page_count)
        ]
    except Exception as exc:
        logger.warning("[MetadataPath] unified chunk resolution failed for %s: %s", doc_id, exc)
    return []
