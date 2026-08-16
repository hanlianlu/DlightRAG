# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata retrieval path: resolve hard filters to a document scope."""

import logging

from dlightrag_rag.ports import MetadataChunkStore, MetadataIndexProtocol
from dlightrag_rag.retrieval import MetadataFilter, MetadataScope

logger = logging.getLogger(__name__)


async def metadata_retrieve(
    *,
    metadata_index: MetadataIndexProtocol,
    stores: MetadataChunkStore,
    filters: MetadataFilter,
) -> MetadataScope:
    """Resolve metadata filters to the documents they select.

    Only the chunk *count* is read back; vector and BM25 adapters filter the
    selected document ids on ``full_doc_id`` directly.
    """
    doc_ids = await metadata_index.query(filters)
    if not doc_ids:
        logger.info("[MetadataPath] filters matched 0 documents")
        return MetadataScope(doc_ids=frozenset(), chunk_count=0)

    chunk_count = await stores.count_chunks_for_docs(doc_ids)
    logger.info(
        "[MetadataPath] filters matched %d doc(s) covering %d chunk(s)",
        len(doc_ids),
        chunk_count,
    )
    return MetadataScope(doc_ids=frozenset(doc_ids), chunk_count=chunk_count)
