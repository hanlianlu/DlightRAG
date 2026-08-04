# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata retrieval path: resolve hard filters to a document scope."""

import logging

from dlightrag.core.retrieval.models import MetadataFilter, MetadataScope
from dlightrag.core.retrieval.protocols import MetadataChunkStore
from dlightrag.storage.protocols import MetadataIndexProtocol

logger = logging.getLogger(__name__)


async def metadata_retrieve(
    *,
    metadata_index: MetadataIndexProtocol,
    stores: MetadataChunkStore,
    filters: MetadataFilter,
) -> MetadataScope:
    """Resolve metadata filters to the documents they select.

    Only the chunk *count* is read back — the ids stay in PostgreSQL, where the
    vector and BM25 queries filter on ``full_doc_id`` directly.
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
