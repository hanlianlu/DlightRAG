# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata retrieval path: resolve hard filters to LightRAG chunk IDs."""

import logging

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.protocols import MetadataChunkStore
from dlightrag.storage.protocols import MetadataIndexProtocol

logger = logging.getLogger(__name__)


async def metadata_retrieve(
    *,
    metadata_index: MetadataIndexProtocol,
    stores: MetadataChunkStore,
    filters: MetadataFilter,
) -> list[str]:
    """Resolve metadata filters to chunk IDs through LightRAG text_chunks."""
    doc_ids = await metadata_index.query(filters)
    if not doc_ids:
        logger.info("[MetadataPath] filters matched 0 documents")
        return []

    chunk_ids = await stores.chunk_ids_for_docs(doc_ids)
    logger.info(
        "[MetadataPath] filters matched %d doc(s), resolved %d chunk(s)",
        len(doc_ids),
        len(chunk_ids),
    )
    return list(chunk_ids)
