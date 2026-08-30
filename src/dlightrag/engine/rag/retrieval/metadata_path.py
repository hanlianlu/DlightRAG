# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata retrieval path: resolve hard filters to a metadata scope."""

import logging

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.ports import MetadataScopeStore

logger = logging.getLogger(__name__)


async def metadata_retrieve(
    *,
    stores: MetadataScopeStore,
    filters: MetadataFilter,
) -> MetadataScope:
    """Resolve metadata filters into scope facts without materializing ids.

    The store owns every data predicate: it reports whether the filter matched
    at least one document plus a bounded matching-chunk count, and selects the
    filename mode. No matching document id set ever crosses this boundary —
    vector, BM25, and graph legs filter by the shared predicate in the database.
    """
    scope = await stores.resolve_scope(filters)
    logger.info(
        "[MetadataPath] filters matched_any=%s candidate_chunks=%s filename_mode=%s",
        scope.doc_exists,
        scope.render_candidate_count(),
        scope.filename_mode,
    )
    return scope
