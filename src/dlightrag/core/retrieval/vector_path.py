# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Vector retrieval path — direct visual embedding search (unified mode only)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


async def vector_retrieve(
    query: str,
    chunks_vdb: Any,
    embedding_func: Callable[..., Any],
    top_k: int = 10,
) -> list[str]:
    """Direct visual embedding search against chunks_vdb.

    This path is only activated in unified mode where chunks_vdb contains
    visual embeddings (page images), complementing KG's text-based search.

    In caption mode, LightRAG's KG path already includes internal vector
    search, so this path would be redundant.
    """
    try:
        query_embedding = await embedding_func([query])
        if isinstance(query_embedding, np.ndarray):
            query_vec = query_embedding[0].tolist()
        elif isinstance(query_embedding, list):
            query_vec = query_embedding[0]
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()
        else:
            logger.warning("[VectorPath] unexpected embedding type: %s", type(query_embedding))
            return []

        results = await chunks_vdb.query(query_vec, top_k=top_k)
        chunk_ids = [r["id"] for r in results] if results else []
        logger.info("[VectorPath] found %d chunks via visual embedding", len(chunk_ids))
        return chunk_ids
    except Exception as exc:
        logger.warning("[VectorPath] visual embedding search failed: %s", exc)
        return []
