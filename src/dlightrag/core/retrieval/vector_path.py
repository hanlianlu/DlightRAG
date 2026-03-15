# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Vector retrieval path — direct visual embedding search (unified mode only)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def vector_retrieve(
    query: str,
    chunks_vdb: Any,
    top_k: int = 10,
) -> list[str]:
    """Direct visual embedding search against chunks_vdb.

    Uses chunks_vdb.query(text, top_k) — embedding is handled
    internally by LightRAG's VDB implementation.

    This path is only activated in unified mode where chunks_vdb contains
    visual embeddings (page images), complementing KG's text-based search.
    """
    try:
        results = await chunks_vdb.query(query, top_k=top_k)
        chunk_ids = [r["id"] for r in results] if results else []
        logger.info("[VectorPath] found %d chunks via visual embedding", len(chunk_ids))
        return chunk_ids
    except Exception as exc:
        logger.warning("[VectorPath] visual embedding search failed: %s", exc)
        return []
