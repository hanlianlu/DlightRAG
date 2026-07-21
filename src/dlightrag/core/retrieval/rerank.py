# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared reranker execution with deterministic RRF fallback."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from dlightrag.core.retrieval.protocols import ContextRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """One rerank attempt and its deterministic fallback metadata."""

    chunks: list[ContextRow]
    reranked: bool
    error_type: str | None = None


async def rerank_with_fallback(
    *,
    query: str,
    chunks: list[ContextRow],
    top_k: int,
    rerank_func: Any | None,
) -> RerankOutcome:
    """Rerank and cap chunks, falling back to the existing RRF order."""
    limit = max(0, top_k)
    if rerank_func is None:
        return RerankOutcome(list(chunks[:limit]), False)
    try:
        reranked = await rerank_func(query=query, chunks=chunks, top_k=limit)
        return RerankOutcome(list(reranked[:limit]), True)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Rerank failed; returning fused chunks", exc_info=True)
        return RerankOutcome(list(chunks[:limit]), False, type(exc).__name__)


__all__ = ["RerankOutcome", "rerank_with_fallback"]
