# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared reranker execution with deterministic RRF fallback."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.rag.retrieval import ContextRow

logger = logging.getLogger(__name__)


class RerankBatchError(RuntimeError):
    """One atomic chat-listwise pass failed at a specific batch."""

    def __init__(
        self,
        *,
        batch_ordinal: int,
        batch_start: int,
        error_type: str,
    ) -> None:
        if (
            not isinstance(batch_ordinal, int)
            or isinstance(batch_ordinal, bool)
            or batch_ordinal < 1
        ):
            raise ValueError("rerank batch ordinal must be a positive integer")
        if not isinstance(batch_start, int) or isinstance(batch_start, bool) or batch_start < 0:
            raise ValueError("rerank batch start must be a non-negative integer")
        if not isinstance(error_type, str) or not error_type.strip():
            raise ValueError("rerank batch error type must be non-empty")
        self.batch_ordinal = batch_ordinal
        self.batch_start = batch_start
        self.error_type = error_type
        super().__init__(f"rerank batch {batch_ordinal} failed with {error_type}")


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """One rerank attempt and its deterministic fallback metadata."""

    chunks: list[ContextRow]
    reranked: bool
    error_type: str | None = None
    failed_batch: int | None = None


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
        if isinstance(exc, RerankBatchError):
            error_type = exc.error_type
            failed_batch = exc.batch_ordinal
        else:
            error_type = type(exc).__name__
            failed_batch = None
        return RerankOutcome(
            list(chunks[:limit]),
            False,
            error_type,
            failed_batch,
        )


__all__ = ["RerankBatchError", "RerankOutcome", "rerank_with_fallback"]
