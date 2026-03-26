# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reciprocal Rank Fusion for multi-path retrieval."""

from __future__ import annotations

from typing import Any


def merge_rrf(
    paths: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple retrieval paths.

    score(chunk) = sum(1 / (k + rank_in_path + 1)) for each path containing chunk.
    Chunks appearing in multiple paths receive higher scores.

    Args:
        paths: List of ranked chunk lists (one per retrieval path).
        k: RRF parameter — higher values give more weight to lower-ranked results.
        top_k: Maximum number of chunks to return.
    """
    if not paths:
        return []

    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}

    for path in paths:
        for rank, chunk in enumerate(path):
            cid = chunk.get("chunk_id", str(id(chunk)))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunks[cid] for cid, _ in ranked]
