# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Result fusion for multi-path retrieval."""

from __future__ import annotations


def reciprocal_rank_fusion(
    path_results: dict[str, list[str]],
    k: int = 60,
) -> list[str]:
    """Merge ranked chunk_id lists from multiple retrieval paths.

    Uses Reciprocal Rank Fusion (RRF): for each chunk, accumulates
    ``1 / (k + rank + 1)`` across all paths. Chunks found by multiple
    paths get boosted scores (reward, not duplicate).

    Args:
        path_results: mapping of path name → ordered list of chunk_ids.
        k: RRF smoothing parameter (default 60, configurable via config.rrf_k).

    Returns:
        Deduplicated chunk_ids sorted by fused score (descending).
    """
    scores: dict[str, float] = {}
    for chunk_ids in path_results.values():
        for rank, chunk_id in enumerate(chunk_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)
