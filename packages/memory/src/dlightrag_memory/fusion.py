# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reciprocal Rank Fusion for recall leg rankings.

The industry-standard fusion (MemMachine ships RRF(k=60) over identity + BM25
rankings). Each leg contributes an ordered ranking of record ids; every rank
position adds ``1 / (k + rank)`` to that id's fused score. Rank-based, never
score-averaging, so legs with different score scales stay comparable.
"""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf_fuse(leg_rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> dict[str, float]:
    """Return one id -> fused RRF score map from per-leg ordered rankings."""
    if k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in leg_rankings:
        for rank, memory_id in enumerate(ranking, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return scores


__all__ = ["RRF_K", "rrf_fuse"]
