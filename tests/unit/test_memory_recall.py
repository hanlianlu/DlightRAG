# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Frozen RRF cases.

Recall over stored records runs against PostgreSQL in tests/integration/test_memory_pg.py.
"""

from dlightrag_memory.fusion import rrf_fuse


def test_rrf_fusion_is_rank_based_and_deterministic() -> None:
    scores = rrf_fuse([["a", "b"], ["b", "a"]], k=60)

    assert scores == {"a": 1 / 61 + 1 / 62, "b": 1 / 61 + 1 / 62}
    assert scores["a"] == scores["b"]


def test_rrf_rewards_consensus_rank() -> None:
    scores = rrf_fuse([["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c"]], k=60)

    assert scores["a"] > scores["b"] > scores["c"]
