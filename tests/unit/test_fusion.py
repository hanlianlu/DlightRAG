# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from dlightrag.core.retrieval.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_path(self) -> None:
        result = reciprocal_rank_fusion({"kg": ["a", "b", "c"]})
        assert result == ["a", "b", "c"]

    def test_two_paths_no_overlap(self) -> None:
        result = reciprocal_rank_fusion({
            "kg": ["a", "b"],
            "metadata": ["c", "d"],
        })
        assert len(result) == 4
        assert set(result) == {"a", "b", "c", "d"}

    def test_overlap_boosts_rank(self) -> None:
        result = reciprocal_rank_fusion({
            "kg": ["a", "b", "c"],
            "metadata": ["b", "d", "a"],
        })
        # "a" and "b" appear in both paths, should be ranked higher
        # "b" is rank 1 in kg and rank 0 in metadata → highest combined score
        assert result[0] in ("a", "b")
        assert "d" in result
        assert "c" in result

    def test_empty_paths(self) -> None:
        result = reciprocal_rank_fusion({})
        assert result == []

    def test_single_empty_path(self) -> None:
        result = reciprocal_rank_fusion({"kg": []})
        assert result == []

    def test_custom_k(self) -> None:
        result_k1 = reciprocal_rank_fusion({"p": ["a", "b"]}, k=1)
        result_k100 = reciprocal_rank_fusion({"p": ["a", "b"]}, k=100)
        # Order should be the same, just different score magnitudes
        assert result_k1 == result_k100 == ["a", "b"]

    def test_three_paths_same_chunk(self) -> None:
        result = reciprocal_rank_fusion({
            "kg": ["a", "b"],
            "metadata": ["a", "c"],
            "vector": ["a", "d"],
        })
        # "a" found by all three paths — should be first
        assert result[0] == "a"

    def test_dedup(self) -> None:
        result = reciprocal_rank_fusion({
            "p1": ["a", "b", "c"],
            "p2": ["b", "c", "a"],
        })
        # No duplicates
        assert len(result) == len(set(result))
