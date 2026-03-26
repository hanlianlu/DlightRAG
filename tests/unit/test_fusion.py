# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RRF fusion."""

from __future__ import annotations

from dlightrag.core.fusion import merge_rrf


class TestMergeRRF:
    def test_single_path(self):
        chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
        result = merge_rrf([chunks], k=60, top_k=3)
        assert [c["chunk_id"] for c in result] == ["a", "b", "c"]

    def test_overlapping_chunks_score_higher(self):
        path_a = [{"chunk_id": "shared"}, {"chunk_id": "a_only"}]
        path_b = [{"chunk_id": "b_only"}, {"chunk_id": "shared"}]
        result = merge_rrf([path_a, path_b], k=60, top_k=3)
        assert result[0]["chunk_id"] == "shared"

    def test_top_k_truncation(self):
        chunks = [{"chunk_id": f"c{i}"} for i in range(10)]
        result = merge_rrf([chunks], k=60, top_k=3)
        assert len(result) == 3

    def test_empty_paths(self):
        result = merge_rrf([], k=60, top_k=10)
        assert result == []

    def test_empty_path_in_list(self):
        path_a = [{"chunk_id": "a"}]
        result = merge_rrf([path_a, []], k=60, top_k=10)
        assert len(result) == 1

    def test_k_parameter_effect(self):
        """Higher k gives more weight to lower-ranked results."""
        path = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        result_low = merge_rrf([path], k=1, top_k=2)
        result_high = merge_rrf([path], k=1000, top_k=2)
        assert result_low[0]["chunk_id"] == "a"
        assert result_high[0]["chunk_id"] == "a"

    def test_preserves_chunk_data(self):
        chunks = [{"chunk_id": "a", "content": "hello", "metadata": {"page": 1}}]
        result = merge_rrf([chunks], k=60, top_k=1)
        assert result[0]["content"] == "hello"
        assert result[0]["metadata"]["page"] == 1
