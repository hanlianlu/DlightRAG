# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval score fusion."""

from dlightrag.core.retrieval.fusion import rrf_fuse


def test_rrf_fuse_deduplicates_by_chunk_id() -> None:
    semantic = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    fused = rrf_fuse([semantic, bm25], k=60)

    assert [row["chunk_id"] for row in fused] == ["b", "a", "c"]
    assert fused[0]["score"] > fused[1]["score"]


def test_rrf_fuse_preserves_best_row_payload() -> None:
    semantic = [{"id": "chunk-a", "content": "semantic"}]
    bm25 = [{"chunk_id": "chunk-a", "content": "bm25", "file_path": "a.md"}]

    fused = rrf_fuse([semantic, bm25])

    assert fused[0]["chunk_id"] == "chunk-a"
    assert fused[0]["content"] == "semantic"
    assert fused[0]["score"] > 0


def test_rrf_fuse_skips_rows_without_chunk_id() -> None:
    fused = rrf_fuse([[{"content": "missing id"}, {"chunk_id": "chunk-a"}]])

    assert [row["chunk_id"] for row in fused] == ["chunk-a"]
