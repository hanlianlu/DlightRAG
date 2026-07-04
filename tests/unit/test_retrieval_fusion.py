# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval score fusion."""

from __future__ import annotations

from dlightrag.core.retrieval.fusion import dedup_chunks_by_content, rrf_fuse


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


def test_dedup_removes_duplicate_prefixes() -> None:
    # Both "a" and "b" share the same first 200 characters
    shared = "The quick brown fox jumps. " + "x" * 173
    chunks = [
        {"chunk_id": "a", "content": shared + " suffix a", "score": 0.9},
        {"chunk_id": "b", "content": shared + " suffix b", "score": 0.8},
        {"chunk_id": "c", "content": "Completely different content here.", "score": 0.7},
    ]
    result = dedup_chunks_by_content(chunks, prefix_len=200)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "a"
    assert result[1]["chunk_id"] == "c"


def test_dedup_handles_empty_list() -> None:
    assert dedup_chunks_by_content([]) == []


def test_dedup_handles_short_content() -> None:
    chunks = [
        {"chunk_id": "x", "content": "Hi", "score": 1.0},
        {"chunk_id": "y", "content": "Hi", "score": 0.5},
    ]
    result = dedup_chunks_by_content(chunks)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "x"
