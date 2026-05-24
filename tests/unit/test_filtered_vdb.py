# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for strict metadata in-filtering context."""

from __future__ import annotations

from unittest.mock import AsyncMock

from dlightrag.core.retrieval.filtered_vdb import (
    _active_filter,
    fetch_chunks_by_ids,
    metadata_filter_scope,
)


async def test_empty_candidate_set_is_active_filter() -> None:
    async with metadata_filter_scope(set()):
        assert _active_filter.get() == set()


async def test_none_candidate_set_is_no_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None


async def test_fetch_chunks_by_ids_uses_explicit_ids_outside_filter_scope() -> None:
    text_chunks = AsyncMock()
    text_chunks.get_by_ids.return_value = [
        {"content": "alpha", "file_path": "/tmp/a.pdf"},
        {"content": "beta", "file_path": "/tmp/b.pdf"},
    ]

    result = await fetch_chunks_by_ids(text_chunks, ["c1", "c2"])

    text_chunks.get_by_ids.assert_awaited_once_with(["c1", "c2"])
    assert result == [
        {"chunk_id": "c1", "content": "alpha", "reference_id": "", "file_path": "/tmp/a.pdf"},
        {"chunk_id": "c2", "content": "beta", "reference_id": "", "file_path": "/tmp/b.pdf"},
    ]
