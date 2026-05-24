# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for strict metadata in-filtering context."""

from __future__ import annotations

from dlightrag.core.retrieval.filtered_vdb import _active_filter, metadata_filter_scope


async def test_empty_candidate_set_is_active_filter() -> None:
    async with metadata_filter_scope(set()):
        assert _active_filter.get() == set()


async def test_none_candidate_set_is_no_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None
