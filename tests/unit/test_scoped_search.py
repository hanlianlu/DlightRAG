# tests/unit/test_scoped_search.py
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for scoped vector search (native DB filtered search)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.retrieval.scoped_search import scoped_vector_search


class TestScopedVectorSearch:
    """Test scoped_vector_search dispatches to correct backend."""

    @pytest.mark.asyncio
    async def test_empty_chunk_ids_returns_empty(self) -> None:
        chunks_vdb = MagicMock()
        result = await scoped_vector_search("query", [], chunks_vdb, top_k=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_few_chunks_returns_all_without_search(self) -> None:
        """When chunk_ids <= top_k, return all without vector search."""
        chunks_vdb = MagicMock()
        result = await scoped_vector_search(
            "query", ["c1", "c2", "c3"], chunks_vdb, top_k=10
        )
        assert set(result) == {"c1", "c2", "c3"}

    @pytest.mark.asyncio
    async def test_fallback_cosine_similarity(self) -> None:
        """Unknown backend falls back to get_vectors_by_ids + cosine sim."""
        chunks_vdb = MagicMock()
        chunks_vdb.__class__.__name__ = "UnknownVDB"

        # Mock embedding func
        import numpy as np
        query_vec = np.array([1.0, 0.0, 0.0])
        chunks_vdb.embedding_func = MagicMock()
        chunks_vdb.embedding_func.func = AsyncMock(return_value=[query_vec])

        # Mock get_vectors_by_ids: c1 is most similar, c2 is least
        chunks_vdb.get_vectors_by_ids = AsyncMock(return_value={
            "c1": np.array([0.9, 0.1, 0.0]),
            "c2": np.array([0.0, 0.0, 1.0]),
            "c3": np.array([0.5, 0.5, 0.0]),
        })

        result = await scoped_vector_search(
            "query", ["c1", "c2", "c3"], chunks_vdb, top_k=2
        )
        assert len(result) == 2
        assert result[0] == "c1"  # highest cosine similarity

    @pytest.mark.asyncio
    async def test_fallback_handles_missing_vectors(self) -> None:
        """Stale chunk_ids (missing from VDB) are gracefully skipped."""
        chunks_vdb = MagicMock()
        chunks_vdb.__class__.__name__ = "UnknownVDB"

        import numpy as np
        chunks_vdb.embedding_func = MagicMock()
        chunks_vdb.embedding_func.func = AsyncMock(
            return_value=[np.array([1.0, 0.0])]
        )
        # Only c1 has a vector; c2 is stale
        chunks_vdb.get_vectors_by_ids = AsyncMock(return_value={
            "c1": np.array([1.0, 0.0]),
        })

        result = await scoped_vector_search(
            "query", ["c1", "c2"], chunks_vdb, top_k=1
        )
        assert result == ["c1"]
