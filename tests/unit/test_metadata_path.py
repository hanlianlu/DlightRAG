# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for metadata retrieval path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter


@pytest.fixture
def mock_metadata_index():
    idx = AsyncMock()
    return idx


@pytest.fixture
def mock_lightrag():
    lr = MagicMock()
    lr.doc_status = AsyncMock()
    lr.full_docs = AsyncMock()
    return lr


class TestMetadataRetrieveCaption:
    @pytest.mark.asyncio
    async def test_resolves_caption_chunks(self, mock_metadata_index, mock_lightrag) -> None:
        mock_metadata_index.query.return_value = ["doc-1"]
        mock_lightrag.doc_status.get_by_id = AsyncMock(
            return_value={"chunks_list": ["chunk-a", "chunk-b"]}
        )
        result = await metadata_retrieve(
            mock_metadata_index,
            MetadataFilter(filename="test.pdf"),
            mock_lightrag,
            rag_mode="caption",
        )
        assert result == ["chunk-a", "chunk-b"]

    @pytest.mark.asyncio
    async def test_no_matching_docs(self, mock_metadata_index, mock_lightrag) -> None:
        mock_metadata_index.query.return_value = []
        result = await metadata_retrieve(
            mock_metadata_index,
            MetadataFilter(filename="missing.pdf"),
            mock_lightrag,
            rag_mode="caption",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_docs(self, mock_metadata_index, mock_lightrag) -> None:
        mock_metadata_index.query.return_value = ["doc-1", "doc-2"]
        mock_lightrag.doc_status.get_by_id = AsyncMock(
            side_effect=[
                {"chunks_list": ["chunk-a"]},
                {"chunks_list": ["chunk-b", "chunk-c"]},
            ]
        )
        result = await metadata_retrieve(
            mock_metadata_index,
            MetadataFilter(doc_author="Zhang San"),
            mock_lightrag,
            rag_mode="caption",
        )
        assert result == ["chunk-a", "chunk-b", "chunk-c"]


class TestMetadataRetrieveUnified:
    @pytest.mark.asyncio
    async def test_resolves_unified_chunks(self, mock_metadata_index, mock_lightrag) -> None:
        mock_metadata_index.query.return_value = ["doc-1"]
        mock_metadata_index.get = AsyncMock(return_value={"page_count": 3})
        with patch("lightrag.utils.compute_mdhash_id") as mock_hash:
            mock_hash.side_effect = lambda s, prefix="": f"{prefix}{s}"
            result = await metadata_retrieve(
                mock_metadata_index,
                MetadataFilter(filename="doc.pdf"),
                mock_lightrag,
                rag_mode="unified",
            )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_unified_missing_doc(self, mock_metadata_index, mock_lightrag) -> None:
        mock_metadata_index.query.return_value = ["doc-missing"]
        mock_metadata_index.get = AsyncMock(return_value=None)
        result = await metadata_retrieve(
            mock_metadata_index,
            MetadataFilter(filename="missing.pdf"),
            mock_lightrag,
            rag_mode="unified",
        )
        assert result == []
