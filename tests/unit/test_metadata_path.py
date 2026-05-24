# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for metadata retrieval path."""

from __future__ import annotations

from unittest.mock import AsyncMock

from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter


async def test_metadata_retrieve_uses_chunk_provenance() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = ["doc-1"]
    chunk_provenance = AsyncMock()
    chunk_provenance.chunk_ids_for_docs.return_value = ["chunk-a", "chunk-b"]

    result = await metadata_retrieve(
        metadata_index=metadata_index,
        chunk_provenance=chunk_provenance,
        filters=MetadataFilter(filename="x.pdf"),
    )

    assert result == ["chunk-a", "chunk-b"]


async def test_metadata_retrieve_empty_docs_short_circuits() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    chunk_provenance = AsyncMock()

    result = await metadata_retrieve(
        metadata_index=metadata_index,
        chunk_provenance=chunk_provenance,
        filters=MetadataFilter(filename="missing.pdf"),
    )

    assert result == []
    chunk_provenance.chunk_ids_for_docs.assert_not_called()
