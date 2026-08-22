# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for metadata retrieval path."""

from unittest.mock import AsyncMock

from dlightrag.rag.retrieval import MetadataFilter
from dlightrag.rag.retrieval.metadata_path import metadata_retrieve


async def test_metadata_retrieve_returns_doc_scope_without_expanding_chunks() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = ["doc-1", "doc-2"]
    stores = AsyncMock()
    stores.count_chunks_for_docs.return_value = 1470

    scope = await metadata_retrieve(
        metadata_index=metadata_index,
        stores=stores,
        filters=MetadataFilter(filename="x.pdf"),
    )

    # The chunk fan-out is counted, never materialized.
    assert scope.doc_ids == frozenset({"doc-1", "doc-2"})
    assert scope.chunk_count == 1470
    stores.count_chunks_for_docs.assert_awaited_once_with(["doc-1", "doc-2"])


async def test_metadata_retrieve_empty_docs_short_circuits() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    stores = AsyncMock()

    scope = await metadata_retrieve(
        metadata_index=metadata_index,
        stores=stores,
        filters=MetadataFilter(filename="missing.pdf"),
    )

    assert not scope
    assert scope.chunk_count == 0
    stores.count_chunks_for_docs.assert_not_called()
