# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified retrieval orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.retriever import UnifiedRetriever


async def test_unified_retriever_empty_metadata_candidates_short_circuits() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    chunk_provenance = AsyncMock()
    backend = AsyncMock()
    bm25 = AsyncMock()
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        chunk_provenance=chunk_provenance,
    )

    result = await retriever.aretrieve("query", metadata_filter=MetadataFilter(filename="x.pdf"))

    assert result.contexts == {"chunks": [], "entities": [], "relationships": []}
    backend.aretrieve.assert_not_called()
    bm25.search.assert_not_called()


async def test_unified_retriever_llm_empty_candidates_falls_back_unfiltered() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    chunk_provenance = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        chunk_provenance=chunk_provenance,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="missing.pdf"),
        metadata_filter_source="llm_inferred",
    )

    assert result.contexts["chunks"] == [{"chunk_id": "semantic-a"}]
    backend.aretrieve.assert_awaited_once()
    bm25.search.assert_awaited_once()
    assert bm25.search.await_args.kwargs["candidate_ids"] is None


async def test_unified_retriever_fuses_lightrag_and_bm25_chunks() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    chunk_provenance = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "shared"}],
            "entities": [{"entity_name": "E"}],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "shared"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        chunk_provenance=chunk_provenance,
        rrf_k=60,
    )

    result = await retriever.aretrieve("query", top_k=3)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == [
        "shared",
        "semantic-a",
        "bm25-b",
    ]
    assert result.contexts["entities"] == [{"entity_name": "E"}]
