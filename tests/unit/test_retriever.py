# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified retrieval orchestration."""

import logging
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.retriever import UnifiedRetriever


async def test_unified_retriever_empty_metadata_candidates_short_circuits() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    stores = AsyncMock()
    backend = AsyncMock()
    bm25 = AsyncMock()
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve("query", metadata_filter=MetadataFilter(filename="x.pdf"))

    assert result.contexts == {"chunks": [], "entities": [], "relationships": []}
    backend.aretrieve.assert_not_called()
    bm25.search.assert_not_called()


async def test_unified_retriever_llm_empty_candidates_falls_back_unfiltered() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    metadata_index.query.return_value = []
    stores = AsyncMock()
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
        stores=stores,
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


async def test_unified_retriever_llm_filtered_empty_falls_back_unfiltered() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    metadata_index.query.return_value = ["doc-1"]
    stores = AsyncMock()
    stores.chunk_ids_for_docs.return_value = ["filtered-chunk"]
    backend = AsyncMock()
    backend.aretrieve.side_effect = [
        RetrievalResult(contexts={"chunks": [], "entities": [], "relationships": []}),
        RetrievalResult(
            contexts={"chunks": [{"chunk_id": "semantic-a"}], "entities": [], "relationships": []}
        ),
    ]
    bm25 = AsyncMock()
    bm25.search.side_effect = [[], []]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="maybe.pdf"),
        metadata_filter_source="llm_inferred",
    )

    assert result.contexts["chunks"] == [{"chunk_id": "semantic-a"}]
    assert result.trace["metadata_filter_relaxed"] is True
    assert result.trace["metadata_candidate_count"] == 1
    assert backend.aretrieve.await_count == 2
    assert bm25.search.await_args_list[0].kwargs["candidate_ids"] == {"filtered-chunk"}
    assert bm25.search.await_args_list[1].kwargs["candidate_ids"] is None


async def test_unified_retriever_explicit_filtered_empty_stays_filtered() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    metadata_index.query.return_value = ["doc-1"]
    stores = AsyncMock()
    stores.chunk_ids_for_docs.return_value = ["filtered-chunk"]
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={"chunks": [], "entities": [], "relationships": []}
    )
    bm25 = AsyncMock()
    bm25.search.return_value = []
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve(
        "query",
        metadata_filter=MetadataFilter(filename="exact.pdf"),
        metadata_filter_source="explicit",
    )

    assert result.contexts["chunks"] == []
    assert result.trace["metadata_filter_relaxed"] is False
    backend.aretrieve.assert_awaited_once()
    bm25.search.assert_awaited_once()


async def test_unified_retriever_fuses_lightrag_and_bm25_chunks() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    stores = AsyncMock()
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
        stores=stores,
        rrf_k=60,
    )

    result = await retriever.aretrieve("query", top_k=3)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == [
        "shared",
        "semantic-a",
        "bm25-b",
    ]
    assert result.contexts["entities"] == [{"entity_name": "E"}]


async def test_unified_retriever_does_not_cap_fused_chunks_to_candidate_limit() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "semantic-b"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-a"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=2)

    assert bm25.search.await_args.kwargs["top_k"] == 2
    assert [c["chunk_id"] for c in result.contexts["chunks"]] == [
        "semantic-a",
        "bm25-a",
        "semantic-b",
        "bm25-b",
    ]


async def test_unified_retriever_keeps_distinct_chunks_with_same_content_prefix() -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    shared = "The quick brown fox jumps. " + "x" * 173
    metadata_index = AsyncMock()
    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [
                {"chunk_id": "semantic-a", "content": shared + " semantic suffix"},
            ],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [
        {"chunk_id": "bm25-b", "content": shared + " bm25 suffix"},
    ]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=2)

    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["semantic-a", "bm25-b"]


async def test_unified_retriever_logs_retrieval_mix_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.core.retrieval.protocols import RetrievalResult

    metadata_index = AsyncMock()
    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.return_value = RetrievalResult(
        contexts={
            "chunks": [{"chunk_id": "semantic-a"}, {"chunk_id": "shared"}],
            "entities": [],
            "relationships": [],
        }
    )
    bm25 = AsyncMock()
    bm25.search.return_value = [
        {"chunk_id": "shared", "bm25_profile": "en", "score": 2.0},
        {"chunk_id": "bm25-b", "bm25_profile": "en", "score": 1.0},
    ]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
        rrf_k=60,
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.core.retrieval.retriever"):
        result = await retriever.aretrieve("query", bm25_query="keyword query", top_k=3)

    assert "[Retriever] mix" in caplog.text
    assert "bm25_enabled=True" in caplog.text
    assert "bm25_query='keyword query'" in caplog.text
    assert "semantic_chunks=2" in caplog.text
    assert "bm25_chunks=2" in caplog.text
    assert "fused_chunks=3" in caplog.text
    assert "fused_sources=semantic_only=1 bm25_only=1 both=1" in caplog.text
    assert "bm25_top=shared:en:2.000,bm25-b:en:1.000" in caplog.text
    assert result.trace["fused_source_counts"] == {
        "semantic_only": 1,
        "bm25_only": 1,
        "both": 1,
    }


async def test_unified_retriever_lightrag_failure_falls_back_to_bm25() -> None:
    """When LightRAG retrieval raises, BM25 results must still be returned."""

    metadata_index = AsyncMock()
    stores = AsyncMock()
    backend = AsyncMock()
    backend.aretrieve.side_effect = RuntimeError("LightRAG backend down")
    bm25 = AsyncMock()
    bm25.search.return_value = [{"chunk_id": "bm25-a"}, {"chunk_id": "bm25-b"}]
    retriever = UnifiedRetriever(
        backend=backend,
        bm25=bm25,
        metadata_index=metadata_index,
        stores=stores,
    )

    result = await retriever.aretrieve("query", top_k=5)

    assert result.trace.get("lightrag_error") is True
    assert len(result.contexts["chunks"]) == 2
    assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["bm25-a", "bm25-b"]
