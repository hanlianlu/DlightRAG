# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VisualRetriever: KG retrieval, visual resolution, reranking, answer generation."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from dlightrag.unifiedrepresent.retriever import VisualRetriever

GRAPH_FIELD_SEP = "<SEP>"

# A tiny valid PNG (1x1 transparent pixel) encoded as base64.
_TINY_PNG_B64 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(default_mode="mix", top_k=60, chunk_top_k=10)


def _make_lightrag_result() -> dict:
    """Return a mock aquery_data result with entities, relationships, chunks."""
    return {
        "data": {
            "entities": [
                {
                    "entity_name": "E1",
                    "entity_type": "CONCEPT",
                    "description": "First entity",
                    "source_id": "chunk-abc",
                },
            ],
            "relationships": [
                {
                    "src_id": "E1",
                    "tgt_id": "E2",
                    "description": "relates to",
                    "source_id": f"chunk-abc{GRAPH_FIELD_SEP}chunk-def",
                },
            ],
            "chunks": [
                {
                    "chunk_id": "chunk-abc",
                    "content": "text from chunk abc",
                    "reference_id": "1",
                    "file_path": "/test/doc.pdf",
                },
            ],
        },
    }


def _make_visual_data() -> list[dict | None]:
    """Visual data list aligned with chunk_id_list from _make_lightrag_result.

    The lightrag result yields chunk IDs: {chunk-abc, chunk-def}.
    get_by_ids is called with a list of those IDs; we return data for both.
    """
    return [
        {
            "image_data": _TINY_PNG_B64,
            "page_index": 0,
            "full_doc_id": "doc-1",
            "file_path": "/test/doc.pdf",
            "doc_title": "Test",
        },
        {
            "image_data": _TINY_PNG_B64,
            "page_index": 1,
            "full_doc_id": "doc-1",
            "file_path": "/test/doc.pdf",
            "doc_title": "Test",
        },
    ]


def _make_retriever(
    *,
    rerank_func: AsyncMock | None = None,
    visual_data: list[dict | None] | None = None,
) -> VisualRetriever:
    lightrag = MagicMock()
    lightrag.aquery_data = AsyncMock(return_value=_make_lightrag_result())
    # Backfill lookup for entity/relationship chunks missing text
    lightrag.text_chunks.get_by_ids = AsyncMock(
        return_value=[{"content": "backfilled text", "file_path": "/test/doc.pdf"}],
    )

    visual_chunks = MagicMock()
    visual_chunks.get_by_ids = AsyncMock(
        return_value=visual_data if visual_data is not None else _make_visual_data(),
    )

    return VisualRetriever(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        config=_make_config(),
        rerank_func=rerank_func,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Test retrieve() — Phases 1-3 without reranking."""

    async def test_returned_structure(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")

        # Top-level keys
        assert "contexts" in result

        # Contexts sub-keys
        assert "entities" in result["contexts"]
        assert "relationships" in result["contexts"]
        assert "chunks" in result["contexts"]

    async def test_entities_passed_through(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")
        entities = result["contexts"]["entities"]
        assert len(entities) == 1
        assert entities[0]["entity_name"] == "E1"

    async def test_relationships_passed_through(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("What is E1?")
        rels = result["contexts"]["relationships"]
        assert len(rels) == 1
        assert rels[0]["src_id"] == "E1"
        assert rels[0]["tgt_id"] == "E2"

    async def test_chunk_ids_extracted_from_all_sources(self) -> None:
        """chunk-abc comes from chunks + entities; chunk-def from relationships."""
        retriever = _make_retriever()
        await retriever.retrieve("query")

        # get_by_ids should be called with a list containing both chunk IDs
        call_args = retriever.visual_chunks.get_by_ids.call_args
        chunk_id_list = call_args[0][0]
        assert set(chunk_id_list) == {"chunk-abc", "chunk-def"}

    async def test_resolved_chunks_in_output(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("query")
        chunks = result["contexts"]["chunks"]
        # Both chunks resolved
        assert len(chunks) == 2
        # chunk-abc has reference_id="1" from chunks; chunk-def has "" (entity-sourced)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "reference_id" in chunk
            assert "file_path" in chunk
            assert "page_idx" in chunk
        chunk_ids = {c["chunk_id"] for c in chunks}
        assert chunk_ids == {"chunk-abc", "chunk-def"}

    async def test_chunks_contain_image_data(self) -> None:
        retriever = _make_retriever()
        result = await retriever.retrieve("query")
        chunks = result["contexts"]["chunks"]
        assert len(chunks) == 2
        assert all(c["image_data"] is not None for c in chunks)

    async def test_none_visual_data_includes_text_only(self) -> None:
        """Chunks without visual data are included as text-only (not dropped)."""

        # Map visual data by chunk_id so result is independent of set iteration order.
        # chunk-abc has text content (from chunks section) but no visual data → text-only.
        # chunk-def has visual data (from relationships source_id) but no text.
        async def _get_by_ids(ids: list[str]) -> list[dict | None]:
            mapping: dict[str, dict | None] = {
                "chunk-def": {
                    "image_data": _TINY_PNG_B64,
                    "page_index": 0,
                    "full_doc_id": "doc-1",
                    "file_path": "/test/doc.pdf",
                    "doc_title": "Test",
                },
            }
            return [mapping.get(cid) for cid in ids]

        retriever = _make_retriever()
        retriever.visual_chunks.get_by_ids = AsyncMock(side_effect=_get_by_ids)

        result = await retriever.retrieve("query")
        chunks = result["contexts"]["chunks"]
        # Both chunks included: chunk-def (visual), chunk-abc (text-only)
        assert len(chunks) == 2
        visual_chunks = [c for c in chunks if c.get("image_data")]
        text_only_chunks = [c for c in chunks if not c.get("image_data")]
        assert len(visual_chunks) == 1
        assert len(text_only_chunks) == 1


# ---------------------------------------------------------------------------
# TestRetrieveNoRerank
# ---------------------------------------------------------------------------


class TestRetrieveNoRerank:
    """Without rerank_func, resolved should be truncated to chunk_top_k."""

    async def test_truncated_to_chunk_top_k(self) -> None:
        # Create visual data with more items than chunk_top_k=2
        many_chunks = {
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "chunk_id": f"c-{i}",
                        "content": f"text {i}",
                        "reference_id": "1",
                        "file_path": "/test/doc.pdf",
                    }
                    for i in range(5)
                ],
            },
        }
        visual_data_list = [
            {
                "image_data": f"img{i}",
                "page_index": i,
                "full_doc_id": "doc-1",
                "file_path": "/test/doc.pdf",
                "doc_title": "Test",
            }
            for i in range(5)
        ]

        lightrag = MagicMock()
        lightrag.aquery_data = AsyncMock(return_value=many_chunks)
        lightrag.text_chunks.get_by_ids = AsyncMock(return_value=[])

        visual_chunks = MagicMock()
        visual_chunks.get_by_ids = AsyncMock(return_value=visual_data_list)

        retriever = VisualRetriever(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=_make_config(),
            rerank_func=None,
        )

        result = await retriever.retrieve("query", chunk_top_k=2)
        assert len(result["contexts"]["chunks"]) == 2


# ---------------------------------------------------------------------------
# TestRerankIntegration
# ---------------------------------------------------------------------------


class TestRerankIntegration:
    """Test unified rerank_func integration in Phase 3."""

    async def test_rerank_func_called_with_chunks(self) -> None:
        """rerank_func receives chunks with content + image_data."""
        received = {}

        async def mock_rerank(query, chunks, top_k):
            received["query"] = query
            received["chunks"] = chunks
            received["top_k"] = top_k
            # Return top 2 chunks with scores
            return [{**c, "rerank_score": 0.9 - i * 0.1} for i, c in enumerate(chunks[:top_k])]

        retriever = _make_retriever(rerank_func=mock_rerank)
        result = await retriever.retrieve("test query", chunk_top_k=2)

        assert received["query"] == "test query"
        assert received["top_k"] == 2
        # Chunks should have content and image_data
        assert any(c.get("image_data") for c in received["chunks"])
        assert len(result["contexts"]["chunks"]) == 2

    async def test_no_rerank_func_truncates(self) -> None:
        """Without rerank_func, candidates are truncated to chunk_top_k."""
        retriever = _make_retriever(rerank_func=None)
        result = await retriever.retrieve("query", chunk_top_k=1)
        assert len(result["contexts"]["chunks"]) == 1
