# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for UnifiedRepresentEngine.

Tests data flow integrity across real components — uses real EntityExtractor
chunk_id formula with mocked external calls (VLM, KG, embedding API) and
dict-backed stores to verify cross-store consistency and ingest→delete invariants.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dlightrag.unifiedrepresent.engine import UnifiedRepresentEngine
from dlightrag.unifiedrepresent.renderer import RenderResult

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    config = MagicMock()
    config.page_render_dpi = 250
    config.embedding_model = "test-model"
    config.embedding_dim = 1024
    config.effective_embedding_provider = "openai"
    config._get_url = MagicMock(return_value="http://localhost:8000/v1")
    config._get_provider_api_key = MagicMock(return_value="test-key")
    config.kg_entity_types = ["Person", "Organization"]
    config.enable_rerank = False
    config.default_mode = "mix"
    config.top_k = 60
    config.chunk_top_k = 10
    return config


def _make_lightrag() -> MagicMock:
    lightrag = MagicMock()
    lightrag.full_docs = MagicMock()
    lightrag.full_docs.upsert = AsyncMock()
    lightrag.text_chunks = MagicMock()
    lightrag.text_chunks.upsert = AsyncMock()
    lightrag.chunks_vdb = MagicMock()
    lightrag.chunks_vdb.upsert = AsyncMock()
    lightrag.chunks_vdb.embedding_func = MagicMock()
    return lightrag


def _make_engine_with_real_extractor(
    lightrag: MagicMock,
    visual_chunks: MagicMock,
    config: MagicMock,
):
    """Build engine with real EntityExtractor (mocked VLM + stubbed KG)."""
    from dlightrag.unifiedrepresent.extractor import EntityExtractor as RealExtractor

    with patch("dlightrag.unifiedrepresent.engine.EntityExtractor"):
        engine = UnifiedRepresentEngine(
            lightrag=lightrag, visual_chunks=visual_chunks, config=config
        )

    engine.extractor = RealExtractor(
        lightrag=lightrag,
        entity_types=config.kg_entity_types,
        vision_model_func=AsyncMock(return_value="Page description text"),
    )
    return engine


# ---------------------------------------------------------------------------
# TestAingestDataConsistency
# ---------------------------------------------------------------------------


class TestAingestDataConsistency:
    """aingest must write consistent chunk_ids and data across all stores.

    Uses real EntityExtractor chunk_id formula and captures actual upsert
    calls to verify chunks_vdb, text_chunks, and visual_chunks all receive
    the same set of chunk_ids with correct data shapes.
    """

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_all_stores_get_consistent_chunk_ids(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()
        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock()

        engine = _make_engine_with_real_extractor(lightrag, visual_chunks, config)

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges", new_callable=AsyncMock
            ),
        ):
            # Renderer: 3 pages
            images = [Image.new("RGB", (50, 50), c) for c in ("white", "blue", "red")]
            engine.renderer.render_file = AsyncMock(
                return_value=RenderResult(
                    pages=list(enumerate(images)),
                    metadata={"title": "T", "page_count": 3},
                )
            )
            engine.embedder.embed_pages = AsyncMock(
                return_value=np.zeros((3, 1024), dtype=np.float32)
            )

            await engine.aingest("/fake/3page.pdf", doc_id="doc-consist")

        # Extract chunk_ids written to each store
        chunks_vdb_keys = set(lightrag.chunks_vdb.upsert.call_args[0][0].keys())
        text_chunks_keys = set(lightrag.text_chunks.upsert.call_args[0][0].keys())
        visual_chunks_keys = set(visual_chunks.upsert.call_args[0][0].keys())

        # Core invariant: all three stores must have identical chunk_id sets
        assert chunks_vdb_keys == text_chunks_keys == visual_chunks_keys
        assert len(chunks_vdb_keys) == 3

        # Verify text_chunks has required fields for retrieval
        tc_data = lightrag.text_chunks.upsert.call_args[0][0]
        for _chunk_id, chunk in tc_data.items():
            assert chunk["full_doc_id"] == "doc-consist"
            assert chunk["source_type"] == "unified_represent"
            assert "chunk_order_index" in chunk
            assert "page_idx" in chunk
            assert chunk["page_idx"] == chunk["chunk_order_index"]
            assert "content" in chunk

        # Verify visual_chunks has required fields for VLM answer generation
        vc_data = visual_chunks.upsert.call_args[0][0]
        for _chunk_id, chunk in vc_data.items():
            assert "image_data" in chunk
            assert chunk["full_doc_id"] == "doc-consist"
            assert "page_index" in chunk

        # Verify full_docs stores page_count (needed for deletion)
        fd_data = lightrag.full_docs.upsert.call_args[0][0]
        assert fd_data["doc-consist"]["page_count"] == 3


# ---------------------------------------------------------------------------
# TestIngestDeleteRoundTrip
# ---------------------------------------------------------------------------


class TestIngestDeleteRoundTrip:
    """chunk_ids written by aingest must match those deleted by adelete_doc.

    Uses real extractor chunk_id formula and dict-backed visual_chunks +
    full_docs stores to verify the ingest→delete invariant end-to-end.
    """

    @patch("dlightrag.unifiedrepresent.engine.VisualRetriever")
    @patch("dlightrag.unifiedrepresent.engine.VisualEmbedder")
    @patch("dlightrag.unifiedrepresent.engine.PageRenderer")
    async def test_delete_removes_exactly_what_ingest_wrote(
        self,
        _renderer_cls: MagicMock,
        _embedder_cls: MagicMock,
        _retriever_cls: MagicMock,
    ) -> None:
        config = _make_config()
        lightrag = _make_lightrag()

        # Dict-backed visual_chunks: tracks upserted and deleted keys
        store: dict[str, dict] = {}

        async def vc_upsert(data: dict) -> None:
            store.update(data)

        async def vc_delete(ids: list[str]) -> None:
            for k in ids:
                store.pop(k, None)

        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock(side_effect=vc_upsert)
        visual_chunks.delete = AsyncMock(side_effect=vc_delete)

        # Dict-backed full_docs: aingest writes page_count, adelete_doc reads it
        full_docs_store: dict[str, dict] = {}

        async def fd_upsert(data: dict) -> None:
            full_docs_store.update(data)

        async def fd_get_by_id(doc_id: str) -> dict | None:
            return full_docs_store.get(doc_id)

        lightrag.full_docs.upsert = AsyncMock(side_effect=fd_upsert)
        lightrag.full_docs.get_by_id = AsyncMock(side_effect=fd_get_by_id)

        engine = _make_engine_with_real_extractor(lightrag, visual_chunks, config)

        with (
            patch(
                "dlightrag.unifiedrepresent.extractor.extract_entities",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "dlightrag.unifiedrepresent.extractor.merge_nodes_and_edges", new_callable=AsyncMock
            ),
        ):
            # Mock renderer: 3 pages
            images = [Image.new("RGB", (100, 100), c) for c in ("white", "blue", "red")]
            engine.renderer.render_file = AsyncMock(
                return_value=RenderResult(
                    pages=list(enumerate(images)),
                    metadata={"title": "Test", "page_count": 3},
                )
            )
            engine.embedder.embed_pages = AsyncMock(
                return_value=np.zeros((3, 1024), dtype=np.float32)
            )

            # --- Ingest ---
            result = await engine.aingest("/fake/test.pdf", doc_id="doc-round")

        assert result["page_count"] == 3
        assert len(store) == 3

        # --- Delete ---
        delete_result = await engine.adelete_doc("doc-round")

        assert delete_result["visual_chunks_deleted"] == 3
        # The store should be empty: every key written by aingest was deleted
        assert store == {}, f"Orphaned visual_chunks after delete: {set(store.keys())}"
