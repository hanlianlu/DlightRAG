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
    # Nested embedding config
    config.embedding.model = "test-model"
    config.embedding.dim = 1024
    config.embedding.base_url = "http://localhost:8000/v1"
    config.embedding.api_key = "test-key"
    config.embedding_func_max_async = 8
    config.kg_entity_types = ["Person", "Organization"]
    # Nested rerank config
    config.rerank.enabled = False
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
    lightrag.doc_status = MagicMock()
    lightrag.doc_status.upsert = AsyncMock()
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
    """chunk_ids written by aingest must match those deleted by unified_delete_files.

    Uses real extractor chunk_id formula and dict-backed visual_chunks +
    doc_status stores to verify the ingest→delete invariant end-to-end
    via the 3-layer cross-backend deletion architecture.
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
        from dlightrag.unifiedrepresent.lifecycle import unified_delete_files

        config = _make_config()
        lightrag = _make_lightrag()

        # Dict-backed visual_chunks: tracks upserted and deleted keys
        vc_store: dict[str, dict] = {}

        async def vc_upsert(data: dict) -> None:
            vc_store.update(data)

        async def vc_delete(ids: list[str]) -> None:
            for k in ids:
                vc_store.pop(k, None)

        visual_chunks = MagicMock()
        visual_chunks.upsert = AsyncMock(side_effect=vc_upsert)
        visual_chunks.delete = AsyncMock(side_effect=vc_delete)

        # Dict-backed doc_status: aingest writes, adelete_by_doc_id reads
        doc_status_store: dict[str, dict] = {}

        async def ds_upsert(data: dict) -> None:
            doc_status_store.update(data)

        lightrag.doc_status = MagicMock()
        lightrag.doc_status.upsert = AsyncMock(side_effect=ds_upsert)
        lightrag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
        lightrag.doc_status.get_docs_by_status = AsyncMock(return_value={})
        lightrag.adelete_by_doc_id = AsyncMock()

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
        assert len(vc_store) == 3

        # Verify doc_status was written
        assert "doc-round" in doc_status_store

        # --- Delete via unified_delete_files (3-layer architecture) ---
        # Mock hash_index that finds our doc
        hash_index = MagicMock()
        hash_index.find_by_name = AsyncMock(
            return_value=("doc-round", "sha256:abc", "/fake/test.pdf")
        )
        hash_index.find_by_path = AsyncMock(return_value=(None, None, None))
        hash_index.remove = AsyncMock(return_value=True)

        # Mock metadata_index with page_count for visual_chunks cleanup
        metadata_index = MagicMock()
        metadata_index.get = AsyncMock(return_value={"page_count": 3})
        metadata_index.delete = AsyncMock()

        delete_results = await unified_delete_files(
            engine=engine,
            hash_index=hash_index,
            lightrag=lightrag,
            filenames=["test.pdf"],
            metadata_index=metadata_index,
        )

        assert len(delete_results) == 1
        assert delete_results[0]["status"] == "deleted"

        # LightRAG adelete_by_doc_id was called (Layer 1)
        lightrag.adelete_by_doc_id.assert_awaited_once_with("doc-round", delete_llm_cache=True)

        # The visual_chunks store should be empty (Layer 2)
        assert vc_store == {}, f"Orphaned visual_chunks after delete: {set(vc_store.keys())}"

        # DlightRAG indexes cleaned (Layer 3)
        hash_index.remove.assert_awaited_once_with("sha256:abc")
        metadata_index.delete.assert_awaited_once_with("doc-round")
