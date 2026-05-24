# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deletion context collection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.ingestion.cleanup import (
    DeletionContext,
    cascade_delete,
    collect_deletion_context,
)


def _make_lightrag(docs: dict[str, str] | None = None):
    """Create a mock lightrag with doc_status API."""
    if docs is None:
        return None

    doc_status = MagicMock()

    async def _get_by_path(fp: str):
        for d_id, stored_fp in docs.items():
            if stored_fp == fp:
                return {"id": d_id, "file_path": stored_fp}
        return None

    async def _get_by_status(_status):
        return {d_id: SimpleNamespace(file_path=fp) for d_id, fp in docs.items()}

    doc_status.get_doc_by_file_path = AsyncMock(side_effect=_get_by_path)
    doc_status.get_docs_by_status = AsyncMock(side_effect=_get_by_status)

    lightrag = MagicMock()
    lightrag.doc_status = doc_status
    return lightrag


class TestCollectDeletionContext:
    """Test multi-strategy doc_id lookup for deletion."""

    async def test_lightrag_doc_status_exact_path(self) -> None:
        lightrag = _make_lightrag({"doc-003": "/storage/docs/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="/storage/docs/report.pdf",
            lightrag=lightrag,
        )
        assert "doc-003" in ctx.doc_ids
        assert "/storage/docs/report.pdf" in ctx.file_paths
        assert "doc_status" in ctx.sources_used

    async def test_lightrag_doc_status_basename(self) -> None:
        lightrag = _make_lightrag({"doc-004": "/storage/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            lightrag=lightrag,
        )
        assert "doc-004" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used

    async def test_stem_match_via_doc_status(self) -> None:
        lightrag = _make_lightrag({"doc-005": "/storage/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.xlsx",
            lightrag=lightrag,
        )
        assert "doc-005" in ctx.doc_ids

    async def test_metadata_index_fallback(self) -> None:
        metadata_index = MagicMock()
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-006"])
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            lightrag=None,
            metadata_index=metadata_index,
        )
        assert ctx.doc_ids == {"doc-006"}
        assert ctx.sources_used == ["metadata_index"]

    async def test_doc_status_exception_falls_back_to_metadata(self) -> None:
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        metadata_index = MagicMock()
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-007"])

        ctx = await collect_deletion_context(
            identifier="file.pdf",
            lightrag=lightrag,
            metadata_index=metadata_index,
        )

        assert ctx.doc_ids == {"doc-007"}
        assert ctx.sources_used == ["metadata_index"]

    async def test_no_matches(self) -> None:
        ctx = await collect_deletion_context(identifier="nonexistent.pdf")
        assert ctx.doc_ids == set()
        assert ctx.file_paths == set()
        assert ctx.sources_used == []


def _make_ctx(doc_ids: set[str] | None = None) -> DeletionContext:
    """Build a minimal DeletionContext for cascade_delete tests."""
    return DeletionContext(identifier="test.pdf", doc_ids=doc_ids or set())


def _make_lightrag_for_delete() -> MagicMock:
    lr = MagicMock()
    lr.adelete_by_doc_id = AsyncMock(return_value=None)
    return lr


def _make_metadata_index(page_count: int = 0) -> MagicMock:
    mi = MagicMock()
    mi.get = AsyncMock(return_value={"page_count": page_count})
    mi.delete = AsyncMock(return_value=None)
    return mi


def _make_visual_chunks() -> MagicMock:
    vc = MagicMock()
    vc.delete = AsyncMock(return_value=None)
    return vc


class TestCascadeDelete:
    """Test cascade_delete with per-layer fault isolation."""

    async def test_cascade_delete_all_layers_called(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-1"})
        lr = _make_lightrag_for_delete()
        vc = _make_visual_chunks()
        mi = _make_metadata_index(page_count=3)

        with patch(
            "lightrag.utils.compute_mdhash_id",
            side_effect=lambda s, prefix="": f"{prefix}{s}",
        ):
            stats = await cascade_delete(
                ctx=ctx,
                lightrag=lr,
                visual_chunks=vc,
                metadata_index=mi,
            )

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-1", delete_llm_cache=True)
        vc.delete.assert_awaited_once()
        mi.delete.assert_awaited_once_with("doc-1")
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []

    async def test_cascade_delete_clears_artifacts_and_chunk_provenance(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-1"})
        lr = _make_lightrag_for_delete()
        artifacts = MagicMock()
        artifacts.delete_doc = AsyncMock(return_value=None)
        provenance = MagicMock()
        provenance.delete_doc = AsyncMock(return_value=None)

        await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            document_artifacts=artifacts,
            chunk_provenance=provenance,
        )

        provenance.delete_doc.assert_awaited_once_with("doc-1")
        artifacts.delete_doc.assert_awaited_once_with("doc-1")

    async def test_cascade_delete_layer1_failure_continues(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-2"})
        lr = _make_lightrag_for_delete()
        lr.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("connection lost"))
        vc = _make_visual_chunks()
        mi = _make_metadata_index(page_count=2)

        with patch(
            "lightrag.utils.compute_mdhash_id",
            side_effect=lambda s, prefix="": f"{prefix}{s}",
        ):
            stats = await cascade_delete(
                ctx=ctx,
                lightrag=lr,
                visual_chunks=vc,
                metadata_index=mi,
            )

        assert any("Layer 1" in e for e in stats["errors"])
        assert stats["docs_deleted"] == 0
        vc.delete.assert_awaited_once()
        mi.delete.assert_awaited_once_with("doc-2")

    async def test_cascade_delete_empty_context(self) -> None:
        ctx = _make_ctx(doc_ids=set())
        lr = _make_lightrag_for_delete()

        stats = await cascade_delete(ctx=ctx, lightrag=lr)

        lr.adelete_by_doc_id.assert_not_called()
        assert stats["docs_deleted"] == 0
        assert stats["errors"] == []

    async def test_cascade_delete_skips_none_stores(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-3"})
        lr = _make_lightrag_for_delete()

        stats = await cascade_delete(ctx=ctx, lightrag=lr)

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-3", delete_llm_cache=True)
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []
