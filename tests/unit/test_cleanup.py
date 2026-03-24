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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hash_index(
    find_by_name_result=(None, None, None),
    find_by_path_result=(None, None, None),
):
    """Create a mock hash index with configurable lookup results."""
    index = MagicMock()
    index.find_by_name = AsyncMock(return_value=find_by_name_result)
    index.find_by_path = AsyncMock(return_value=find_by_path_result)
    return index


def _make_lightrag(docs: dict[str, str] | None = None):
    """Create a mock lightrag with doc_status API.

    Args:
        docs: mapping of doc_id -> file_path for processed docs
    """
    if docs is None:
        return None

    doc_status = MagicMock()

    # get_doc_by_file_path: exact match on file_path
    async def _get_by_path(fp: str):
        for d_id, stored_fp in docs.items():
            if stored_fp == fp:
                return {"id": d_id, "file_path": stored_fp}
        return None

    doc_status.get_doc_by_file_path = AsyncMock(side_effect=_get_by_path)

    # get_docs_by_status: return all docs as SimpleNamespace objects
    async def _get_by_status(_status):
        return {d_id: SimpleNamespace(file_path=fp) for d_id, fp in docs.items()}

    doc_status.get_docs_by_status = AsyncMock(side_effect=_get_by_status)

    lightrag = MagicMock()
    lightrag.doc_status = doc_status
    return lightrag


# ---------------------------------------------------------------------------
# TestCollectDeletionContext
# ---------------------------------------------------------------------------


class TestCollectDeletionContext:
    """Test multi-strategy doc_id lookup for deletion."""

    async def test_hash_index_finds_by_name(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
        )
        assert "doc-001" in ctx.doc_ids
        assert "sha256:abc" in ctx.content_hashes
        assert "/path/file.pdf" in ctx.file_paths
        assert "hash_index" in ctx.sources_used

    async def test_hash_index_finds_by_path(self) -> None:
        index = _make_hash_index(
            find_by_name_result=(None, None, None),
            find_by_path_result=("doc-002", "sha256:def", "/full/path/doc.pdf"),
        )
        ctx = await collect_deletion_context(
            identifier="/full/path/doc.pdf",
            hash_index=index,
        )
        assert "doc-002" in ctx.doc_ids
        assert "hash_index" in ctx.sources_used

    async def test_lightrag_doc_status_fallback(self) -> None:
        index = _make_hash_index()  # Returns nothing
        lightrag = _make_lightrag({"doc-003": "/storage/sources/local/report.pdf"})
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        assert "doc-003" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used

    async def test_both_strategies_merge(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        lightrag = _make_lightrag({"doc-002": "/storage/sources/local/file.pdf"})
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        assert ctx.doc_ids == {"doc-001", "doc-002"}

    async def test_no_matches(self) -> None:
        index = _make_hash_index()
        ctx = await collect_deletion_context(
            identifier="nonexistent.pdf",
            hash_index=index,
        )
        assert ctx.doc_ids == set()
        assert ctx.file_paths == set()

    async def test_stem_match_via_doc_status(self) -> None:
        index = _make_hash_index()
        lightrag = _make_lightrag({"doc-004": "/storage/report.pdf"})
        # Query with different extension but same stem
        ctx = await collect_deletion_context(
            identifier="report.xlsx",
            hash_index=index,
            lightrag=lightrag,
        )
        assert "doc-004" in ctx.doc_ids

    async def test_doc_status_exception_handled(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", "sha256:abc", "/path/file.pdf"))
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=lightrag,
        )
        # Hash index result should still be present despite doc_status failure
        assert "doc-001" in ctx.doc_ids

    async def test_no_lightrag(self) -> None:
        index = _make_hash_index(find_by_name_result=("doc-001", None, None))
        ctx = await collect_deletion_context(
            identifier="file.pdf",
            hash_index=index,
            lightrag=None,
        )
        assert "doc-001" in ctx.doc_ids

    async def test_no_hash_index_uses_doc_status(self) -> None:
        lightrag = _make_lightrag({"doc-005": "/storage/data.pdf"})
        ctx = await collect_deletion_context(
            identifier="data.pdf",
            hash_index=None,
            lightrag=lightrag,
        )
        assert "doc-005" in ctx.doc_ids
        assert "doc_status" in ctx.sources_used


# ---------------------------------------------------------------------------
# Helpers for cascade_delete tests
# ---------------------------------------------------------------------------


def _make_ctx(
    doc_ids: set[str] | None = None,
    content_hashes: set[str] | None = None,
) -> DeletionContext:
    """Build a minimal DeletionContext for cascade_delete tests."""
    return DeletionContext(
        identifier="test.pdf",
        doc_ids=doc_ids or set(),
        content_hashes=content_hashes or set(),
    )


def _make_lightrag_for_delete() -> MagicMock:
    """Mock lightrag with a working adelete_by_doc_id."""
    lr = MagicMock()
    lr.adelete_by_doc_id = AsyncMock(return_value=None)
    return lr


def _make_metadata_index(page_count: int = 0) -> MagicMock:
    """Mock metadata_index with get() returning page_count and delete()."""
    mi = MagicMock()
    mi.get = AsyncMock(return_value={"page_count": page_count})
    mi.delete = AsyncMock(return_value=None)
    return mi


def _make_visual_chunks() -> MagicMock:
    """Mock visual_chunks with a delete() method."""
    vc = MagicMock()
    vc.delete = AsyncMock(return_value=None)
    return vc


def _make_hash_index_for_delete() -> MagicMock:
    """Mock hash index with a remove() method."""
    hi = MagicMock()
    hi.remove = AsyncMock(return_value=True)
    return hi


# ---------------------------------------------------------------------------
# TestCascadeDelete
# ---------------------------------------------------------------------------


class TestCascadeDelete:
    """Test 5-layer cascade_delete with per-layer fault isolation."""

    async def test_cascade_delete_all_layers_called(self) -> None:
        """All stores are called when doc_ids and content_hashes are present."""
        ctx = _make_ctx(doc_ids={"doc-1"}, content_hashes={"sha256:aaa"})
        lr = _make_lightrag_for_delete()
        vc = _make_visual_chunks()
        mi = _make_metadata_index(page_count=3)
        hi = _make_hash_index_for_delete()

        # Patch compute_mdhash_id so we don't need lightrag installed
        with patch(
            "dlightrag.core.ingestion.cleanup.cascade_delete.__wrapped__"
            if False
            else "lightrag.utils.compute_mdhash_id",
            side_effect=lambda s, prefix="": f"{prefix}{s}",
        ):
            stats = await cascade_delete(
                ctx=ctx,
                lightrag=lr,
                visual_chunks=vc,
                hash_index=hi,
                metadata_index=mi,
            )

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-1", delete_llm_cache=True)
        vc.delete.assert_awaited_once()
        mi.delete.assert_awaited_once_with("doc-1")
        hi.remove.assert_awaited_once_with("sha256:aaa")
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []

    async def test_cascade_delete_layer1_failure_continues(self) -> None:
        """Layer 1 failure is recorded but Layers 2-4 still execute."""
        ctx = _make_ctx(doc_ids={"doc-2"}, content_hashes={"sha256:bbb"})
        lr = _make_lightrag_for_delete()
        lr.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("connection lost"))
        vc = _make_visual_chunks()
        mi = _make_metadata_index(page_count=2)
        hi = _make_hash_index_for_delete()

        with patch(
            "lightrag.utils.compute_mdhash_id",
            side_effect=lambda s, prefix="": f"{prefix}{s}",
        ):
            stats = await cascade_delete(
                ctx=ctx,
                lightrag=lr,
                visual_chunks=vc,
                hash_index=hi,
                metadata_index=mi,
            )

        # Layer 1 error recorded
        assert any("Layer 1" in e for e in stats["errors"])
        # docs_deleted NOT incremented because Layer 1 raised
        assert stats["docs_deleted"] == 0
        # Layers 2-4 still called despite Layer 1 failure
        vc.delete.assert_awaited_once()
        mi.delete.assert_awaited_once_with("doc-2")
        hi.remove.assert_awaited_once_with("sha256:bbb")

    async def test_cascade_delete_empty_context(self) -> None:
        """No doc_ids → returns 0 deleted and no errors."""
        ctx = _make_ctx(doc_ids=set(), content_hashes=set())
        lr = _make_lightrag_for_delete()

        stats = await cascade_delete(ctx=ctx, lightrag=lr)

        lr.adelete_by_doc_id.assert_not_called()
        assert stats["docs_deleted"] == 0
        assert stats["errors"] == []

    async def test_cascade_delete_skips_none_stores(self) -> None:
        """visual_chunks=None and hash_index=None → Layers 2 and 4 skipped."""
        ctx = _make_ctx(doc_ids={"doc-3"}, content_hashes={"sha256:ccc"})
        lr = _make_lightrag_for_delete()
        mi = _make_metadata_index(page_count=0)

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            visual_chunks=None,
            hash_index=None,
            metadata_index=mi,
        )

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-3", delete_llm_cache=True)
        mi.delete.assert_awaited_once_with("doc-3")
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []
