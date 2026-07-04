# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for deletion context collection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from dlightrag.core.ingestion.cleanup import (
    DeletionContext,
    cascade_delete,
    collect_deletion_context,
    remove_deleted_files,
)


def _make_lightrag(docs: dict[str, str] | None = None):
    """Create a mock lightrag with doc_status API."""
    if docs is None:
        return None

    doc_status = MagicMock()

    async def _get_by_path(fp: str):
        for _d_id, stored_fp in docs.items():
            if stored_fp == fp:
                # Real LightRAG strips 'id' from the return dict
                return {"file_path": stored_fp}
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
        metadata_index.find_by_file_path = AsyncMock(return_value=[])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-006"])
        ctx = await collect_deletion_context(
            identifier="report.pdf",
            lightrag=None,
            metadata_index=metadata_index,
        )
        assert ctx.doc_ids == {"doc-006"}
        assert ctx.sources_used == ["metadata_index"]

    async def test_metadata_index_exact_remote_path_precedes_filename(self) -> None:
        lightrag = _make_lightrag(
            {"doc-remote": "/inputs/default/__remote_ingest__/s3/b1/report__abc.pdf"}
        )
        metadata_index = MagicMock()
        metadata_index.find_by_file_path = AsyncMock(return_value=["doc-remote"])
        metadata_index.find_by_filename = AsyncMock(return_value=["doc-wrong"])

        ctx = await collect_deletion_context(
            identifier="s3://bucket/team-a/report.pdf",
            lightrag=lightrag,
            metadata_index=metadata_index,
        )

        assert ctx.doc_ids == {"doc-remote"}
        assert ctx.file_paths == {"/inputs/default/__remote_ingest__/s3/b1/report__abc.pdf"}
        metadata_index.find_by_file_path.assert_awaited_once_with("s3://bucket/team-a/report.pdf")
        metadata_index.find_by_filename.assert_not_awaited()

    async def test_doc_status_exception_falls_back_to_metadata(self) -> None:
        lightrag = MagicMock()
        lightrag.doc_status = MagicMock()
        lightrag.doc_status.get_doc_by_file_path = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        metadata_index = MagicMock()
        metadata_index.find_by_file_path = AsyncMock(return_value=[])
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


class TestRemoveDeletedFiles:
    """Test physical source cleanup boundaries."""

    def test_remote_uri_does_not_delete_local_same_basename(self, tmp_path) -> None:
        workspace_input = tmp_path / "inputs" / "default"
        workspace_input.mkdir(parents=True)
        local_copy = workspace_input / "report.pdf"
        local_copy.write_bytes(b"local")

        removed = remove_deleted_files(
            {
                "azure://container/team-a/report.pdf",
                "s3://bucket/team-b/report.pdf",
                "https://api.bynder.com/docs/report.pdf",
            },
            str(workspace_input),
        )

        assert removed == 0
        assert local_copy.read_bytes() == b"local"

    def test_nested_remote_parser_path_removes_artifacts_not_remote_source(self, tmp_path) -> None:
        workspace_input = tmp_path / "inputs" / "default"
        batch_root = workspace_input / "__remote_ingest__" / "s3" / "batch-1"
        parsed_root = batch_root / "__parsed__"
        artifact_dir = parsed_root / "report__abc.parsed"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "blocks.jsonl").write_text("{}\n")
        archived_source = parsed_root / "report__abc.pdf"
        archived_source.write_bytes(b"%PDF")
        direct_source = batch_root / "report__abc.pdf"
        direct_source.write_bytes(b"%PDF")

        removed = remove_deleted_files({str(direct_source)}, str(workspace_input))

        assert removed == 3
        assert not direct_source.exists()
        assert not archived_source.exists()
        assert not artifact_dir.exists()


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


class TestCascadeDelete:
    """Test cascade_delete with per-layer fault isolation."""

    async def test_cascade_delete_lightrag_and_metadata_called(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-1"})
        lr = _make_lightrag_for_delete()
        mi = _make_metadata_index()

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            metadata_index=mi,
        )

        lr.adelete_by_doc_id.assert_awaited_once_with("doc-1", delete_llm_cache=True)
        mi.delete.assert_awaited_once_with("doc-1")
        assert stats["docs_deleted"] == 1
        assert stats["errors"] == []

    async def test_cascade_delete_lightrag_failure_continues_to_metadata(self) -> None:
        ctx = _make_ctx(doc_ids={"doc-2"})
        lr = _make_lightrag_for_delete()
        lr.adelete_by_doc_id = AsyncMock(side_effect=RuntimeError("connection lost"))
        mi = _make_metadata_index()

        stats = await cascade_delete(
            ctx=ctx,
            lightrag=lr,
            metadata_index=mi,
        )

        assert any("Layer 1" in e for e in stats["errors"])
        assert stats["docs_deleted"] == 0
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
