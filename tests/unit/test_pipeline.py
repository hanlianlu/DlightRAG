# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for IngestionPipeline core logic."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.config import DlightragConfig
from dlightrag.core.ingestion.pipeline import (
    IngestionCancelledError,
    IngestionPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    test_config: DlightragConfig,
    *,
    cancel_checker=None,
    mineru_backend=None,
) -> IngestionPipeline:
    """Build an IngestionPipeline with a fully-mocked RAGAnything."""
    rag = MagicMock()
    rag.parse_document = AsyncMock(
        return_value=(
            [{"type": "text", "text": "hello world"}],
            "doc-001",
        )
    )
    rag.insert_content_list = AsyncMock()
    rag._ensure_lightrag_initialized = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.lightrag = MagicMock()

    pipeline = IngestionPipeline(
        rag,
        config=test_config,
        max_concurrent=2,
        cancel_checker=cancel_checker,
        mineru_backend=mineru_backend,
    )
    # Stub out converter to never trigger Excel conversion
    pipeline.converter = MagicMock()
    pipeline.converter.should_convert.return_value = False
    return pipeline


# ---------------------------------------------------------------------------
# TestIngestSingleFileWithPolicy
# ---------------------------------------------------------------------------


class TestIngestSingleFileWithPolicy:
    """Core parse -> filter -> insert pipeline."""

    async def test_successful_ingest(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(
            file_path, artifacts, content_hash="sha256:abc123"
        )

        assert result.status == "success"
        assert result.processed == 1
        assert result.doc_id == "doc-001"
        pipeline.rag.insert_content_list.assert_awaited_once()
        # Hash should be registered
        assert pipeline._hash_index.lookup("sha256:abc123") is not None

    async def test_all_content_filtered_by_policy(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        # Return only discarded blocks
        pipeline.rag.parse_document.return_value = (
            [{"type": "discarded", "text": "noise"}],
            "doc-002",
        )
        file_path = tmp_path / "noisy.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "success"
        assert result.processed == 1
        assert result.stats is not None
        assert result.stats.indexed == 0
        pipeline.rag.insert_content_list.assert_not_awaited()

    async def test_parse_document_error(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        pipeline.rag.parse_document.side_effect = RuntimeError("parse failed")
        file_path = tmp_path / "bad.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "error"
        assert "parse failed" in result.error

    async def test_no_hash_registration_when_hash_is_none(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(
            file_path, artifacts, content_hash=None
        )

        assert result.status == "success"
        # Hash index should remain empty since no hash was provided
        assert await pipeline._hash_index.list_all() == []

    async def test_mineru_backend_passed_to_parse_kwargs(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config, mineru_backend="hybrid-auto-engine")
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("content")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        call_kwargs = pipeline.rag.parse_document.call_args
        assert call_kwargs.kwargs.get("backend") == "hybrid-auto-engine"


# ---------------------------------------------------------------------------
# TestAingestFromLocal
# ---------------------------------------------------------------------------


class TestAingestFromLocal:
    """Local ingestion workflow: dedup, copy, process."""

    async def test_single_file_new(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "new.pdf"
        src.write_text("content")

        result = await pipeline.aingest_from_local(src)

        assert result.source_type == "local"
        assert result.total_files == 1
        assert result.processed == 1
        assert result.skipped == 0

    async def test_single_file_duplicate_skipped(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "dup.pdf"
        src.write_text("content")

        # Ingest once
        await pipeline.aingest_from_local(src)

        # Second ingest should skip
        result = await pipeline.aingest_from_local(src)

        assert result.processed == 0
        assert result.skipped == 1

    async def test_path_not_found(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)
        with pytest.raises(FileNotFoundError, match="Path not found"):
            await pipeline.aingest_from_local(Path("/nonexistent/file.pdf"))

    async def test_directory_multiple_files(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.pdf").write_text("aaa")
        (d / "b.pdf").write_text("bbb")
        (d / "c.pdf").write_text("ccc")

        result = await pipeline.aingest_from_local(d)

        assert result.total_files == 3
        assert result.processed == 3
        assert result.skipped == 0

    async def test_directory_empty(self, test_config: DlightragConfig, tmp_path: Path) -> None:
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "empty_dir"
        d.mkdir()

        result = await pipeline.aingest_from_local(d)

        assert result.processed == 0
        assert result.total_files == 0

    @pytest.mark.asyncio
    async def test_single_file_records_original_path(self, test_config):
        """Ingested file metadata records the original path, not a sources/ copy."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        result = await pipeline.aingest_from_local(test_file)

        assert result.status == "success"
        assert result.processed == 1
        assert "sources" not in (result.source_path or "")
        assert str(test_file.resolve()) in (result.source_path or "")

    @pytest.mark.asyncio
    async def test_no_copy_to_sources_dir(self, test_config):
        """Files are NOT copied to sources/ anymore."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        await pipeline.aingest_from_local(test_file)

        sources_local = test_config.working_dir_path / "sources" / "local"
        assert not sources_local.exists() or not list(sources_local.iterdir())

    @pytest.mark.asyncio
    async def test_temp_cleaned_up_after_ingestion(self, test_config):
        """Temp dir is cleaned up after ingestion completes."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "outside" / "report.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")

        await pipeline.aingest_from_local(test_file)

        tmp_dir = test_config.temp_dir
        assert not tmp_dir.exists() or not list(tmp_dir.iterdir())


# ---------------------------------------------------------------------------
# TestCheckCancelled
# ---------------------------------------------------------------------------


class TestCheckCancelled:
    """Test cancellation detection."""

    async def test_checker_returns_false(self, test_config: DlightragConfig) -> None:
        checker = AsyncMock(return_value=False)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        await pipeline._check_cancelled()

    async def test_checker_returns_true(self, test_config: DlightragConfig) -> None:
        checker = AsyncMock(return_value=True)
        pipeline = _make_pipeline(test_config, cancel_checker=checker)
        with pytest.raises(IngestionCancelledError, match="cancelled by caller"):
            await pipeline._check_cancelled()

    async def test_asyncio_cancellation(self, test_config: DlightragConfig) -> None:
        pipeline = _make_pipeline(test_config)

        async def run():
            task = asyncio.current_task()
            task.cancel()
            await asyncio.sleep(0)  # Yield to allow cancellation
            await pipeline._check_cancelled()

        with pytest.raises(asyncio.CancelledError):
            await run()


# ---------------------------------------------------------------------------
# TestTempDirAndSourceUri
# ---------------------------------------------------------------------------


class TestTempDirAndSourceUri:
    """Tests for temp dir creation and source_uri flow."""

    @pytest.mark.asyncio
    async def test_source_uri_passed_to_insert_content_list(self, test_config):
        """source_uri (not parse_path) is passed to insert_content_list."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.txt"
        test_file.write_text("hello")

        await pipeline._ingest_single_file_with_policy(
            file_path=test_file,
            artifacts_dir=test_config.artifacts_dir,
            source_uri="/original/path/test.txt",
        )

        # insert_content_list should receive source_uri, not parse_path
        call_args = pipeline.rag.insert_content_list.call_args
        assert call_args is not None
        # Check both positional and keyword args
        file_path_arg = call_args.kwargs.get("file_path") or call_args[1].get("file_path")
        assert file_path_arg == "/original/path/test.txt"

    @pytest.mark.asyncio
    async def test_source_uri_stored_in_hash_index(self, test_config):
        """source_uri is stored in hash_index, not parse_path."""
        pipeline = _make_pipeline(test_config)
        test_file = test_config.working_dir_path / "test.txt"
        test_file.write_text("hello")

        await pipeline._ingest_single_file_with_policy(
            file_path=test_file,
            artifacts_dir=test_config.artifacts_dir,
            content_hash="abc123",
            source_uri="/original/path/test.txt",
        )

        entries = await pipeline._hash_index.list_all()
        assert len(entries) == 1
        assert entries[0]["file_path"] == "/original/path/test.txt"


# ---------------------------------------------------------------------------
# TestAingestFromAzureBlob
# ---------------------------------------------------------------------------


class TestAingestFromAzureBlob:
    """Azure Blob ingestion workflow: temp download, dedup, ingest, cleanup."""

    @pytest.mark.asyncio
    async def test_azure_source_uri_format(self, test_config):
        """Azure blobs record azure://container/path as source_uri."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        result = await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mycontainer",
            blob_path="data/report.pdf",
        )

        assert result.status == "success"
        pipeline.rag.insert_content_list.assert_awaited_once()
        call_kwargs = pipeline.rag.insert_content_list.call_args
        file_path_arg = call_kwargs.kwargs.get("file_path", "")
        assert file_path_arg == "azure://mycontainer/data/report.pdf"

    @pytest.mark.asyncio
    async def test_azure_no_permanent_download(self, test_config):
        """Azure blobs are NOT permanently stored in sources/."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mycontainer",
            blob_path="data/report.pdf",
        )

        sources_azure = test_config.working_dir_path / "sources" / "azure_blobs"
        assert not sources_azure.exists() or not list(sources_azure.rglob("*"))

    @pytest.mark.asyncio
    async def test_azure_temp_cleaned_up(self, test_config):
        """Temp dir is cleaned up after Azure ingestion."""
        pipeline = _make_pipeline(test_config)
        mock_source = AsyncMock()
        mock_source.aload_document = AsyncMock(return_value=b"blob content")

        await pipeline.aingest_from_azure_blob(
            source=mock_source,
            container_name="mycontainer",
            blob_path="data/report.pdf",
        )

        tmp_dir = test_config.temp_dir
        assert not tmp_dir.exists() or not list(tmp_dir.iterdir())


# ---------------------------------------------------------------------------
# TestAingestFromLocalEdgeCases
# ---------------------------------------------------------------------------


class TestAingestFromLocalEdgeCases:
    """Additional local ingestion edge cases."""

    async def test_replace_mode_bypasses_dedup(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """replace=True re-ingests even when hash exists."""
        pipeline = _make_pipeline(test_config)
        src = tmp_path / "doc.pdf"
        src.write_text("content")

        # First ingest
        await pipeline.aingest_from_local(src)
        # Second ingest with replace — should NOT skip
        result = await pipeline.aingest_from_local(src, replace=True)
        assert result.processed == 1
        assert result.skipped == 0

    async def test_directory_partial_failure(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """One file failing does not prevent others from processing."""
        pipeline = _make_pipeline(test_config)
        d = tmp_path / "docs"
        d.mkdir()
        (d / "good.pdf").write_text("ok")
        (d / "bad.pdf").write_text("fail")

        call_count = 0
        original_parse = pipeline.rag.parse_document

        async def maybe_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            file_path = str(kwargs.get("file_path", args[0] if args else ""))
            if "bad" in file_path:
                raise RuntimeError("parse failed")
            return await original_parse(*args, **kwargs)

        pipeline.rag.parse_document = AsyncMock(side_effect=maybe_fail)
        result = await pipeline.aingest_from_local(d)

        assert result.total_files == 2
        # At least one succeeded
        assert result.processed >= 1


# ---------------------------------------------------------------------------
# TestVlmParserRouting
# ---------------------------------------------------------------------------


class TestVlmParserRouting:
    """Test that parser='vlm' routes to VlmOcrParser."""

    @pytest.mark.asyncio
    async def test_vlm_parser_used_when_configured(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """When vlm_parser is set, pipeline uses it instead of rag.parse_document."""
        from dlightrag.core.ingestion.vlm_parser import VlmOcrParser

        mock_vlm_parser = MagicMock(spec=VlmOcrParser)
        mock_vlm_parser.parse = AsyncMock(
            return_value=(
                [{"type": "text", "text": "VLM extracted", "page_idx": 0}],
                "doc-vlm-001",
            )
        )

        pipeline = _make_pipeline(test_config)
        pipeline.vlm_parser = mock_vlm_parser

        file_path = tmp_path / "test.pdf"
        file_path.write_bytes(b"fake")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)
        assert result.status == "success"
        mock_vlm_parser.parse.assert_called_once()
        pipeline.rag.parse_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_parser_without_vlm(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """Without vlm_parser, pipeline uses rag.parse_document."""
        pipeline = _make_pipeline(test_config)

        file_path = tmp_path / "test.pdf"
        file_path.write_bytes(b"fake")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)
        assert result.status == "success"
        pipeline.rag.parse_document.assert_called_once()


# ---------------------------------------------------------------------------
# TestDoclingPostprocessing
# ---------------------------------------------------------------------------


class TestDoclingPostprocessing:
    """Test Docling JSON post-processing step (step 2.5) in the pipeline."""

    @pytest.mark.asyncio
    async def test_docling_parser_rebuilds_text_items(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """When parser=docling and Docling JSON exists, text items are rebuilt with headings."""
        test_config.parser = "docling"
        pipeline = _make_pipeline(test_config)

        # parse_document returns generic text + an image item
        pipeline.rag.parse_document.return_value = (
            [
                {"type": "text", "text": "plain paragraph"},
                {"type": "image", "img_path": "/img/fig1.png", "page_idx": 0},
            ],
            "doc-docling-001",
        )

        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"fake")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Create Docling JSON at the expected path
        docling_dir = artifacts / "report" / "docling"
        docling_dir.mkdir(parents=True)
        docling_json = {
            "body": {
                "children": [
                    {"$ref": "texts/0"},
                    {"$ref": "texts/1"},
                ],
            },
            "texts": [
                {
                    "orig": "Introduction",
                    "label": "section_header",
                    "level": 2,
                    "prov": [{"page_no": 1}],
                },
                {
                    "orig": "Some body text here.",
                    "label": "paragraph",
                    "prov": [{"page_no": 1}],
                },
            ],
        }
        (docling_dir / "report.json").write_text(json.dumps(docling_json))

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "success"
        call_args = pipeline.rag.insert_content_list.call_args
        content_list = call_args.kwargs.get("content_list") or call_args[1]["content_list"]

        # Rebuilt texts come first: heading with ## prefix, then paragraph
        text_items = [i for i in content_list if i["type"] == "text"]
        assert text_items[0]["text"] == "## Introduction"
        assert text_items[0]["page_idx"] == 1
        assert text_items[1]["text"] == "Some body text here."

        # Non-text items (image) are preserved
        non_text = [i for i in content_list if i["type"] != "text"]
        assert len(non_text) == 1
        assert non_text[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_mineru_parser_no_postprocessing(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """When parser=mineru, original content_list passes through unchanged."""
        test_config.parser = "mineru"
        pipeline = _make_pipeline(test_config)

        original_items = [
            {"type": "text", "text": "original text"},
            {"type": "image", "img_path": "/img/fig.png"},
        ]
        pipeline.rag.parse_document.return_value = (original_items, "doc-mineru-001")

        file_path = tmp_path / "report.pdf"
        file_path.write_bytes(b"fake")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "success"
        call_args = pipeline.rag.insert_content_list.call_args
        content_list = call_args.kwargs.get("content_list") or call_args[1]["content_list"]
        # Content should be the original items (after policy, which keeps them)
        assert content_list == original_items

    @pytest.mark.asyncio
    async def test_docling_json_missing_falls_back(
        self, test_config: DlightragConfig, tmp_path: Path
    ) -> None:
        """When parser=docling but JSON doesn't exist, original content_list is used."""
        test_config.parser = "docling"
        pipeline = _make_pipeline(test_config)

        original_items = [{"type": "text", "text": "fallback text"}]
        pipeline.rag.parse_document.return_value = (original_items, "doc-docling-002")

        file_path = tmp_path / "missing.pdf"
        file_path.write_bytes(b"fake")
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # Deliberately do NOT create the docling JSON

        result = await pipeline._ingest_single_file_with_policy(file_path, artifacts)

        assert result.status == "success"
        call_args = pipeline.rag.insert_content_list.call_args
        content_list = call_args.kwargs.get("content_list") or call_args[1]["content_list"]
        assert content_list == original_items
