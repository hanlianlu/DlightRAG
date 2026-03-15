# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PGMetadataIndex (mocked asyncpg)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.core.retrieval.metadata_index import _SYSTEM_FIELDS, PGMetadataIndex
from dlightrag.core.retrieval.models import MetadataFilter


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool with context manager support."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


@pytest.fixture
def index(mock_pool):
    """Create a PGMetadataIndex with mocked pool."""
    pool, _ = mock_pool
    idx = PGMetadataIndex(workspace="test_ws")
    idx._pool = pool
    return idx


class TestPGMetadataIndexUpsert:
    @pytest.mark.asyncio
    async def test_upsert_system_fields(self, index, mock_pool) -> None:
        _, conn = mock_pool
        await index.upsert(
            "doc-1",
            {
                "filename": "report.pdf",
                "filename_stem": "report",
                "file_extension": ".pdf",
                "doc_title": "Annual Report",
            },
        )
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert args[1] == "test_ws"  # workspace
        assert args[2] == "doc-1"  # doc_id
        assert args[3] == "report.pdf"  # filename

    @pytest.mark.asyncio
    async def test_upsert_custom_fields_go_to_jsonb(self, index, mock_pool) -> None:
        _, conn = mock_pool
        await index.upsert(
            "doc-1",
            {
                "filename": "test.pdf",
                "department": "finance",
                "tags": ["confidential"],
            },
        )
        args = conn.execute.call_args[0]
        # Last arg is custom_metadata JSON
        import json

        custom = json.loads(args[-1])
        assert custom["department"] == "finance"
        assert custom["tags"] == ["confidential"]


class TestPGMetadataIndexQuery:
    @pytest.mark.asyncio
    async def test_query_filename(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = [{"doc_id": "doc-1"}]
        result = await index.query(MetadataFilter(filename="test.pdf"))
        assert result == ["doc-1"]
        sql = conn.fetch.call_args[0][0]
        assert "LOWER(filename) = LOWER($2)" in sql

    @pytest.mark.asyncio
    async def test_query_multiple_filters(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        result = await index.query(
            MetadataFilter(
                doc_author="Zhang San",
                date_from=datetime(2024, 1, 1),
            )
        )
        assert len(result) == 2
        sql = conn.fetch.call_args[0][0]
        assert "LOWER(doc_author) = LOWER($2)" in sql
        assert "creation_date >= $3" in sql

    @pytest.mark.asyncio
    async def test_query_custom_metadata(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = []
        await index.query(MetadataFilter(custom={"department": "finance"}))
        sql = conn.fetch.call_args[0][0]
        assert "custom_metadata @>" in sql

    @pytest.mark.asyncio
    async def test_query_filename_pattern(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = [{"doc_id": "doc-1"}]
        await index.query(MetadataFilter(filename_pattern="%report%"))
        sql = conn.fetch.call_args[0][0]
        assert "filename ILIKE" in sql

    @pytest.mark.asyncio
    async def test_query_file_extension(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = []
        await index.query(MetadataFilter(file_extension=".png"))
        sql = conn.fetch.call_args[0][0]
        assert "LOWER(file_extension) = LOWER($2)" in sql


class TestPGMetadataIndexLifecycle:
    @pytest.mark.asyncio
    async def test_get_existing(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetchrow.return_value = {"doc_id": "doc-1", "filename": "test.pdf"}
        result = await index.get("doc-1")
        assert result is not None
        assert result["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_get_missing(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        result = await index.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, index, mock_pool) -> None:
        _, conn = mock_pool
        await index.delete("doc-1")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE" in sql
        assert "doc_id=$2" in sql

    @pytest.mark.asyncio
    async def test_clear(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 5"
        await index.clear()
        sql = conn.execute.call_args[0][0]
        assert "DELETE" in sql
        assert "workspace=$1" in sql

    @pytest.mark.asyncio
    async def test_find_by_filename(self, index, mock_pool) -> None:
        _, conn = mock_pool
        conn.fetch.return_value = [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]
        result = await index.find_by_filename("report.pdf")
        assert result == ["doc-1", "doc-2"]


class TestSystemFieldsSeparation:
    def test_system_fields_complete(self) -> None:
        expected = {
            "filename",
            "filename_stem",
            "file_path",
            "file_extension",
            "doc_title",
            "doc_author",
            "creation_date",
            "original_format",
            "page_count",
            "rag_mode",
        }
        assert _SYSTEM_FIELDS == expected


class TestExtractSystemMetadata:
    def test_basic_extraction(self) -> None:
        from dlightrag.core.retrieval.metadata_index import extract_system_metadata

        meta = extract_system_metadata("/data/reports/annual-report.pdf", rag_mode="caption")
        assert meta["filename"] == "annual-report.pdf"
        assert meta["filename_stem"] == "annual-report"
        assert meta["file_extension"] == "pdf"
        assert meta["rag_mode"] == "caption"
        assert meta.get("page_count") is None

    def test_with_page_count(self) -> None:
        from dlightrag.core.retrieval.metadata_index import extract_system_metadata

        meta = extract_system_metadata("/tmp/doc.pdf", rag_mode="unified", page_count=5)
        assert meta["page_count"] == 5
        assert meta["rag_mode"] == "unified"

    def test_no_extension(self) -> None:
        from dlightrag.core.retrieval.metadata_index import extract_system_metadata

        meta = extract_system_metadata("/tmp/README")
        assert meta["file_extension"] is None
        assert meta["filename"] == "README"

    def test_pathlib_input(self) -> None:
        from pathlib import Path

        from dlightrag.core.retrieval.metadata_index import extract_system_metadata

        meta = extract_system_metadata(Path("/docs/slides.pptx"), rag_mode="caption")
        assert meta["filename"] == "slides.pptx"
        assert meta["file_extension"] == "pptx"
