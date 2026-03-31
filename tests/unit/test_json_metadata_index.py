# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for JsonMetadataIndex — JSON file-based metadata fallback (TDD)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage.json_metadata_index import JsonMetadataIndex


@pytest.fixture()
async def index(tmp_path: Path) -> JsonMetadataIndex:
    idx = JsonMetadataIndex(tmp_path, workspace="test")
    await idx.initialize()
    return idx


# ---------------------------------------------------------------------------
# TestCRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "report.pdf", "file_extension": "pdf"})
        row = await index.get("doc1")
        assert row is not None
        assert row["filename"] == "report.pdf"
        assert row["file_extension"] == "pdf"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, index: JsonMetadataIndex) -> None:
        result = await index.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_merges(self, index: JsonMetadataIndex) -> None:
        """Second upsert with None fields should NOT overwrite existing (COALESCE)."""
        await index.upsert("doc1", {"filename": "report.pdf", "doc_title": "Old Title"})
        await index.upsert("doc1", {"doc_title": "New Title"})  # filename not set → keep old
        row = await index.get("doc1")
        assert row is not None
        assert row["filename"] == "report.pdf"  # preserved
        assert row["doc_title"] == "New Title"  # updated

    @pytest.mark.asyncio
    async def test_upsert_merges_custom_metadata(self, index: JsonMetadataIndex) -> None:
        """Custom metadata keys merge additively (like JSONB || operator)."""
        await index.upsert("doc1", {"project": "alpha"})
        await index.upsert("doc1", {"department": "eng"})
        row = await index.get("doc1")
        assert row is not None
        assert row["custom_metadata"]["project"] == "alpha"
        assert row["custom_metadata"]["department"] == "eng"

    @pytest.mark.asyncio
    async def test_delete(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "a.pdf"})
        await index.delete("doc1")
        assert await index.get("doc1") is None

    @pytest.mark.asyncio
    async def test_clear(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "a.pdf"})
        await index.upsert("doc2", {"filename": "b.pdf"})
        await index.clear()
        assert await index.get("doc1") is None
        assert await index.get("doc2") is None


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_by_filename_substring(self, index: JsonMetadataIndex) -> None:
        """Trigram fields use case-insensitive substring matching."""
        await index.upsert("doc1", {"filename": "Annual_Report_2025.pdf"})
        await index.upsert("doc2", {"filename": "budget.xlsx"})

        result = await index.query(MetadataFilter(filename="report"))
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_by_exact_extension(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"file_extension": "pdf"})
        await index.upsert("doc2", {"file_extension": "docx"})

        result = await index.query(MetadataFilter(file_extension="pdf"))
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_case_insensitive(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "Report.PDF"})
        result = await index.query(MetadataFilter(filename="report"))
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_date_range(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"creation_date": "2025-06-15T00:00:00+00:00"})
        await index.upsert("doc2", {"creation_date": "2024-01-01T00:00:00+00:00"})
        result = await index.query(
            MetadataFilter(
                date_from=datetime(2025, 1, 1, tzinfo=UTC),
                date_to=datetime(2025, 12, 31, tzinfo=UTC),
            )
        )
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_custom_containment(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"project": "alpha", "filename": "a.pdf"})
        await index.upsert("doc2", {"project": "beta", "filename": "b.pdf"})

        result = await index.query(MetadataFilter(custom={"project": "alpha"}))
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_empty_filter_returns_all(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "a.pdf"})
        await index.upsert("doc2", {"filename": "b.pdf"})

        result = await index.query(MetadataFilter())
        assert set(result) == {"doc1", "doc2"}

    @pytest.mark.asyncio
    async def test_query_rag_mode(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"rag_mode": "caption"})
        await index.upsert("doc2", {"rag_mode": "native"})

        result = await index.query(MetadataFilter(rag_mode="caption"))
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_query_filename_pattern(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "Q1_report_2025.pdf"})
        await index.upsert("doc2", {"filename": "budget_2025.xlsx"})

        result = await index.query(MetadataFilter(filename_pattern="%report%"))
        assert result == ["doc1"]


# ---------------------------------------------------------------------------
# TestFindByFilename
# ---------------------------------------------------------------------------


class TestFindByFilename:
    @pytest.mark.asyncio
    async def test_find_case_insensitive(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "Report.PDF"})
        result = await index.find_by_filename("report.pdf")
        assert result == ["doc1"]

    @pytest.mark.asyncio
    async def test_find_no_match(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "report.pdf"})
        result = await index.find_by_filename("budget.xlsx")
        assert result == []


# ---------------------------------------------------------------------------
# TestFieldSchema
# ---------------------------------------------------------------------------


class TestFieldSchema:
    @pytest.mark.asyncio
    async def test_returns_columns_and_custom_keys(self, index: JsonMetadataIndex) -> None:
        await index.upsert("doc1", {"filename": "a.pdf", "project": "alpha", "dept": "eng"})
        schema: dict[str, Any] = await index.get_field_schema()

        # columns should come from METADATA_FIELDS, excluding internal cols
        col_names = {c["name"] for c in schema["columns"]}
        assert "filename" in col_names
        assert "file_extension" in col_names
        assert "doc_title" in col_names
        # internal/structural columns excluded
        assert "custom_metadata" not in col_names
        assert "ingested_at" not in col_names
        assert "workspace" not in col_names
        assert "doc_id" not in col_names

        # custom_keys scanned from actual data
        assert set(schema["custom_keys"]) == {"project", "dept"}

    @pytest.mark.asyncio
    async def test_empty_custom_keys_when_no_data(self, index: JsonMetadataIndex) -> None:
        schema = await index.get_field_schema()
        assert schema["custom_keys"] == []


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_data_survives_reload(self, tmp_path: Path) -> None:
        idx1 = JsonMetadataIndex(tmp_path, workspace="test")
        await idx1.initialize()
        await idx1.upsert("doc1", {"filename": "report.pdf"})

        # Create a new instance pointing to the same directory
        idx2 = JsonMetadataIndex(tmp_path, workspace="test")
        await idx2.initialize()
        row = await idx2.get("doc1")
        assert row is not None
        assert row["filename"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, tmp_path: Path) -> None:
        idx_a = JsonMetadataIndex(tmp_path, workspace="ws_a")
        await idx_a.initialize()
        await idx_a.upsert("doc1", {"filename": "a.pdf"})

        idx_b = JsonMetadataIndex(tmp_path, workspace="ws_b")
        await idx_b.initialize()

        # ws_b should NOT see ws_a's data
        assert await idx_b.get("doc1") is None
