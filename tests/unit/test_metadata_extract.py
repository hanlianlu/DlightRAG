# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for core.ingestion.metadata_extract."""

from __future__ import annotations

from pathlib import Path

from dlightrag.core.ingestion.metadata_extract import extract_system_metadata


class TestExtractSystemMetadata:
    """Unit tests for extract_system_metadata."""

    def test_basic_extraction(self) -> None:
        result = extract_system_metadata(Path("/data/report.pdf"))
        assert result["filename"] == "report.pdf"
        assert result["filename_stem"] == "report"
        assert result["file_extension"] == "pdf"
        assert result["original_format"] == "pdf"
        assert result["rag_mode"] == "caption"
        assert "page_count" not in result

    def test_rag_mode_override(self) -> None:
        result = extract_system_metadata(Path("/data/report.pdf"), rag_mode="unified")
        assert result["rag_mode"] == "unified"

    def test_page_count_included(self) -> None:
        result = extract_system_metadata(Path("/data/report.pdf"), page_count=10)
        assert result["page_count"] == 10

    def test_no_extension(self) -> None:
        result = extract_system_metadata(Path("/data/README"))
        assert result["file_extension"] is None
        assert result["original_format"] is None

    def test_string_path_accepted(self) -> None:
        result = extract_system_metadata("/data/report.pdf")
        assert result["filename"] == "report.pdf"

    def test_file_path_is_string(self) -> None:
        result = extract_system_metadata(Path("/data/report.pdf"))
        assert isinstance(result["file_path"], str)
