# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MetadataFilter data model."""

from __future__ import annotations

from datetime import datetime

from dlightrag.core.retrieval.models import MetadataFilter


class TestMetadataFilter:
    def test_empty_filter(self) -> None:
        f = MetadataFilter()
        assert f.is_empty()

    def test_filename_makes_non_empty(self) -> None:
        f = MetadataFilter(filename="test.pdf")
        assert not f.is_empty()

    def test_filename_stem_makes_non_empty(self) -> None:
        f = MetadataFilter(filename_stem="test")
        assert not f.is_empty()

    def test_custom_metadata_makes_non_empty(self) -> None:
        f = MetadataFilter(custom={"department": "finance"})
        assert not f.is_empty()

    def test_file_extension_makes_non_empty(self) -> None:
        f = MetadataFilter(file_extension=".png")
        assert not f.is_empty()

    def test_date_range_makes_non_empty(self) -> None:
        f = MetadataFilter(date_from=datetime(2024, 1, 1))
        assert not f.is_empty()

    def test_all_none_is_empty(self) -> None:
        f = MetadataFilter(
            filename=None,
            filename_stem=None,
            filename_pattern=None,
            file_extension=None,
            doc_title=None,
            doc_author=None,
            date_from=None,
            date_to=None,
            rag_mode=None,
            custom=None,
        )
        assert f.is_empty()
