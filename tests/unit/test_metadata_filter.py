# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MetadataFilter and RetrievalPlan data models."""

from __future__ import annotations

from datetime import datetime

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan


class TestMetadataFilter:
    def test_empty_filter(self) -> None:
        f = MetadataFilter()
        assert f.is_empty()

    def test_filename_makes_non_empty(self) -> None:
        f = MetadataFilter(filename="test.pdf")
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
            filename=None, filename_pattern=None, file_extension=None,
            doc_title=None, doc_author=None, date_from=None, date_to=None,
            rag_mode=None, custom=None,
        )
        assert f.is_empty()


class TestRetrievalPlan:
    def test_default_paths(self) -> None:
        plan = RetrievalPlan(semantic_query="test", metadata_filters=None)
        assert plan.paths == ["kg"]

    def test_with_filters(self) -> None:
        f = MetadataFilter(filename="test.pdf")
        plan = RetrievalPlan(
            semantic_query="test", metadata_filters=f,
            paths=["metadata", "kg"],
        )
        assert "metadata" in plan.paths
        assert "kg" in plan.paths

    def test_original_query_preserved(self) -> None:
        plan = RetrievalPlan(
            semantic_query="cleaned", metadata_filters=None,
            original_query="original query with file.pdf",
        )
        assert plan.original_query == "original query with file.pdf"


class TestRrfKConfig:
    def test_rrf_k_default(self) -> None:
        from dlightrag.config import DlightragConfig

        config = DlightragConfig(openai_api_key="test-key")  # type: ignore[call-arg]
        assert config.rrf_k == 60

    def test_rrf_k_custom(self) -> None:
        from dlightrag.config import DlightragConfig

        config = DlightragConfig(openai_api_key="test-key", rrf_k=100)  # type: ignore[call-arg]
        assert config.rrf_k == 100
