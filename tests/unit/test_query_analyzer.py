# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for QueryAnalyzer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.query_analyzer import QueryAnalyzer, fast_detect


class TestFastDetect:
    def test_pdf_filename(self) -> None:
        result = fast_detect("What's in report.pdf?")
        assert result is not None
        assert result.filename == "report.pdf"

    def test_png_filename(self) -> None:
        result = fast_detect("Describe img_9551.png")
        assert result is not None
        assert result.filename == "img_9551.png"

    def test_xlsx_filename(self) -> None:
        result = fast_detect("data from budget-2024.xlsx please")
        assert result is not None
        assert result.filename == "budget-2024.xlsx"

    def test_no_filename(self) -> None:
        result = fast_detect("What are the key revenue insights?")
        assert result is None

    def test_case_insensitive(self) -> None:
        result = fast_detect("open MyDoc.PDF")
        assert result is not None
        assert result.filename == "MyDoc.PDF"

    def test_filename_with_dots(self) -> None:
        result = fast_detect("check v2.1.report.pdf")
        assert result is not None
        assert result.filename == "v2.1.report.pdf"


class TestQueryAnalyzerExplicitFilters:
    @pytest.mark.asyncio
    async def test_explicit_filters_bypass_detection(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter(doc_author="Zhang San")
        plan = await analyzer.analyze("revenue insights", explicit_filters=filters)
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.doc_author == "Zhang San"
        assert "metadata" in plan.paths
        assert "kg" in plan.paths

    @pytest.mark.asyncio
    async def test_explicit_filters_with_empty_query(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter(doc_author="Zhang San")
        plan = await analyzer.analyze("", explicit_filters=filters)
        assert plan.paths == ["metadata"]
        assert "kg" not in plan.paths

    @pytest.mark.asyncio
    async def test_empty_explicit_filters_ignored(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter()  # all None
        plan = await analyzer.analyze("revenue insights", explicit_filters=filters)
        # Should fall through to next layer
        assert plan.paths == ["kg"]


class TestQueryAnalyzerFastDetect:
    @pytest.mark.asyncio
    async def test_filename_detected(self) -> None:
        analyzer = QueryAnalyzer()
        plan = await analyzer.analyze("What's in img_9551.png?")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "img_9551.png"
        assert "metadata" in plan.paths

    @pytest.mark.asyncio
    async def test_filename_with_semantic_query(self) -> None:
        analyzer = QueryAnalyzer()
        plan = await analyzer.analyze("key insights from report.pdf")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "report.pdf"
        assert "kg" in plan.paths
        assert "metadata" in plan.paths
        # Semantic query should have filename removed
        assert "report.pdf" not in plan.semantic_query

    @pytest.mark.asyncio
    async def test_no_filename_falls_through(self) -> None:
        analyzer = QueryAnalyzer()
        plan = await analyzer.analyze("What are the revenue trends?")
        assert plan.metadata_filters is None
        assert plan.paths == ["kg"]


class TestQueryAnalyzerLLM:
    @pytest.mark.asyncio
    async def test_llm_extraction(self) -> None:
        llm_response = json.dumps(
            {
                "semantic_query": "revenue insights",
                "filters": {"doc_author": "Zhang San", "date_from": "2024-01-01"},
                "paths": ["metadata", "kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("revenue insights from Zhang San's docs in 2024")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.doc_author == "Zhang San"
        assert plan.semantic_query == "revenue insights"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self) -> None:
        llm_func = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("some complex query")
        # Should fall back to pure KG
        assert plan.paths == ["kg"]
        assert plan.metadata_filters is None

    @pytest.mark.asyncio
    async def test_llm_returns_empty_filters(self) -> None:
        llm_response = json.dumps(
            {
                "semantic_query": "what is quantum computing",
                "filters": {},
                "paths": ["kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("what is quantum computing")
        assert plan.metadata_filters is None
        assert plan.paths == ["kg"]

    @pytest.mark.asyncio
    async def test_llm_called_even_when_fast_detect_matches(self) -> None:
        """LLM runs alongside regex to extract fields regex can't detect."""
        llm_response = json.dumps(
            {
                "semantic_query": "what's in the report",
                "filters": {"filename": "report.pdf", "doc_author": "Zhang San"},
                "paths": ["metadata", "kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("What's in Zhang San's report.pdf?")
        llm_func.assert_called_once()
        # Regex captures filename, LLM fills in author
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "report.pdf"
        assert plan.metadata_filters.doc_author == "Zhang San"

    @pytest.mark.asyncio
    async def test_regex_only_when_no_llm_func(self) -> None:
        """Without llm_func, regex-only still works (no regression)."""
        analyzer = QueryAnalyzer()  # no LLM
        plan = await analyzer.analyze("What's in report.pdf?")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "report.pdf"
        assert plan.metadata_filters.doc_author is None


class TestQueryAnalyzerMerge:
    """Tests for collaborative regex + LLM filter merging."""

    @pytest.mark.asyncio
    async def test_regex_filename_priority_over_llm(self) -> None:
        """Regex-extracted filename takes priority over LLM's."""
        llm_response = json.dumps(
            {
                "semantic_query": "key insights",
                "filters": {"filename": "wrong-name.pdf", "doc_author": "Li Si"},
                "paths": ["metadata", "kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("key insights from report.pdf by Li Si")
        assert plan.metadata_filters.filename == "report.pdf"  # regex wins
        assert plan.metadata_filters.doc_author == "Li Si"  # LLM fills in

    @pytest.mark.asyncio
    async def test_merge_dates_from_llm(self) -> None:
        """LLM-extracted date filters merge with regex filename."""
        llm_response = json.dumps(
            {
                "semantic_query": "summary",
                "filters": {
                    "filename": "annual-report.pdf",
                    "date_from": "2024-01-01",
                    "date_to": "2024-12-31",
                },
                "paths": ["metadata", "kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("summarize annual-report.pdf from 2024")
        assert plan.metadata_filters.filename == "annual-report.pdf"
        assert plan.metadata_filters.date_from is not None
        assert plan.metadata_filters.date_to is not None

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_regex(self) -> None:
        """If LLM fails, regex results still work."""
        llm_func = AsyncMock(side_effect=RuntimeError("timeout"))
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("insights from report.pdf")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "report.pdf"
        assert plan.metadata_filters.doc_author is None  # no LLM fallback

    @pytest.mark.asyncio
    async def test_llm_empty_filters_uses_regex_only(self) -> None:
        """If LLM returns empty filters, regex results are used alone."""
        llm_response = json.dumps(
            {
                "semantic_query": "what's in the report",
                "filters": {},
                "paths": ["kg"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("What's in report.pdf?")
        assert plan.metadata_filters.filename == "report.pdf"


class TestQueryAnalyzerFallback:
    @pytest.mark.asyncio
    async def test_no_llm_pure_semantic(self) -> None:
        analyzer = QueryAnalyzer()  # no LLM
        plan = await analyzer.analyze("explain quantum entanglement")
        assert plan.paths == ["kg"]
        assert plan.semantic_query == "explain quantum entanglement"
        assert plan.original_query == "explain quantum entanglement"
