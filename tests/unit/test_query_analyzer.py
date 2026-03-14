# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for QueryAnalyzer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan
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
        llm_response = json.dumps({
            "semantic_query": "revenue insights",
            "filters": {"doc_author": "Zhang San", "date_from": "2024-01-01"},
            "paths": ["metadata", "kg"],
        })
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
        llm_response = json.dumps({
            "semantic_query": "what is quantum computing",
            "filters": {},
            "paths": ["kg"],
        })
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("what is quantum computing")
        assert plan.metadata_filters is None
        assert plan.paths == ["kg"]

    @pytest.mark.asyncio
    async def test_llm_not_called_when_fast_detect_matches(self) -> None:
        llm_func = AsyncMock()
        analyzer = QueryAnalyzer(llm_func=llm_func)
        await analyzer.analyze("What's in report.pdf?")
        llm_func.assert_not_called()


class TestQueryAnalyzerFallback:
    @pytest.mark.asyncio
    async def test_no_llm_pure_semantic(self) -> None:
        analyzer = QueryAnalyzer()  # no LLM
        plan = await analyzer.analyze("explain quantum entanglement")
        assert plan.paths == ["kg"]
        assert plan.semantic_query == "explain quantum entanglement"
        assert plan.original_query == "explain quantum entanglement"
