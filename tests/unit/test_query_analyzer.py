# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for QueryAnalyzer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.core.retrieval.query_analyzer import QueryAnalyzer


class TestQueryAnalyzerExplicitFilters:
    @pytest.mark.asyncio
    async def test_explicit_filters_bypass_detection(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter(doc_author="Zhang San")
        plan = await analyzer.analyze("revenue insights", explicit_filters=filters)
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.doc_author == "Zhang San"
        assert "metafilters" in plan.paths
        assert "kgvector" in plan.paths

    @pytest.mark.asyncio
    async def test_explicit_filters_with_empty_query(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter(doc_author="Zhang San")
        plan = await analyzer.analyze("", explicit_filters=filters)
        assert plan.paths == ["metafilters"]
        assert "kgvector" not in plan.paths

    @pytest.mark.asyncio
    async def test_empty_explicit_filters_ignored(self) -> None:
        analyzer = QueryAnalyzer()
        filters = MetadataFilter()  # all None
        plan = await analyzer.analyze("revenue insights", explicit_filters=filters)
        # Should fall through to next layer
        assert plan.paths == ["kgvector"]


class TestQueryAnalyzerLLM:
    @pytest.mark.asyncio
    async def test_llm_extraction(self) -> None:
        llm_response = json.dumps(
            {
                "semantic_query": "revenue insights",
                "filters": {"doc_author": "Zhang San", "date_from": "2024-01-01"},
                "paths": ["metafilters", "kgvector"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("revenue insights from Zhang San's docs in 2024")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.doc_author == "Zhang San"
        assert plan.semantic_query == "revenue insights"

    @pytest.mark.asyncio
    async def test_llm_extracts_filename(self) -> None:
        """LLM handles filenames with spaces, wrong case, etc."""
        llm_response = json.dumps(
            {
                "semantic_query": "what is this about",
                "filters": {"filename": "IMG_9551.PNG"},
                "paths": ["metafilters", "kgvector"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("what is IMG 9551.png about")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename == "IMG_9551.PNG"

    @pytest.mark.asyncio
    async def test_llm_uses_filename_pattern_for_ambiguous(self) -> None:
        """LLM can use filename_pattern with ILIKE wildcards."""
        llm_response = json.dumps(
            {
                "semantic_query": "what is this about",
                "filters": {"filename_pattern": "%IMG%9551%"},
                "paths": ["metafilters", "kgvector"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("what is IMG 9551 about")
        assert plan.metadata_filters is not None
        assert plan.metadata_filters.filename_pattern == "%IMG%9551%"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self) -> None:
        llm_func = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("some complex query")
        # Should fall back to pure KG
        assert plan.paths == ["kgvector"]
        assert plan.metadata_filters is None

    @pytest.mark.asyncio
    async def test_llm_returns_empty_filters(self) -> None:
        llm_response = json.dumps(
            {
                "semantic_query": "what is quantum computing",
                "filters": {},
                "paths": ["kgvector"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("what is quantum computing")
        assert plan.metadata_filters is None
        assert plan.paths == ["kgvector"]

    @pytest.mark.asyncio
    async def test_llm_extracts_dates(self) -> None:
        llm_response = json.dumps(
            {
                "semantic_query": "summary",
                "filters": {
                    "filename": "annual-report.pdf",
                    "date_from": "2024-01-01",
                    "date_to": "2024-12-31",
                },
                "paths": ["metafilters", "kgvector"],
            }
        )
        llm_func = AsyncMock(return_value=llm_response)
        analyzer = QueryAnalyzer(llm_func=llm_func)
        plan = await analyzer.analyze("summarize annual-report.pdf from 2024")
        assert plan.metadata_filters.filename == "annual-report.pdf"
        assert plan.metadata_filters.date_from is not None
        assert plan.metadata_filters.date_to is not None


class TestQueryAnalyzerFallback:
    @pytest.mark.asyncio
    async def test_no_llm_pure_semantic(self) -> None:
        analyzer = QueryAnalyzer()  # no LLM
        plan = await analyzer.analyze("explain quantum entanglement")
        assert plan.paths == ["kgvector"]
        assert plan.semantic_query == "explain quantum entanglement"
        assert plan.original_query == "explain quantum entanglement"

    @pytest.mark.asyncio
    async def test_no_llm_with_filename_in_query(self) -> None:
        """Without LLM, even filename queries fall to pure KG."""
        analyzer = QueryAnalyzer()  # no LLM
        plan = await analyzer.analyze("What's in report.pdf?")
        assert plan.paths == ["kgvector"]
        assert plan.metadata_filters is None
