# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for QueryPlanner -- replaces old _rewrite_query tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from dlightrag.core.query_planner import QueryPlanner
from dlightrag.core.retrieval.models import MetadataFilter


class TestQueryPlannerNoHistory:
    """Test that no-history queries pass through with no rewrite."""

    async def test_no_history_returns_original(self):
        """First turn (no history) returns query unchanged."""
        llm = AsyncMock(
            return_value=json.dumps({"standalone_query": "What is revenue?", "filters": {}})
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("What is revenue?")
        assert plan.standalone_query == "What is revenue?"
        assert plan.original_query == "What is revenue?"

    async def test_no_llm_returns_fallback(self):
        """No LLM configured returns query unchanged."""
        planner = QueryPlanner(llm_func=None)
        plan = await planner.plan("What is revenue?")
        assert plan.standalone_query == "What is revenue?"
        assert plan.metadata_filter is None


class TestQueryPlannerWithHistory:
    """Test conversation rewriting via QueryPlanner."""

    async def test_with_history_rewrites(self):
        """With conversation history, rewrites into standalone query."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "What were the Q3 revenue numbers?",
                    "filters": {},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        history = [
            {"role": "user", "content": "Tell me about revenue"},
            {"role": "assistant", "content": "Revenue grew 20% in Q3."},
        ]
        plan = await planner.plan("more details", conversation_history=history)
        assert plan.standalone_query == "What were the Q3 revenue numbers?"
        assert plan.original_query == "more details"
        llm.assert_awaited_once()

    async def test_llm_failure_returns_fallback(self):
        """LLM failure returns original query as fallback."""
        llm = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = QueryPlanner(llm_func=llm)
        history = [{"role": "user", "content": "hello"}]
        plan = await planner.plan("follow up", conversation_history=history)
        assert plan.standalone_query == "follow up"
        assert plan.metadata_filter is None


class TestQueryPlannerFilters:
    """Test metadata filter extraction."""

    async def test_extracts_filters(self):
        """Extracts metadata filters from query."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "张三写的2024年财报分析",
                    "filters": {
                        "doc_author": "张三",
                        "date_from": "2024-01-01",
                        "date_to": "2024-12-31",
                    },
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("张三写的2024年财报分析")
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "张三"
        assert plan.metadata_filter.date_from is not None
        assert plan.metadata_filter.date_to is not None

    async def test_extracts_filename(self):
        """Extracts filename from query."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "summarize annual-report.pdf",
                    "filters": {"filename": "annual-report.pdf"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("summarize annual-report.pdf")
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.filename == "annual-report.pdf"

    async def test_empty_filters_returns_none(self):
        """Empty filters result in None metadata_filter."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "what are the main revenue trends",
                    "filters": {},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("what are the main revenue trends")
        assert plan.metadata_filter is None

    async def test_invalid_json_returns_fallback(self):
        """Invalid JSON response returns fallback plan."""
        llm = AsyncMock(return_value="not json at all")
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("some query")
        assert plan.standalone_query == "some query"
        assert plan.metadata_filter is None


class TestQueryPlannerMerge:
    """Test explicit + LLM filter merging."""

    async def test_explicit_filter_wins(self):
        """Explicit filters override LLM-extracted ones."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "revenue insights",
                    "filters": {"doc_author": "LLM Author", "file_extension": "pdf"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        explicit = MetadataFilter(doc_author="Explicit Author")
        plan = await planner.plan("revenue insights", explicit_filter=explicit)
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "Explicit Author"
        # LLM-only field preserved
        assert plan.metadata_filter.file_extension == "pdf"

    async def test_explicit_filter_only(self):
        """Explicit filter with no LLM extraction still works."""
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "query", "filters": {}}))
        planner = QueryPlanner(llm_func=llm)
        explicit = MetadataFilter(doc_author="Author")
        plan = await planner.plan("query", explicit_filter=explicit)
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "Author"


class TestQueryPlannerSchema:
    """Test schema TTL caching."""

    async def test_schema_provider_called(self):
        """Schema provider is called and result cached."""
        schema = {
            "columns": [{"name": "filename", "type": "character varying"}],
            "custom_keys": ["project"],
        }
        provider = AsyncMock(return_value=schema)
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "test", "filters": {}}))
        planner = QueryPlanner(llm_func=llm, schema_provider=provider, schema_ttl=300.0)
        await planner.plan("test")
        provider.assert_awaited_once()

        # Second call uses cache
        await planner.plan("test2")
        assert provider.await_count == 1

    async def test_schema_provider_failure_graceful(self):
        """Schema provider failure doesn't break planning."""
        provider = AsyncMock(side_effect=RuntimeError("DB down"))
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "test", "filters": {}}))
        planner = QueryPlanner(llm_func=llm, schema_provider=provider)
        plan = await planner.plan("test")
        assert plan.standalone_query == "test"
