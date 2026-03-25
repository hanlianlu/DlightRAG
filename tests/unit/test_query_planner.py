# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for QueryPlanner -- unified query understanding."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from dlightrag.core.query_planner import (
    QueryPlan,
    QueryPlanner,
    _build_custom_keys_hint,
    _build_schema_section,
)
from dlightrag.core.retrieval.models import MetadataFilter

# ---------------------------------------------------------------------------
# _build_schema_section / _build_custom_keys_hint
# ---------------------------------------------------------------------------


class TestBuildSchemaSection:
    def test_none_schema(self):
        assert _build_schema_section(None) == ""

    def test_empty_schema(self):
        assert _build_schema_section({}) == ""

    def test_no_columns(self):
        assert _build_schema_section({"columns": []}) == ""

    def test_with_columns(self):
        schema = {
            "columns": [
                {"name": "filename", "type": "character varying"},
                {"name": "ingested_at", "type": "timestamp with time zone"},
            ]
        }
        result = _build_schema_section(schema)
        assert "filename (text)" in result
        assert "ingested_at (timestamp)" in result


class TestBuildCustomKeysHint:
    def test_none_schema(self):
        assert _build_custom_keys_hint(None) == ""

    def test_no_custom_keys(self):
        assert _build_custom_keys_hint({"custom_keys": []}) == ""

    def test_with_keys(self):
        result = _build_custom_keys_hint({"custom_keys": ["project", "team"]})
        assert "project" in result
        assert "team" in result


# ---------------------------------------------------------------------------
# QueryPlan dataclass
# ---------------------------------------------------------------------------


class TestQueryPlan:
    def test_defaults(self):
        plan = QueryPlan(original_query="test", standalone_query="test")
        assert plan.metadata_filter is None

    def test_with_filter(self):
        mf = MetadataFilter(doc_author="Author")
        plan = QueryPlan(original_query="q", standalone_query="q", metadata_filter=mf)
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "Author"


# ---------------------------------------------------------------------------
# QueryPlanner.plan()
# ---------------------------------------------------------------------------


class TestPlanNoLLM:
    async def test_no_llm_returns_fallback(self):
        planner = QueryPlanner(llm_func=None)
        plan = await planner.plan("hello")
        assert plan.original_query == "hello"
        assert plan.standalone_query == "hello"
        assert plan.metadata_filter is None


class TestPlanWithLLM:
    async def test_basic_query(self):
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "what is X", "filters": {}}))
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("what is X")
        assert plan.standalone_query == "what is X"
        assert plan.metadata_filter is None

    async def test_rewrite_with_history(self):
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "What is the GDP of France in 2023?",
                    "filters": {},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        history = [
            {"role": "user", "content": "Tell me about France"},
            {"role": "assistant", "content": "France is a country in Europe."},
        ]
        plan = await planner.plan("what about GDP in 2023?", conversation_history=history)
        assert plan.standalone_query == "What is the GDP of France in 2023?"
        assert plan.original_query == "what about GDP in 2023?"

    async def test_filter_extraction(self):
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "find report.pdf",
                    "filters": {"filename": "report.pdf", "file_extension": "pdf"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("find report.pdf")
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.filename == "report.pdf"
        assert plan.metadata_filter.file_extension == "pdf"

    async def test_date_filter_extraction(self):
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "2024 reports",
                    "filters": {"date_from": "2024-01-01", "date_to": "2024-12-31"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("2024 reports")
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.date_from is not None
        assert plan.metadata_filter.date_to is not None

    async def test_invalid_date_ignored(self):
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "query",
                    "filters": {"date_from": "not-a-date", "doc_author": "Auth"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("query")
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.date_from is None
        assert plan.metadata_filter.doc_author == "Auth"


class TestPlanFallback:
    async def test_llm_exception_returns_fallback(self):
        llm = AsyncMock(side_effect=RuntimeError("LLM error"))
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("query")
        assert plan.standalone_query == "query"
        assert plan.metadata_filter is None

    async def test_empty_response_returns_fallback(self):
        llm = AsyncMock(return_value="")
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("query")
        assert plan.standalone_query == "query"

    async def test_invalid_json_returns_fallback(self):
        llm = AsyncMock(return_value="this is not json")
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("query")
        assert plan.standalone_query == "query"

    async def test_markdown_fenced_json_parsed(self):
        llm = AsyncMock(return_value='```json\n{"standalone_query": "parsed", "filters": {}}\n```')
        planner = QueryPlanner(llm_func=llm)
        plan = await planner.plan("query")
        assert plan.standalone_query == "parsed"


# ---------------------------------------------------------------------------
# Filter merging
# ---------------------------------------------------------------------------


class TestFilterMerge:
    async def test_explicit_overrides_llm(self):
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "q",
                    "filters": {"doc_author": "LLM", "file_extension": "pdf"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        explicit = MetadataFilter(doc_author="Explicit")
        plan = await planner.plan("q", explicit_filter=explicit)
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "Explicit"
        assert plan.metadata_filter.file_extension == "pdf"

    async def test_empty_explicit_ignored(self):
        """Empty explicit filter does not trigger merge."""
        llm = AsyncMock(
            return_value=json.dumps(
                {
                    "standalone_query": "q",
                    "filters": {"doc_author": "LLM"},
                }
            )
        )
        planner = QueryPlanner(llm_func=llm)
        explicit = MetadataFilter()  # all None
        plan = await planner.plan("q", explicit_filter=explicit)
        assert plan.metadata_filter is not None
        assert plan.metadata_filter.doc_author == "LLM"

    def test_merge_filters_static(self):
        explicit = MetadataFilter(doc_author="Explicit", filename="f.pdf")
        llm = MetadataFilter(doc_author="LLM", file_extension="pdf")
        merged = QueryPlanner._merge_filters(explicit, llm)
        assert merged.doc_author == "Explicit"
        assert merged.filename == "f.pdf"
        assert merged.file_extension == "pdf"

    def test_merge_filters_llm_none(self):
        explicit = MetadataFilter(doc_author="Explicit")
        merged = QueryPlanner._merge_filters(explicit, None)
        assert merged is explicit


# ---------------------------------------------------------------------------
# Schema caching
# ---------------------------------------------------------------------------


class TestSchemaCache:
    async def test_caches_schema(self):
        schema = {"columns": [{"name": "x", "type": "text"}]}
        provider = AsyncMock(return_value=schema)
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "q", "filters": {}}))
        planner = QueryPlanner(llm_func=llm, schema_provider=provider, schema_ttl=300.0)

        await planner.plan("q1")
        await planner.plan("q2")
        # Provider only called once due to TTL
        assert provider.await_count == 1

    async def test_ttl_expired_refreshes(self):
        schema = {"columns": [{"name": "x", "type": "text"}]}
        provider = AsyncMock(return_value=schema)
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "q", "filters": {}}))
        planner = QueryPlanner(llm_func=llm, schema_provider=provider, schema_ttl=0.0)

        await planner.plan("q1")
        await planner.plan("q2")
        # TTL=0 means always refresh
        assert provider.await_count == 2

    async def test_provider_failure_uses_stale(self):
        schema = {"columns": [{"name": "x", "type": "text"}]}
        provider = AsyncMock(return_value=schema)
        llm = AsyncMock(return_value=json.dumps({"standalone_query": "q", "filters": {}}))
        planner = QueryPlanner(llm_func=llm, schema_provider=provider, schema_ttl=0.0)

        # First call succeeds
        await planner.plan("q1")
        assert provider.await_count == 1

        # Second call: provider fails but stale schema is used
        provider.side_effect = RuntimeError("DB down")
        plan = await planner.plan("q2")
        assert plan.standalone_query == "q"


# ---------------------------------------------------------------------------
# History truncation
# ---------------------------------------------------------------------------


class TestHistoryTruncation:
    def test_empty_history(self):
        result = QueryPlanner._truncate_history(None, max_turns=10, max_tokens=10000)
        assert result == []

    def test_truncates_by_turns(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(100)]
        result = QueryPlanner._truncate_history(history, max_turns=5, max_tokens=100000)
        # max_turns=5 => max_messages=10
        assert len(result) <= 10
