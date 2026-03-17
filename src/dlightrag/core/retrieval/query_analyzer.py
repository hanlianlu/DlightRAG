# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query intent analysis and metadata filter extraction."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """\
Extract metadata filters and a cleaned semantic query from the user's search query.

{schema_section}Output format (JSON only, no markdown):
{{"semantic_query": "...", "filters": {{...}}}}

Filter fields (use null for unmentioned):
- filename: best-guess normalized filename (underscores, correct extension case)
- filename_pattern: SQL ILIKE pattern with % wildcards when reference is ambiguous
- file_extension: e.g. "pdf", "png" (lowercase, no dot)
- doc_title: document title reference
- doc_author: author name
- date_from / date_to: ISO 8601 dates for time ranges
- custom: {{"key": "value"}} for custom metadata{custom_keys_hint}

Guidelines:
- semantic_query: the core question with file/author/date references removed. Empty string if pure enumeration.
- Filename normalization: users often use spaces instead of underscores, wrong case, or abbreviations. Normalize to the most likely actual filename. When uncertain, use filename_pattern instead.
- Dates: convert relative references ("last month", "去年") to absolute ISO 8601 ranges.
- The query may be in any language; extract filters regardless of language.

Examples:

Query: "what is IMG 9551.png about"
{{"semantic_query": "what is this about", "filters": {{"filename_pattern": "%IMG_9551%", "file_extension": "png"}}}}

Query: "张三写的2024年财报分析"
{{"semantic_query": "财报分析", "filters": {{"doc_author": "张三", "date_from": "2024-01-01", "date_to": "2024-12-31"}}}}

Query: "summarize the key findings in annual-report.pdf"
{{"semantic_query": "summarize the key findings", "filters": {{"filename": "annual-report.pdf"}}}}

Query: "what are the main revenue trends"
{{"semantic_query": "what are the main revenue trends", "filters": {{}}}}

Query: "list all PDF files from Dr. Wang"
{{"semantic_query": "", "filters": {{"file_extension": "pdf", "doc_author": "Dr. Wang"}}}}

Now extract from this query:
Query: "{query}"
"""

# PG data_type → human-readable short form
_TYPE_ALIASES: dict[str, str] = {
    "character varying": "text",
    "timestamp with time zone": "timestamp",
    "jsonb": "json",
    "integer": "int",
}


def _build_schema_section(schema: dict[str, Any] | None) -> str:
    """Build schema context from live table structure."""
    if not schema or not schema.get("columns"):
        return ""

    col_lines = [
        f"  {col['name']} ({_TYPE_ALIASES.get(col['type'], col['type'])})"
        for col in schema["columns"]
    ]

    return "Available table columns:\n" + "\n".join(col_lines) + "\n\n"


def _build_custom_keys_hint(schema: dict[str, Any] | None) -> str:
    """Build hint for custom metadata keys."""
    if not schema:
        return ""
    custom_keys = schema.get("custom_keys")
    if not custom_keys:
        return ""
    return f" (known keys: {', '.join(custom_keys)})"


def _resolve_paths(filters: MetadataFilter | None, semantic_query: str | None) -> list[str]:
    """Determine which retrieval paths to activate."""
    paths: list[str] = []
    if filters and not filters.is_empty():
        paths.append("metadata")
    if semantic_query:
        paths.append("kg")
    return paths or ["kg"]


class QueryAnalyzer:
    """Analyzes queries to produce a RetrievalPlan.

    Two-layer strategy:
    1. Explicit API filters → direct construction (short-circuit)
    2. LLM extraction → broad intent analysis (filename, author, dates, custom)

    When no LLM is available, falls back to pure KG search.
    """

    def __init__(
        self,
        llm_func: Callable[..., Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._llm_func = llm_func
        self._schema = schema

    async def analyze(
        self,
        query: str,
        explicit_filters: MetadataFilter | None = None,
    ) -> RetrievalPlan:
        """Analyze query and produce a retrieval plan."""
        # Layer 1: explicit API filters (short-circuit — user intent is explicit)
        if explicit_filters and not explicit_filters.is_empty():
            paths = ["metadata"]
            if query.strip():
                paths.append("kg")
            logger.info(
                "[QueryAnalyzer] Explicit filters provided, paths=%s", paths
            )
            return RetrievalPlan(
                semantic_query=query,
                metadata_filters=explicit_filters,
                paths=paths,
                original_query=query,
            )

        # Layer 2: LLM extraction
        if self._llm_func:
            logger.info("[QueryAnalyzer] Running LLM extraction for: %.120s", query)
            llm_plan = await self._llm_extract(query)
            if llm_plan:
                logger.info(
                    "[QueryAnalyzer] LLM result: paths=%s, has_filters=%s, semantic_query=%.80s",
                    llm_plan.paths,
                    llm_plan.metadata_filters is not None,
                    llm_plan.semantic_query,
                )
                return llm_plan

        # Fallback: pure KG search
        logger.info("[QueryAnalyzer] No LLM or extraction failed, fallback to pure KG")
        return RetrievalPlan(
            semantic_query=query,
            metadata_filters=None,
            paths=["kg"],
            original_query=query,
        )

    async def _llm_extract(self, query: str) -> RetrievalPlan | None:
        """Use LLM to extract structured filters from natural language."""
        assert self._llm_func is not None  # guarded by caller
        try:
            schema_section = _build_schema_section(self._schema)
            custom_keys_hint = _build_custom_keys_hint(self._schema)
            prompt = _ANALYSIS_PROMPT.format(
                query=query,
                schema_section=schema_section,
                custom_keys_hint=custom_keys_hint,
            )
            response = await self._llm_func(
                prompt,
                system_prompt="You are a metadata filter extractor. Return JSON only, no explanation.",
            )
            from dlightrag.utils.text import extract_json

            raw = extract_json(response)
            data = json.loads(raw)

            filters_data = data.get("filters", {})
            mf = MetadataFilter(
                filename=filters_data.get("filename"),
                filename_pattern=filters_data.get("filename_pattern"),
                file_extension=filters_data.get("file_extension"),
                doc_title=filters_data.get("doc_title"),
                doc_author=filters_data.get("doc_author"),
                custom=filters_data.get("custom"),
            )
            for date_field in ("date_from", "date_to"):
                val = filters_data.get(date_field)
                if val:
                    from datetime import datetime

                    try:
                        setattr(mf, date_field, datetime.fromisoformat(val))
                    except (ValueError, TypeError):
                        pass

            if mf.is_empty():
                mf = None

            semantic = data.get("semantic_query", query)
            paths = _resolve_paths(mf, semantic)
            return RetrievalPlan(
                semantic_query=semantic,
                metadata_filters=mf,
                paths=paths,
                original_query=query,
            )
        except Exception as exc:
            logger.warning("QueryAnalyzer LLM extraction failed: %s", exc)
            return None
