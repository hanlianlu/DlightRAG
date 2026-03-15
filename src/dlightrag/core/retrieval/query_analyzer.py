# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query intent analysis and metadata filter extraction."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter, RetrievalPlan

logger = logging.getLogger(__name__)

_FILE_PATTERN = re.compile(
    r"\b([\w\-\.]+\.(pdf|png|jpg|jpeg|docx|pptx|xlsx|csv|txt|md|html))\b",
    re.IGNORECASE,
)

_ANALYSIS_PROMPT = """\
Analyze this search query. Extract metadata filters and a cleaned semantic query.

Query: "{query}"

Return JSON only:
{{"semantic_query": "core question without file/author/date references",
  "filters": {{"filename": null, "filename_pattern": null, "file_extension": null,
               "doc_title": null, "doc_author": null, "date_from": null, "date_to": null,
               "custom": null}},
  "paths": ["metadata", "kg"]}}

Rules:
- semantic_query: remove filename/author/date references, keep the actual question
- filters: extract any document identifiers mentioned. Use null for unmentioned fields.
- paths: "metadata" if any filter extracted, "kg" if semantic question exists, both if mixed
- If pure enumeration (no semantic question), paths=["metadata"] and semantic_query=""
"""


def fast_detect(query: str) -> MetadataFilter | None:
    """Regex-based fast detection of file references in query text."""
    match = _FILE_PATTERN.search(query)
    if match:
        return MetadataFilter(filename=match.group(1))
    return None


def _merge_filters(regex: MetadataFilter, llm: MetadataFilter) -> MetadataFilter:
    """Merge regex (high-confidence) and LLM (broad) extracted filters.

    Regex results take priority for fields it can detect (filename).
    LLM fills in fields regex cannot extract (author, dates, custom, etc.).
    """
    return MetadataFilter(
        filename=regex.filename or llm.filename,
        filename_pattern=regex.filename_pattern or llm.filename_pattern,
        file_extension=regex.file_extension or llm.file_extension,
        doc_title=llm.doc_title,
        doc_author=llm.doc_author,
        date_from=llm.date_from,
        date_to=llm.date_to,
        rag_mode=llm.rag_mode,
        custom=llm.custom,
    )


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

    Three-layer strategy:
    1. Explicit API filters → direct construction (short-circuit)
    2. fast_detect regex → high-confidence filename extraction (free)
    3. LLM extraction → broad intent analysis (author, dates, custom)

    Layers 2 and 3 run collaboratively: regex provides high-confidence
    filename, LLM fills in fields regex cannot detect. Results are merged
    with regex taking priority for overlapping fields.
    """

    def __init__(self, llm_func: Callable[..., Any] | None = None) -> None:
        self._llm_func = llm_func

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
            return RetrievalPlan(
                semantic_query=query,
                metadata_filters=explicit_filters,
                paths=paths,
                original_query=query,
            )

        # Layer 2: fast regex detection (free, instant)
        regex_filters = fast_detect(query)
        regex_semantic: str | None = None
        if regex_filters:
            regex_semantic = _FILE_PATTERN.sub("", query).strip()
            regex_semantic = re.sub(r"\s{2,}", " ", regex_semantic).strip() or None

        # Layer 3: LLM extraction (if available)
        llm_plan: RetrievalPlan | None = None
        if self._llm_func:
            llm_plan = await self._llm_extract(query)

        # Merge layers 2 + 3
        if regex_filters and llm_plan and llm_plan.metadata_filters:
            merged = _merge_filters(regex_filters, llm_plan.metadata_filters)
            semantic = llm_plan.semantic_query or regex_semantic or query
            return RetrievalPlan(
                semantic_query=semantic,
                metadata_filters=merged,
                paths=_resolve_paths(merged, semantic),
                original_query=query,
            )

        # Regex only (no LLM available, or LLM found nothing extra)
        if regex_filters:
            paths = ["metadata"]
            if regex_semantic:
                paths.append("kg")
            return RetrievalPlan(
                semantic_query=regex_semantic or query,
                metadata_filters=regex_filters,
                paths=paths,
                original_query=query,
            )

        # LLM only (regex found nothing)
        if llm_plan:
            return llm_plan

        # Fallback: pure KG search
        return RetrievalPlan(
            semantic_query=query,
            metadata_filters=None,
            paths=["kg"],
            original_query=query,
        )

    async def _llm_extract(self, query: str) -> RetrievalPlan | None:
        """Use LLM to extract structured filters from natural language."""
        try:
            prompt = _ANALYSIS_PROMPT.format(query=query)
            response = await self._llm_func(
                prompt,
                system_prompt="You are a query analyzer. Return JSON only.",
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
            # Parse dates if present
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

            paths = data.get("paths", ["kg"])
            return RetrievalPlan(
                semantic_query=data.get("semantic_query", query),
                metadata_filters=mf,
                paths=paths,
                original_query=query,
            )
        except Exception as exc:
            logger.warning("QueryAnalyzer LLM extraction failed: %s", exc)
            return None
