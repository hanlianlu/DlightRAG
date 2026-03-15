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


class QueryAnalyzer:
    """Analyzes queries to produce a RetrievalPlan.

    Three-layer strategy:
    1. Explicit API filters → direct construction
    2. fast_detect regex → skip LLM for obvious filenames
    3. LLM extraction → full intent analysis
    """

    def __init__(self, llm_func: Callable[..., Any] | None = None) -> None:
        self._llm_func = llm_func

    async def analyze(
        self,
        query: str,
        explicit_filters: MetadataFilter | None = None,
    ) -> RetrievalPlan:
        """Analyze query and produce a retrieval plan."""
        # Layer 1: explicit API filters
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

        # Layer 2: fast regex detection
        detected = fast_detect(query)
        if detected:
            semantic = _FILE_PATTERN.sub("", query).strip()
            # Clean up leftover punctuation/whitespace
            semantic = re.sub(r"\s{2,}", " ", semantic).strip()
            paths = ["metadata"]
            if semantic:
                paths.append("kg")
            return RetrievalPlan(
                semantic_query=semantic or query,
                metadata_filters=detected,
                paths=paths,
                original_query=query,
            )

        # Layer 3: LLM extraction (if available)
        if self._llm_func:
            plan = await self._llm_extract(query)
            if plan:
                return plan

        # Fallback: pure KG search (current behavior)
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
