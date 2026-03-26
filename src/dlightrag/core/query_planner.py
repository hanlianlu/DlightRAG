# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified query understanding -- rewrite + analyze in one LLM call.

Replaces the two-step pipeline (web/routes._rewrite_query + retrieval/query_analyzer)
with a single QueryPlanner that produces a QueryPlan from one LLM call.

Consumers:
- ServiceManager -- owns the planner singleton, calls plan() before retrieval
- Web/API routes -- no longer do query processing, just pass raw history
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.utils.text import extract_json
from dlightrag.utils.tokens import truncate_conversation_history

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Output of QueryPlanner -- full query understanding in one shot."""

    original_query: str  # User's raw input (before rewrite)
    standalone_query: str  # Rewritten standalone query (= original if no history)
    metadata_filter: MetadataFilter | None = None


# PG data_type -> human-readable short form
_TYPE_ALIASES: dict[str, str] = {
    "character varying": "text",
    "timestamp with time zone": "timestamp",
    "jsonb": "json",
    "integer": "int",
}

_SYSTEM_PROMPT = """\
You are a document domain query understanding system. Given a user query (and optionally
conversation history), produce a JSON response with these keys:

- "standalone_query": If conversation history is provided, rewrite the follow-up into
  a self-contained query capturing full intent. If no history or the query is already
  standalone, return it unchanged. This is the primary search query -- keep it complete.
- "filters": An object with applicable fields from the metadata schema below.
  Only include fields you are highly confident about. Leave out uncertain fields.

Filter fields (use null for unmentioned):
- filename: best-guess normalized filename (underscores, correct extension case)
- filename_pattern: SQL ILIKE pattern with % wildcards when reference is ambiguous
- file_extension: e.g. "pdf", "png" (lowercase, no dot)
- doc_title: document title reference
- doc_author: author name
- date_from / date_to: ISO 8601 dates for time ranges
- custom: {{"key": "value"}} for custom metadata

{schema_section}{custom_keys_hint}\
{history_section}\
Examples:

Query: "summarize the key findings in annual-report.pdf"
{{"standalone_query": "summarize the key findings in annual-report.pdf", "filters": {{"filename": "annual-report.pdf"}}}}

Query: "what are the main revenue trends"
{{"standalone_query": "what are the main revenue trends", "filters": {{}}}}

Query: "张三写的2024年财报分析"
{{"standalone_query": "张三写的2024年财报分析", "filters": {{"doc_author": "张三", "date_from": "2024-01-01", "date_to": "2024-12-31"}}}}

Return valid JSON only, no markdown fences."""

_HISTORY_TEMPLATE = """\
Conversation history:
{history_text}

Current follow-up message: {query}

"""

_NO_HISTORY_TEMPLATE = """\
Query: {query}

"""


def _build_schema_section(schema: dict[str, Any] | None) -> str:
    if not schema or not schema.get("columns"):
        return ""
    col_lines = [
        f"  {col['name']} ({_TYPE_ALIASES.get(col['type'], col['type'])})"
        for col in schema["columns"]
    ]
    return "Available metadata columns:\n" + "\n".join(col_lines) + "\n\n"


def _build_custom_keys_hint(schema: dict[str, Any] | None) -> str:
    if not schema:
        return ""
    custom_keys = schema.get("custom_keys")
    if not custom_keys:
        return ""
    return f"Known custom metadata keys: {', '.join(custom_keys)}\n\n"


def _is_empty_filter(f: MetadataFilter) -> bool:
    return f.is_empty()


class QueryPlanner:
    """Unified query understanding -- rewrite + analyze in one LLM call."""

    def __init__(
        self,
        llm_func: Callable[..., Any] | None = None,
        *,
        schema_provider: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        schema_ttl: float = 300.0,
    ) -> None:
        self._llm_func = llm_func
        self._schema: dict[str, Any] | None = None
        self._schema_provider = schema_provider
        self._schema_ts: float = 0.0
        self._schema_ttl = schema_ttl

    async def plan(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        explicit_filter: MetadataFilter | None = None,
        max_turns: int = 25,
        max_tokens: int = 65536,
    ) -> QueryPlan:
        """Produce a full QueryPlan from one LLM call.

        Handles conversation rewriting, semantic query extraction, and metadata
        filter extraction in a single prompt.
        """
        fallback = QueryPlan(
            original_query=query,
            standalone_query=query,
        )

        if self._llm_func is None:
            return fallback

        # Truncate history
        history = self._truncate_history(conversation_history, max_turns, max_tokens)

        # Refresh schema (TTL-cached)
        t0 = time.monotonic()
        schema = await self._refresh_schema()
        t1 = time.monotonic()
        logger.info("[Planner] schema refresh: %.1fs", t1 - t0)

        # Build prompt
        schema_section = _build_schema_section(schema)
        custom_keys_hint = _build_custom_keys_hint(schema)

        if history:
            history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
            history_section = _HISTORY_TEMPLATE.format(history_text=history_text, query=query)
        else:
            history_section = _NO_HISTORY_TEMPLATE.format(query=query)

        system_prompt = _SYSTEM_PROMPT.format(
            schema_section=schema_section,
            custom_keys_hint=custom_keys_hint,
            history_section=history_section,
        )

        # Single LLM call
        try:
            response = await self._llm_func(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
            )
            logger.info("[Planner] LLM call: %.1fs", time.monotonic() - t1)
        except Exception:
            logger.warning(
                "QueryPlanner LLM call failed (%.1fs)", time.monotonic() - t1, exc_info=True
            )
            return fallback

        # Parse response
        plan = self._parse_response(response, query)
        if plan is None:
            return fallback

        # Merge explicit filter (explicit wins)
        if explicit_filter is not None and not _is_empty_filter(explicit_filter):
            plan.metadata_filter = self._merge_filters(explicit_filter, plan.metadata_filter)

        return plan

    def _parse_response(self, response: str, query: str) -> QueryPlan | None:
        """Parse LLM JSON response into a QueryPlan."""
        if not response:
            return None
        try:
            raw = extract_json(response)
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("QueryPlanner: invalid JSON for query: %r", query[:80])
            return None

        standalone = data.get("standalone_query", query)
        raw_filters = data.get("filters", {})

        # Validate filters
        clean = {k: v for k, v in raw_filters.items() if v is not None}

        # Handle date fields specially
        metadata_filter: MetadataFilter | None = None
        if clean:
            date_from_str = clean.pop("date_from", None)
            date_to_str = clean.pop("date_to", None)
            try:
                mf = MetadataFilter(**clean)
                if date_from_str:
                    from datetime import datetime

                    try:
                        mf.date_from = datetime.fromisoformat(date_from_str)
                    except (ValueError, TypeError):
                        pass
                if date_to_str:
                    from datetime import datetime

                    try:
                        mf.date_to = datetime.fromisoformat(date_to_str)
                    except (ValueError, TypeError):
                        pass
                metadata_filter = mf if not mf.is_empty() else None
            except Exception:
                logger.warning("QueryPlanner: invalid filter values for query: %r", query[:80])
                metadata_filter = None

        return QueryPlan(
            original_query=query,
            standalone_query=standalone,
            metadata_filter=metadata_filter,
        )

    @staticmethod
    def _merge_filters(
        explicit: MetadataFilter,
        llm_extracted: MetadataFilter | None,
    ) -> MetadataFilter:
        """Merge explicit and LLM-extracted filters. Explicit fields win."""
        if llm_extracted is None:
            return explicit

        # Build merged filter field by field
        merged_kwargs: dict[str, Any] = {}
        filter_fields = [
            "filename",
            "filename_pattern",
            "file_extension",
            "doc_title",
            "doc_author",
            "date_from",
            "date_to",
            "rag_mode",
            "custom",
        ]
        for field in filter_fields:
            explicit_val = getattr(explicit, field, None)
            llm_val = getattr(llm_extracted, field, None)
            if explicit_val is not None:
                merged_kwargs[field] = explicit_val
            elif llm_val is not None:
                merged_kwargs[field] = llm_val
        return MetadataFilter(**merged_kwargs)

    @staticmethod
    def _truncate_history(
        history: list[dict[str, str]] | None,
        max_turns: int,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        if not history:
            return []
        max_messages = max_turns * 2
        return truncate_conversation_history(
            history, max_messages=max_messages, max_tokens=max_tokens
        )

    async def _refresh_schema(self) -> dict[str, Any] | None:
        """Return cached schema, refreshing if TTL expired."""
        now = time.monotonic()
        if self._schema is not None and (now - self._schema_ts) < self._schema_ttl:
            return self._schema
        if self._schema_provider is not None:
            try:
                self._schema = await self._schema_provider()
                self._schema_ts = now
            except Exception:
                logger.debug("Schema refresh failed, using stale cache")
        return self._schema
