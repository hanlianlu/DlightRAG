# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified query understanding -- rewrite + analyze in one LLM call.

Replaces the two-step pipeline (web/routes._rewrite_query + retrieval/query_analyzer)
with a single QueryPlanner that produces a QueryPlan from one LLM call.

Consumers:
- ServiceManager -- owns the planner singleton, calls plan() before retrieval
- Web/API routes -- no longer do query processing, just pass raw history
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.models.structured import StructuredOutput
from dlightrag.prompts import (
    PLANNER_HISTORY_TEMPLATE,
    PLANNER_NO_HISTORY_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)
from dlightrag.utils.tokens import truncate_conversation_history

logger = logging.getLogger(__name__)


def _convert_history_to_text(history: list[dict[str, Any]] | None) -> str:
    """Convert multimodal conversation_history to plain text for the planner LLM.

    Text content is preserved inline.  ``image_url`` blocks are replaced
    with ``[user shared N image(s)]`` placeholders so the planner can
    understand referential context without needing vision capability.
    """
    if not history:
        return ""
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            text_parts: list[str] = []
            image_count = 0
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    image_count += 1
            text = "".join(text_parts)
            if image_count > 0:
                s = "s" if image_count > 1 else ""
                text += f" [user shared {image_count} image{s}]"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


@dataclass
class QueryPlan:
    """Output of QueryPlanner -- full query understanding in one shot."""

    original_query: str  # User's raw input (before rewrite)
    standalone_query: str  # Rewritten standalone query (= original if no history)
    bm25_query: str | None = None
    referenced_image_ids: list[str] | None = None
    metadata_filter: MetadataFilter | None = None
    metadata_filter_source: str | None = None
    metadata_filter_confidence: str | None = None
    metadata_filter_evidence: list[dict[str, Any]] | None = None


class QueryPlannerFilterEvidence(BaseModel):
    """Evidence attached to one LLM-proposed metadata filter."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: str = ""
    evidence_span: str
    intent_basis: str = ""


class QueryPlannerFilters(BaseModel):
    """Structured filter proposal emitted by the planner LLM."""

    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    filename_stem: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    custom: dict[str, str | int | float | bool | None] | None = None


class QueryPlannerStructuredResponse(BaseModel):
    """Pydantic schema for planner structured-output calls."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    bm25_query: str | None = None
    referenced_image_ids: list[str] = Field(default_factory=list)
    filters: QueryPlannerFilters = Field(default_factory=QueryPlannerFilters)
    filter_confidence: Literal["high", "low"] = "low"
    filter_evidence: list[QueryPlannerFilterEvidence] = Field(default_factory=list)


QUERY_PLAN_STRUCTURED_OUTPUT = StructuredOutput(
    name="query_plan",
    schema=QueryPlannerStructuredResponse,
)


# PG data_type -> human-readable short form
_TYPE_ALIASES: dict[str, str] = {
    "character varying": "text",
    "timestamp with time zone": "timestamp",
    "jsonb": "json",
    "integer": "int",
}


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


def _format_filter_evidence(evidence: list[dict[str, Any]] | None, *, limit: int = 3) -> str:
    if not evidence:
        return "[]"
    parts: list[str] = []
    for item in evidence[:limit]:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "?")
        basis = str(item.get("intent_basis") or "?")
        span = str(item.get("evidence_span") or "").replace("\n", " ")[:80]
        parts.append(f"{field}:{basis}:{span!r}")
    if len(evidence) > limit:
        parts.append(f"+{len(evidence) - limit}")
    return ",".join(parts) if parts else "[]"


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

    def _make_llm_call(self) -> Callable[..., Any]:
        """Return a closure that calls ``self._llm_func`` with the right signature.

        LightRAG-adapted wrappers expect ``(prompt, *, system_prompt, ...)``
        while raw completion funcs expect ``(messages, **kw)``. We detect
        which one we have at init time so the hot path doesn't branch.
        """
        llm_func = self._llm_func
        if llm_func is None:
            raise RuntimeError("Query planning requires an LLM function")

        try:
            sig = inspect.signature(llm_func)
            params = sig.parameters
        except ValueError, TypeError:
            params = {}

        if "prompt" in params and "messages" not in params:
            # LightRAG-adapted wrapper: (prompt, *, system_prompt, ...)

            async def _call(query: str, system_prompt: str) -> str:
                return await llm_func(
                    prompt=query,
                    system_prompt=system_prompt,
                    structured_output=QUERY_PLAN_STRUCTURED_OUTPUT,
                )

            return _call

        # Raw completion func (messages-first, default path)

        async def _call(query: str, system_prompt: str) -> str:
            return await llm_func(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                structured_output=QUERY_PLAN_STRUCTURED_OUTPUT,
            )

        return _call

    async def aclose(self) -> None:
        """Release model-function worker resources owned by this planner."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self._llm_func)

    async def plan(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        explicit_filter: MetadataFilter | None = None,
        max_turns: int = 25,
        max_tokens: int = 65536,
        schema: dict[str, Any] | None = None,
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
        schema = schema if schema is not None else await self._refresh_schema()
        t1 = time.monotonic()
        logger.info("[Planner] schema refresh: %.1fs", t1 - t0)

        # Build prompt
        schema_section = _build_schema_section(schema)
        custom_keys_hint = _build_custom_keys_hint(schema)

        if history:
            history_text = _convert_history_to_text(history)
            history_section = PLANNER_HISTORY_TEMPLATE.format(
                history_text=history_text, query=query
            )
        else:
            history_section = PLANNER_NO_HISTORY_TEMPLATE.format(query=query)

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            schema_section=schema_section,
            custom_keys_hint=custom_keys_hint,
            history_section=history_section,
        )

        # Detect callable signature: LightRAG-adapted wrappers expect
        # (prompt, *, system_prompt, ...) while raw completion funcs
        # expect (messages, **kw).
        _llm_call = self._make_llm_call()

        # LLM call with adaptive retry (up to 2 retries with exponential backoff)
        _MAX_RETRIES = 2
        response: str | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await _llm_call(query, system_prompt)
                logger.info(
                    "[Planner] LLM call: %.1fs (attempt %d)", time.monotonic() - t1, attempt
                )
                break
            except Exception:
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt  # 1s, 2s
                    logger.warning(
                        "QueryPlanner LLM call failed (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "QueryPlanner LLM call failed after %d attempts (%.1fs)",
                        _MAX_RETRIES + 1,
                        time.monotonic() - t1,
                        exc_info=True,
                    )
                    return fallback

        # Parse response (response is guaranteed str here; None means all retries failed)
        if response is None:
            return fallback
        plan = self._parse_response(response, query)
        if plan is None:
            return fallback

        # Merge explicit filter (explicit wins)
        if explicit_filter is not None and not _is_empty_filter(explicit_filter):
            plan.metadata_filter = self._merge_filters(explicit_filter, plan.metadata_filter)
            plan.metadata_filter_source = "explicit"
            plan.metadata_filter_confidence = "high"

        logger.info(
            "[Planner] result: standalone=%r, bm25_query=%r, filter_source=%s, "
            "filter_confidence=%s, filter_evidence=%s, filter=%s",
            plan.standalone_query[:60],
            plan.bm25_query,
            plan.metadata_filter_source,
            plan.metadata_filter_confidence,
            _format_filter_evidence(plan.metadata_filter_evidence),
            plan.metadata_filter,
        )
        return plan

    def _parse_response(self, response: str, query: str) -> QueryPlan | None:
        """Parse LLM JSON response into a QueryPlan."""
        if not response:
            return None
        try:
            parsed = QUERY_PLAN_STRUCTURED_OUTPUT.parse(response)
        except ValidationError, ValueError, TypeError:
            logger.warning("QueryPlanner: invalid structured output for query: %r", query[:80])
            return None

        data = parsed.model_dump()
        standalone = data.get("standalone_query", query)
        raw_filters = data.get("filters", {}) or {}
        filter_confidence = str(data.get("filter_confidence") or "").lower() or None
        filter_evidence = data.get("filter_evidence") or []

        # Validate filters
        clean = {k: v for k, v in raw_filters.items() if v is not None}
        if clean and filter_confidence == "low":
            logger.info("QueryPlanner: ignored low-confidence metadata filter proposal")
            return QueryPlan(
                original_query=query,
                standalone_query=standalone,
                bm25_query=data.get("bm25_query") or None,
                referenced_image_ids=_clean_image_ids(data.get("referenced_image_ids")),
                metadata_filter=None,
                metadata_filter_source=None,
                metadata_filter_confidence=filter_confidence or "low",
                metadata_filter_evidence=filter_evidence
                if isinstance(filter_evidence, list)
                else None,
            )

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
                    except ValueError, TypeError:
                        pass
                if date_to_str:
                    from datetime import datetime

                    try:
                        mf.date_to = datetime.fromisoformat(date_to_str)
                    except ValueError, TypeError:
                        pass
                metadata_filter = mf if not mf.is_empty() else None
            except Exception:
                logger.warning("QueryPlanner: invalid filter values for query: %r", query[:80])
                metadata_filter = None

        return QueryPlan(
            original_query=query,
            standalone_query=standalone,
            bm25_query=data.get("bm25_query") or None,
            referenced_image_ids=_clean_image_ids(data.get("referenced_image_ids")),
            metadata_filter=metadata_filter,
            metadata_filter_source="llm_inferred" if metadata_filter is not None else None,
            metadata_filter_confidence=filter_confidence,
            metadata_filter_evidence=filter_evidence if isinstance(filter_evidence, list) else None,
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
            "filename_stem",
            "filename_pattern",
            "file_extension",
            "doc_title",
            "doc_author",
            "date_from",
            "date_to",
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
        history: list[dict[str, Any]] | None,
        max_turns: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
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
                logger.debug("Schema refresh failed, keeping existing cache")
        return self._schema


def _clean_image_ids(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    ids = [item for item in raw if isinstance(item, str) and item.startswith("img_")]
    return ids or None
