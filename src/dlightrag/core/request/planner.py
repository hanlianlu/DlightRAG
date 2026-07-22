# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified query understanding -- rewrite + analyze in one LLM call.

Replaces the two-step pipeline (web/routes._rewrite_query + retrieval/query_analyzer)
with a single QueryPlanner that produces a QueryPlan from one LLM call.

Consumers:
- ServiceManager -- owns the planner singleton, calls plan() before retrieval
- Web/API routes -- no longer do query processing, just pass raw history
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlightrag.core.request.attachment_digest import ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.models.structured import StructuredOutput
from dlightrag.prompts import (
    PLANNER_CURRENT_IMAGE_TEMPLATE,
    PLANNER_HISTORY_TEMPLATE,
    PLANNER_NO_HISTORY_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_WEB_SELECTION_TEMPLATE,
    WEB_PLANNER_CURRENT_ATTACHMENT_TEMPLATE,
    WEB_PLANNER_HISTORY_ATTACHMENT_TEMPLATE,
    WEB_PLANNER_SYSTEM_PROMPT,
)
from dlightrag.utils import log_safe
from dlightrag.utils.tokens import (
    estimate_tokens,
    truncate_conversation_history,
    truncate_to_estimated_tokens,
)

logger = logging.getLogger(__name__)

_WEB_PLANNER_INPUT_TOKEN_ENVELOPE = 102_400
_WEB_PLANNER_QUERY_TOKEN_BUDGET = 8_192
_WEB_PLANNER_SCHEMA_TOKEN_BUDGET = 8_192
_WEB_PLANNER_CURRENT_IMAGE_DESCRIPTION_TOKEN_BUDGET = 4_096
_WEB_PLANNER_HISTORY_CATALOG_TOKEN_BUDGET = 4_096
_WEB_PLANNER_HISTORY_ATTACHMENT_SUMMARY_MAX_TOKENS = 1_024
_WEB_PLANNER_HISTORY_IMAGE_DESCRIPTION_MAX_TOKENS = 512
_WEB_PLANNER_ESTIMATION_MARGIN = 64


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
    metadata_filter: MetadataFilter | None = None
    metadata_filter_source: str | None = None
    metadata_filter_confidence: str | None = None
    metadata_filter_evidence: list[dict[str, Any]] | None = None
    selected_history_image_ids: tuple[str, ...] = ()
    # Web-only conversation planner field (empty for stateless REST/MCP/SDK/retrieve).
    selected_history_attachment_ids: tuple[str, ...] = ()


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
    filters: QueryPlannerFilters = Field(default_factory=QueryPlannerFilters)
    filter_confidence: Literal["high", "low"] = "low"
    filter_evidence: list[QueryPlannerFilterEvidence] = Field(default_factory=list)


QUERY_PLAN_STRUCTURED_OUTPUT = StructuredOutput(
    name="query_plan",
    schema=QueryPlannerStructuredResponse,
)


class WebQueryPlannerStructuredResponse(QueryPlannerStructuredResponse):
    """Web-variant planner schema: adds scoped history-image selection.

    Used only when a conversation image catalog is supplied.  The public schema
    stays byte-identical, so REST/MCP/Python planning is provably unchanged.
    """

    selected_history_image_ids: list[str] = Field(default_factory=list)


QUERY_PLAN_WEB_STRUCTURED_OUTPUT = StructuredOutput(
    name="query_plan",
    schema=WebQueryPlannerStructuredResponse,
)


class WebConversationPlannerStructuredResponse(QueryPlannerStructuredResponse):
    """Web conversation planner schema: adds scoped document/image selection.

    Used only by ``QueryPlanner.plan_web_conversation`` (the Web ``/web/answer``
    path). The stateless ``QueryPlanner.plan`` contract is unaffected -- its
    public schema forbids these extra fields.
    """

    selected_history_image_ids: list[str] = Field(default_factory=list)
    selected_history_attachment_ids: list[str] = Field(default_factory=list)


QUERY_PLAN_WEB_CONVERSATION_STRUCTURED_OUTPUT = StructuredOutput(
    name="query_plan",
    schema=WebConversationPlannerStructuredResponse,
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


def _scope_ids(
    selected: tuple[str, ...],
    allowed: set[str],
    limit: int,
) -> tuple[str, ...]:
    """Keep only selected ids present in the scoped catalog, deduped + truncated.

    LLM output is not authorization: the planner may only return ids that were
    supplied in the request-local catalog.
    """
    deduped: dict[str, None] = {}
    for item in selected:
        if str(item) in allowed:
            deduped.setdefault(str(item), None)
    return tuple(list(deduped)[: max(0, limit)])


def _equal_demand_token_budgets(demands: dict[str, int], token_budget: int) -> dict[str, int]:
    """Water-fill one token pool fairly across independently capped demands."""
    budgets = dict.fromkeys(demands, 0)
    remaining = max(0, token_budget)
    active = {key for key, demand in demands.items() if demand > 0}
    while active and remaining:
        share = remaining // len(active)
        satisfied = {key for key in active if demands[key] - budgets[key] <= share}
        if satisfied:
            for key in sorted(satisfied):
                grant = demands[key] - budgets[key]
                budgets[key] += grant
                remaining -= grant
            active -= satisfied
            continue
        for key in sorted(active):
            grant = min(share, demands[key] - budgets[key])
            budgets[key] += grant
            remaining -= grant
        for key in sorted(active):
            if remaining <= 0:
                break
            if budgets[key] < demands[key]:
                budgets[key] += 1
                remaining -= 1
        break
    return budgets


def _budget_current_attachment_catalog(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Defensively enforce the current-turn 8192-token digest contract."""
    if not rows:
        return []
    demands = {
        str(row["attachment_id"]): estimate_tokens(str(row.get("parse_summary") or ""))
        for row in rows
    }
    budgets = _equal_demand_token_budgets(
        demands,
        ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET,
    )
    return [
        {
            **row,
            "parse_summary": truncate_to_estimated_tokens(
                str(row.get("parse_summary") or ""),
                budgets[str(row["attachment_id"])],
            ),
        }
        for row in rows
    ]


def _budget_current_image_descriptions(descriptions: list[str] | None) -> list[str]:
    """Bound current image semantics as one planner-context pool."""
    if not descriptions:
        return []
    demands = {
        str(index): estimate_tokens(description) for index, description in enumerate(descriptions)
    }
    budgets = _equal_demand_token_budgets(
        demands,
        _WEB_PLANNER_CURRENT_IMAGE_DESCRIPTION_TOKEN_BUDGET,
    )
    return [
        truncate_to_estimated_tokens(description, budgets[str(index)])
        for index, description in enumerate(descriptions)
    ]


def _render_web_history_catalog_sections(
    image_rows: list[dict[str, Any]],
    attachment_rows: list[dict[str, Any]],
    *,
    allowed_image_count: int,
    allowed_attachment_count: int,
) -> str:
    sections: list[str] = []
    if image_rows:
        image_lines = "\n".join(
            f"{row['image_id']} | turn {row['turn_number']} | "
            f"ord {row['ordinal']} | {row['vlm_description']}"
            for row in image_rows
        )
        sections.append(
            PLANNER_WEB_SELECTION_TEMPLATE.format(
                allowed_count=max(0, allowed_image_count),
                catalog_lines=image_lines,
            )
        )
    if attachment_rows:
        attachment_lines = "\n".join(
            f"{row['attachment_id']} | turn {row['turn_number']} | "
            f"ord {row['ordinal']} | {row['filename']} | {row['parse_summary']}"
            for row in attachment_rows
        )
        sections.append(
            WEB_PLANNER_HISTORY_ATTACHMENT_TEMPLATE.format(
                allowed_count=max(0, allowed_attachment_count),
                catalog_lines=attachment_lines,
            )
        )
    return "".join(sections)


def _pack_web_history_catalogs(
    image_catalog: list[dict[str, Any]] | None,
    attachment_catalog: list[dict[str, Any]] | None,
    *,
    allowed_image_count: int,
    allowed_attachment_count: int,
    token_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Pack the most recent cross-modal history catalog into one token cap."""
    entries: list[tuple[int, int, str, dict[str, Any]]] = []
    for row in image_catalog or []:
        prepared = {
            **row,
            "vlm_description": truncate_to_estimated_tokens(
                str(row.get("vlm_description") or ""),
                _WEB_PLANNER_HISTORY_IMAGE_DESCRIPTION_MAX_TOKENS,
            ),
        }
        entries.append(
            (
                int(row.get("turn_number") or -1),
                int(row.get("ordinal") or 0),
                "image",
                prepared,
            )
        )
    for row in attachment_catalog or []:
        prepared = {
            **row,
            "parse_summary": truncate_to_estimated_tokens(
                str(row.get("parse_summary") or ""),
                _WEB_PLANNER_HISTORY_ATTACHMENT_SUMMARY_MAX_TOKENS,
            ),
        }
        entries.append(
            (
                int(row.get("turn_number") or -1),
                int(row.get("ordinal") or 0),
                "attachment",
                prepared,
            )
        )

    selected_images: list[dict[str, Any]] = []
    selected_attachments: list[dict[str, Any]] = []
    budget = max(0, token_budget)
    for _, _, kind, row in sorted(entries, key=lambda item: item[:3], reverse=True):
        candidate_images = [*selected_images, row] if kind == "image" else selected_images
        candidate_attachments = (
            [*selected_attachments, row] if kind == "attachment" else selected_attachments
        )
        candidate = _render_web_history_catalog_sections(
            candidate_images,
            candidate_attachments,
            allowed_image_count=allowed_image_count,
            allowed_attachment_count=allowed_attachment_count,
        )
        if estimate_tokens(candidate) <= budget:
            selected_images = candidate_images
            selected_attachments = candidate_attachments

    rendered = _render_web_history_catalog_sections(
        selected_images,
        selected_attachments,
        allowed_image_count=allowed_image_count,
        allowed_attachment_count=allowed_attachment_count,
    )
    return selected_images, selected_attachments, rendered


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

    async def _call_llm(
        self,
        query: str,
        system_prompt: str,
        *,
        structured_output: StructuredOutput = QUERY_PLAN_STRUCTURED_OUTPUT,
    ) -> str:
        """Call the planner LLM using DlightRAG's messages-first contract."""
        llm_func = self._llm_func
        if llm_func is None:
            raise RuntimeError("Query planning requires an LLM function")

        return await llm_func(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            structured_output=structured_output,
        )

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
        image_catalog: list[dict[str, Any]] | None = None,
        allowed_history_image_count: int = 0,
        current_image_descriptions: list[str] | None = None,
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

        # Web-variant: append the image catalog + selection guidance and switch to
        # the schema that carries selected_history_image_ids. The public prompt and
        # schema stay byte-identical when no catalog is supplied.
        structured_output = QUERY_PLAN_STRUCTURED_OUTPUT
        if image_catalog:
            structured_output = QUERY_PLAN_WEB_STRUCTURED_OUTPUT
            catalog_lines = "\n".join(
                f"{row['image_id']} | turn {row['turn_number']} | "
                f"ord {row['ordinal']} | {row['vlm_description']}"
                for row in image_catalog
            )
            system_prompt += PLANNER_WEB_SELECTION_TEMPLATE.format(
                allowed_count=max(0, allowed_history_image_count),
                catalog_lines=catalog_lines,
            )

        # Current-turn images are always incorporated (no selection): fold their
        # descriptions into the query so bm25/filters/rewrite become image-aware
        # even for stateless REST/MCP calls with no conversation history.
        if current_image_descriptions:
            system_prompt += PLANNER_CURRENT_IMAGE_TEMPLATE.format(
                image_lines="\n".join(f"- {desc}" for desc in current_image_descriptions)
            )

        # LLM call with adaptive retry (up to 2 retries with exponential backoff)
        _MAX_RETRIES = 2
        response: str | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._call_llm(
                    query, system_prompt, structured_output=structured_output
                )
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
        plan = self._parse_response(response, query, structured_output=structured_output)
        if plan is None:
            return fallback

        # LLM output is not authorization: keep only ids present in the scoped
        # catalog, deduped and truncated to the allowed count.
        if image_catalog:
            allowed = {str(row["image_id"]) for row in image_catalog}
            deduped: dict[str, None] = {}
            for image_id in plan.selected_history_image_ids:
                if str(image_id) in allowed:
                    deduped.setdefault(str(image_id), None)
            plan.selected_history_image_ids = tuple(
                list(deduped)[: max(0, allowed_history_image_count)]
            )

        # Merge explicit filter (explicit wins)
        if explicit_filter is not None and not _is_empty_filter(explicit_filter):
            plan.metadata_filter = self._merge_filters(explicit_filter, plan.metadata_filter)
            plan.metadata_filter_source = "explicit"
            plan.metadata_filter_confidence = "high"

        logger.info(
            "[Planner] result: standalone=%r, bm25_query=%r, filter_source=%s, "
            "filter_confidence=%s, filter_evidence=%s, filter=%s",
            log_safe(plan.standalone_query, max_length=60),
            log_safe(plan.bm25_query),
            log_safe(plan.metadata_filter_source),
            log_safe(plan.metadata_filter_confidence),
            log_safe(_format_filter_evidence(plan.metadata_filter_evidence)),
            log_safe(plan.metadata_filter),
        )
        return plan

    async def _call_llm_with_retry(
        self,
        query: str,
        system_prompt: str,
        *,
        structured_output: StructuredOutput,
        start_time: float,
    ) -> str | None:
        """Call the planner LLM with adaptive exponential-backoff retry.

        Returns the raw response string, or ``None`` when all attempts fail.
        """
        _MAX_RETRIES = 2
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._call_llm(
                    query, system_prompt, structured_output=structured_output
                )
                logger.info(
                    "[Planner] LLM call: %.1fs (attempt %d)",
                    time.monotonic() - start_time,
                    attempt,
                )
                return response
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
                        time.monotonic() - start_time,
                        exc_info=True,
                    )
                    return None
        return None

    async def plan_web_conversation(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        explicit_filter: MetadataFilter | None = None,
        max_turns: int = 25,
        max_tokens: int = 65536,
        schema: dict[str, Any] | None = None,
        image_catalog: list[dict[str, Any]] | None = None,
        attachment_catalog: list[dict[str, Any]] | None = None,
        current_attachment_catalog: list[dict[str, Any]] | None = None,
        allowed_history_image_count: int = 0,
        allowed_history_attachment_count: int = 0,
        current_image_descriptions: list[str] | None = None,
    ) -> QueryPlan:
        """Plan one Web conversation turn (Web ``/web/answer`` only).

        Separate contract from the stateless :meth:`plan` used by REST/MCP/SDK/
        retrieve. Reuses the shared planner helpers (history truncation, schema
        rendering, structured-output parsing, filter merging) and adds scoped
        selection of prior Composer documents and images.
        """
        fallback = QueryPlan(
            original_query=query,
            standalone_query=query,
        )

        if self._llm_func is None:
            return fallback

        planner_query = truncate_to_estimated_tokens(
            query,
            _WEB_PLANNER_QUERY_TOKEN_BUDGET,
        )

        t0 = time.monotonic()
        schema = schema if schema is not None else await self._refresh_schema()
        t1 = time.monotonic()
        logger.info("[Planner] schema refresh: %.1fs", t1 - t0)

        schema_section = _build_schema_section(schema)
        custom_keys_hint = _build_custom_keys_hint(schema)
        schema_section = truncate_to_estimated_tokens(
            schema_section + custom_keys_hint,
            _WEB_PLANNER_SCHEMA_TOKEN_BUDGET,
        )
        custom_keys_hint = ""

        current_sections: list[str] = []
        current_attachment_rows = _budget_current_attachment_catalog(current_attachment_catalog)
        if current_attachment_rows:
            current_lines = "\n".join(
                f"{row['attachment_id']} | {row['filename']} | {row['parse_summary']}"
                for row in current_attachment_rows
            )
            current_sections.append(
                WEB_PLANNER_CURRENT_ATTACHMENT_TEMPLATE.format(catalog_lines=current_lines)
            )
        planned_image_descriptions = _budget_current_image_descriptions(current_image_descriptions)
        if planned_image_descriptions:
            current_sections.append(
                PLANNER_CURRENT_IMAGE_TEMPLATE.format(
                    image_lines="\n".join(f"- {desc}" for desc in planned_image_descriptions)
                )
            )
        current_context = "".join(current_sections)

        no_history_section = PLANNER_NO_HISTORY_TEMPLATE.format(query=planner_query)
        fixed_no_catalog_prompt = (
            WEB_PLANNER_SYSTEM_PROMPT.format(
                schema_section=schema_section,
                custom_keys_hint=custom_keys_hint,
                history_section=no_history_section,
            )
            + current_context
        )
        available_catalog_tokens = max(
            0,
            _WEB_PLANNER_INPUT_TOKEN_ENVELOPE
            - estimate_tokens(fixed_no_catalog_prompt)
            - estimate_tokens(planner_query)
            - _WEB_PLANNER_ESTIMATION_MARGIN,
        )
        catalog_budget = min(
            _WEB_PLANNER_HISTORY_CATALOG_TOKEN_BUDGET,
            available_catalog_tokens,
        )
        planned_image_catalog, planned_attachment_catalog, history_catalog_context = (
            _pack_web_history_catalogs(
                image_catalog,
                attachment_catalog,
                allowed_image_count=allowed_history_image_count,
                allowed_attachment_count=allowed_history_attachment_count,
                token_budget=catalog_budget,
            )
        )

        history = self._truncate_history(
            conversation_history,
            max_turns,
            max(0, max_tokens),
        )

        def render_prompt(messages: list[dict[str, Any]]) -> str:
            history_section = (
                PLANNER_HISTORY_TEMPLATE.format(
                    history_text=_convert_history_to_text(messages),
                    query=planner_query,
                )
                if messages
                else no_history_section
            )
            return (
                WEB_PLANNER_SYSTEM_PROMPT.format(
                    schema_section=schema_section,
                    custom_keys_hint=custom_keys_hint,
                    history_section=history_section,
                )
                + history_catalog_context
                + current_context
            )

        system_prompt = render_prompt(history)
        while history and (
            estimate_tokens(system_prompt) + estimate_tokens(planner_query)
            > _WEB_PLANNER_INPUT_TOKEN_ENVELOPE
        ):
            history = history[1:]
            system_prompt = render_prompt(history)
        planner_input_tokens = estimate_tokens(system_prompt) + estimate_tokens(planner_query)
        if planner_input_tokens > _WEB_PLANNER_INPUT_TOKEN_ENVELOPE:
            logger.error(
                "[Planner] refusing Web plan above the %d-token envelope: %d tokens",
                _WEB_PLANNER_INPUT_TOKEN_ENVELOPE,
                planner_input_tokens,
            )
            return fallback
        logger.debug(
            "[Planner] Web input budget: envelope=%d input=%d history_messages=%d "
            "catalog_images=%d catalog_attachments=%d",
            _WEB_PLANNER_INPUT_TOKEN_ENVELOPE,
            planner_input_tokens,
            len(history),
            len(planned_image_catalog),
            len(planned_attachment_catalog),
        )

        response = await self._call_llm_with_retry(
            planner_query,
            system_prompt,
            structured_output=QUERY_PLAN_WEB_CONVERSATION_STRUCTURED_OUTPUT,
            start_time=t1,
        )
        if response is None:
            return fallback

        plan = self._parse_response(
            response,
            query,
            structured_output=QUERY_PLAN_WEB_CONVERSATION_STRUCTURED_OUTPUT,
        )
        if plan is None:
            return fallback

        # LLM output is not authorization: keep only ids present in the scoped
        # catalogs, deduped and truncated to the allowed count.
        plan.selected_history_image_ids = _scope_ids(
            plan.selected_history_image_ids,
            {str(row["image_id"]) for row in planned_image_catalog},
            allowed_history_image_count,
        )
        allowed_attachments = {str(row["attachment_id"]) for row in planned_attachment_catalog}
        plan.selected_history_attachment_ids = _scope_ids(
            plan.selected_history_attachment_ids,
            allowed_attachments,
            allowed_history_attachment_count,
        )

        # Merge explicit filter (explicit wins)
        if explicit_filter is not None and not _is_empty_filter(explicit_filter):
            plan.metadata_filter = self._merge_filters(explicit_filter, plan.metadata_filter)
            plan.metadata_filter_source = "explicit"
            plan.metadata_filter_confidence = "high"

        logger.info(
            "[Planner] web result: standalone=%r, history_images=%d, history_attachments=%d",
            log_safe(plan.standalone_query, max_length=60),
            len(plan.selected_history_image_ids),
            len(plan.selected_history_attachment_ids),
        )
        return plan

    def _parse_response(
        self,
        response: str,
        query: str,
        *,
        structured_output: StructuredOutput = QUERY_PLAN_STRUCTURED_OUTPUT,
    ) -> QueryPlan | None:
        """Parse LLM JSON response into a QueryPlan."""
        if not response:
            return None
        try:
            parsed = structured_output.parse(response)
        except ValidationError, ValueError, TypeError:
            logger.warning(
                "QueryPlanner: invalid structured output for query: %s",
                log_safe(query, max_length=80),
            )
            return None

        data = parsed.model_dump()
        standalone = data.get("standalone_query", query)
        selected = tuple(str(i) for i in data.get("selected_history_image_ids", []) if i)
        # Web-only fields: absent (defaulted) for the stateless public schema.
        selected_attachments = tuple(
            str(i) for i in data.get("selected_history_attachment_ids", []) if i
        )
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
                metadata_filter=None,
                metadata_filter_source=None,
                metadata_filter_confidence=filter_confidence or "low",
                metadata_filter_evidence=filter_evidence
                if isinstance(filter_evidence, list)
                else None,
                selected_history_image_ids=selected,
                selected_history_attachment_ids=selected_attachments,
            )

        # Handle date fields specially
        metadata_filter: MetadataFilter | None = None
        if clean:
            date_from_str = clean.pop("date_from", None)
            date_to_str = clean.pop("date_to", None)
            try:
                mf = MetadataFilter(**clean)
                if date_from_str:
                    with suppress(ValueError, TypeError):
                        mf.date_from = datetime.fromisoformat(date_from_str)
                if date_to_str:
                    with suppress(ValueError, TypeError):
                        mf.date_to = datetime.fromisoformat(date_to_str)
                metadata_filter = mf if not mf.is_empty() else None
            except Exception:
                logger.warning(
                    "QueryPlanner: invalid filter values for query: %s",
                    log_safe(query, max_length=80),
                )
                metadata_filter = None

        return QueryPlan(
            original_query=query,
            standalone_query=standalone,
            bm25_query=data.get("bm25_query") or None,
            metadata_filter=metadata_filter,
            metadata_filter_source="llm_inferred" if metadata_filter is not None else None,
            metadata_filter_confidence=filter_confidence,
            metadata_filter_evidence=filter_evidence if isinstance(filter_evidence, list) else None,
            selected_history_image_ids=selected,
            selected_history_attachment_ids=selected_attachments,
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
