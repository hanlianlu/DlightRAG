# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Plan one retrieval query: semantic text, lexical text, and metadata scope."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE, AnswerCapacity
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.models.structured import StructuredOutput
from dlightrag.prompts import (
    RETRIEVAL_PLANNER_IMAGE_CONTEXT_GUIDANCE,
    RETRIEVAL_PLANNER_SYSTEM_PROMPT,
)
from dlightrag.utils import log_safe
from dlightrag.utils.tokens import (
    estimate_tokens,
)

logger = logging.getLogger(__name__)

# Planning packs into the same declared answer context window as control and
# final calls, minus the shared final-generation reserve. There is no separate
# fixed planner envelope.
_DEFAULT_RETRIEVAL_PLANNER_INPUT_TOKEN_ENVELOPE = (
    AnswerCapacity(260_000).context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
)


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


RetrievalPlannerOutcome = Literal[
    "planned",
    "fallback_no_model",
    "fallback_provider_error",
    "fallback_invalid_response",
    "fallback_input_overflow",
]
RetrievalPlannerFallbackOutcome = Literal[
    "fallback_no_model",
    "fallback_provider_error",
    "fallback_invalid_response",
    "fallback_input_overflow",
]


@dataclass
class RetrievalPlan:
    """The semantic, lexical, and metadata inputs one retrieval will execute."""

    standalone_query: str  # Rewritten standalone query (= original if no history)
    bm25_query: str | None = None
    metadata_filter: MetadataFilter | None = None
    metadata_filter_source: str | None = None
    metadata_filter_confidence: str | None = None
    metadata_filter_evidence: list[dict[str, Any]] | None = None
    outcome: RetrievalPlannerOutcome = "planned"

    @classmethod
    def fallback(cls, query: str, outcome: RetrievalPlannerFallbackOutcome) -> RetrievalPlan:
        return cls(
            standalone_query=query,
            outcome=outcome,
        )


class RetrievalFilterEvidence(BaseModel):
    """Evidence attached to one LLM-proposed metadata filter."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: str = ""
    evidence_span: str
    intent_basis: str = ""


class RetrievalFilters(BaseModel):
    """Structured filter proposal emitted by the planner LLM."""

    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    file_extension: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date_from: str | None = None
    creation_date_to: str | None = None
    custom: dict[str, str | int | float | bool | None] | None = None


class RetrievalPlannerResponse(BaseModel):
    """Pydantic schema for planner structured-output calls."""

    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    bm25_query: str | None = None
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    filter_confidence: Literal["high", "low"] = "low"
    filter_evidence: list[RetrievalFilterEvidence] = Field(default_factory=list)


RETRIEVAL_PLAN_STRUCTURED_OUTPUT = StructuredOutput(
    name="retrieval_plan",
    schema=RetrievalPlannerResponse,
)


def _build_schema_section(schema: dict[str, Any] | None) -> str:
    if not schema or not schema.get("filters"):
        return ""
    available = ", ".join(schema["filters"])
    return (
        f"Only these filters hold data here: {available}. "
        "Every other filter field must stay null.\n\n"
    )


def _build_custom_keys_hint(schema: dict[str, Any] | None) -> str:
    if not schema:
        return ""
    custom_keys = schema.get("custom_keys")
    if not custom_keys:
        return ""
    return f"Known custom metadata keys: {', '.join(custom_keys)}\n\n"


def _build_schema_context(schema: dict[str, Any] | None) -> str:
    return (_build_schema_section(schema) + _build_custom_keys_hint(schema)).strip()


def _planner_user_payload(
    query: str,
    *,
    metadata_schema: str | None = None,
    history: list[dict[str, Any]] | None = None,
    current_images: list[str] | None = None,
    preserve_query: bool = False,
) -> str:
    payload: dict[str, Any] = {"query": query}
    if preserve_query:
        payload["preserve_query"] = True
    if metadata_schema:
        payload["metadata_schema"] = metadata_schema
    history_text = _convert_history_to_text(history)
    if history_text:
        payload["conversation_history"] = history_text
    if current_images:
        payload["current_images"] = current_images
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _planner_request_tokens(system_tokens: int, user_payload: str) -> int:
    return system_tokens + estimate_tokens(user_payload)


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


class RetrievalPlanner:
    """Unified query understanding -- rewrite + analyze in one LLM call."""

    def __init__(
        self,
        llm_func: Callable[..., Any] | None = None,
        *,
        input_token_envelope: int = _DEFAULT_RETRIEVAL_PLANNER_INPUT_TOKEN_ENVELOPE,
    ) -> None:
        self._llm_func = llm_func
        self._input_token_envelope = max(1, int(input_token_envelope))

    async def _call_llm(
        self,
        query: str,
        system_prompt: str,
        *,
        structured_output: StructuredOutput = RETRIEVAL_PLAN_STRUCTURED_OUTPUT,
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
        conversation_history: PriorTurns | None = None,
        schema: dict[str, Any] | None = None,
        current_image_descriptions: list[str] | None = None,
    ) -> RetrievalPlan:
        """Produce one retrieval plan from one LLM call.

        Handles conversation rewriting, lexical query derivation, and metadata
        filter extraction in one prompt. Stateless retrieval preserves the
        caller's semantic query; history enables coreference rewriting.
        """

        if self._llm_func is None:
            return RetrievalPlan.fallback(query, "fallback_no_model")

        # Truncate history
        history = conversation_history or PriorTurns()
        preserve_query = not bool(history)

        # Schema is fetched and cached by the service manager, then passed in.
        schema_context = _build_schema_context(schema)
        system_prompt = RETRIEVAL_PLANNER_SYSTEM_PROMPT

        structured_output = RETRIEVAL_PLAN_STRUCTURED_OUTPUT
        if current_image_descriptions:
            system_prompt += "\n\n" + RETRIEVAL_PLANNER_IMAGE_CONTEXT_GUIDANCE

        def render_input(messages: list[dict[str, Any]], metadata_schema: str) -> str:
            return _planner_user_payload(
                query,
                metadata_schema=metadata_schema,
                history=messages,
                current_images=current_image_descriptions,
                preserve_query=preserve_query,
            )

        system_tokens = estimate_tokens(system_prompt)
        envelope = self._input_token_envelope

        def fit_to_envelope() -> tuple[str, bool]:
            """Drop the schema, then the oldest turns, until the request fits."""
            metadata_schema = schema_context
            payload = render_input(history.messages, metadata_schema)
            if metadata_schema and _planner_request_tokens(system_tokens, payload) > envelope:
                metadata_schema = ""
            kept = history.fit(
                envelope,
                lambda messages: _planner_request_tokens(
                    system_tokens, render_input(messages, metadata_schema)
                ),
            )
            payload = render_input(kept, metadata_schema)
            fits = _planner_request_tokens(system_tokens, payload) <= envelope
            return payload, fits

        planner_input, fits_envelope = await asyncio.to_thread(fit_to_envelope)
        if not fits_envelope:
            return RetrievalPlan.fallback(query, "fallback_input_overflow")

        llm_start = time.monotonic()
        response = await self._call_llm_with_retry(
            planner_input,
            system_prompt,
            structured_output=structured_output,
            start_time=llm_start,
        )
        if response is None:
            return RetrievalPlan.fallback(query, "fallback_provider_error")
        plan = self._parse_response(response, query, structured_output=structured_output)
        if plan is None:
            return RetrievalPlan.fallback(query, "fallback_invalid_response")
        if preserve_query:
            plan.standalone_query = query

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
                        "RetrievalPlanner LLM call failed (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "RetrievalPlanner LLM call failed after %d attempts (%.1fs)",
                        _MAX_RETRIES + 1,
                        time.monotonic() - start_time,
                        exc_info=True,
                    )
                    return None
        return None

    def _parse_response(
        self,
        response: str,
        query: str,
        *,
        structured_output: StructuredOutput = RETRIEVAL_PLAN_STRUCTURED_OUTPUT,
    ) -> RetrievalPlan | None:
        """Parse one LLM response into a retrieval plan."""
        if not response:
            return None
        try:
            parsed = structured_output.parse(response)
        except ValidationError, ValueError, TypeError:
            logger.warning(
                "RetrievalPlanner: invalid structured output for query: %s",
                log_safe(query, max_length=80),
            )
            return None

        data = parsed.model_dump()
        standalone = data.get("standalone_query", query)
        raw_filters = data.get("filters", {}) or {}
        filter_confidence = str(data.get("filter_confidence") or "").lower() or None
        filter_evidence = data.get("filter_evidence") or []

        # Validate filters
        clean = {k: v for k, v in raw_filters.items() if v is not None}
        if clean and filter_confidence == "low":
            logger.info("RetrievalPlanner: ignored low-confidence metadata filter proposal")
            return RetrievalPlan(
                standalone_query=standalone,
                bm25_query=data.get("bm25_query") or None,
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
            creation_date_from_str = clean.pop("creation_date_from", None)
            creation_date_to_str = clean.pop("creation_date_to", None)
            try:
                mf = MetadataFilter(**clean)
                if creation_date_from_str:
                    with suppress(ValueError, TypeError):
                        mf.creation_date_from = datetime.fromisoformat(creation_date_from_str)
                if creation_date_to_str:
                    with suppress(ValueError, TypeError):
                        mf.creation_date_to = datetime.fromisoformat(creation_date_to_str)
                metadata_filter = mf if not mf.is_empty() else None
            except Exception:
                logger.warning(
                    "RetrievalPlanner: invalid filter values for query: %s",
                    log_safe(query, max_length=80),
                )
                metadata_filter = None

        return RetrievalPlan(
            standalone_query=standalone,
            bm25_query=data.get("bm25_query") or None,
            metadata_filter=metadata_filter,
            metadata_filter_source="llm_inferred" if metadata_filter is not None else None,
            metadata_filter_confidence=filter_confidence,
            metadata_filter_evidence=filter_evidence if isinstance(filter_evidence, list) else None,
        )
