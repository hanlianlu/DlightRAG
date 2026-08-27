# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Plan one retrieval query: semantic text, lexical text, and metadata scope."""

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.rag.retrieval.models import MetadataFilter
from dlightrag.rag.retrieval.planner_prompt import (
    RETRIEVAL_PLANNER_IMAGE_CONTEXT_GUIDANCE,
    RETRIEVAL_PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

type PlannerMessage = Mapping[str, object]
type PlannerHistory = Sequence[PlannerMessage]


def _convert_history_to_text(history: PlannerHistory | None) -> str:
    """Convert text conversation history to a planner transcript."""
    if not history:
        return ""
    lines: list[str] = []
    for message in history:
        role = message.get("role", "user")
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


RetrievalPlannerOutcome = Literal[
    "planned",
    "fallback_no_model",
    "fallback_provider_error",
    "fallback_invalid_response",
    "planner_input_overflow",
]
RetrievalPlannerFallbackOutcome = Literal[
    "fallback_no_model",
    "fallback_provider_error",
    "fallback_invalid_response",
    "planner_input_overflow",
]


@dataclass
class RetrievalPlan:
    """The semantic, lexical, and metadata inputs one retrieval will execute."""

    standalone_query: str
    bm25_query: str | None = None
    metadata_filter: MetadataFilter | None = None
    metadata_filter_source: str | None = None
    metadata_filter_confidence: str | None = None
    metadata_filter_evidence: list[dict[str, Any]] | None = None
    outcome: RetrievalPlannerOutcome = "planned"

    @classmethod
    def fallback(cls, query: str, outcome: RetrievalPlannerFallbackOutcome) -> RetrievalPlan:
        return cls(standalone_query=query, outcome=outcome)


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
    history: PlannerHistory | None = None,
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


def _planner_messages(system_prompt: str, user_payload: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]


def _planner_request_tokens(system_prompt: str, user_payload: str) -> int:
    return estimate_messages_tokens(_planner_messages(system_prompt, user_payload))


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
    """Rewrite and analyze one retrieval query in one bounded model call."""

    def __init__(
        self,
        llm_func: Callable[..., Any] | None = None,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
    ) -> None:
        self._llm_func = llm_func
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._input_limit_tokens = context_policy.hard_input_limit(model_profile)

    async def _call_llm(
        self,
        query: str,
        system_prompt: str,
        *,
        structured_output: StructuredOutput = RETRIEVAL_PLAN_STRUCTURED_OUTPUT,
        max_tokens: int | None = None,
    ) -> str:
        llm_func = self._llm_func
        if llm_func is None:
            raise RuntimeError("Query planning requires an LLM function")
        kwargs: dict[str, Any] = {
            "messages": _planner_messages(system_prompt, query),
            "structured_output": structured_output,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return await llm_func(**kwargs)

    def history_input_measure(
        self,
        query: str,
        *,
        schema: dict[str, Any] | None = None,
        current_image_descriptions: list[str] | None = None,
        preserve_query: bool | None = None,
    ) -> Callable[[PlannerHistory], int]:
        """Return the exact planner serializer used by the shared projector."""
        system_prompt, render_input = self._input_renderer(
            query,
            schema=schema,
            current_image_descriptions=current_image_descriptions,
            preserve_query=preserve_query,
        )
        return lambda messages: _planner_request_tokens(system_prompt, render_input(messages))

    def _input_renderer(
        self,
        query: str,
        *,
        schema: dict[str, Any] | None,
        current_image_descriptions: list[str] | None,
        preserve_query: bool | None,
    ) -> tuple[str, Callable[[PlannerHistory], str]]:
        system_prompt = RETRIEVAL_PLANNER_SYSTEM_PROMPT
        if current_image_descriptions:
            system_prompt += "\n\n" + RETRIEVAL_PLANNER_IMAGE_CONTEXT_GUIDANCE
        metadata_schema = _build_schema_context(schema)

        def render(messages: PlannerHistory, schema_text: str) -> str:
            return _planner_user_payload(
                query,
                metadata_schema=schema_text,
                history=messages,
                current_images=current_image_descriptions,
                preserve_query=not bool(messages) if preserve_query is None else preserve_query,
            )

        if metadata_schema:
            fixed_payload = render((), metadata_schema)
            if _planner_request_tokens(system_prompt, fixed_payload) > self._input_limit_tokens:
                metadata_schema = ""
        return system_prompt, lambda messages: render(messages, metadata_schema)

    async def plan(
        self,
        query: str,
        *,
        conversation_history: PlannerHistory | None = None,
        schema: dict[str, Any] | None = None,
        current_image_descriptions: list[str] | None = None,
        preserve_query: bool | None = None,
    ) -> RetrievalPlan:
        """Produce one bounded retrieval plan from one model call."""
        if self._llm_func is None:
            return RetrievalPlan.fallback(query, "fallback_no_model")

        history = tuple(conversation_history or ())
        system_prompt, render_input = self._input_renderer(
            query,
            schema=schema,
            current_image_descriptions=current_image_descriptions,
            preserve_query=preserve_query,
        )

        def fit_to_envelope() -> tuple[str, int, bool]:
            payload = render_input(history)
            input_tokens = _planner_request_tokens(system_prompt, payload)
            return payload, input_tokens, input_tokens <= self._input_limit_tokens

        planner_input, input_tokens, fits_envelope = await asyncio.to_thread(fit_to_envelope)
        if not fits_envelope:
            return RetrievalPlan.fallback(query, "planner_input_overflow")
        max_tokens = self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )
        response = await self._call_llm_with_retry(
            planner_input,
            system_prompt,
            structured_output=RETRIEVAL_PLAN_STRUCTURED_OUTPUT,
            start_time=time.monotonic(),
            max_tokens=max_tokens,
        )
        if response is None:
            return RetrievalPlan.fallback(query, "fallback_provider_error")
        plan = self._parse_response(
            response,
            query,
            structured_output=RETRIEVAL_PLAN_STRUCTURED_OUTPUT,
        )
        if plan is None:
            return RetrievalPlan.fallback(query, "fallback_invalid_response")
        if preserve_query is True or (preserve_query is None and not history):
            plan.standalone_query = query

        logger.info(
            "[Planner] result: standalone=%r, bm25_query=%r, filter_source=%s, "
            "filter_confidence=%s, filter_evidence=%s, filter=%s",
            safe_log_text(plan.standalone_query, max_length=60),
            safe_log_text(plan.bm25_query),
            safe_log_text(plan.metadata_filter_source),
            safe_log_text(plan.metadata_filter_confidence),
            safe_log_text(_format_filter_evidence(plan.metadata_filter_evidence)),
            safe_log_text(plan.metadata_filter),
        )
        return plan

    async def _call_llm_with_retry(
        self,
        query: str,
        system_prompt: str,
        *,
        structured_output: StructuredOutput,
        start_time: float,
        max_tokens: int | None,
    ) -> str | None:
        _MAX_RETRIES = 2
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._call_llm(
                    query,
                    system_prompt,
                    structured_output=structured_output,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "[Planner] LLM call: %.1fs (attempt %d)",
                    time.monotonic() - start_time,
                    attempt,
                )
                return response
            except Exception:
                if attempt < _MAX_RETRIES:
                    delay = 2**attempt
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
        if not response:
            return None
        try:
            parsed = structured_output.parse(response)
        except ValidationError, ValueError, TypeError:
            logger.warning(
                "RetrievalPlanner: invalid structured output for query: %s",
                safe_log_text(query, max_length=80),
            )
            return None

        data = parsed.model_dump()
        standalone = data.get("standalone_query", query)
        raw_filters = data.get("filters", {}) or {}
        filter_confidence = str(data.get("filter_confidence") or "").lower() or None
        filter_evidence = data.get("filter_evidence") or []
        clean = {key: value for key, value in raw_filters.items() if value is not None}
        if clean and filter_confidence == "low":
            logger.info("RetrievalPlanner: ignored low-confidence metadata filter proposal")
            return RetrievalPlan(
                standalone_query=standalone,
                bm25_query=data.get("bm25_query") or None,
                metadata_filter=None,
                metadata_filter_source=None,
                metadata_filter_confidence=filter_confidence or "low",
                metadata_filter_evidence=(
                    filter_evidence if isinstance(filter_evidence, list) else None
                ),
            )

        metadata_filter: MetadataFilter | None = None
        if clean:
            creation_date_from_str = clean.pop("creation_date_from", None)
            creation_date_to_str = clean.pop("creation_date_to", None)
            try:
                if creation_date_from_str:
                    with suppress(ValueError, TypeError):
                        clean["creation_date_from"] = datetime.fromisoformat(creation_date_from_str)
                if creation_date_to_str:
                    with suppress(ValueError, TypeError):
                        clean["creation_date_to"] = datetime.fromisoformat(creation_date_to_str)
                candidate = MetadataFilter(**clean)
                metadata_filter = candidate if not candidate.is_empty() else None
            except Exception:
                logger.warning(
                    "RetrievalPlanner: invalid filter values for query: %s",
                    safe_log_text(query, max_length=80),
                )

        return RetrievalPlan(
            standalone_query=standalone,
            bm25_query=data.get("bm25_query") or None,
            metadata_filter=metadata_filter,
            metadata_filter_source="llm_inferred" if metadata_filter is not None else None,
            metadata_filter_confidence=filter_confidence,
            metadata_filter_evidence=(
                filter_evidence if isinstance(filter_evidence, list) else None
            ),
        )


__all__ = [
    "RetrievalPlan",
    "RetrievalPlanner",
]
