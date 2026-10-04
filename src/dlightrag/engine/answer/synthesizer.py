# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Final answer synthesis.

Receives merged retrieval/evidence contexts from any path and generates the
single final answer with proper citations. Lives in the Answer domain
level -- shared across all workspaces.

The synthesizer accepts a single ``model_func`` callable that follows the
messages-first interface: it receives ``messages=`` (OpenAI-format list) and an
optional ``stream=`` keyword argument.  Images are inlined as ``image_url``
content blocks so there is no separate VLM path -- the provider decides how to
handle multimodal content.

Both streaming and non-streaming paths use the same freetext system prompt and
the same evidence preparation. Sources are projected from validated inline
citation markers. Input packing is bounded by the answer model's resolved
profile and the current immutable context policy.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast

from dlightrag.engine.agent.context import ContextContribution, ContextProjector
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_content_tokens, estimate_messages_tokens
from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.citations.streaming import AnswerStream, aclose_answer_stream
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.evidence import EvidenceLedger, has_unrepresentable_text
from dlightrag.engine.answer.images import AnswerImageBudget, AnswerImagePolicy
from dlightrag.engine.answer.memory import standing_memory_message
from dlightrag.engine.answer.prompts import clock_line, fast_answer_prompt
from dlightrag.engine.answer.synthesis_context import AnswerContextPacker
from dlightrag.engine.rag.retrieval import RetrievalContexts

logger = logging.getLogger(__name__)

NO_CONTEXT_DISCLAIMER = (
    "**General Knowledge Notice:** The answer below is NOT grounded in your knowledge base."
)


@dataclass
class _PreparedEvidence:
    contexts: RetrievalContexts
    blocks: list[dict[str, Any]]
    indexer: CitationIndexer
    trace: dict[str, Any]


@dataclass
class _PreparedModelCall:
    contexts: RetrievalContexts
    messages: list[dict[str, Any]]
    indexer: CitationIndexer
    trace: dict[str, Any]
    no_context: bool
    max_output_tokens: int | None


class AnswerSynthesizer:
    """Mode-agnostic final answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first interface.
    Current-request and chunk images are inlined as ``image_url`` content blocks
    under one budget -- no separate VLM routing is needed.

    ``generate_stream()`` uses the unified freetext system prompt and identical
    evidence preparation. Sources are projected from validated inline ``[n]``
    and ``[n-m]`` markers.
    """

    def __init__(
        self,
        *,
        image_policy: AnswerImagePolicy,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        model_func: Callable[..., Any] | None = None,
    ) -> None:
        self.model_func = model_func
        self._image_policy = image_policy
        self._model_profile = model_profile
        self._context_policy = context_policy

    def history_input_measure(
        self,
        query: str,
        memory_text: str = "",
        episodic_summary: str = "",
        current_images: list[dict[str, Any]] | None = None,
    ) -> Callable[..., int]:
        """Return the exact zero-evidence final-call serializer for history fitting."""

        def measure(
            history: list[dict[str, Any]],
            projected_summary: str = "",
        ) -> int:
            budget = self._image_policy.new_budget()
            current_image_blocks = self._prepare_current_image_blocks(
                current_images,
                image_budget=budget,
            )
            # Zero evidence renders nothing: the request is its images, clock and question.
            messages = self._compose_user_messages(
                fast_answer_prompt(),
                query,
                [],
                current_image_blocks=current_image_blocks,
                history_messages=history,
                episodic_summary="\n\n".join(
                    part for part in (episodic_summary, projected_summary) if part.strip()
                ),
                memory_text=memory_text,
            )
            return estimate_messages_tokens(messages)

        return measure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
        current_images: list[dict[str, Any]] | None = None,
        image_budget: AnswerImageBudget | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming final answer generation.

        Uses the same freetext prompt and identical evidence preparation as
        ``generate()``. Wraps the token stream with :class:`AnswerStream` for
        post-stream citation index validation.
        """
        if self.model_func is None:
            logger.info("[AS] generate_stream: no model_func, returning None")
            return contexts, None

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            conversation_history=conversation_history,
            memory_text=memory_text,
            current_images=current_images,
            image_budget=image_budget,
        )

        logger.info(
            "[AS] generate_stream: input_chunks=%d packed_chunks=%d images_sent=%d "
            "images_skipped=%d query=%s",
            len(contexts.get("chunks", [])),
            prepared.trace["answer_context_chunks"],
            prepared.trace["answer_context_images_sent"],
            prepared.trace["answer_context_images_skipped"],
            query[:60],
        )

        usage: dict[str, Any] = {}
        prepared.trace["usage"] = usage
        call_kwargs: dict[str, Any] = {
            "messages": prepared.messages,
            "stream": True,
            "usage_holder": usage,
            "model_profile": self._model_profile,
        }
        if prepared.max_output_tokens is not None:
            call_kwargs["max_tokens"] = prepared.max_output_tokens
        token_iterator = await self.model_func(**call_kwargs)
        if prepared.no_context:
            token_iterator = _prepend_no_context_stream(token_iterator)

        if hasattr(token_iterator, "__aiter__"):
            token_iterator = AnswerStream(token_iterator, indexer=prepared.indexer)
            cast(Any, token_iterator).trace = prepared.trace

        return prepared.contexts, token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_model_call(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
        current_images: list[dict[str, Any]] | None = None,
        image_budget: AnswerImageBudget | None = None,
    ) -> _PreparedModelCall:
        prior_turns = conversation_history or PriorTurns()
        original_history = list(prior_turns.messages)

        def build(
            history: list[dict[str, Any]],
            candidate_contexts: RetrievalContexts,
        ) -> tuple[_PreparedModelCall, int, int]:
            # The executor has already admitted current and retained images to
            # the Fast Run budget. Packing trials start at that same ceiling;
            # neither recharge those occurrences nor grant chunks a fresh budget.
            budget = (
                replace(image_budget)
                if image_budget is not None
                else self._image_policy.new_budget()
            )
            current_image_blocks = (
                list(current_images or ())
                if image_budget is not None
                else self._prepare_current_image_blocks(current_images, image_budget=budget)
            )
            evidence = self._prepare_evidence(candidate_contexts, image_budget=budget)
            no_context = not any(
                evidence.contexts.get(key) for key in ("chunks", "entities", "relationships")
            )
            if no_context:
                evidence.trace["answer_no_context"] = True
            self._apply_image_trace(
                evidence.trace,
                budget=budget,
                current_image_count=len(current_images or ()),
            )
            messages = self._compose_user_messages(
                fast_answer_prompt(),
                query,
                evidence.blocks,
                current_image_blocks=current_image_blocks,
                history_messages=history,
                episodic_summary=prior_turns.episodic_summary,
                memory_text=memory_text,
            )
            evidence_tokens = estimate_content_tokens(evidence.blocks)
            total_tokens = estimate_messages_tokens(messages)
            call = _PreparedModelCall(
                contexts=evidence.contexts,
                messages=messages,
                indexer=evidence.indexer,
                trace=evidence.trace,
                no_context=no_context,
                max_output_tokens=None,
            )
            return call, evidence_tokens, total_tokens

        input_limit = self._context_policy.hard_input_limit(self._model_profile)
        retrieved_chunk_count = len(contexts.get("chunks", []))
        result, evidence_tokens, total_tokens = build(original_history, contexts)
        capacity_chunks = [dict(chunk) for chunk in result.contexts.get("chunks", [])]
        base_contexts: RetrievalContexts = {
            key: [dict(item) for item in value] for key, value in result.contexts.items()
        }
        capacity_dropped_chunk_count = 0
        while total_tokens > input_limit and capacity_chunks:
            capacity_chunks.pop()
            capacity_dropped_chunk_count += 1
            candidate_contexts: RetrievalContexts = {
                key: [dict(item) for item in value] for key, value in base_contexts.items()
            }
            candidate_contexts["chunks"] = [dict(chunk) for chunk in capacity_chunks]
            result, evidence_tokens, total_tokens = build(
                original_history,
                candidate_contexts,
            )
        if total_tokens > input_limit:
            raise AnswerInputOverflowError(
                "Answer input exceeds the resolved model input limit after all whole "
                "capacity-adjustable chunks were removed; the retained non-chunk context "
                f"and fixed envelope use {total_tokens} > {input_limit} estimated input tokens"
            )
        admitted_chunk_count = len(result.contexts.get("chunks", []))
        overhead_tokens = total_tokens - evidence_tokens
        evidence_capacity = max(0, input_limit - overhead_tokens)
        result.max_output_tokens = self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=total_tokens,
        )
        result.trace.update(
            {
                "answer_input_limit_tokens": input_limit,
                "context_policy_revision": self._context_policy.revision,
                "answer_evidence_tokens": evidence_tokens,
                "answer_evidence_capacity_tokens": evidence_capacity,
                "answer_input_tokens": total_tokens,
                "answer_history_messages": len(original_history),
                "answer_retrieved_chunk_count": retrieved_chunk_count,
                "answer_capacity_admitted_chunk_count": admitted_chunk_count,
                "answer_capacity_dropped_chunk_count": capacity_dropped_chunk_count,
            }
        )
        if result.max_output_tokens is not None:
            result.trace["answer_output_allowance_tokens"] = result.max_output_tokens
        return result

    def _compose_user_messages(
        self,
        system_prompt: str,
        query: str,
        evidence_blocks: list[dict[str, Any]],
        *,
        current_image_blocks: list[dict[str, Any]] | None = None,
        history_messages: list[dict[str, Any]],
        episodic_summary: str = "",
        memory_text: str = "",
    ) -> list[dict[str, Any]]:
        """Place budgeted image blocks into the final message structure.

        The standing memory block rides as its own user-role message after the
        current request — never inside the system prompt (Pi/Kimi convention).

        The clock is the current request's own line, placed before the question that
        can depend on it: Fast has no tools, so nothing in the environment can tell
        the model what "now" is. It costs one line and no cache — this message is
        fresh every turn and never enters the reusable prefix.
        """
        content: list[dict[str, Any]] = []
        content.extend(current_image_blocks or ())
        content.extend(evidence_blocks)
        content.append({"type": "text", "text": clock_line(datetime.now(UTC))})
        content.append({"type": "text", "text": f"## Question\n{query}"})
        contributions = [
            ContextContribution(
                source="answer.system",
                authority="system",
                messages=({"role": "system", "content": system_prompt},),
            )
        ]
        if episodic_summary:
            contributions.append(
                ContextContribution(
                    source="conversation.episodic",
                    authority="conversation",
                    messages=({"role": "user", "content": episodic_summary},),
                )
            )
        if history_messages:
            contributions.append(
                ContextContribution(
                    source="conversation.tail",
                    authority="conversation",
                    messages=tuple(history_messages),
                )
            )
        contributions.append(
            ContextContribution(
                source="answer.evidence" if evidence_blocks else "answer.question",
                authority="evidence" if evidence_blocks else "user",
                messages=({"role": "user", "content": content},),
            )
        )
        memory_message = standing_memory_message(memory_text)
        if memory_message is not None:
            contributions.append(
                ContextContribution(
                    source="profile.memory",
                    authority="profile",
                    messages=(memory_message,),
                )
            )
        return list(ContextProjector().project(contributions).messages)

    @staticmethod
    def _prepare_current_image_blocks(
        current_images: list[dict[str, Any]] | None,
        *,
        image_budget: AnswerImageBudget,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for index, image in enumerate(current_images or (), start=1):
            bounded = image_budget.add_user_image(image, label=f"current_image_{index}")
            if bounded is None:
                raise CurrentImagePayloadError(
                    f"current image current_image_{index} could not fit the answer image budget"
                )
            blocks.extend(
                (
                    {"type": "text", "text": f"[current image {index}]"},
                    bounded,
                )
            )
        return blocks

    @staticmethod
    def _apply_image_trace(
        trace: dict[str, Any],
        *,
        budget: AnswerImageBudget,
        current_image_count: int,
    ) -> None:
        rag_context = int(trace.get("answer_context_images_sent", 0))
        trace["answer_images_current"] = current_image_count
        trace["answer_images_rag"] = rag_context
        trace["answer_images_total"] = current_image_count + rag_context
        trace["answer_image_budget_used_bytes"] = budget.used_bytes

    @staticmethod
    def _prepare_evidence(
        contexts: RetrievalContexts,
        *,
        image_budget: AnswerImageBudget,
    ) -> _PreparedEvidence:
        """Pack retrieval and render it the way the Evidence ledger renders all evidence.

        Fast and Research share one renderer and one citation identity: the ledger
        numbers documents by first appearance, labels each excerpt from that numbering,
        and its rows are the contexts the answer is finalized against. A retrieved
        row's own reference id ranks documents by frequency, so it is never what the
        model is shown.

        A row the ledger would refuse, because no durable store can hold its text, is
        dropped before packing, so its picture spends none of the image budget.
        """
        admissible: RetrievalContexts = {
            key: [row for row in rows if not has_unrepresentable_text(row)]
            for key, rows in contexts.items()
        }
        packed = AnswerContextPacker().pack(admissible, image_budget=image_budget)
        evidence = EvidenceLedger()
        evidence.add_contexts(packed.contexts)
        blocks, indexer = evidence.render_blocks(packed.image_blocks_by_context_key)
        return _PreparedEvidence(
            contexts=evidence.contexts,
            blocks=blocks,
            indexer=indexer,
            trace=dict(packed.trace),
        )


async def _prepend_no_context_stream(token_iterator: Any) -> AsyncIterator[str]:
    yield f"{NO_CONTEXT_DISCLAIMER}\n\n"
    if isinstance(token_iterator, str):
        yield token_iterator
        return
    if token_iterator is None:
        return
    try:
        async for token in token_iterator:
            yield token
    finally:
        await aclose_answer_stream(token_iterator)


__all__ = ["NO_CONTEXT_DISCLAIMER", "AnswerSynthesizer"]
