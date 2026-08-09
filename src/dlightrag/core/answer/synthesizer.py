# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Final answer synthesis.

Receives merged retrieval/evidence contexts from any path and generates the
single final answer with proper citations.  Lives at the RAGServiceManager
level -- shared across all workspaces.

The synthesizer accepts a single ``model_func`` callable that follows the
messages-first interface: it receives ``messages=`` (OpenAI-format list) and an
optional ``stream=`` keyword argument.  Images are inlined as ``image_url``
content blocks so there is no separate VLM path -- the provider decides how to
handle multimodal content.

Both streaming and non-streaming paths use the same freetext system prompt and
the same evidence preparation.  Sources are projected from validated inline
citation markers.  Input packing is bounded by one :class:`AnswerCapacity`:
evidence is at most the capacity evidence ceiling, recent history is retained
first, and fixed evidence that cannot fit beside the generation reserve is
rejected rather than silently trimmed.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE, AnswerCapacity
from dlightrag.core.answer.context import AnswerContextPacker
from dlightrag.core.answer.errors import AnswerInputOverflowError, CurrentImagePayloadError
from dlightrag.core.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.prompts import answer_core
from dlightrag.utils.tokens import estimate_content_tokens, estimate_messages_tokens

logger = logging.getLogger(__name__)

NO_CONTEXT_DISCLAIMER = (
    "**General Knowledge Notice:** The answer below is NOT grounded in your knowledge base."
)


@dataclass
class _PreparedAnswerPrompt:
    contexts: RetrievalContexts
    user_prompt: str
    kg_context: str
    indexer: CitationIndexer
    chunk_image_blocks: dict[str, dict[str, Any]]
    trace: dict[str, Any]


@dataclass
class _PreparedModelCall:
    contexts: RetrievalContexts
    messages: list[dict[str, Any]]
    indexer: CitationIndexer
    trace: dict[str, Any]
    no_context: bool


class AnswerSynthesizer:
    """Mode-agnostic final answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first interface.
    Images found in chunks are inlined as ``image_url`` content blocks -- no
    separate VLM routing is needed.

    Both ``generate()`` and ``generate_stream()`` use the same unified freetext
    system prompt and identical evidence preparation.  Sources are projected
    from validated inline ``[n]`` and ``[n-m]`` markers.
    """

    def __init__(
        self,
        *,
        image_max_pixels: int,
        model_func: Callable[..., Any] | None = None,
        effective_max_images: int = 0,
        image_max_bytes: int = 3_000_000,
        image_max_total_bytes: int = 24_000_000,
        image_max_px: int = 1536,
        image_min_px: int = 1024,
        image_quality: int = 89,
        image_min_quality: int = 79,
        context_window_tokens: int = 260_000,
    ) -> None:
        self.model_func = model_func
        self._effective_max_images = effective_max_images
        self._image_max_bytes = image_max_bytes
        self._image_max_total_bytes = image_max_total_bytes
        self._image_max_pixels = image_max_pixels
        self._image_max_px = image_max_px
        self._image_min_px = image_min_px
        self._image_quality = image_quality
        self._image_min_quality = image_min_quality
        self._capacity = AnswerCapacity(max(1, context_window_tokens))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> RetrievalResult:
        """Non-streaming final answer generation.

        Returns a :class:`RetrievalResult` with ``answer``, ``contexts``,
        ``references``, and one stable ``warnings`` list populated.  Uses the
        same freetext prompt as streaming; references are derived from validated
        inline markers.

        ``query_images`` are user-attached ``image_url`` content blocks inlined
        ahead of the retrieved-document section, letting the model see the
        user's input images in addition to retrieved chunks.
        """
        collected_warnings = list(warnings or [])
        if self.model_func is None:
            logger.info("[AS] generate: no model_func available, returning None answer")
            return RetrievalResult(answer=None, contexts=contexts, warnings=collected_warnings)

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            query_images=query_images,
            history_images=history_images,
            conversation_history=conversation_history,
        )

        logger.info(
            "[AS] generate: input_chunks=%d packed_chunks=%d images_sent=%d "
            "images_skipped=%d query=%s",
            len(contexts.get("chunks", [])),
            prepared.trace["answer_context_chunks"],
            prepared.trace["answer_context_images_sent"],
            prepared.trace["answer_context_images_skipped"],
            query[:60],
        )

        raw = await self.model_func(messages=prepared.messages)
        if prepared.no_context:
            raw = _prepend_no_context_disclaimer(str(raw))

        # Extract references programmatically from validated inline markers,
        # not from model-generated reference-section text.
        from dlightrag.citations import finalize_answer

        finalized = finalize_answer(raw, prepared.contexts, indexer=prepared.indexer)

        from dlightrag.core.answer.media import (
            answer_blocks_from_markdown,
            answer_images_from_sources,
        )
        from dlightrag.models.schemas import Reference

        references = [Reference(id=s.id, title=s.title or "Source") for s in finalized.sources]
        answer_images = answer_images_from_sources(finalized.sources, contexts=prepared.contexts)

        return RetrievalResult(
            answer=finalized.answer,
            contexts=prepared.contexts,
            references=references,
            sources=finalized.sources,
            answer_images=answer_images,
            answer_blocks=answer_blocks_from_markdown(finalized.answer, answer_images),
            trace=prepared.trace,
            warnings=collected_warnings,
        )

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming final answer generation.

        Uses the same freetext prompt and identical evidence preparation as
        ``generate()``.  Wraps the token stream with :class:`AnswerStream` for
        post-stream citation index validation and exposes one stable
        ``warnings`` list on the returned stream.
        """
        collected_warnings = list(warnings or [])
        if self.model_func is None:
            logger.info("[AS] generate_stream: no model_func, returning None")
            return contexts, None

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            query_images=query_images,
            history_images=history_images,
            conversation_history=conversation_history,
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

        token_iterator = await self.model_func(messages=prepared.messages, stream=True)
        if prepared.no_context:
            token_iterator = _prepend_no_context_stream(token_iterator)

        if hasattr(token_iterator, "__aiter__"):
            token_iterator = AnswerStream(token_iterator, indexer=prepared.indexer)
            cast(Any, token_iterator).trace = prepared.trace
            cast(Any, token_iterator).warnings = collected_warnings

        return prepared.contexts, token_iterator

    # ------------------------------------------------------------------
    # Research path: single final owner over a tool transcript
    # ------------------------------------------------------------------

    async def synthesize_research(
        self,
        messages: list[dict[str, Any]],
        contexts: RetrievalContexts,
        *,
        complete: Callable[..., Any],
        indexer: CitationIndexer,
        trace: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> RetrievalResult:
        """Own the tools-disabled final answer for the non-streaming research path.

        The orchestrator supplies the packed tool transcript ``messages`` (which
        carries provider-native reasoning signatures), the ledger's citable
        ``contexts``, and the matching ``indexer``.  This method makes the single
        tools-disabled final call and owns no-context handling, citation
        finalization, warnings, and answer media so both answer branches share
        one final owner.
        """
        from dlightrag.citations import finalize_answer
        from dlightrag.core.answer.media import (
            answer_blocks_from_markdown,
            answer_images_from_sources,
        )
        from dlightrag.models.schemas import Reference

        collected_warnings = list(warnings or [])
        result_trace = dict(trace or {})
        no_context = not _has_research_evidence(contexts)

        raw = await complete(messages=messages)
        text = str(raw)
        if no_context:
            text = _prepend_no_context_disclaimer(text)
            result_trace["answer_no_context"] = True

        finalized = finalize_answer(text, contexts, indexer=indexer)
        answer_images = answer_images_from_sources(finalized.sources, contexts=contexts)
        return RetrievalResult(
            answer=finalized.answer,
            contexts=contexts,
            references=[
                Reference(id=source.id, title=source.title or "Source")
                for source in finalized.sources
            ],
            sources=finalized.sources,
            answer_images=answer_images,
            answer_blocks=answer_blocks_from_markdown(finalized.answer, answer_images),
            trace=result_trace,
            warnings=collected_warnings,
        )

    async def synthesize_research_stream(
        self,
        messages: list[dict[str, Any]],
        contexts: RetrievalContexts,
        *,
        stream: Callable[..., AsyncIterator[str]],
        indexer: CitationIndexer,
        trace: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> tuple[RetrievalContexts, AnswerStream]:
        """Own the tools-disabled final stream for the research path.

        Streaming analogue of :meth:`synthesize_research`.  Wraps the provider's
        native token stream with :class:`AnswerStream` for post-stream citation
        validation and exposes one stable ``warnings`` list plus no-context
        handling, keeping the streaming wrapper under the single final owner.
        """
        collected_warnings = list(warnings or [])
        result_trace = dict(trace or {})
        no_context = not _has_research_evidence(contexts)

        token_iterator: AsyncIterator[str] = stream(messages=messages)
        if no_context:
            token_iterator = _prepend_no_context_stream(token_iterator)
            result_trace["answer_no_context"] = True

        wrapped = AnswerStream(token_iterator, indexer=indexer)
        cast(Any, wrapped).trace = result_trace
        cast(Any, wrapped).warnings = collected_warnings
        cast(Any, wrapped).image_descriptions = {}
        return contexts, wrapped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_model_call(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        query_images: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> _PreparedModelCall:
        original_history = list(conversation_history or [])

        def build(history: list[dict[str, Any]]) -> tuple[_PreparedModelCall, int, int]:
            system_prompt = answer_core()
            budget = self._new_image_budget()
            current_blocks = self._budget_current_images(query_images, budget)
            selected_history_blocks = self._budget_history_images(history_images, budget)
            history_messages, message_history_blocks = self._build_history_messages(history, budget)
            prepared = self._prepare_prompt_context(query, contexts, image_budget=budget)
            no_context = not _has_answer_evidence(
                prepared.contexts,
                query_images=query_images,
                history_images=history_images,
                conversation_history=history,
            )
            if no_context:
                prepared.trace["answer_no_context"] = True
            self._apply_image_trace(
                prepared.trace,
                current_count=len(current_blocks),
                history_count=len(selected_history_blocks) + len(message_history_blocks),
                budget=budget,
            )
            excerpt_blocks = self._build_excerpt_blocks(
                prepared.contexts,
                prepared.indexer,
                image_blocks_by_context_key=prepared.chunk_image_blocks,
            )
            messages = self._compose_user_messages(
                system_prompt,
                prepared.user_prompt,
                excerpt_blocks,
                current_blocks=current_blocks,
                selected_history_blocks=selected_history_blocks,
                history_messages=history_messages,
            )
            evidence_tokens = estimate_content_tokens(excerpt_blocks) + estimate_content_tokens(
                prepared.kg_context
            )
            total_tokens = estimate_messages_tokens(messages)
            call = _PreparedModelCall(
                contexts=prepared.contexts,
                messages=messages,
                indexer=prepared.indexer,
                trace=prepared.trace,
                no_context=no_context,
            )
            return call, evidence_tokens, total_tokens

        kept_history = list(original_history)
        result, evidence_tokens, total_tokens = build(kept_history)
        input_budget = self._capacity.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
        while kept_history and total_tokens > input_budget:
            kept_history = kept_history[_oldest_history_turn_width(kept_history) :]
            result, evidence_tokens, total_tokens = build(kept_history)
        if total_tokens > input_budget:
            raise AnswerInputOverflowError(
                "Fixed answer input does not fit beside the generation reserve: "
                f"{total_tokens} > {input_budget} estimated input tokens"
            )
        overhead_tokens = total_tokens - evidence_tokens
        evidence_ceiling = self._capacity.evidence_ceiling(fixed_input_tokens=overhead_tokens)
        if evidence_tokens > evidence_ceiling:
            raise AnswerInputOverflowError(
                "Fixed answer evidence exceeds the packable evidence ceiling: "
                f"{evidence_tokens} > {evidence_ceiling} estimated evidence tokens"
            )
        result.trace.update(
            {
                "answer_context_window_tokens": self._capacity.context_window_tokens,
                "answer_evidence_tokens": evidence_tokens,
                "answer_evidence_ceiling": evidence_ceiling,
                "answer_input_tokens": total_tokens,
                "answer_history_messages_input": len(original_history),
                "answer_history_messages_kept": len(kept_history),
                "answer_history_messages_dropped": len(original_history) - len(kept_history),
            }
        )
        return result

    def _budget_current_images(
        self,
        query_images: list[dict[str, Any]] | None,
        budget: AnswerImageBudget,
    ) -> list[dict[str, Any]]:
        """Reserve budget for current-turn images; raise on any overflow.

        Current images are explicit user input with no silent fallback: if they
        exceed the effective count, or a payload cannot fit the byte/quality
        budget, the request fails and names the offending image.
        """
        current = query_images or []
        if len(current) > self._effective_max_images:
            raise CurrentImagePayloadError(
                f"{len(current)} current-turn images exceed the effective "
                f"answer-image capacity of {self._effective_max_images}"
            )
        blocks: list[dict[str, Any]] = []
        for idx, img in enumerate(current, start=1):
            label = f"query_image_{idx}"
            block = budget.add_user_image(img, label=label)
            if block is None:
                raise CurrentImagePayloadError(
                    f"current image {label} could not fit the answer image budget"
                )
            blocks.append(block)
        return blocks

    def _build_history_messages(
        self,
        conversation_history: list[dict[str, Any]] | None,
        budget: AnswerImageBudget,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Budget history-turn images into leftover slots, keeping turn text.

        History images that miss a slot are dropped from transport while their
        surrounding text is preserved, so overflow images still contribute
        their stored descriptions.
        """
        history_messages: list[dict[str, Any]] = []
        history_blocks: list[dict[str, Any]] = []
        if not conversation_history:
            return history_messages, history_blocks
        for hmsg in conversation_history:
            hcontent = hmsg.get("content")
            if not isinstance(hcontent, list):
                history_messages.append(hmsg)
                continue
            budgeted: list[Any] = []
            for block in hcontent:
                if isinstance(block, str) or block.get("type") != "image_url":
                    budgeted.append(block)
                    continue
                bounded = budget.add_user_image(block, label=f"history_img_{budget.count + 1}")
                if bounded is not None:
                    budgeted.append(bounded)
                    history_blocks.append(bounded)
            history_messages.append({"role": hmsg["role"], "content": budgeted})
        return history_messages, history_blocks

    @staticmethod
    def _budget_history_images(
        history_images: list[dict[str, Any]] | None,
        budget: AnswerImageBudget,
    ) -> list[dict[str, Any]]:
        """Add planner-selected history pixels best-effort to the shared budget."""
        blocks: list[dict[str, Any]] = []
        for idx, image in enumerate(history_images or [], start=1):
            block = budget.add_user_image(image, label=f"selected_history_image_{idx}")
            if block is not None:
                blocks.append(block)
        return blocks

    def _compose_user_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        excerpt_blocks: list[dict[str, Any]],
        *,
        current_blocks: list[dict[str, Any]],
        selected_history_blocks: list[dict[str, Any]],
        history_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Place budgeted image blocks into the final message structure."""
        content: list[dict[str, Any]] = []
        if current_blocks:
            content.append({"type": "text", "text": "## User-attached images\n"})
            content.extend(current_blocks)
        if selected_history_blocks:
            content.append({"type": "text", "text": "## Referenced conversation images\n"})
            content.extend(selected_history_blocks)
        content.extend(excerpt_blocks)
        content.append({"type": "text", "text": user_prompt})
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _apply_image_trace(
        trace: dict[str, Any],
        *,
        current_count: int,
        history_count: int,
        budget: AnswerImageBudget,
    ) -> None:
        rag_context = int(trace.get("answer_context_images_sent", 0))
        trace["answer_images_current"] = current_count
        trace["answer_images_history"] = history_count
        trace["answer_images_rag"] = rag_context
        trace["answer_images_total"] = current_count + history_count + rag_context
        trace["answer_image_budget_used_bytes"] = budget.used_bytes

    def _new_image_budget(self) -> AnswerImageBudget:
        """Create one fresh transport budget shared by every answer visual lane."""
        return AnswerImageBudget(
            max_images=self._effective_max_images,
            max_total_bytes=self._image_max_total_bytes,
            max_bytes_per_image=self._image_max_bytes,
            max_pixels=self._image_max_pixels,
            max_px=self._image_max_px,
            min_px=self._image_min_px,
            quality=self._image_quality,
            min_quality=self._image_min_quality,
        )

    async def aclose(self) -> None:
        """Release model-function worker resources owned by this synthesizer."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self.model_func)

    def _prepare_prompt_context(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        image_budget: AnswerImageBudget | None = None,
    ) -> _PreparedAnswerPrompt:
        if image_budget is None:
            image_budget = self._new_image_budget()
        packed = AnswerContextPacker().pack(contexts, image_budget=image_budget)
        indexer = self._build_citation_indexer(packed.contexts)
        kg_context = self._format_kg_context(packed.contexts, indexer=indexer)
        user_prompt = "\n\n".join(
            [
                f"## Knowledge Graph Context\n{kg_context}",
                f"## Question\n{query}",
            ]
        )
        trace = dict(packed.trace)
        return _PreparedAnswerPrompt(
            contexts=packed.contexts,
            user_prompt=user_prompt,
            kg_context=kg_context,
            indexer=indexer,
            chunk_image_blocks=packed.image_blocks_by_context_key,
            trace=trace,
        )

    @staticmethod
    def _build_citation_indexer(contexts: RetrievalContexts) -> CitationIndexer:
        """Flatten contexts and build a CitationIndexer."""
        flat: list[dict[str, Any]] = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)
        return indexer

    @staticmethod
    def _format_kg_context(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
    ) -> str:
        """Format entities/relationships as markdown text (max 20 each).

        When *indexer* is provided, each entity/relationship is annotated with
        citation tags derived from its ``source_id``, so the LLM knows which
        document each KG fact originated from.
        """
        return format_kg_context(contexts, indexer)

    @staticmethod
    def _build_excerpt_blocks(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
        image_blocks_by_context_key: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build lane-labelled per-document blocks with interleaved images."""
        chunks = contexts.get("chunks", [])
        if not chunks:
            return []

        composer_chunks: list[dict[str, Any]] = []
        rag_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            source_type = str((chunk.get("metadata") or {}).get("source_type") or "")
            if source_type == "web_attachment":
                composer_chunks.append(chunk)
            else:
                rag_chunks.append(chunk)

        blocks: list[dict[str, Any]] = []
        if composer_chunks:
            blocks.append({"type": "text", "text": "## User-attached documents"})
            blocks.extend(
                build_excerpt_lane_blocks(
                    composer_chunks,
                    indexer=indexer,
                    image_blocks_by_context_key=image_blocks_by_context_key,
                )
            )
        if rag_chunks:
            blocks.append({"type": "text", "text": "## Knowledge-base evidence"})
            blocks.extend(
                build_excerpt_lane_blocks(
                    rag_chunks,
                    indexer=indexer,
                    image_blocks_by_context_key=image_blocks_by_context_key,
                )
            )
        return blocks

    def _build_user_prompt(
        self,
        query: str,
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
    ) -> tuple[str, CitationIndexer]:
        """Combine KG context + question.

        Document excerpts are NOT included in the text prompt because they are
        rendered as interleaved content blocks (with images) by
        :meth:`_build_excerpt_blocks`, which is also where every ``[n]`` and
        ``[n-m]`` marker is defined.
        """
        if indexer is None:
            indexer = self._build_citation_indexer(contexts)
        kg_context = self._format_kg_context(contexts, indexer=indexer)
        prompt_parts = [
            f"## Knowledge Graph Context\n{kg_context}",
            f"## Question\n{query}",
        ]
        return "\n\n".join(prompt_parts), indexer


def _prepend_no_context_disclaimer(answer: str) -> str:
    answer = answer.strip()
    if not answer:
        return NO_CONTEXT_DISCLAIMER
    return f"{NO_CONTEXT_DISCLAIMER}\n\n{answer}"


async def _prepend_no_context_stream(token_iterator: Any) -> AsyncIterator[str]:
    yield f"{NO_CONTEXT_DISCLAIMER}\n\n"
    if isinstance(token_iterator, str):
        yield token_iterator
        return
    if token_iterator is None:
        return
    async for token in token_iterator:
        yield token


def _has_research_evidence(contexts: RetrievalContexts) -> bool:
    """A research answer is grounded when the ledger accumulated any context."""
    return any(contexts.get(key) for key in ("chunks", "entities", "relationships"))


def _has_answer_evidence(
    contexts: RetrievalContexts,
    *,
    query_images: list[dict[str, Any]] | None,
    history_images: list[dict[str, Any]] | None,
    conversation_history: list[dict[str, Any]] | None,
) -> bool:
    if any(contexts.get(key) for key in ("chunks", "entities", "relationships")):
        return True
    if query_images or history_images:
        return True
    if not conversation_history:
        return False
    for message in conversation_history:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "image_url" for block in content
        ):
            return True
    return False


def _oldest_history_turn_width(messages: list[dict[str, Any]]) -> int:
    """Drop a user/assistant pair together when the history shape permits."""
    if len(messages) >= 2:
        first_role = str(messages[0].get("role") or "")
        second_role = str(messages[1].get("role") or "")
        if first_role == "user" and second_role == "assistant":
            return 2
    return 1


__all__ = ["NO_CONTEXT_DISCLAIMER", "AnswerSynthesizer"]
