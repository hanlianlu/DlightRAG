# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized answer generation engine.

Receives merged retrieval contexts from any backend and generates answers
with proper citations.  Lives at the RAGServiceManager level -- shared across
all workspaces.

The engine accepts a single ``model_func`` callable that follows the
messages-first interface: it receives ``messages=`` (OpenAI-format list)
and optional ``stream=`` keyword arguments.  Images are inlined as
``image_url`` content blocks so there is no separate VLM path -- the
provider decides how to handle multimodal content.

Both streaming and non-streaming paths use the same freetext system prompt.
Sources are projected from validated inline citation markers.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.answer.context import AnswerContextPacker
from dlightrag.core.answer.errors import CurrentImagePayloadError
from dlightrag.core.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.prompts import answer_core
from dlightrag.utils.tokens import estimate_messages_tokens, truncate_conversation_history

logger = logging.getLogger(__name__)

NO_CONTEXT_DISCLAIMER = (
    "**General Knowledge Notice:** The answer below is NOT grounded in your knowledge base."
)
ANSWER_INPUT_TOKEN_ENVELOPE = 102_400


class AnswerInputOverflowError(RuntimeError):
    """Fixed answer evidence exceeds the configured model-input envelope."""


@dataclass
class _PreparedAnswerPrompt:
    contexts: RetrievalContexts
    user_prompt: str
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


class AnswerEngine:
    """Mode-agnostic answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first
    interface.  Images found in chunks are inlined as ``image_url``
    content blocks -- no separate VLM routing is needed.

    Both ``generate()`` and ``generate_stream()`` use the same unified
    freetext system prompt.  Sources are projected from validated inline
    ``[n]`` and ``[n-m]`` markers.
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
        context_top_k: int | None = 30,
        input_token_envelope: int = ANSWER_INPUT_TOKEN_ENVELOPE,
        history_token_ceiling: int = 81_920,
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
        self._context_top_k = context_top_k
        self._input_token_envelope = max(1, input_token_envelope)
        self._history_token_ceiling = max(0, history_token_ceiling)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
        separate_composer_visual_budget: bool = False,
        history_images: list[dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        """Non-streaming answer generation.

        Returns a :class:`RetrievalResult` with ``answer``, ``contexts``,
        and ``references`` populated.  Uses the same freetext prompt as
        streaming; references are derived from validated inline markers.

        ``query_images`` are user-attached ``image_url`` content blocks
        inlined ahead of the
        retrieved-document section, letting the model see the user's input
        images in addition to retrieved chunks. Designed for multi-turn chat
        where the user uploads images alongside their question.
        """
        if self.model_func is None:
            logger.info("[AE] generate: no model_func available, returning None answer")
            return RetrievalResult(answer=None, contexts=contexts)

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            query_images=query_images,
            history_images=history_images,
            conversation_history=conversation_history,
            context_top_k=context_top_k,
            separate_composer_visual_budget=separate_composer_visual_budget,
        )

        logger.info(
            "[AE] generate: input_chunks=%d packed_chunks=%d target=%s images_sent=%d "
            "images_skipped=%d query=%s",
            len(contexts.get("chunks", [])),
            prepared.trace["answer_context_chunks"],
            prepared.trace["answer_context_target_chunks"],
            prepared.trace["answer_context_images_sent"],
            prepared.trace["answer_context_images_skipped"],
            query[:60],
        )

        raw = await self.model_func(messages=prepared.messages)
        if prepared.no_context:
            raw = _prepend_no_context_disclaimer(str(raw))

        logger.info(
            "[AE] generate: LLM returned type=%s len=%d first200=%s",
            type(raw).__name__,
            len(raw) if isinstance(raw, str) else -1,
            repr(raw[:200]) if isinstance(raw, str) else repr(raw),
        )

        # Extract references programmatically from validated inline markers,
        # not from model-generated reference-section text.
        from dlightrag.citations import finalize_answer

        finalized = finalize_answer(raw, prepared.contexts, indexer=prepared.indexer)

        logger.info(
            "[AE] generate: parsed sources=%d answer_len=%d",
            len(finalized.sources),
            len(finalized.answer) if finalized.answer else 0,
        )

        # Convert sources to Reference objects for RetrievalResult
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
        )

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
        separate_composer_visual_budget: bool = False,
        history_images: list[dict[str, Any]] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer generation.

        Uses the same freetext prompt as ``generate()``.  Wraps the
        token stream with :class:`AnswerStream` for post-stream citation
        index validation.

        ``query_images`` mirrors ``generate()``: user-attached ``image_url``
        content blocks are inlined before retrieved-document context.
        """
        if self.model_func is None:
            logger.info("[AE] generate_stream: no model_func, returning None")
            return contexts, None

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            query_images=query_images,
            history_images=history_images,
            conversation_history=conversation_history,
            context_top_k=context_top_k,
            separate_composer_visual_budget=separate_composer_visual_budget,
        )

        logger.info(
            "[AE] generate_stream: input_chunks=%d packed_chunks=%d target=%s "
            "images_sent=%d images_skipped=%d query=%s",
            len(contexts.get("chunks", [])),
            prepared.trace["answer_context_chunks"],
            prepared.trace["answer_context_target_chunks"],
            prepared.trace["answer_context_images_sent"],
            prepared.trace["answer_context_images_skipped"],
            query[:60],
        )

        token_iterator = await self.model_func(messages=prepared.messages, stream=True)
        if prepared.no_context:
            token_iterator = _prepend_no_context_stream(token_iterator)

        logger.info(
            "[AE] generate_stream: model_func returned type=%s",
            type(token_iterator).__name__,
        )

        # Always wrap with AnswerStream (passthrough + post-stream citation validation)
        if hasattr(token_iterator, "__aiter__"):
            logger.info("[AE] generate_stream: wrapping with AnswerStream")
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
        query_images: list[dict[str, Any]] | None = None,
        history_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
        separate_composer_visual_budget: bool = False,
    ) -> _PreparedModelCall:
        original_history = list(conversation_history or [])
        bounded_history = truncate_conversation_history(
            original_history,
            max_messages=len(original_history),
            max_tokens=self._history_token_ceiling,
        )

        def build(history: list[dict[str, Any]]) -> _PreparedModelCall:
            system_prompt = answer_core()
            composer_budget = self._new_image_budget()
            rag_budget = (
                self._new_image_budget() if separate_composer_visual_budget else composer_budget
            )
            # Direct/history Composer images and parsed document visuals share
            # composer_budget. RAG visual transport has an independent budget in
            # Web mode and is never allocated from Composer's remainder.
            current_blocks = self._budget_current_images(query_images, composer_budget)
            selected_history_blocks = self._budget_history_images(
                history_images,
                composer_budget,
            )
            history_messages, message_history_blocks = self._build_history_messages(
                history,
                composer_budget,
            )
            prepared = self._prepare_prompt_context(
                query,
                contexts,
                context_top_k=context_top_k,
                composer_image_budget=composer_budget,
                rag_image_budget=rag_budget,
            )
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
                composer_budget=composer_budget,
                rag_budget=rag_budget,
            )
            messages = self._compose_user_messages(
                system_prompt,
                prepared.user_prompt,
                prepared.contexts,
                prepared.indexer,
                current_blocks=current_blocks,
                selected_history_blocks=selected_history_blocks,
                history_messages=history_messages,
                chunk_image_blocks_by_context_key=prepared.chunk_image_blocks,
            )
            return _PreparedModelCall(
                contexts=prepared.contexts,
                messages=messages,
                indexer=prepared.indexer,
                trace=prepared.trace,
                no_context=no_context,
            )

        kept_history = list(bounded_history)
        result = build(kept_history)
        input_tokens = estimate_messages_tokens(result.messages)
        while kept_history and input_tokens > self._input_token_envelope:
            kept_history = kept_history[_oldest_history_turn_width(kept_history) :]
            result = build(kept_history)
            input_tokens = estimate_messages_tokens(result.messages)
        if input_tokens > self._input_token_envelope:
            raise AnswerInputOverflowError(
                "Fixed answer evidence exceeds the input envelope: "
                f"{input_tokens} > {self._input_token_envelope} estimated tokens"
            )
        result.trace.update(
            {
                "answer_input_token_envelope": self._input_token_envelope,
                "answer_input_tokens": input_tokens,
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
        """Add planner-selected history pixels best-effort to the Composer budget."""
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
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None,
        *,
        current_blocks: list[dict[str, Any]],
        selected_history_blocks: list[dict[str, Any]],
        history_messages: list[dict[str, Any]],
        chunk_image_blocks_by_context_key: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Place budgeted image blocks into the final message structure."""
        content: list[dict[str, Any]] = []
        if current_blocks:
            content.append({"type": "text", "text": "## User-attached images\n"})
            content.extend(current_blocks)
        if selected_history_blocks:
            content.append({"type": "text", "text": "## Referenced conversation images\n"})
            content.extend(selected_history_blocks)
        content.extend(
            self._build_excerpt_blocks(
                contexts,
                indexer,
                image_blocks_by_context_key=chunk_image_blocks_by_context_key,
            )
        )
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
        composer_budget: AnswerImageBudget,
        rag_budget: AnswerImageBudget,
    ) -> None:
        composer_context = int(trace.get("answer_context_composer_images_sent", 0))
        rag_context = int(trace.get("answer_context_rag_images_sent", 0))
        trace["answer_images_current"] = current_count
        trace["answer_images_history"] = history_count
        trace["answer_images_composer"] = composer_context
        trace["answer_images_rag"] = rag_context
        trace["answer_images_total"] = (
            current_count + history_count + composer_context + rag_context
        )
        if composer_budget is rag_budget:
            trace["answer_image_budget_used_bytes"] = composer_budget.used_bytes
        else:
            trace["answer_composer_image_budget_used_bytes"] = composer_budget.used_bytes
            trace["answer_rag_image_budget_used_bytes"] = rag_budget.used_bytes
            trace["answer_image_budget_used_bytes"] = (
                composer_budget.used_bytes + rag_budget.used_bytes
            )

    def _new_image_budget(self) -> AnswerImageBudget:
        """Create one fresh transport budget for a single visual lane."""
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
        """Release model-function worker resources owned by this engine."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self.model_func)

    def _prepare_prompt_context(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        context_top_k: int | None = None,
        composer_image_budget: AnswerImageBudget | None = None,
        rag_image_budget: AnswerImageBudget | None = None,
    ) -> _PreparedAnswerPrompt:
        if composer_image_budget is None:
            composer_image_budget = self._new_image_budget()
        if rag_image_budget is None:
            rag_image_budget = composer_image_budget
        effective_context_top_k = self._context_top_k if context_top_k is None else context_top_k
        packed = AnswerContextPacker().pack(
            contexts,
            composer_image_budget=composer_image_budget,
            rag_image_budget=rag_image_budget,
            context_top_k=effective_context_top_k,
        )
        user_prompt, indexer = self._build_user_prompt(query, packed.contexts)
        trace = dict(packed.trace)
        return _PreparedAnswerPrompt(
            contexts=packed.contexts,
            user_prompt=user_prompt,
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

        When *indexer* is provided, each entity/relationship is annotated
        with citation tags derived from its ``source_id``, so the LLM
        knows which document each KG fact originated from.
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

        Document excerpts are NOT included in the text prompt because
        they are now rendered as interleaved content blocks (with images)
        by :meth:`_build_excerpt_blocks`, which is also where every
        ``[n]`` and ``[n-m]`` marker is defined.

        Returns the prompt string **and** the indexer so that
        :meth:`_build_excerpt_blocks` can label inline images with their
        ``[n-m]`` citation markers.
        """
        # Build indexer first so KG context includes citation tags
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


__all__ = ["AnswerEngine", "AnswerInputOverflowError"]
