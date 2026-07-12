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
from pathlib import Path
from typing import Any, cast

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.answer_context import AnswerContextPacker
from dlightrag.core.answer_images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.prompts import get_answer_system_prompt
from dlightrag.utils.images import image_data_uri

logger = logging.getLogger(__name__)

NO_CONTEXT_DISCLAIMER = (
    "**General Knowledge Notice:** The answer below is NOT grounded in your knowledge base."
)


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
        model_func: Callable[..., Any] | None = None,
        max_images: int = 6,
        max_user_images: int = 3,
        image_max_bytes: int = 3_000_000,
        image_max_total_bytes: int = 24_000_000,
        image_max_px: int = 1536,
        image_min_px: int = 1024,
        image_quality: int = 89,
        image_min_quality: int = 79,
        context_top_k: int | None = 30,
    ) -> None:
        self.model_func = model_func
        self._max_images = max_images
        self._max_user_images = max_user_images
        self._image_max_bytes = image_max_bytes
        self._image_max_total_bytes = image_max_total_bytes
        self._image_max_px = image_max_px
        self._image_min_px = image_min_px
        self._image_quality = image_quality
        self._image_min_quality = image_min_quality
        self._context_top_k = context_top_k

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
            conversation_history=conversation_history,
            context_top_k=context_top_k,
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

        finalized = finalize_answer(raw, prepared.contexts)

        logger.info(
            "[AE] generate: parsed sources=%d answer_len=%d",
            len(finalized.sources),
            len(finalized.answer) if finalized.answer else 0,
        )

        # Convert sources to Reference objects for RetrievalResult
        from dlightrag.core.answer_media import (
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
            conversation_history=conversation_history,
            context_top_k=context_top_k,
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
        conversation_history: list[dict[str, Any]] | None = None,
        context_top_k: int | None = None,
    ) -> _PreparedModelCall:
        system_prompt = get_answer_system_prompt()
        prepared = self._prepare_prompt_context(
            query,
            contexts,
            context_top_k=context_top_k,
        )
        no_context = not _has_answer_evidence(
            prepared.contexts,
            query_images=query_images,
            conversation_history=conversation_history,
        )
        if no_context:
            prepared.trace["answer_no_context"] = True
        messages = self._build_messages(
            system_prompt,
            prepared.user_prompt,
            prepared.contexts,
            indexer=prepared.indexer,
            conversation_history=conversation_history,
            query_images=query_images,
            chunk_image_blocks_by_chunk_id=prepared.chunk_image_blocks,
            trace=prepared.trace,
        )
        return _PreparedModelCall(
            contexts=prepared.contexts,
            messages=messages,
            indexer=prepared.indexer,
            trace=prepared.trace,
            no_context=no_context,
        )

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
        query_images: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        chunk_image_blocks_by_chunk_id: dict[str, dict[str, Any]] | None = None,
        trace: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with independent dual image budgets.

        RAG context images (from ``_prepare_prompt_context``) use a separate
        budget from user images (``query_images`` + ``conversation_history``).
        Within the user budget, current-turn ``query_images`` take priority;
        history images consume remaining slots.
        """
        user_budget = self._new_user_image_budget()

        # ---- Phase 1: current-turn query_images reserve user image slots first ----
        resolved_query_blocks: list[dict[str, Any]] = []
        if query_images:
            for idx, img in enumerate(query_images, start=1):
                block = user_budget.add_user_image(img, label=f"query_image_{idx}")
                if block is not None:
                    resolved_query_blocks.append(block)

        # ---- Phase 2: conversation_history gets leftovers ----
        history_messages: list[dict[str, Any]] = []
        if conversation_history:
            for hmsg in conversation_history:
                hcontent = hmsg.get("content")
                if isinstance(hcontent, list):
                    budgeted: list[Any] = []
                    for block in hcontent:
                        if isinstance(block, str):
                            budgeted.append(block)
                        elif block.get("type") == "text":
                            budgeted.append(block)
                        elif block.get("type") == "image_url":
                            bounded = user_budget.add_user_image(
                                block,
                                label=f"history_img_{user_budget.count + 1}",
                            )
                            if bounded is not None:
                                budgeted.append(bounded)
                        else:
                            budgeted.append(block)
                    history_messages.append({"role": hmsg["role"], "content": budgeted})
                else:
                    history_messages.append(hmsg)

        # ---- Phase 3: RAG context uses its own budget ----
        if chunk_image_blocks_by_chunk_id is None:
            prepared = self._prepare_prompt_context("", contexts)
            contexts = prepared.contexts
            indexer = prepared.indexer
            chunk_image_blocks_by_chunk_id = prepared.chunk_image_blocks

        content: list[dict[str, Any]] = []

        if resolved_query_blocks:
            content.append({"type": "text", "text": "## User-attached images\n"})
            content.extend(resolved_query_blocks)

        content.extend(
            self._build_excerpt_blocks(
                contexts,
                indexer,
                image_blocks_by_chunk_id=chunk_image_blocks_by_chunk_id,
            )
        )

        content.append({"type": "text", "text": user_prompt})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": content})
        if trace is not None:
            trace["answer_user_images_sent"] = user_budget.count
            trace["answer_user_image_budget_used_bytes"] = user_budget.used_bytes
        return messages

    def _new_rag_budget(self) -> AnswerImageBudget:
        """RAG context images — independent from user images."""
        return AnswerImageBudget(
            max_images=self._max_images,
            max_total_bytes=self._image_max_total_bytes,
            max_bytes_per_image=self._image_max_bytes,
            max_px=self._image_max_px,
            min_px=self._image_min_px,
            quality=self._image_quality,
            min_quality=self._image_min_quality,
        )

    def _new_user_image_budget(self) -> AnswerImageBudget:
        """User images (query + history) — independent from RAG context."""
        return AnswerImageBudget(
            max_images=self._max_user_images,
            max_total_bytes=self._image_max_total_bytes // 2,
            max_bytes_per_image=self._image_max_bytes,
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
    ) -> _PreparedAnswerPrompt:
        image_budget = self._new_rag_budget()  # user images handled in _build_messages
        effective_context_top_k = self._context_top_k if context_top_k is None else context_top_k
        packed = AnswerContextPacker().pack(
            contexts,
            image_budget=image_budget,
            context_top_k=effective_context_top_k,
        )
        user_prompt, indexer = self._build_user_prompt(query, packed.contexts)
        trace = dict(packed.trace)
        trace["answer_context_image_budget_count"] = image_budget.count
        trace["answer_context_image_budget_used_bytes"] = image_budget.used_bytes
        return _PreparedAnswerPrompt(
            contexts=packed.contexts,
            user_prompt=user_prompt,
            indexer=indexer,
            chunk_image_blocks=packed.image_blocks_by_chunk_id,
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
        parts: list[str] = []

        entities = contexts.get("entities", [])
        if entities:
            parts.append("## Entities")
            for e in entities[:20]:
                name = e.get("entity_name", "")
                etype = e.get("entity_type", "")
                desc = e.get("description", "")
                cite = ""
                if indexer:
                    tags = indexer.get_doc_tags(e.get("source_id"))
                    if tags:
                        cite = f" (from {', '.join(tags)})"
                parts.append(f"- **{name}** ({etype}): {desc}{cite}")

        rels = contexts.get("relationships", [])
        if rels:
            parts.append("\n## Relationships")
            for r in rels[:20]:
                src = r.get("src_id", "")
                tgt = r.get("tgt_id", "")
                desc = r.get("description", "")
                cite = ""
                if indexer:
                    tags = indexer.get_doc_tags(r.get("source_id"))
                    if tags:
                        cite = f" (from {', '.join(tags)})"
                parts.append(f"- {src} -> {tgt}: {desc}{cite}")

        return "\n".join(parts) if parts else "No knowledge graph context available."

    @staticmethod
    def _build_excerpt_blocks(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
        image_blocks_by_chunk_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build per-document content blocks with interleaved images.

        Groups chunks by ``reference_id`` (document), then for each
        document renders a section header followed by its images and
        text excerpts.  This lets the LLM associate each image with
        its source document rather than seeing a flat image list.

        Returns a list of OpenAI-format content blocks (text + image_url dicts).
        """
        chunks = contexts.get("chunks", [])
        if not chunks:
            return []

        # Group chunks by reference_id, preserving first-seen order
        doc_groups: dict[str, list[dict[str, Any]]] = {}
        doc_order: list[str] = []
        for chunk in chunks:
            ref_id = str(chunk.get("reference_id", ""))
            if ref_id not in doc_groups:
                doc_order.append(ref_id)
                doc_groups[ref_id] = [chunk]
            else:
                doc_groups[ref_id].append(chunk)

        blocks: list[dict[str, Any]] = []
        blocks.append({"type": "text", "text": "## Document Excerpts"})

        for ref_id in doc_order:
            doc_chunks = doc_groups[ref_id]

            # Document section header
            first = doc_chunks[0]
            file_path = first.get("file_path", "")
            filename = Path(file_path).name if file_path else f"Source {ref_id}"

            # Collect document-level metadata from first chunk
            meta = first.get("metadata") or {}
            meta_parts: list[str] = []
            for k, v in meta.items():
                if v is not None and str(v).strip():
                    display_key = k.removeprefix("doc_").replace("_", " ")
                    meta_parts.append(f"{display_key}: {v}")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""

            workspace = indexer.get_doc_workspace(ref_id) if indexer is not None else None
            workspace_label = f" [workspace: {workspace}]" if workspace else ""
            header = f"### Document [{ref_id}]{workspace_label}: {filename}{meta_suffix}"
            blocks.append({"type": "text", "text": header})

            # Per-chunk: image label + image + text content + dynamic metadata
            for chunk in doc_chunks:
                content = chunk.get("content", "").strip()
                chunk_id = chunk.get("chunk_id", "")
                page_idx = chunk.get("page_idx")
                img_data = chunk.get("image_data")

                # Build citation tag
                cite_tag = ""
                if indexer and ref_id and chunk_id:
                    cidx = indexer.get_chunk_idx(ref_id, chunk_id)
                    if cidx is not None:
                        cite_tag = f"[{ref_id}-{cidx}]"

                # Image with enriched label. The label is emitted only when the
                # corresponding image block is actually sent to the answer model.
                if img_data:
                    if image_blocks_by_chunk_id is not None:
                        block = image_blocks_by_chunk_id.get(str(chunk_id))
                    else:
                        block = {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri(img_data)},
                        }
                    if block is not None:
                        label = _build_image_label(
                            cite_tag=cite_tag,
                            chunk=chunk,
                            filename=filename,
                        )
                        blocks.append({"type": "text", "text": label})
                        blocks.append(block)

                # Text excerpt
                if content:
                    if cite_tag:
                        if page_idx:
                            label_line = f"{cite_tag} {filename}, Page {page_idx}"
                        else:
                            label_line = f"{cite_tag} {filename}"
                    else:
                        if page_idx:
                            label_line = f"[{filename}, Page {page_idx}]"
                        else:
                            label_line = f"[{filename}]"
                    blocks.append({"type": "text", "text": f"{label_line}\n{content}"})

                # Dynamic metadata line for non-internal chunk annotations.
                meta_line = _format_chunk_metadata(chunk)
                if meta_line:
                    blocks.append({"type": "text", "text": meta_line})

        return blocks

    def _build_user_prompt(
        self,
        query: str,
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
    ) -> tuple[str, CitationIndexer]:
        """Combine KG context + reference list + question.

        Document excerpts are NOT included in the text prompt because
        they are now rendered as interleaved content blocks (with images)
        by :meth:`_build_excerpt_blocks`.

        Returns the prompt string **and** the indexer so that
        :meth:`_build_messages` can label inline images with their
        ``[n-m]`` citation markers.
        """
        # Build indexer first so KG context includes citation tags
        if indexer is None:
            indexer = self._build_citation_indexer(contexts)
        kg_context = self._format_kg_context(contexts, indexer=indexer)
        ref_list = indexer.format_reference_list()

        prompt_parts = [
            f"## Knowledge Graph Context\n{kg_context}",
            f"## Reference List\n{ref_list}",
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
    conversation_history: list[dict[str, Any]] | None,
) -> bool:
    if any(contexts.get(key) for key in ("chunks", "entities", "relationships")):
        return True
    if query_images:
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


# ── Internal keys & metadata formatting ──────────────────────────────────
# Fields that are internal plumbing — never sent to the LLM context.
# Everything else in the chunk dict auto-surfaces as structured metadata.
_INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "chunk_idx",
        "content",
        "bbox",
        "bm25_profile",
        "distance",
        "file_path",
        "full_doc_id",
        "image_data",
        "image_mime_type",
        "image_url",
        "metadata",
        "page_idx",
        "pipeline_stage",
        "reference_id",
        "relevance_score",
        "rerank_score",
        "score",
        "sidecar",
        "sidecar_location",
        "thumbnail_url",
        "_answer_image_sent",
        "_workspace",
    }
)


def _format_chunk_metadata(
    chunk: dict[str, Any],
    *,
    internal_keys: frozenset[str] = _INTERNAL_KEYS,
) -> str:
    """Serialize non-internal chunk fields into a compact metadata line.

    Returns a string like ``[meta: sidecar.type=drawing, sidecar.id=im-hash-xxx]``
    or an empty string when there are no extra fields.
    """
    extra: dict[str, Any] = {}
    for k, v in chunk.items():
        if k in internal_keys or k.startswith("_"):
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        extra[k] = v

    if not extra:
        return ""

    parts: list[str] = []
    for k, v in extra.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if sv is not None and str(sv).strip():
                    parts.append(f"{k}.{sk}={sv}")
        elif isinstance(v, list):
            items = [str(x) for x in v[:5] if str(x).strip()]
            if len(v) > 5:
                items.append(f"...({len(v)} total)")
            parts.append(f"{k}=[{', '.join(items)}]")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)):
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        else:
            s = str(v).strip()
            if len(s) > 120:
                s = s[:117] + "..."
            parts.append(f"{k}={s}")

    if not parts:
        return ""

    return "[meta: " + ", ".join(parts) + "]"


def _build_image_label(
    *,
    cite_tag: str,
    chunk: dict[str, Any],
    filename: str,
) -> str:
    """Build an enriched image label with sidecar awareness.

    Produces labels like::

        [1-2] "2025 Annual Report" Page 7 (VLM-generated drawing)

    The ``(VLM-generated drawing)`` suffix appears when the chunk's
    sidecar indicates a drawing type, helping the LLM distinguish
    real document images from VLM-generated illustrations.
    """
    meta = chunk.get("metadata") or {}
    title = meta.get("doc_title", "")
    page_idx = chunk.get("page_idx")
    sidecar = chunk.get("sidecar")

    label_parts: list[str] = []
    if cite_tag:
        label_parts.append(cite_tag)
    if title:
        label_parts.append(f'"{title}"')
    if page_idx is not None:
        label_parts.append(f"Page {page_idx}")
    elif filename:
        label_parts.append(filename)
    else:
        label_parts.append("Page image")

    # Annotate VLM-generated drawings so the LLM can distinguish them
    # from actual document photographs/scans.
    if isinstance(sidecar, dict):
        stype = sidecar.get("type", "")
        if stype == "drawing":
            sid = sidecar.get("id", "")
            if sid:
                label_parts.append(f"(VLM drawing: {sid[:24]})")
            else:
                label_parts.append("(VLM-generated drawing)")
        elif stype:
            label_parts.append(f"(sidecar: {stype})")

    return " ".join(label_parts)


__all__ = ["AnswerEngine"]
