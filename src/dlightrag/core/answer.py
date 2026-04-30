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
References are extracted from the ``### References`` section in the LLM
output.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.streaming import AnswerStream
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.prompts import get_answer_system_prompt

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Mode-agnostic answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first
    interface.  Images found in chunks are inlined as ``image_url``
    content blocks -- no separate VLM routing is needed.

    Both ``generate()`` and ``generate_stream()`` use the same unified
    freetext system prompt.  References are always extracted from the
    ``### References`` markdown section.
    """

    def __init__(
        self,
        *,
        model_func: Callable[..., Any] | None = None,
    ) -> None:
        self.model_func = model_func

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[str | dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        """Non-streaming answer generation.

        Returns a :class:`RetrievalResult` with ``answer``, ``contexts``,
        and ``references`` populated.  Uses the same freetext prompt as
        streaming; references are extracted from ``### References``.

        ``query_images`` are user-attached images (URLs or base64 data URIs)
        inlined as OpenAI ``image_url`` content blocks ahead of the
        retrieved-document section, letting the model see the user's input
        images in addition to retrieved chunks. Designed for multi-turn chat
        where the user uploads images alongside their question.
        """
        if self.model_func is None:
            logger.info("[AE] generate: no model_func available, returning None answer")
            return RetrievalResult(answer=None, contexts=contexts)

        system_prompt = get_answer_system_prompt()
        user_prompt, indexer = self._build_user_prompt(query, contexts)
        messages = self._build_messages(
            system_prompt,
            user_prompt,
            contexts,
            indexer=indexer,
            query_images=query_images,
        )

        logger.info(
            "[AE] generate: chunks=%d query=%s",
            len(contexts.get("chunks", [])),
            query[:60],
        )

        raw = await self.model_func(messages=messages)

        logger.info(
            "[AE] generate: LLM returned type=%s len=%d first200=%s",
            type(raw).__name__,
            len(raw) if isinstance(raw, str) else -1,
            repr(raw[:200]) if isinstance(raw, str) else repr(raw),
        )

        # Extract references programmatically via CitationProcessor
        # (not from LLM-generated ### References section)
        from dlightrag.citations.processor import CitationProcessor
        from dlightrag.citations.source_builder import build_sources

        chunks = contexts.get("chunks", [])
        sources = build_sources(contexts)
        processor = CitationProcessor(chunks, sources)
        processed = processor.process(raw)

        logger.info(
            "[AE] generate: parsed sources=%d answer_len=%d",
            len(processed.sources),
            len(processed.answer) if processed.answer else 0,
        )

        # Convert sources to Reference objects for RetrievalResult
        from dlightrag.models.schemas import Reference

        references = [
            Reference(id=i + 1, title=s.title or s.path) for i, s in enumerate(processed.sources)
        ]

        return RetrievalResult(
            answer=processed.answer,
            contexts=contexts,
            references=references,
        )

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
        query_images: list[str | dict[str, Any]] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer generation.

        Uses the same freetext prompt as ``generate()``.  Wraps the
        token stream with :class:`AnswerStream` for post-stream citation
        validation via :class:`CitationProcessor`.

        ``query_images`` mirrors ``generate()``: user-attached images
        (URLs or base64 data URIs) are inlined as OpenAI ``image_url``
        content blocks before retrieved-document context.
        """
        if self.model_func is None:
            logger.info("[AE] generate_stream: no model_func, returning None")
            return contexts, None

        system_prompt = get_answer_system_prompt()
        user_prompt, indexer = self._build_user_prompt(query, contexts)
        messages = self._build_messages(
            system_prompt,
            user_prompt,
            contexts,
            indexer=indexer,
            query_images=query_images,
        )

        logger.info(
            "[AE] generate_stream: chunks=%d query=%s",
            len(contexts.get("chunks", [])),
            query[:60],
        )

        token_iterator = await self.model_func(messages=messages, stream=True)

        logger.info(
            "[AE] generate_stream: model_func returned type=%s",
            type(token_iterator).__name__,
        )

        # Always wrap with AnswerStream (passthrough + post-stream citation validation)
        if hasattr(token_iterator, "__aiter__"):
            logger.info("[AE] generate_stream: wrapping with AnswerStream")
            token_iterator = AnswerStream(token_iterator, indexer=indexer)

        return contexts, token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
        query_images: list[str | dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with per-document grouped images.

        Instead of flat-listing all images at the start, images are
        interleaved with their document's text excerpts so the LLM can
        associate each image with its source context.

        When ``query_images`` is non-empty, those user-attached images are
        prepended (with a clear section header) so the LLM sees them
        before retrieved-document context. Any item that already looks
        like an OpenAI ``image_url`` block dict is passed through verbatim;
        plain strings are wrapped as ``{"type": "image_url", "image_url":
        {"url": s}}`` so callers can pass either URLs or pre-built blocks.
        """
        content: list[dict[str, Any]] = []

        if query_images:
            content.append({"type": "text", "text": "## User-attached images\n"})
            for img in query_images:
                if isinstance(img, dict) and img.get("type") == "image_url":
                    content.append(img)
                else:
                    content.append({"type": "image_url", "image_url": {"url": img}})

        content.extend(self._build_excerpt_blocks(contexts, indexer))

        content.append({"type": "text", "text": user_prompt})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

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
            doc_groups.setdefault(ref_id, []).append(chunk)

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

            header = f"### Document [{ref_id}]: {filename}{meta_suffix}"
            blocks.append({"type": "text", "text": header})

            # Per-chunk: image label + image + text content
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

                # Image with enriched label
                if img_data:
                    label = _build_image_label(
                        cite_tag=cite_tag,
                        chunk=chunk,
                        filename=filename,
                    )
                    blocks.append({"type": "text", "text": label})
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_data}"},
                        }
                    )

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

        return blocks

    @staticmethod
    def _format_chunk_excerpts(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
    ) -> str:
        """Format chunk text content for the LLM prompt.

        When *indexer* is provided, each excerpt is labelled with its
        ``[ref_id-chunk_idx]`` citation marker so the LLM can directly
        see which marker corresponds to which content.

        When chunks carry ``metadata`` (injected by _enrich_chunks_with_metadata),
        non-empty fields are appended to the label so the LLM knows the
        document title, author, etc. Fields are dynamic -- any key present
        in metadata is rendered.
        """
        chunks = contexts.get("chunks", [])
        if not chunks:
            return "No document excerpts available."

        parts: list[str] = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            ref_id = str(chunk.get("reference_id", ""))
            chunk_id = chunk.get("chunk_id", "")
            page_idx = chunk.get("page_idx")
            file_path = chunk.get("file_path", "")
            filename = Path(file_path).name if file_path else f"Source {ref_id}"

            # Build citation tag from indexer when available
            cite_tag = ""
            if indexer and ref_id and chunk_id:
                cidx = indexer.get_chunk_idx(ref_id, chunk_id)
                if cidx is not None:
                    cite_tag = f"[{ref_id}-{cidx}] "

            # Build metadata suffix from dynamic fields
            meta = chunk.get("metadata") or {}
            meta_parts: list[str] = []
            for k, v in meta.items():
                if v is not None and str(v).strip():
                    # Render key as human-readable: doc_title -> "title", doc_author -> "author"
                    display_key = k.removeprefix("doc_").replace("_", " ")
                    meta_parts.append(f"{display_key}: {v}")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""

            if cite_tag:
                if page_idx:
                    label = f"{cite_tag}{filename}, Page {page_idx}{meta_suffix}"
                else:
                    label = f"{cite_tag}{filename}{meta_suffix}"
            else:
                if page_idx:
                    label = f"[{filename}, Page {page_idx}{meta_suffix}]"
                else:
                    label = f"[{filename}{meta_suffix}]"
            parts.append(f"{label}\n{content}")

        return "\n\n".join(parts) if parts else "No document excerpts available."

    def _build_user_prompt(
        self, query: str, contexts: RetrievalContexts
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
        indexer = self._build_citation_indexer(contexts)
        kg_context = self._format_kg_context(contexts, indexer=indexer)
        ref_list = indexer.format_reference_list()

        prompt_parts = [
            f"## Knowledge Graph Context\n{kg_context}",
            f"## Reference List\n{ref_list}",
            f"## Question\n{query}",
        ]
        return "\n\n".join(prompt_parts), indexer

    # _parse_response removed: references are now extracted programmatically
    # by CitationProcessor from inline [n]/[n-m] markers, not from
    # LLM-generated ### References sections.


def _build_image_label(
    *,
    cite_tag: str,
    chunk: dict[str, Any],
    filename: str,
) -> str:
    """Build an enriched image label with metadata.

    Uses chunk metadata (doc_title, page_idx) to produce labels like::

        [1-2] "2025 Annual Report" Page 7

    instead of bare ``[1-2] Page image``.
    """
    meta = chunk.get("metadata") or {}
    title = meta.get("doc_title", "")
    page_idx = chunk.get("page_idx")

    label_parts: list[str] = []
    if cite_tag:
        label_parts.append(cite_tag)
    if title:
        label_parts.append(f'"{title}"')
    if page_idx:
        label_parts.append(f"Page {page_idx}")
    elif filename:
        label_parts.append(filename)
    else:
        label_parts.append("Page image")

    return " ".join(label_parts)


__all__ = ["AnswerEngine"]
