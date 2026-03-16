# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized answer generation engine.

Receives merged retrieval contexts from any backend and generates answers
with proper citations.  Lives at the RAGServiceManager level -- shared across
all workspaces.

The engine accepts a single ``model_func`` callable that follows the
messages-first interface: it receives ``messages=`` (OpenAI-format list)
and optional ``response_format=`` / ``stream=`` keyword arguments.  Images
are inlined as ``image_url`` content blocks so there is no separate VLM
path -- the provider decides how to handle multimodal content.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.parser import extract_references
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.models.schemas import StructuredAnswer
from dlightrag.models.streaming import AnswerStream
from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt
from dlightrag.utils.logging import log_answer_llm_output, log_references

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Mode-agnostic answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first
    interface.  Images found in chunks are inlined as ``image_url``
    content blocks -- no separate VLM routing is needed.
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
    ) -> RetrievalResult:
        """Non-streaming answer generation.

        Returns a :class:`RetrievalResult` with ``answer``, ``contexts``,
        and ``references`` populated.  Always requests structured JSON
        output via ``response_format``.
        """
        if self.model_func is None:
            logger.info("[AE] generate: no model_func available, returning None answer")
            return RetrievalResult(answer=None, contexts=contexts)

        system_prompt = get_answer_system_prompt(structured=True)
        user_prompt = self._build_user_prompt(query, contexts)
        messages = self._build_messages(system_prompt, user_prompt, contexts)

        logger.info(
            "[AE] generate: structured=True chunks=%d query=%s",
            len(contexts.get("chunks", [])),
            query[:60],
        )

        log_answer_llm_output(
            "answer_engine.generate",
            structured=True,
            query=query,
        )

        raw = await self.model_func(
            messages=messages,
            response_format=StructuredAnswer,
        )

        logger.info(
            "[AE] generate: LLM returned type=%s len=%d first200=%s",
            type(raw).__name__,
            len(raw) if isinstance(raw, str) else -1,
            repr(raw[:200]) if isinstance(raw, str) else repr(raw),
        )

        # Parse response
        result = self._parse_response(raw, structured=True, query=query)

        logger.info(
            "[AE] generate: parsed refs=%d answer_len=%d",
            len(result.references),
            len(result.answer) if result.answer else 0,
        )

        log_references(
            "answer_engine.generate",
            result.references,
            query=query,
            structured=True,
        )

        return RetrievalResult(
            answer=result.answer,
            contexts=contexts,
            references=result.references,
        )

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming answer generation.

        Always uses freetext prompt since streaming cannot enforce
        ``response_format``.  Wraps the token stream with
        :class:`AnswerStream` for post-stream reference extraction.
        """
        if self.model_func is None:
            logger.info("[AE] generate_stream: no model_func, returning None")
            return contexts, None

        # Always freetext for streaming -- structured prompt is wrong here
        system_prompt = get_answer_system_prompt(structured=False)
        user_prompt = self._build_user_prompt(query, contexts)
        messages = self._build_messages(system_prompt, user_prompt, contexts)

        logger.info(
            "[AE] generate_stream: chunks=%d query=%s",
            len(contexts.get("chunks", [])),
            query[:60],
        )

        log_answer_llm_output(
            "answer_engine.generate_stream",
            structured=False,
            query=query,
        )

        token_iterator = await self.model_func(messages=messages, stream=True)

        logger.info(
            "[AE] generate_stream: model_func returned type=%s",
            type(token_iterator).__name__,
        )

        # Always wrap with AnswerStream (passthrough + post-stream ref extraction)
        if hasattr(token_iterator, "__aiter__"):
            logger.info("[AE] generate_stream: wrapping with AnswerStream")
            token_iterator = AnswerStream(token_iterator)

        return contexts, token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        contexts: RetrievalContexts,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages array with inline images if present."""
        content: list[dict[str, Any]] = []

        # Add images from chunks
        for chunk in contexts.get("chunks", []):
            img_data = chunk.get("image_data")
            if img_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )

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
    def _format_kg_context(contexts: RetrievalContexts) -> str:
        """Format entities/relationships as markdown text (max 20 each)."""
        parts: list[str] = []

        entities = contexts.get("entities", [])
        if entities:
            parts.append("## Entities")
            for e in entities[:20]:
                name = e.get("entity_name", "")
                etype = e.get("entity_type", "")
                desc = e.get("description", "")
                parts.append(f"- **{name}** ({etype}): {desc}")

        rels = contexts.get("relationships", [])
        if rels:
            parts.append("\n## Relationships")
            for r in rels[:20]:
                src = r.get("src_id", "")
                tgt = r.get("tgt_id", "")
                desc = r.get("description", "")
                parts.append(f"- {src} -> {tgt}: {desc}")

        return "\n".join(parts) if parts else "No knowledge graph context available."

    def _build_user_prompt(self, query: str, contexts: RetrievalContexts) -> str:
        """Combine KG context + reference list + question."""
        kg_context = self._format_kg_context(contexts)
        indexer = self._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()

        prompt_parts = [
            f"Knowledge Graph Context:\n{kg_context}",
            f"Reference Document List:\n{ref_list}",
            f"Question: {query}",
        ]
        return "\n\n".join(prompt_parts)

    def _parse_response(
        self,
        raw: str,
        structured: bool,
        query: str,
    ) -> StructuredAnswer:
        """Parse LLM response into a StructuredAnswer.

        Structured mode: parse JSON. On failure, degrade to freetext parsing.
        Freetext mode: extract references from ``### References`` section.
        """
        if structured:
            log_answer_llm_output(
                "answer_engine.generate",
                structured=structured,
                query=query,
                raw=raw,
            )
            try:
                result = StructuredAnswer.model_validate_json(raw)
                logger.info(
                    "[AE] _parse_response: JSON parse OK, refs=%d answer_len=%d",
                    len(result.references),
                    len(result.answer),
                )
                return result
            except Exception as e:
                logger.warning(
                    "[AE] _parse_response: JSON parse FAILED (%s: %s), "
                    "raw_first200=%s — falling back to freetext parsing",
                    type(e).__name__,
                    e,
                    repr(raw[:200]),
                )
                log_answer_llm_output(
                    "answer_engine.generate",
                    structured=structured,
                    query=query,
                    raw=raw,
                    parse_error=e,
                )
                answer_text, refs = extract_references(raw)
                logger.info(
                    "[AE] _parse_response: freetext fallback got refs=%d",
                    len(refs),
                )
                return StructuredAnswer(answer=answer_text, references=refs)
        else:
            log_answer_llm_output(
                "answer_engine.generate",
                structured=structured,
                query=query,
                answer_text=raw,
            )
            logger.info("[AE] _parse_response: freetext mode, extracting refs from raw")
            answer_text, refs = extract_references(raw)
            logger.info(
                "[AE] _parse_response: freetext extracted refs=%d",
                len(refs),
            )
            return StructuredAnswer(answer=answer_text, references=refs)


__all__ = ["AnswerEngine"]
