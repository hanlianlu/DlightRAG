# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized answer generation engine.

Receives merged retrieval contexts from any backend and generates answers
with proper citations. Lives at the RAGServiceManager level — shared across
all workspaces.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
from dlightrag.models.llm import provider_supports_structured_vision
from dlightrag.models.schemas import StructuredAnswer
from dlightrag.models.streaming import AnswerStream, StreamingAnswerParser
from dlightrag.unifiedrepresent.prompts import get_answer_system_prompt
from dlightrag.utils.logging import log_answer_llm_output, log_references

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Mode-agnostic answer generator with citation support.

    Inspects chunks for ``image_data`` to route between VLM (multimodal)
    and LLM (text-only) paths. Uses ``provider_supports_structured_vision``
    to auto-detect whether to request structured JSON output.
    """

    def __init__(
        self,
        *,
        llm_model_func: Callable[..., Any] | None = None,
        vision_model_func: Callable[..., Any] | None = None,
        provider: str = "openai",
    ) -> None:
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.provider = provider

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
        and ``references`` populated.
        """
        has_images = self._has_images(contexts)
        model_func = self._select_model_func(has_images)

        if model_func is None:
            return RetrievalResult(answer=None, contexts=contexts)

        structured = provider_supports_structured_vision(self.provider)
        system_prompt = get_answer_system_prompt(structured=structured)
        user_prompt = self._build_user_prompt(query, contexts)

        log_answer_llm_output(
            "answer_engine.generate",
            structured=structured,
            provider=self.provider,
            query=query,
        )

        if has_images:
            messages = self._build_vlm_messages(
                system_prompt, user_prompt, contexts["chunks"]
            )
            if structured:
                raw = await model_func(
                    user_prompt,
                    messages=messages,
                    response_schema=StructuredAnswer,
                )
            else:
                raw = await model_func(user_prompt, messages=messages)
        else:
            # Text-only LLM path
            if structured:
                raw = await model_func(
                    user_prompt,
                    system_prompt=system_prompt,
                )
            else:
                raw = await model_func(
                    user_prompt,
                    system_prompt=system_prompt,
                )

        # Parse response
        result = self._parse_response(raw, structured, query)

        log_references(
            "answer_engine.generate",
            result.references,
            query=query,
            provider=self.provider,
            structured=structured,
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

        Returns ``(contexts, async_iterator)`` where the iterator yields
        answer text tokens. If no model funcs are available, returns
        ``(contexts, None)``.
        """
        has_images = self._has_images(contexts)
        model_func = self._select_model_func(has_images)

        if model_func is None:
            return contexts, None

        structured = provider_supports_structured_vision(self.provider)
        system_prompt = get_answer_system_prompt(structured=structured)
        user_prompt = self._build_user_prompt(query, contexts)

        log_answer_llm_output(
            "answer_engine.generate_stream",
            structured=structured,
            provider=self.provider,
            query=query,
        )

        if has_images:
            messages = self._build_vlm_messages(
                system_prompt, user_prompt, contexts["chunks"]
            )
            token_iterator = await model_func(
                user_prompt,
                messages=messages,
                stream=True,
                response_schema=StructuredAnswer if structured else None,
            )
        else:
            token_iterator = await model_func(
                user_prompt,
                system_prompt=system_prompt,
                stream=True,
            )

        if structured and hasattr(token_iterator, "__aiter__"):
            parser = StreamingAnswerParser()
            token_iterator = AnswerStream(token_iterator, parser)

        return contexts, token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_images(contexts: RetrievalContexts) -> bool:
        """Check if any chunk contains image data."""
        return any(
            chunk.get("image_data")
            for chunk in contexts.get("chunks", [])
        )

    def _select_model_func(self, has_images: bool) -> Callable[..., Any] | None:
        """Select appropriate model function based on content type."""
        if has_images:
            return self.vision_model_func or self.llm_model_func
        return self.llm_model_func

    @staticmethod
    def _build_vlm_messages(
        system_prompt: str,
        user_prompt: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format multimodal messages with inline base64 images."""
        content: list[dict[str, Any]] = []
        for item in chunks:
            img_data = item.get("image_data")
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

        Structured mode: parse JSON. On failure, degrade to raw text with
        empty references.
        Freetext mode: wrap raw text directly.
        """
        if structured:
            log_answer_llm_output(
                "answer_engine.generate",
                structured=structured,
                provider=self.provider,
                query=query,
                raw=raw,
            )
            try:
                return StructuredAnswer.model_validate_json(raw)
            except Exception as e:
                log_answer_llm_output(
                    "answer_engine.generate",
                    structured=structured,
                    provider=self.provider,
                    query=query,
                    raw=raw,
                    parse_error=e,
                )
                logger.warning("Structured answer parse failed, degrading to raw text")
                return StructuredAnswer(answer=raw, references=[])
        else:
            log_answer_llm_output(
                "answer_engine.generate",
                structured=structured,
                provider=self.provider,
                query=query,
                answer_text=raw,
            )
            return StructuredAnswer(answer=raw, references=[])


__all__ = ["AnswerEngine"]
