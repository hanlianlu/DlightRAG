# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerEngine messages-first interface."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.retrieval.protocols import RetrievalContexts

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO"
    "+/p9sAAAAASUVORK5CYII="
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _text_contexts() -> RetrievalContexts:
    """Contexts with text-only chunks (no image_data)."""
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue grew 15%.",
                "page_idx": 3,
            },
        ],
        "entities": [
            {
                "entity_name": "Revenue",
                "entity_type": "Metric",
                "description": "Total revenue",
                "source_id": "c1",
            },
        ],
        "relationships": [],
    }


def _image_contexts() -> RetrievalContexts:
    """Contexts with image data in chunks."""
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/chart.pdf",
                "content": "Chart showing growth",
                "page_idx": 1,
                "image_data": _PNG_B64,
            },
        ],
        "entities": [],
        "relationships": [],
    }


def _multi_doc_contexts() -> RetrievalContexts:
    """Contexts with chunks from multiple documents."""
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue data.",
                "page_idx": 3,
                "image_data": _PNG_B64,
                "metadata": {"doc_title": "2025 Annual Report"},
            },
            {
                "chunk_id": "c2",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Expenses data.",
                "page_idx": 7,
                "image_data": _PNG_B64,
            },
            {
                "chunk_id": "c3",
                "reference_id": "2",
                "file_path": "/docs/other.pdf",
                "content": "Other info.",
                "page_idx": 1,
            },
        ],
        "entities": [],
        "relationships": [],
    }


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerate
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerate:
    """Test non-streaming generate() with unified freetext prompt."""

    @pytest.mark.asyncio
    async def test_generate_with_freetext_response(self) -> None:
        """generate() validates inline citations and returns cited references."""
        raw = "AI is artificial intelligence [1-1].\n\n### References\n- [1] AI Overview"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        contexts = _text_contexts()
        result = await engine.generate("What is AI?", contexts)

        assert "AI is artificial intelligence" in result.answer
        assert "### References" not in result.answer
        assert len(result.references) == 1
        model_func.assert_called_once()
        call_kwargs = model_func.call_args.kwargs
        assert "messages" in call_kwargs
        # Must NOT pass response_format (unified freetext prompt)
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_preserves_citation_reference_ids(self) -> None:
        model_func = AsyncMock(return_value="The other document applies here [2-1].")
        engine = AnswerEngine(model_func=model_func)

        result = await engine.generate("Which document applies?", _multi_doc_contexts())

        assert len(result.references) == 1
        assert result.references[0].id == "2"
        assert result.references[0].title == "other.pdf"

    @pytest.mark.asyncio
    async def test_generate_no_model_func(self) -> None:
        """generate() returns None answer when model_func is None."""
        engine = AnswerEngine(model_func=None)
        contexts: RetrievalContexts = {"chunks": []}
        result = await engine.generate("test", contexts)
        assert result.answer is None
        assert result.contexts is contexts

    @pytest.mark.asyncio
    async def test_generate_with_images(self) -> None:
        """generate() includes images in messages content array."""
        raw = "ok\n\n### References\n- [1] chart.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        contexts = _image_contexts()
        await engine.generate("describe", contexts)

        messages = model_func.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Should have image_url entry + text entries
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)

    @pytest.mark.asyncio
    async def test_query_images_take_budget_before_retrieved_visual_chunks(self) -> None:
        raw = "ok"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func, max_images=1)
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/chart.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                }
            ],
            "entities": [],
            "relationships": [],
        }

        result = await engine.generate("describe this", contexts, query_images=[_PNG_B64])

        assert result.contexts["chunks"] == []
        assert result.trace["answer_context_query_images_sent"] == 1
        assert result.trace["answer_context_images_skipped"] == 1
        assert result.trace["answer_context_skipped_image_only_chunks"] == 1
        messages = model_func.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert sum(1 for item in user_content if item.get("type") == "image_url") == 1
        assert any(
            item.get("type") == "text" and "User-attached images" in item.get("text", "")
            for item in user_content
        )

    @pytest.mark.asyncio
    async def test_generate_no_response_format(self) -> None:
        """generate() must NOT pass response_format (unified freetext prompt)."""
        raw = "Revenue grew 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        await engine.generate("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_returns_answer_packed_contexts(self) -> None:
        """generate() returns the contexts actually sent to the answer model."""
        raw = "answer\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func, max_images=0)
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/figures.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                },
                *_text_contexts()["chunks"],
            ],
            "entities": [],
            "relationships": [],
        }
        result = await engine.generate("q", contexts)

        assert result.contexts is not contexts
        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["c1"]
        assert result.trace["answer_context_skipped_image_only_chunks"] == 1

    @pytest.mark.asyncio
    async def test_generate_limits_final_prompt_contexts(self) -> None:
        """generate() limits final prompt chunks while retrieval can over-fetch."""
        raw = "The first item matters [1-1]."
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func, context_top_k=1)
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/a.pdf",
                    "content": "First candidate.",
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "2",
                    "file_path": "/docs/b.pdf",
                    "content": "Second candidate.",
                },
            ],
            "entities": [],
            "relationships": [],
        }

        result = await engine.generate("query", contexts)

        assert [c["chunk_id"] for c in result.contexts["chunks"]] == ["c1"]
        assert result.trace["answer_context_target_chunks"] == 1
        messages = model_func.call_args.kwargs["messages"]
        user_text = "\n".join(
            item["text"] for item in messages[1]["content"] if item.get("type") == "text"
        )
        assert "First candidate." in user_text
        assert "Second candidate." not in user_text

    @pytest.mark.asyncio
    async def test_generate_strips_model_generated_references_tail(self) -> None:
        """generate() ignores generated References tails and trusts inline markers."""
        raw = "Growth is 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        result = await engine.generate("query", _text_contexts())

        assert "Growth is 15%" in result.answer
        assert "### References" not in result.answer
        assert len(result.references) == 1
        assert result.references[0].title == "report.pdf"


# ---------------------------------------------------------------------------
# TestAnswerEngineStream
# ---------------------------------------------------------------------------


class TestAnswerEngineStream:
    """Test streaming generate_stream() with messages-first interface."""

    @pytest.mark.asyncio
    async def test_generate_stream_wraps_with_answer_stream(self) -> None:
        """generate_stream() wraps async iterator with AnswerStream."""

        async def mock_tokens():
            for token in ["Hello", " world"]:
                yield token

        model_func = AsyncMock(return_value=mock_tokens())
        engine = AnswerEngine(model_func=model_func)

        contexts: RetrievalContexts = {"chunks": []}
        ctx, token_iter = await engine.generate_stream("test", contexts)

        from dlightrag.citations.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)

    @pytest.mark.asyncio
    async def test_generate_stream_no_model_func(self) -> None:
        """generate_stream() returns None when no model_func."""
        engine = AnswerEngine(model_func=None)
        contexts: RetrievalContexts = {"chunks": []}
        ctx, token_iter = await engine.generate_stream("test", contexts)
        assert token_iter is None
        assert ctx is contexts

    @pytest.mark.asyncio
    async def test_generate_stream_returns_packed_contexts_and_tokens(self) -> None:
        """generate_stream() returns answer-packed contexts and consumable tokens."""

        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        contexts = _text_contexts()
        result_contexts, token_iter = await engine.generate_stream("query", contexts)

        assert result_contexts is not contexts
        assert result_contexts["chunks"] == contexts["chunks"]
        assert token_iter is not None
        tokens = [t async for t in token_iter]
        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_generate_stream_no_response_format(self) -> None:
        """generate_stream() must NOT pass response_format."""

        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        await engine.generate_stream("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("stream") is True

    @pytest.mark.asyncio
    async def test_generate_stream_passes_messages(self) -> None:
        """generate_stream() passes messages kwarg."""

        async def mock_stream():
            yield "token"

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        await engine.generate_stream("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "messages" in call_kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_stream_passes_indexer_to_answer_stream(self) -> None:
        """generate_stream() passes indexer to AnswerStream for citation validation."""

        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        _, token_iter = await engine.generate_stream("query", _text_contexts())

        from dlightrag.citations.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)
        assert token_iter._indexer is not None


# ---------------------------------------------------------------------------
# TestBuildMessages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Test _build_messages method."""

    def test_text_only(self) -> None:
        """_build_messages without images returns text content."""
        contexts: RetrievalContexts = {"chunks": [{"content": "text chunk"}]}
        engine = AnswerEngine()
        messages = engine._build_messages("system", "user prompt", contexts)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system"
        assert messages[1]["role"] == "user"
        # user content should be a list with text entries
        user_content = messages[1]["content"]
        # Should contain at least the user prompt text block
        text_blocks = [e for e in user_content if e.get("type") == "text"]
        assert any("user prompt" in e["text"] for e in text_blocks)

    def test_with_images_grouped_by_document(self) -> None:
        """_build_messages groups images by document with section headers."""
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "content": "text",
                    "image_data": _PNG_B64,
                    "file_path": "/docs/report.pdf",
                    "page_idx": 3,
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "1",
                    "content": "more text",
                    "file_path": "/docs/report.pdf",
                    "page_idx": 5,
                },
            ],
        }
        engine = AnswerEngine()
        messages = engine._build_messages("system", "user prompt", contexts)

        user_content = messages[1]["content"]
        image_entries = [e for e in user_content if e.get("type") == "image_url"]
        text_entries = [e for e in user_content if e.get("type") == "text"]

        # One image from chunk c1
        assert len(image_entries) == 1
        assert "base64" in image_entries[0]["image_url"]["url"]

        # Should have Document Excerpts header and document section header
        all_text = " ".join(e["text"] for e in text_entries)
        assert "Document Excerpts" in all_text
        assert "Document [1]" in all_text
        assert "report.pdf" in all_text

    def test_with_images_labelled_by_indexer(self) -> None:
        """_build_messages with indexer labels images with enriched metadata."""
        from dlightrag.citations.indexer import CitationIndexer

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "2",
                    "content": "chart data",
                    "image_data": _PNG_B64,
                    "file_path": "/docs/report.pdf",
                    "page_idx": 7,
                    "metadata": {"doc_title": "2025 Annual Report"},
                },
            ],
        }
        flat = list(contexts["chunks"])
        indexer = CitationIndexer()
        indexer.build_index(flat)

        engine = AnswerEngine()
        messages = engine._build_messages("sys", "prompt", contexts, indexer=indexer)
        user_content = messages[1]["content"]

        # Find image label text
        label_entries = [
            e for e in user_content if e.get("type") == "text" and "[2-1]" in e.get("text", "")
        ]
        assert len(label_entries) >= 1
        # Should have enriched label with metadata
        label_text = label_entries[0]["text"]
        assert '"2025 Annual Report"' in label_text
        assert "Page 7" in label_text

    def test_multiple_images_from_different_docs(self) -> None:
        """_build_messages with images from multiple documents groups them correctly."""
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "content": "a",
                    "image_data": _PNG_B64,
                    "file_path": "/docs/doc1.pdf",
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "2",
                    "content": "b",
                    "image_data": _PNG_B64,
                    "file_path": "/docs/doc2.pdf",
                },
                {
                    "chunk_id": "c3",
                    "reference_id": "2",
                    "content": "c",
                    "file_path": "/docs/doc2.pdf",
                },
            ],
        }
        engine = AnswerEngine()
        messages = engine._build_messages("sys", "prompt", contexts)
        user_content = messages[1]["content"]

        image_entries = [e for e in user_content if e.get("type") == "image_url"]
        assert len(image_entries) == 2

        # Check document grouping headers exist
        all_text = " ".join(e["text"] for e in user_content if e.get("type") == "text")
        assert "Document [1]" in all_text
        assert "Document [2]" in all_text

    def test_empty_chunks(self) -> None:
        """_build_messages with empty chunks returns text-only content."""
        contexts: RetrievalContexts = {"chunks": []}
        engine = AnswerEngine()
        messages = engine._build_messages("sys", "prompt", contexts)
        user_content = messages[1]["content"]
        # Should have at least the user prompt text block
        assert any(e.get("type") == "text" for e in user_content)


# ---------------------------------------------------------------------------
# TestAnswerEngine -- Internal helpers (preserved)
# ---------------------------------------------------------------------------


class TestAnswerEngineHelpers:
    """Test static/internal helper methods."""

    def test_format_kg_context_with_entities_and_rels(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Acme",
                    "entity_type": "Company",
                    "description": "A company",
                    "source_id": "s1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "generates",
                    "source_id": "s1",
                },
            ],
        }
        result = AnswerEngine._format_kg_context(contexts)
        assert "## Entities" in result
        assert "**Acme**" in result
        assert "## Relationships" in result
        assert "Acme -> Revenue" in result

    def test_format_kg_context_empty(self) -> None:
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        result = AnswerEngine._format_kg_context(contexts)
        assert result == "No knowledge graph context available."

    def test_format_kg_context_limits_to_20(self) -> None:
        entities = [
            {
                "entity_name": f"E{i}",
                "entity_type": "Type",
                "description": f"desc{i}",
                "source_id": "s1",
            }
            for i in range(30)
        ]
        contexts: RetrievalContexts = {"chunks": [], "entities": entities, "relationships": []}
        result = AnswerEngine._format_kg_context(contexts)
        lines = [line for line in result.split("\n") if line.strip()]
        entity_lines = [line for line in lines if line.startswith("- **")]
        assert len(entity_lines) == 20

    def test_format_kg_context_includes_doc_level_tags(self) -> None:
        """KG entities/relationships should include doc-level citation tags when indexer is provided."""
        from dlightrag.citations.indexer import CitationIndexer

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Revenue data.",
                    "page_idx": 1,
                },
            ],
            "entities": [
                {
                    "entity_name": "Revenue",
                    "entity_type": "Metric",
                    "description": "Total revenue grew 15%",
                    "source_id": "c1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "reports",
                    "source_id": "c1",
                },
            ],
        }
        flat: list = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)

        result = AnswerEngine._format_kg_context(contexts, indexer=indexer)
        assert "(from [1])" in result
        # Both entity and relationship should have doc-level tag
        lines = result.split("\n")
        entity_line = [ln for ln in lines if "Revenue" in ln and "Metric" in ln][0]
        rel_line = [ln for ln in lines if "Acme -> Revenue" in ln][0]
        assert "(from [1])" in entity_line
        assert "(from [1])" in rel_line

    def test_build_user_prompt_contains_all_parts(self) -> None:
        engine = AnswerEngine()
        contexts = _text_contexts()
        prompt, _indexer = engine._build_user_prompt("What is revenue?", contexts)

        assert "Knowledge Graph Context" in prompt
        assert "Reference List" in prompt
        assert "Question" in prompt
        assert "What is revenue?" in prompt

    def test_build_citation_indexer(self) -> None:
        contexts = _text_contexts()
        indexer = AnswerEngine._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()
        assert "report.pdf" in ref_list


# ---------------------------------------------------------------------------
# TestBuildExcerptBlocks
# ---------------------------------------------------------------------------


class TestBuildExcerptBlocks:
    """Test _build_excerpt_blocks for document-grouped image interleaving."""

    def test_groups_chunks_by_document(self) -> None:
        """Chunks from the same document appear under the same header."""
        contexts = _multi_doc_contexts()
        from dlightrag.citations.indexer import CitationIndexer

        indexer = CitationIndexer()
        flat = list(contexts["chunks"])
        indexer.build_index(flat)

        blocks = AnswerEngine._build_excerpt_blocks(contexts, indexer=indexer)

        text_blocks = [b for b in blocks if b.get("type") == "text"]
        all_text = "\n".join(b["text"] for b in text_blocks)

        # Should have document grouping headers
        assert "Document [1]" in all_text
        assert "Document [2]" in all_text
        assert "report.pdf" in all_text
        assert "other.pdf" in all_text

    def test_images_interleaved_with_document(self) -> None:
        """Images appear within their document group, not flat at the start."""
        contexts = _multi_doc_contexts()
        blocks = AnswerEngine._build_excerpt_blocks(contexts)

        image_blocks = [b for b in blocks if b.get("type") == "image_url"]
        # c1 and c2 have image_data, c3 does not
        assert len(image_blocks) == 2

        # Verify images appear after their document header
        doc1_header_idx = None
        first_image_idx = None
        for i, b in enumerate(blocks):
            if b.get("type") == "text" and "Document [1]" in b.get("text", ""):
                doc1_header_idx = i
            if b.get("type") == "image_url" and first_image_idx is None:
                first_image_idx = i
        assert doc1_header_idx is not None
        assert first_image_idx is not None
        assert first_image_idx > doc1_header_idx

    def test_enriched_image_labels(self) -> None:
        """Image labels include metadata like doc_title and page number."""
        contexts = _multi_doc_contexts()
        from dlightrag.citations.indexer import CitationIndexer

        indexer = CitationIndexer()
        flat = list(contexts["chunks"])
        indexer.build_index(flat)

        blocks = AnswerEngine._build_excerpt_blocks(contexts, indexer=indexer)

        text_blocks = [b for b in blocks if b.get("type") == "text"]
        all_text = "\n".join(b["text"] for b in text_blocks)

        # c1 has metadata doc_title="2025 Annual Report" and page_idx=3
        assert '"2025 Annual Report"' in all_text
        assert "Page 3" in all_text

    def test_empty_chunks_returns_empty_blocks(self) -> None:
        contexts: RetrievalContexts = {"chunks": []}
        blocks = AnswerEngine._build_excerpt_blocks(contexts)
        assert blocks == []

    def test_skipped_image_has_no_orphan_label(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/figures.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                }
            ]
        }
        engine = AnswerEngine(max_images=0)

        messages = engine._build_messages("sys", "prompt", contexts)

        user_content = messages[1]["content"]
        assert not any(block.get("type") == "image_url" for block in user_content)
        all_text = "\n".join(block["text"] for block in user_content if block.get("type") == "text")
        assert "Page image" not in all_text
        assert "figures.pdf" not in all_text


# ---------------------------------------------------------------------------
# TestBuildUserPromptExcludedExcerpts
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """_build_user_prompt() excludes Document Excerpts (now in content blocks)."""

    def test_prompt_excludes_excerpts_section(self) -> None:
        """Excerpts are rendered as content blocks, not in the text prompt."""
        engine = AnswerEngine()
        contexts = _text_contexts()
        prompt, _indexer = engine._build_user_prompt("What is revenue?", contexts)

        # Excerpts should NOT be in the text prompt (they are in content blocks now)
        assert "Document Excerpts:" not in prompt
        # But KG context and reference list should be present
        assert "Knowledge Graph Context" in prompt
        assert "Reference List" in prompt
        assert "Question" in prompt
        assert "What is revenue?" in prompt

    def test_prompt_contains_reference_list(self) -> None:
        engine = AnswerEngine()
        contexts = _text_contexts()
        prompt, _indexer = engine._build_user_prompt("query", contexts)
        assert "report.pdf" in prompt
