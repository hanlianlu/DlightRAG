# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerEngine messages-first interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.retrieval.protocols import RetrievalContexts

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
                "image_data": "iVBORw0KGgoAAAANS",
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
                "image_data": "img_report_p3",
                "metadata": {"doc_title": "2025 Annual Report"},
            },
            {
                "chunk_id": "c2",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Expenses data.",
                "page_idx": 7,
                "image_data": "img_report_p7",
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
        """generate() parses freetext response with ### References."""
        raw = "AI is artificial intelligence [1-1].\n\n### References\n- [1] AI Overview"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        contexts = _text_contexts()
        result = await engine.generate("What is AI?", contexts)

        assert "AI is artificial intelligence" in result.answer
        assert len(result.references) == 1
        model_func.assert_called_once()
        call_kwargs = model_func.call_args.kwargs
        assert "messages" in call_kwargs
        # Must NOT pass response_format (unified freetext prompt)
        assert "response_format" not in call_kwargs

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
    async def test_generate_no_response_format(self) -> None:
        """generate() must NOT pass response_format (unified freetext prompt)."""
        raw = "Revenue grew 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        await engine.generate("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_contexts_passed_through_unchanged(self) -> None:
        """The original contexts dict should be returned as-is."""
        raw = "answer\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)
        contexts = _text_contexts()
        result = await engine.generate("q", contexts)

        assert result.contexts is contexts
        assert result.contexts["chunks"] == contexts["chunks"]
        assert result.contexts["entities"] == contexts["entities"]

    @pytest.mark.asyncio
    async def test_generate_freetext_references(self) -> None:
        """generate() extracts references from ### References section."""
        raw = "Growth is 15% [1-1].\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        result = await engine.generate("query", _text_contexts())

        assert "Growth is 15%" in result.answer
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

        from dlightrag.models.streaming import AnswerStream

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
    async def test_generate_stream_returns_contexts_and_tokens(self) -> None:
        """generate_stream() returns original contexts and consumable tokens."""

        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        contexts = _text_contexts()
        result_contexts, token_iter = await engine.generate_stream("query", contexts)

        assert result_contexts is contexts
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

        from dlightrag.models.streaming import AnswerStream

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
                    "image_data": "abc123",
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
                    "image_data": "abc123",
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
                    "image_data": "img1",
                    "file_path": "/docs/doc1.pdf",
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "2",
                    "content": "b",
                    "image_data": "img2",
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
# TestAnswerEngine -- Logging
# ---------------------------------------------------------------------------


class TestAnswerEngineLogging:
    """Test that logging functions are called correctly."""

    @pytest.mark.asyncio
    async def test_generate_calls_log_answer_llm_output(self) -> None:
        raw = "answer\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        with patch("dlightrag.core.answer.log_answer_llm_output") as mock_log:
            await engine.generate("query", _text_contexts())
            # Called once before LLM call (refs now via CitationProcessor, not parse)
            assert mock_log.call_count >= 1

    @pytest.mark.asyncio
    async def test_generate_calls_log_references(self) -> None:
        raw = "answer\n\n### References\n- [1] report.pdf"
        model_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(model_func=model_func)

        with patch("dlightrag.core.answer.log_references") as mock_log:
            await engine.generate("query", _text_contexts())
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "answer_engine.generate"

    @pytest.mark.asyncio
    async def test_generate_stream_calls_log_answer_llm_output(self) -> None:
        async def mock_stream():
            yield "token"

        model_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(model_func=model_func)

        with patch("dlightrag.core.answer.log_answer_llm_output") as mock_log:
            await engine.generate_stream("query", _text_contexts())
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "answer_engine.generate_stream"


# ---------------------------------------------------------------------------
# TestAnswerEngine -- Freetext reference parsing
# ---------------------------------------------------------------------------


class TestAnswerEngineFreetextReferences:
    """Test parse_freetext_references extracts references from LLM output."""

    def test_parses_references_section(self) -> None:
        from dlightrag.citations.parser import parse_freetext_references

        raw = (
            "Revenue grew 15% [1-1].\n\n"
            "### References\n"
            "- [1] report.pdf\n"
            "- [2] quarterly_report.pdf\n"
        )
        answer, refs = parse_freetext_references(raw)
        assert "Revenue grew 15%" in answer
        assert "### References" not in answer
        assert len(refs) == 2
        assert refs[0].id == 1
        assert refs[0].title == "report.pdf"
        assert refs[1].id == 2
        assert refs[1].title == "quarterly_report.pdf"

    def test_no_references_section_returns_empty(self) -> None:
        from dlightrag.citations.parser import parse_freetext_references

        raw = "Just a plain answer with no references section."
        answer, refs = parse_freetext_references(raw)
        assert answer == raw
        assert refs == []

    def test_case_insensitive_heading(self) -> None:
        from dlightrag.citations.parser import parse_freetext_references

        raw = "Answer text.\n\n## references\n[1] doc.pdf"
        answer, refs = parse_freetext_references(raw)
        assert len(refs) == 1
        assert refs[0].title == "doc.pdf"


# ---------------------------------------------------------------------------
# TestFormatChunkExcerpts
# ---------------------------------------------------------------------------


class TestFormatChunkExcerpts:
    """Test _format_chunk_excerpts static method."""

    def test_excerpts_include_content(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Revenue grew 15%.",
                    "page_idx": 3,
                },
            ],
        }
        result = AnswerEngine._format_chunk_excerpts(contexts)
        assert "Revenue grew 15%." in result
        assert "report.pdf" in result
        assert "Page 3" in result

    def test_empty_chunks(self) -> None:
        contexts: RetrievalContexts = {"chunks": []}
        result = AnswerEngine._format_chunk_excerpts(contexts)
        assert result == "No document excerpts available."

    def test_chunks_without_content_skipped(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {"chunk_id": "c1", "content": ""},
                {"chunk_id": "c2", "content": "  "},
                {"chunk_id": "c3", "content": "actual text", "file_path": "/doc.pdf"},
            ],
        }
        result = AnswerEngine._format_chunk_excerpts(contexts)
        assert "actual text" in result
        assert result.count("[") == 1  # Only one label

    def test_chunk_without_page_idx(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [
                {"chunk_id": "c1", "content": "text", "file_path": "/doc.pdf"},
            ],
        }
        result = AnswerEngine._format_chunk_excerpts(contexts)
        assert "[doc.pdf]" in result
        assert "Page" not in result

    def test_excerpts_include_citation_tags_with_indexer(self) -> None:
        """When indexer is provided, excerpt labels include [ref_id-chunk_idx]."""
        from dlightrag.citations.indexer import CitationIndexer

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Revenue grew 15%.",
                    "page_idx": 3,
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "2",
                    "file_path": "/docs/whitepaper.pdf",
                    "content": "Parser benchmarks.",
                    "page_idx": 1,
                },
            ],
        }
        indexer = CitationIndexer()
        flat = list(contexts["chunks"])
        indexer.build_index(flat)

        result = AnswerEngine._format_chunk_excerpts(contexts, indexer=indexer)

        # Citation markers should appear before filenames
        assert "[1-1] report.pdf, Page 3" in result
        assert "[2-1] whitepaper.pdf, Page 1" in result
        # Content still present
        assert "Revenue grew 15%." in result
        assert "Parser benchmarks." in result

    def test_citation_tags_match_reference_list(self) -> None:
        """Citation tags in excerpts must match the reference list numbering."""
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "First doc content.",
                    "page_idx": 1,
                },
                {
                    "chunk_id": "c2",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Second page content.",
                    "page_idx": 5,
                },
                {
                    "chunk_id": "c3",
                    "reference_id": "2",
                    "file_path": "/docs/other.pdf",
                    "content": "Other doc content.",
                    "page_idx": 2,
                },
            ],
            "entities": [],
            "relationships": [],
        }
        engine = AnswerEngine()
        prompt, _indexer = engine._build_user_prompt("test query", contexts)

        # Reference list should have matching entries
        assert "[1] report.pdf" in prompt
        assert "[2] other.pdf" in prompt


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
