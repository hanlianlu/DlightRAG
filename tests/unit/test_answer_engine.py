# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerEngine messages-first interface."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.retrieval.protocols import RetrievalContexts
from dlightrag.models.schemas import StructuredAnswer

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


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerate
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerate:
    """Test non-streaming generate() with messages-first interface."""

    @pytest.mark.asyncio
    async def test_generate_with_structured_response(self) -> None:
        """generate() parses JSON response into StructuredAnswer."""
        response_json = json.dumps(
            {
                "answer": "AI is artificial intelligence.",
                "references": [{"id": 1, "title": "AI Overview"}],
            }
        )
        model_func = AsyncMock(return_value=response_json)
        engine = AnswerEngine(model_func=model_func)

        contexts = _text_contexts()
        result = await engine.generate("What is AI?", contexts)

        assert result.answer == "AI is artificial intelligence."
        assert len(result.references) == 1
        model_func.assert_called_once()
        call_kwargs = model_func.call_args.kwargs
        assert "messages" in call_kwargs
        assert "response_format" in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_no_model_func(self) -> None:
        """generate() returns None answer when model_func is None."""
        engine = AnswerEngine(model_func=None)
        contexts = {"chunks": []}
        result = await engine.generate("test", contexts)
        assert result.answer is None
        assert result.contexts is contexts

    @pytest.mark.asyncio
    async def test_generate_with_images(self) -> None:
        """generate() includes images in messages content array."""
        response_json = json.dumps({"answer": "ok", "references": []})
        model_func = AsyncMock(return_value=response_json)
        engine = AnswerEngine(model_func=model_func)

        contexts = _image_contexts()
        await engine.generate("describe", contexts)

        messages = model_func.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Should have image_url entry + text entry
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)

    @pytest.mark.asyncio
    async def test_generate_always_passes_response_format(self) -> None:
        """generate() always passes response_format=StructuredAnswer."""
        response_json = json.dumps(
            {"answer": "Revenue grew 15%.", "references": [{"id": 1, "title": "report.pdf"}]}
        )
        model_func = AsyncMock(return_value=response_json)
        engine = AnswerEngine(model_func=model_func)

        await engine.generate("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert call_kwargs.get("response_format") is StructuredAnswer

    @pytest.mark.asyncio
    async def test_contexts_passed_through_unchanged(self) -> None:
        """The original contexts dict should be returned as-is."""
        response_json = json.dumps({"answer": "answer", "references": []})
        model_func = AsyncMock(return_value=response_json)
        engine = AnswerEngine(model_func=model_func)
        contexts = _text_contexts()
        result = await engine.generate("q", contexts)

        assert result.contexts is contexts
        assert result.contexts["chunks"] == contexts["chunks"]
        assert result.contexts["entities"] == contexts["entities"]

    @pytest.mark.asyncio
    async def test_generate_freetext_fallback(self) -> None:
        """generate() falls back to freetext parsing when JSON parse fails."""
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

        contexts = {"chunks": []}
        ctx, token_iter = await engine.generate_stream("test", contexts)

        from dlightrag.models.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)

    @pytest.mark.asyncio
    async def test_generate_stream_no_model_func(self) -> None:
        """generate_stream() returns None when no model_func."""
        engine = AnswerEngine(model_func=None)
        contexts = {"chunks": []}
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


# ---------------------------------------------------------------------------
# TestBuildMessages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Test _build_messages static method."""

    def test_text_only(self) -> None:
        """_build_messages without images returns text content."""
        contexts: RetrievalContexts = {"chunks": [{"content": "text chunk"}]}
        messages = AnswerEngine._build_messages("system", "user prompt", contexts)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system"
        assert messages[1]["role"] == "user"
        # user content should be a list with text entry
        user_content = messages[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"
        assert user_content[0]["text"] == "user prompt"

    def test_with_images(self) -> None:
        """_build_messages with images includes image_url entries."""
        contexts: RetrievalContexts = {
            "chunks": [
                {"content": "text", "image_data": "abc123"},
                {"content": "more text"},
            ],
        }
        messages = AnswerEngine._build_messages("system", "user prompt", contexts)

        user_content = messages[1]["content"]
        image_entries = [e for e in user_content if e.get("type") == "image_url"]
        text_entries = [e for e in user_content if e.get("type") == "text"]
        assert len(image_entries) == 1
        assert len(text_entries) == 1
        assert "base64" in image_entries[0]["image_url"]["url"]

    def test_multiple_images(self) -> None:
        """_build_messages with multiple images includes all image_url entries."""
        contexts: RetrievalContexts = {
            "chunks": [
                {"content": "a", "image_data": "img1"},
                {"content": "b", "image_data": "img2"},
                {"content": "c"},
            ],
        }
        messages = AnswerEngine._build_messages("sys", "prompt", contexts)
        user_content = messages[1]["content"]
        image_entries = [e for e in user_content if e.get("type") == "image_url"]
        assert len(image_entries) == 2

    def test_empty_chunks(self) -> None:
        """_build_messages with empty chunks returns text-only content."""
        contexts: RetrievalContexts = {"chunks": []}
        messages = AnswerEngine._build_messages("sys", "prompt", contexts)
        user_content = messages[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"


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

    def test_build_user_prompt_contains_all_parts(self) -> None:
        engine = AnswerEngine()
        contexts = _text_contexts()
        prompt = engine._build_user_prompt("What is revenue?", contexts)

        assert "Knowledge Graph Context:" in prompt
        assert "Reference Document List:" in prompt
        assert "Question: What is revenue?" in prompt

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
        response_json = json.dumps({"answer": "answer", "references": []})
        model_func = AsyncMock(return_value=response_json)
        engine = AnswerEngine(model_func=model_func)

        with patch("dlightrag.core.answer.log_answer_llm_output") as mock_log:
            await engine.generate("query", _text_contexts())
            # Should be called at least twice: pre-call + parse
            assert mock_log.call_count >= 2

    @pytest.mark.asyncio
    async def test_generate_calls_log_references(self) -> None:
        response_json = json.dumps({"answer": "answer", "references": []})
        model_func = AsyncMock(return_value=response_json)
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
