# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerEngine centralized answer generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from dlightrag.core.answer import AnswerEngine
from dlightrag.core.retrieval.protocols import RetrievalContexts, RetrievalResult
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
# TestAnswerEngineGenerate — LLM text path
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerateLLM:
    """Test text-only LLM path (no images in chunks)."""

    async def test_llm_text_path_calls_llm_model_func(self) -> None:
        llm_func = AsyncMock(return_value="The answer is 42.")
        engine = AnswerEngine(
            llm_model_func=llm_func,
            provider="ollama",  # freetext provider
        )
        result = await engine.generate("What is revenue?", _text_contexts())

        llm_func.assert_awaited_once()
        assert isinstance(result, RetrievalResult)
        assert result.answer == "The answer is 42."
        assert result.references == []

    async def test_llm_text_path_passes_system_prompt(self) -> None:
        llm_func = AsyncMock(return_value="Answer.")
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")
        await engine.generate("query", _text_contexts())

        _, kwargs = llm_func.call_args
        assert "system_prompt" in kwargs
        assert len(kwargs["system_prompt"]) > 0

    async def test_contexts_passed_through_unchanged(self) -> None:
        """The original contexts dict should be returned as-is."""
        llm_func = AsyncMock(return_value="answer")
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")
        contexts = _text_contexts()
        result = await engine.generate("q", contexts)

        assert result.contexts is contexts
        assert result.contexts["chunks"] == contexts["chunks"]
        assert result.contexts["entities"] == contexts["entities"]


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerate — VLM multimodal path
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerateVLM:
    """Test VLM path (chunks contain image_data)."""

    async def test_vlm_path_with_images(self) -> None:
        vlm_func = AsyncMock(return_value="Image analysis result.")
        engine = AnswerEngine(
            vision_model_func=vlm_func,
            provider="ollama",  # freetext
        )
        result = await engine.generate("Describe the chart", _image_contexts())

        vlm_func.assert_awaited_once()
        assert result.answer == "Image analysis result."

    async def test_vlm_path_uses_messages_with_images(self) -> None:
        vlm_func = AsyncMock(return_value="result")
        engine = AnswerEngine(vision_model_func=vlm_func, provider="ollama")
        await engine.generate("query", _image_contexts())

        _, kwargs = vlm_func.call_args
        assert "messages" in kwargs
        messages = kwargs["messages"]
        # System message + user message with image content
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # User content includes image_url block
        user_content = messages[1]["content"]
        assert any(block.get("type") == "image_url" for block in user_content)

    async def test_vlm_fallback_to_llm_when_no_vision_func(self) -> None:
        """If no vision_model_func, fall back to llm_model_func for images."""
        llm_func = AsyncMock(return_value="fallback answer")
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")
        result = await engine.generate("query", _image_contexts())

        # Should use llm_func since vision_model_func is None
        llm_func.assert_awaited_once()
        assert result.answer == "fallback answer"


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerate — Structured JSON parsing
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerateStructured:
    """Test structured JSON parsing with providers that support it."""

    async def test_structured_vlm_json_parsing(self) -> None:
        """VLM structured path parses JSON with references."""
        structured_json = json.dumps(
            {
                "answer": "Revenue grew 15% [1-1].",
                "references": [{"id": 1, "title": "report.pdf"}],
            }
        )
        vlm_func = AsyncMock(return_value=structured_json)
        engine = AnswerEngine(vision_model_func=vlm_func, provider="openai")

        result = await engine.generate("What is revenue?", _image_contexts())

        assert result.answer == "Revenue grew 15% [1-1]."
        assert len(result.references) == 1
        assert result.references[0].id == 1
        assert result.references[0].title == "report.pdf"

    async def test_text_only_never_passes_response_schema(self) -> None:
        """Text-only path always uses freetext — no response_schema to LLM."""
        raw = "Answer text.\n\n### References\n[1] report.pdf"
        llm_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(llm_model_func=llm_func, provider="openai")

        await engine.generate("query", _text_contexts())

        _, kwargs = llm_func.call_args
        assert "response_schema" not in kwargs

    async def test_text_only_freetext_extracts_references(self) -> None:
        """Text-only path extracts references from freetext output."""
        raw = "Revenue grew 15% [1-1].\n\n### References\n- [1] report.pdf"
        llm_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(llm_model_func=llm_func, provider="openai")

        result = await engine.generate("query", _text_contexts())

        assert "Revenue grew 15%" in result.answer
        assert len(result.references) == 1
        assert result.references[0].id == 1
        assert result.references[0].title == "report.pdf"

    async def test_structured_vlm_sends_response_schema(self) -> None:
        structured_json = json.dumps({"answer": "Chart shows growth [1-1].", "references": []})
        vlm_func = AsyncMock(return_value=structured_json)
        engine = AnswerEngine(vision_model_func=vlm_func, provider="openai")

        await engine.generate("describe chart", _image_contexts())

        _, kwargs = vlm_func.call_args
        assert kwargs.get("response_schema") is StructuredAnswer


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerate — No model funcs
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerateNoModel:
    """Test behavior when no model functions are provided."""

    async def test_no_model_funcs_returns_none_answer(self) -> None:
        engine = AnswerEngine()
        contexts = _text_contexts()
        result = await engine.generate("query", contexts)

        assert result.answer is None
        assert result.contexts is contexts
        assert result.references == []

    async def test_no_model_funcs_with_images(self) -> None:
        engine = AnswerEngine()
        contexts = _image_contexts()
        result = await engine.generate("query", contexts)

        assert result.answer is None
        assert result.contexts is contexts


# ---------------------------------------------------------------------------
# TestAnswerEngineGenerateStream
# ---------------------------------------------------------------------------


class TestAnswerEngineGenerateStream:
    """Test streaming answer generation."""

    async def test_stream_returns_contexts_and_iterator(self) -> None:
        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        llm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        contexts = _text_contexts()
        result_contexts, token_iter = await engine.generate_stream("query", contexts)

        assert result_contexts is contexts
        assert token_iter is not None
        tokens = [t async for t in token_iter]
        assert tokens == ["Hello", " ", "world"]

    async def test_stream_no_model_funcs_returns_none_iterator(self) -> None:
        engine = AnswerEngine()
        contexts = _text_contexts()
        result_contexts, token_iter = await engine.generate_stream("query", contexts)

        assert result_contexts is contexts
        assert token_iter is None

    async def test_stream_vlm_path_with_images(self) -> None:
        async def mock_stream():
            for token in ["Image", " analysis"]:
                yield token

        vlm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(vision_model_func=vlm_func, provider="ollama")

        contexts = _image_contexts()
        result_contexts, token_iter = await engine.generate_stream("describe", contexts)

        assert result_contexts is contexts
        assert token_iter is not None
        vlm_func.assert_awaited_once()
        # Verify messages were sent with images
        _, kwargs = vlm_func.call_args
        assert "messages" in kwargs
        assert kwargs["stream"] is True

    async def test_stream_vlm_structured_wraps_with_answer_stream(self) -> None:
        """VLM structured providers should wrap the iterator with AnswerStream."""

        async def mock_stream():
            for token in ['{"answer": "hello', " world", '", "references": []}']:
                yield token

        vlm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(vision_model_func=vlm_func, provider="openai")

        contexts = _image_contexts()
        _, token_iter = await engine.generate_stream("query", contexts)

        assert token_iter is not None
        # For VLM structured providers, should be wrapped in AnswerStream
        from dlightrag.models.streaming import AnswerStream as AnswerStreamCls

        assert isinstance(token_iter, AnswerStreamCls)

    async def test_stream_text_only_not_wrapped(self) -> None:
        """Text-only path should NOT wrap with AnswerStream (always freetext)."""

        async def mock_stream():
            yield "plain answer text"

        llm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(llm_model_func=llm_func, provider="openai")

        contexts = _text_contexts()
        _, token_iter = await engine.generate_stream("query", contexts)

        assert token_iter is not None
        from dlightrag.models.streaming import AnswerStream as AnswerStreamCls

        assert not isinstance(token_iter, AnswerStreamCls)


# ---------------------------------------------------------------------------
# TestAnswerEngine — Internal helpers
# ---------------------------------------------------------------------------


class TestAnswerEngineHelpers:
    """Test static/internal helper methods."""

    def test_has_images_true(self) -> None:
        assert AnswerEngine._has_images(_image_contexts()) is True

    def test_has_images_false(self) -> None:
        assert AnswerEngine._has_images(_text_contexts()) is False

    def test_has_images_empty(self) -> None:
        empty: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        assert AnswerEngine._has_images(empty) is False

    def test_build_vlm_messages_structure(self) -> None:
        chunks = [
            {"image_data": "abc123", "content": "text"},
            {"content": "no image"},
        ]
        messages = AnswerEngine._build_vlm_messages("sys", "user prompt", chunks)

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "sys"}
        user_content = messages[1]["content"]
        # One image_url + one text
        types = [block["type"] for block in user_content]
        assert "image_url" in types
        assert "text" in types
        # Only 1 image (second chunk has no image_data)
        assert types.count("image_url") == 1

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
        # Should have header + 20 entities = 21 lines in entity section
        lines = [line for line in result.split("\n") if line.strip()]
        entity_lines = [line for line in lines if line.startswith("- **")]
        assert len(entity_lines) == 20

    def test_build_user_prompt_contains_all_parts(self) -> None:
        engine = AnswerEngine(provider="ollama")
        contexts = _text_contexts()
        prompt = engine._build_user_prompt("What is revenue?", contexts)

        assert "Knowledge Graph Context:" in prompt
        assert "Reference Document List:" in prompt
        assert "Question: What is revenue?" in prompt
        # Should NOT contain conversation_history or FREETEXT_REMINDER
        assert "Conversation History" not in prompt

    def test_build_citation_indexer(self) -> None:
        contexts = _text_contexts()
        indexer = AnswerEngine._build_citation_indexer(contexts)
        ref_list = indexer.format_reference_list()
        assert "report.pdf" in ref_list


# ---------------------------------------------------------------------------
# TestAnswerEngine — Logging
# ---------------------------------------------------------------------------


class TestAnswerEngineLogging:
    """Test that logging functions are called correctly."""

    async def test_generate_calls_log_answer_llm_output(self) -> None:
        llm_func = AsyncMock(return_value="answer")
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        with patch("dlightrag.core.answer.log_answer_llm_output") as mock_log:
            await engine.generate("query", _text_contexts())
            # Should be called at least twice: path branch + output
            assert mock_log.call_count >= 2

    async def test_generate_calls_log_references(self) -> None:
        llm_func = AsyncMock(return_value="answer")
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        with patch("dlightrag.core.answer.log_references") as mock_log:
            await engine.generate("query", _text_contexts())
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args
            assert call_kwargs[0][0] == "answer_engine.generate"

    async def test_generate_stream_calls_log_answer_llm_output(self) -> None:
        async def mock_stream():
            yield "token"

        llm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        with patch("dlightrag.core.answer.log_answer_llm_output") as mock_log:
            await engine.generate_stream("query", _text_contexts())
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "answer_engine.generate_stream"


# ---------------------------------------------------------------------------
# TestAnswerEngine — Freetext reference parsing
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

    def test_freetext_generate_extracts_references(self) -> None:
        """End-to-end: freetext provider generates references section, engine parses it."""
        import asyncio

        raw = "Growth is 15% [1-1].\n\n### References\n- [1] report.pdf"
        llm_func = AsyncMock(return_value=raw)
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        result = asyncio.get_event_loop().run_until_complete(
            engine.generate("query", _text_contexts())
        )

        assert "Growth is 15%" in result.answer
        assert len(result.references) == 1
        assert result.references[0].title == "report.pdf"


# ---------------------------------------------------------------------------
# TestAnswerEngine — Structured text path passes response_schema
# ---------------------------------------------------------------------------


class TestAnswerEngineStreamStructured:
    """Test streaming paths handle response_schema correctly."""

    async def test_stream_text_never_passes_response_schema(self) -> None:
        """Text-only path should never pass response_schema (always freetext)."""

        async def mock_stream():
            yield "plain text answer"

        llm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(llm_model_func=llm_func, provider="openai")

        await engine.generate_stream("query", _text_contexts())

        _, kwargs = llm_func.call_args
        assert "response_schema" not in kwargs
        assert kwargs.get("stream") is True

    async def test_stream_text_freetext_no_response_schema(self) -> None:
        async def mock_stream():
            yield "plain text"

        llm_func = AsyncMock(return_value=mock_stream())
        engine = AnswerEngine(llm_model_func=llm_func, provider="ollama")

        await engine.generate_stream("query", _text_contexts())

        _, kwargs = llm_func.call_args
        assert kwargs.get("response_schema") is None
