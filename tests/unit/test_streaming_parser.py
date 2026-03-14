# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for StreamingAnswerParser state machine."""

from __future__ import annotations

import pytest

from dlightrag.models.streaming import AnswerStream, StreamingAnswerParser


class TestStreamingAnswerParser:
    def test_complete_json_single_chunk(self) -> None:
        """Full JSON in one chunk — extract answer, parse references."""
        parser = StreamingAnswerParser()
        raw = '{"answer": "The findings [1-1].", "references": [{"id": 1, "title": "report.pdf"}]}'
        emitted = parser.feed(raw)
        result = parser.finish()
        assert "The findings [1-1]." in emitted
        assert len(result.references) == 1
        assert result.references[0].title == "report.pdf"

    def test_streamed_token_by_token(self) -> None:
        """Tokens arrive one at a time — answer emitted incrementally."""
        parser = StreamingAnswerParser()
        tokens = [
            '{"answer',
            '": "Hello',
            " world",
            '", "ref',
            'erences": [{"id": 1, "title": "doc.pdf"}]}',
        ]
        emitted_parts = []
        for t in tokens:
            out = parser.feed(t)
            if out:
                emitted_parts.append(out)
        result = parser.finish()
        full = "".join(emitted_parts)
        assert full == "Hello world"
        assert result.references[0].id == 1

    def test_json_escape_handling(self) -> None:
        """JSON escapes in answer (\\n, \\") are unescaped."""
        parser = StreamingAnswerParser()
        raw = '{"answer": "line1\\nline2 \\"quoted\\"", "references": []}'
        emitted = parser.feed(raw)
        result = parser.finish()
        assert 'line1\nline2 "quoted"' in emitted
        assert result.references == []

    def test_passthrough_on_non_json(self) -> None:
        """Non-JSON input triggers passthrough — all text emitted as-is."""
        parser = StreamingAnswerParser()
        out1 = parser.feed("The findings are")
        out2 = parser.feed(" important [1-1].")
        result = parser.finish()
        full = out1 + out2
        assert "The findings are important [1-1]." in full
        assert result.references == []

    def test_empty_references(self) -> None:
        """Valid JSON with empty references array."""
        parser = StreamingAnswerParser()
        raw = '{"answer": "No refs here.", "references": []}'
        emitted = parser.feed(raw)
        result = parser.finish()
        assert "No refs here." in emitted
        assert result.references == []

    def test_malformed_references_degrades(self) -> None:
        """Valid JSON but bad references — degrades gracefully."""
        parser = StreamingAnswerParser()
        raw = '{"answer": "Text.", "references": "not-a-list"}'
        parser.feed(raw)
        result = parser.finish()
        assert result.answer == "Text."
        assert result.references == []

    def test_multiple_references(self) -> None:
        """Multiple references parsed correctly."""
        parser = StreamingAnswerParser()
        raw = '{"answer": "A [1-1] B [2-1]", "references": [{"id": 1, "title": "a.pdf"}, {"id": 2, "title": "b.pdf"}]}'
        parser.feed(raw)
        result = parser.finish()
        assert len(result.references) == 2
        assert result.references[1].title == "b.pdf"

    def test_escape_at_chunk_boundary(self) -> None:
        """Backslash at end of chunk boundary is handled correctly."""
        parser = StreamingAnswerParser()
        out1 = parser.feed('{"answer": "hello\\')
        out2 = parser.feed('nworld", "references": []}')
        result = parser.finish()
        full = out1 + out2
        assert "hello\nworld" in full
        assert result.references == []


@pytest.mark.asyncio
class TestAnswerStream:
    async def test_wraps_async_iterator(self) -> None:
        """AnswerStream yields answer tokens and populates references."""

        async def fake_stream():
            yield '{"answer": "Hello'
            yield " world"
            yield '", "references": [{"id": 1, "title": "doc.pdf"}]}'

        parser = StreamingAnswerParser()
        stream = AnswerStream(fake_stream(), parser)
        parts = []
        async for token in stream:
            parts.append(token)
        assert "".join(parts) == "Hello world"
        assert len(stream.references) == 1
        assert stream.references[0].title == "doc.pdf"

    async def test_passthrough_for_non_json(self) -> None:
        """Non-JSON stream passes through as-is with empty references."""

        async def fake_stream():
            yield "Plain text answer"
            yield " continues here."

        parser = StreamingAnswerParser()
        stream = AnswerStream(fake_stream(), parser)
        parts = []
        async for token in stream:
            parts.append(token)
        assert "Plain text answer continues here." in "".join(parts)
        assert stream.references == []

    def test_markdown_code_fence_stripped(self) -> None:
        """JSON wrapped in ```json ... ``` fences should be parsed correctly."""
        parser = StreamingAnswerParser()
        tokens = [
            "```json\n",
            '{"answer": "Hello',
            ' world", "references": []}',
            "\n```",
        ]
        parts = []
        for t in tokens:
            out = parser.feed(t)
            if out:
                parts.append(out)
        result = parser.finish()
        assert "".join(parts) == "Hello world"
        assert result.answer == "Hello world"
        assert result.references == []

    def test_code_fence_single_chunk(self) -> None:
        """Full fenced JSON in one chunk."""
        parser = StreamingAnswerParser()
        raw = '```json\n{"answer": "Test answer.", "references": [{"id": 1, "title": "doc.pdf"}]}\n```'
        emitted = parser.feed(raw)
        result = parser.finish()
        assert "Test answer." in emitted
        assert result.answer == "Test answer."
        assert len(result.references) == 1
