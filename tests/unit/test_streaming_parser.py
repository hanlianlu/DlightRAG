# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for simplified AnswerStream (passthrough + post-stream ref extraction)."""

from __future__ import annotations

import pytest

from dlightrag.models.streaming import AnswerStream


@pytest.mark.asyncio
class TestAnswerStream:
    async def test_passthrough_tokens(self) -> None:
        """All tokens are yielded as-is."""

        async def fake_stream():
            yield "Hello "
            yield "world."

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert "".join(parts) == "Hello world."

    async def test_extracts_references_from_freetext(self) -> None:
        """Post-stream extraction finds ### References section."""

        async def fake_stream():
            yield "Growth is 15% [1].\n\n"
            yield "### References\n"
            yield "- [1] report.pdf"

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert len(stream.references) == 1
        assert stream.references[0].title == "report.pdf"
        assert "### References" not in stream.answer

    async def test_extracts_references_from_json_block(self) -> None:
        """Post-stream extraction finds trailing JSON block."""

        async def fake_stream():
            yield "Based on the data [1-1].\n\n"
            yield '{"answer": "Based on the data [1-1].", '
            yield '"references": [{"id": 1, "title": "report.pdf"}]}'

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        # Raw stream includes JSON (inherent streaming limitation)
        assert '{"answer"' in "".join(parts)
        # But .answer is cleaned
        assert '{"answer"' not in stream.answer
        assert len(stream.references) == 1

    async def test_no_references(self) -> None:
        """Plain text — empty references, answer equals full text."""

        async def fake_stream():
            yield "Just a plain answer."

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass
        assert stream.references == []
        assert stream.answer == "Just a plain answer."

    async def test_empty_stream(self) -> None:
        """Empty stream — no crash."""

        async def fake_stream():
            return
            yield  # make it an async generator

        stream = AnswerStream(fake_stream())
        parts = []
        async for token in stream:
            parts.append(token)
        assert parts == []
        assert stream.references == []
        assert stream.answer == ""
