# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerStream (passthrough + post-stream citation validation)."""

from __future__ import annotations

import pytest

from dlightrag.citations.indexer import CitationIndexer
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
        """Plain text -- empty references, answer equals full text."""

        async def fake_stream():
            yield "Just a plain answer."

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass
        assert stream.references == []
        assert stream.answer == "Just a plain answer."

    async def test_empty_stream(self) -> None:
        """Empty stream -- no crash."""

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


@pytest.mark.asyncio
class TestAnswerStreamCitationValidation:
    """Test post-stream citation validation with indexer."""

    async def test_invalid_citations_cleaned_with_indexer(self) -> None:
        """When indexer is provided, invalid citations are removed from answer."""
        # Build an indexer with only ref_id=1, chunk_idx=1
        indexer = CitationIndexer()
        indexer.build_index(
            [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "content": "Real content.",
                    "file_path": "/docs/report.pdf",
                },
            ]
        )

        async def fake_stream():
            # [1-1] is valid, [9-9] is invalid
            yield "Valid citation [1-1] and invalid [9-9].\n\n"
            yield "### References\n"
            yield "- [1] report.pdf"

        stream = AnswerStream(fake_stream(), indexer=indexer)
        async for _ in stream:
            pass

        # Invalid citation should be removed
        assert "[1-1]" in stream.answer or "[1]" in stream.answer
        assert "[9-9]" not in stream.answer
        assert len(stream.references) == 1

    async def test_no_indexer_passes_through_all_citations(self) -> None:
        """Without indexer, all citations pass through (backward compat)."""

        async def fake_stream():
            yield "Citation [1-1] and [9-9].\n\n"
            yield "### References\n"
            yield "- [1] report.pdf\n"
            yield "- [9] phantom.pdf"

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass

        # Both citations remain
        assert "[1-1]" in stream.answer or "[1]" in stream.answer
        assert "[9-9]" in stream.answer or "[9]" in stream.answer
