# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for AnswerStream (passthrough + post-stream citation validation)."""

from __future__ import annotations

import pytest

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.streaming import AnswerStream


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

    async def test_answer_preserves_full_text(self) -> None:
        """Post-stream .answer contains the full accumulated text."""

        async def fake_stream():
            yield "Growth is 15% [1].\n\n"
            yield "### References\n"
            yield "- [1] report.pdf"

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass
        assert stream.answer == "Growth is 15% [1].\n\n### References\n- [1] report.pdf"

    async def test_no_references_attr(self) -> None:
        """AnswerStream no longer exposes .references."""

        async def fake_stream():
            yield "Just a plain answer."

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass
        assert not hasattr(stream, "references")
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
        assert stream.answer == ""


@pytest.mark.asyncio
class TestAnswerStreamCitationValidation:
    """Test post-stream citation validation with indexer."""

    async def test_invalid_citations_cleaned_with_indexer(self) -> None:
        """When indexer is provided, invalid citations are removed from answer."""
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
            yield "Valid citation [1-1] and invalid [9-9]."

        stream = AnswerStream(fake_stream(), indexer=indexer)
        async for _ in stream:
            pass

        assert "[1-1]" in stream.answer
        assert "[9-9]" not in stream.answer

    async def test_no_indexer_passes_through_all_citations(self) -> None:
        """Without indexer, all citations pass through."""

        async def fake_stream():
            yield "Citation [1-1] and [9-9]."

        stream = AnswerStream(fake_stream())
        async for _ in stream:
            pass

        assert "[1-1]" in stream.answer
        assert "[9-9]" in stream.answer
