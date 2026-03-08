# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Docling HybridChunker wrapper."""

from __future__ import annotations

from typing import Any

from dlightrag.core.chunking import docling_hybrid_chunking_func


def _word_tokenizer() -> Any:
    """Mock tokenizer that counts words (split by whitespace) as tokens."""
    return None  # The function ignores this; HybridChunker uses its own tokenizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PLAIN_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is the first paragraph with enough words to form a chunk.\n\n"
    "The second paragraph provides additional content for testing. "
    "It contains several sentences to verify chunking behaviour."
)

_HEADED_TEXT = (
    "# Introduction\n\n"
    "This section introduces the topic with some detail.\n\n"
    "## Background\n\n"
    "Background information provides historical context for the reader.\n\n"
    "# Methods\n\n"
    "The methods section describes the experimental approach."
)


# ---------------------------------------------------------------------------
# TestPlainParagraphs
# ---------------------------------------------------------------------------


class TestPlainParagraphs:
    """Plain text (no headings) produces chunks with correct fields."""

    def test_plain_paragraphs(self) -> None:
        result = docling_hybrid_chunking_func(
            tokenizer=_word_tokenizer(),
            content=_PLAIN_TEXT,
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=0,
            chunk_token_size=500,
        )
        assert len(result) >= 1
        for chunk in result:
            assert "content" in chunk
            assert "tokens" in chunk
            assert "chunk_order_index" in chunk
            assert "_raw_content" in chunk
            assert isinstance(chunk["content"], str)
            assert isinstance(chunk["tokens"], int)
            assert chunk["tokens"] > 0
            assert len(chunk["content"]) > 0


# ---------------------------------------------------------------------------
# TestMarkdownHeadingsContextualised
# ---------------------------------------------------------------------------


class TestMarkdownHeadingsContextualised:
    """Heading text is prepended via contextualize()."""

    def test_markdown_headings_contextualised(self) -> None:
        result = docling_hybrid_chunking_func(
            tokenizer=_word_tokenizer(),
            content=_HEADED_TEXT,
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=0,
            chunk_token_size=500,
        )
        assert len(result) >= 1
        # At least one chunk should contain heading text from the document
        all_content = " ".join(c["content"] for c in result)
        assert "Introduction" in all_content or "Methods" in all_content


# ---------------------------------------------------------------------------
# TestRawContentFieldPresent
# ---------------------------------------------------------------------------


class TestRawContentFieldPresent:
    """_raw_content exists and is findable in the original content."""

    def test_raw_content_field_present(self) -> None:
        result = docling_hybrid_chunking_func(
            tokenizer=_word_tokenizer(),
            content=_HEADED_TEXT,
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=0,
            chunk_token_size=500,
        )
        assert len(result) >= 1
        for chunk in result:
            assert "_raw_content" in chunk
            raw = chunk["_raw_content"]
            assert isinstance(raw, str)
            assert len(raw) > 0
            # _raw_content (original text without heading prefix) must appear
            # somewhere in the original content string
            assert raw in _HEADED_TEXT or raw.strip() in _HEADED_TEXT


# ---------------------------------------------------------------------------
# TestRespectsChunkTokenSize
# ---------------------------------------------------------------------------


class TestRespectsChunkTokenSize:
    """Chunks don't greatly exceed the requested max token size."""

    def test_respects_chunk_token_size(self) -> None:
        long_text = "\n\n".join(
            f"Paragraph {i} with several words to fill up the content block." for i in range(50)
        )
        small_limit = 30
        result = docling_hybrid_chunking_func(
            tokenizer=_word_tokenizer(),
            content=long_text,
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=0,
            chunk_token_size=small_limit,
        )
        assert len(result) >= 2, "Expected multiple chunks with a small token limit"
        for chunk in result:
            # Allow some overhead (HybridChunker may add heading context),
            # but tokens should not be wildly above the limit.
            assert chunk["tokens"] <= small_limit * 3, (
                f"Chunk has {chunk['tokens']} tokens, "
                f"expected at most ~{small_limit * 3} (3x limit)"
            )


# ---------------------------------------------------------------------------
# TestEmptyContent
# ---------------------------------------------------------------------------


class TestEmptyContent:
    """Empty input returns empty list."""

    def test_empty_content(self) -> None:
        assert (
            docling_hybrid_chunking_func(
                tokenizer=_word_tokenizer(),
                content="",
                split_by_character=None,
                split_by_character_only=False,
                chunk_overlap_token_size=0,
                chunk_token_size=500,
            )
            == []
        )

    def test_whitespace_only_content(self) -> None:
        assert (
            docling_hybrid_chunking_func(
                tokenizer=_word_tokenizer(),
                content="   \n\n   ",
                split_by_character=None,
                split_by_character_only=False,
                chunk_overlap_token_size=0,
                chunk_token_size=500,
            )
            == []
        )


# ---------------------------------------------------------------------------
# TestChunkOrderIndicesSequential
# ---------------------------------------------------------------------------


class TestChunkOrderIndicesSequential:
    """Chunk order indices are 0, 1, 2, ..."""

    def test_chunk_order_indices_sequential(self) -> None:
        long_text = "\n\n".join(
            f"Paragraph number {i} has enough words to be meaningful content." for i in range(20)
        )
        result = docling_hybrid_chunking_func(
            tokenizer=_word_tokenizer(),
            content=long_text,
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=0,
            chunk_token_size=50,
        )
        assert len(result) >= 2
        indices = [c["chunk_order_index"] for c in result]
        assert indices == list(range(len(result)))
