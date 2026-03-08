# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared VLM OCR utilities (vlm_ocr.py)."""

from __future__ import annotations

import json

from dlightrag.core.vlm_ocr import blocks_to_text, parse_vlm_response

# ---------------------------------------------------------------------------
# TestBlocksToText
# ---------------------------------------------------------------------------


class TestBlocksToText:
    """blocks_to_text converts structured OCR blocks to plain text for entity extraction."""

    def test_all_block_types(self) -> None:
        blocks = [
            {"type": "heading", "level": 2, "text": "Results"},
            {"type": "text", "text": "The experiment showed significant improvement."},
            {"type": "table", "html": "<table><tr><th>Metric</th><th>Value</th></tr></table>"},
            {"type": "formula", "latex": "\\[ E = mc^2 \\]"},
            {"type": "figure", "description": "Bar chart of quarterly revenue."},
        ]
        result = blocks_to_text(blocks)

        assert "## Results" in result
        assert "The experiment showed significant improvement." in result
        assert "<table>" in result
        assert "\\[ E = mc^2 \\]" in result
        assert "[Figure: Bar chart of quarterly revenue.]" in result
        # Blocks separated by double newlines
        assert result.count("\n\n") == 4

    def test_heading_levels(self) -> None:
        blocks = [
            {"type": "heading", "level": 1, "text": "Title"},
            {"type": "heading", "level": 3, "text": "Subsection"},
        ]
        result = blocks_to_text(blocks)
        assert result.startswith("# Title")
        assert "### Subsection" in result

    def test_empty_blocks(self) -> None:
        assert blocks_to_text([]) == ""

    def test_skips_empty_content(self) -> None:
        """Blocks with empty text/html/latex/description are skipped."""
        blocks = [
            {"type": "text", "text": ""},
            {"type": "table", "html": ""},
            {"type": "formula", "latex": ""},
            {"type": "figure", "description": ""},
            {"type": "text", "text": "Only real content"},
        ]
        assert blocks_to_text(blocks) == "Only real content"

    def test_unknown_block_type_ignored(self) -> None:
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "unknown_future_type", "data": "whatever"},
            {"type": "text", "text": "World"},
        ]
        assert blocks_to_text(blocks) == "Hello\n\nWorld"


# ---------------------------------------------------------------------------
# TestParseVlmResponse
# ---------------------------------------------------------------------------


class TestParseVlmResponse:
    """parse_vlm_response extracts blocks from VLM JSON output."""

    def test_valid_json(self) -> None:
        raw = json.dumps({"blocks": [{"type": "text", "text": "Hello"}]})
        result = parse_vlm_response(raw)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_empty_blocks(self) -> None:
        """Empty blocks list returns [] (not None)."""
        raw = json.dumps({"blocks": []})
        result = parse_vlm_response(raw)
        assert result == []

    def test_json_with_markdown_fences(self) -> None:
        """Regex fallback extracts JSON from markdown code fences."""
        raw = '```json\n{"blocks": [{"type": "text", "text": "Extracted"}]}\n```'
        result = parse_vlm_response(raw)
        assert result is not None
        assert result[0]["text"] == "Extracted"

    def test_invalid_json_returns_none(self) -> None:
        assert parse_vlm_response("This is not JSON at all {{{") is None

    def test_valid_json_without_blocks_key_returns_none(self) -> None:
        assert parse_vlm_response('{"data": [1, 2, 3]}') is None


# ---------------------------------------------------------------------------
# TestDescribePageHappyPath
# ---------------------------------------------------------------------------


class TestDescribePageStructuredOcr:
    """_describe_page with valid JSON exercises the full OCR → blocks → text path."""

    async def test_structured_json_converted_to_text(self) -> None:
        """VLM returns valid structured JSON → blocks_to_text produces formatted text."""
        from unittest.mock import AsyncMock, MagicMock

        from dlightrag.unifiedrepresent.extractor import EntityExtractor

        vlm_response = json.dumps(
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Introduction"},
                    {"type": "text", "text": "This paper presents novel findings."},
                    {"type": "table", "html": "<table><tr><td>A</td></tr></table>"},
                ]
            }
        )
        vision_fn = AsyncMock(return_value=vlm_response)

        lightrag = MagicMock()
        ext = EntityExtractor(lightrag, ["person"], vision_fn)
        result = await ext._describe_page(MagicMock(), page_index=0)

        assert "# Introduction" in result
        assert "This paper presents novel findings." in result
        assert "<table>" in result
