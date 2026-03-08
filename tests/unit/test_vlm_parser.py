# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VlmOcrParser (VLM-based document OCR extraction)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from dlightrag.core.ingestion.vlm_parser import VlmOcrParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser(vlm_response: str | list[str]) -> VlmOcrParser:
    """Create a VlmOcrParser with a mock vision model that returns *vlm_response*.

    If *vlm_response* is a list, successive calls return successive items.
    """
    if isinstance(vlm_response, list):
        vision_func = AsyncMock(side_effect=vlm_response)
    else:
        vision_func = AsyncMock(return_value=vlm_response)
    return VlmOcrParser(vision_model_func=vision_func)


def _patch_renderer(parser: VlmOcrParser, page_count: int = 1):
    """Return a context manager that patches the parser's renderer.

    Produces *page_count* fake pages, each a MagicMock with a ``save`` method.
    """
    mock_images = []
    pages = []
    for i in range(page_count):
        img = MagicMock()
        img.save = MagicMock()
        mock_images.append(img)
        pages.append((i, img))

    ctx = patch.object(parser, "renderer")

    class _CM:
        def __enter__(self_cm):
            mock_renderer = ctx.__enter__()
            mock_renderer.render_file = AsyncMock(
                return_value=MagicMock(
                    pages=pages,
                    metadata={"original_format": "pdf", "page_count": page_count},
                )
            )
            self_cm.mock_renderer = mock_renderer
            self_cm.mock_images = mock_images
            return self_cm

        def __exit__(self_cm, *args):
            ctx.__exit__(*args)

    return _CM()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextAndHeadingEntries:
    """Headings become text entries with ``# `` prefix; text entries preserved."""

    async def test_text_and_heading_entries(self, tmp_path: Path) -> None:
        vlm_response = json.dumps(
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Introduction"},
                    {"type": "text", "text": "Some paragraph content."},
                    {"type": "heading", "level": 2, "text": "Background"},
                ]
            }
        )
        parser = _make_parser(vlm_response)
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, doc_id = await parser.parse("/fake/doc.pdf", output_dir)

        # Filter out the page-image entry
        text_entries = [e for e in content_list if e.get("text") is not None]

        assert len(text_entries) == 3
        assert text_entries[0]["type"] == "text"
        assert text_entries[0]["text"] == "# Introduction"
        assert text_entries[0]["text_level"] == 1

        assert text_entries[1]["type"] == "text"
        assert text_entries[1]["text"] == "Some paragraph content."

        assert text_entries[2]["type"] == "text"
        assert text_entries[2]["text"] == "## Background"
        assert text_entries[2]["text_level"] == 2


class TestTableEntries:
    """Table blocks become type='table' with HTML in table_body."""

    async def test_table_entries(self, tmp_path: Path) -> None:
        vlm_response = json.dumps(
            {
                "blocks": [
                    {
                        "type": "table",
                        "html": "<table><tr><th>A</th></tr></table>",
                    }
                ]
            }
        )
        parser = _make_parser(vlm_response)
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        table_entries = [e for e in content_list if e.get("type") == "table"]
        assert len(table_entries) == 1
        assert table_entries[0]["table_body"] == "<table><tr><th>A</th></tr></table>"
        assert table_entries[0]["page_idx"] == 0


class TestFigureEntries:
    """Figure blocks become image entries with img_path=None."""

    async def test_figure_entries_have_no_img_path(self, tmp_path: Path) -> None:
        vlm_response = json.dumps(
            {"blocks": [{"type": "figure", "description": "A bar chart showing revenue."}]}
        )
        parser = _make_parser(vlm_response)
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        figure_entries = [
            e for e in content_list if e.get("type") == "image" and e.get("img_path") is None
        ]
        assert len(figure_entries) == 1
        assert figure_entries[0]["image_caption"] == ["A bar chart showing revenue."]
        assert figure_entries[0]["page_idx"] == 0


class TestPageImageEntry:
    """Each page gets an image entry with img_path pointing to saved file."""

    async def test_page_image_entry_has_img_path(self, tmp_path: Path) -> None:
        vlm_response = json.dumps({"blocks": []})
        parser = _make_parser(vlm_response)
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        page_img_entries = [
            e for e in content_list if e.get("type") == "image" and e.get("img_path") is not None
        ]
        assert len(page_img_entries) == 1
        assert "page_0.png" in page_img_entries[0]["img_path"]
        assert page_img_entries[0]["page_idx"] == 0


class TestInvalidJsonFallback:
    """Invalid JSON response falls back to a single text block."""

    async def test_invalid_json_fallback(self, tmp_path: Path) -> None:
        parser = _make_parser("This is not valid JSON at all {{{")
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        # Should have the fallback text entry + page image entry
        text_entries = [e for e in content_list if e.get("type") == "text"]
        assert len(text_entries) == 1
        assert "not valid JSON" in text_entries[0]["text"]

        # Page image entry should also be present
        img_entries = [e for e in content_list if e.get("type") == "image"]
        assert len(img_entries) == 1


class TestEmptyPage:
    """Empty blocks list returns only the page image entry."""

    async def test_empty_page(self, tmp_path: Path) -> None:
        vlm_response = json.dumps({"blocks": []})
        parser = _make_parser(vlm_response)
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        assert len(content_list) == 1
        assert content_list[0]["type"] == "image"
        assert content_list[0]["img_path"] is not None
        assert content_list[0]["page_idx"] == 0


class TestMultiPage:
    """Multi-page documents produce entries with correct page_idx."""

    async def test_multi_page(self, tmp_path: Path) -> None:
        page0_response = json.dumps({"blocks": [{"type": "text", "text": "Page zero content."}]})
        page1_response = json.dumps(
            {"blocks": [{"type": "heading", "level": 1, "text": "Page One Title"}]}
        )
        parser = _make_parser([page0_response, page1_response])
        output_dir = str(tmp_path / "out")

        with _patch_renderer(parser, page_count=2):
            content_list, _ = await parser.parse("/fake/doc.pdf", output_dir)

        # Entries from page 0
        p0 = [e for e in content_list if e.get("page_idx") == 0]
        assert any(e.get("text") == "Page zero content." for e in p0)
        assert any(e.get("img_path") is not None for e in p0)

        # Entries from page 1
        p1 = [e for e in content_list if e.get("page_idx") == 1]
        assert any(e.get("text") == "# Page One Title" for e in p1)
        assert any(e.get("img_path") is not None for e in p1)


class TestVlmOcrEndToEnd:
    """End-to-end: VLM parser output survives IngestionPolicy filtering."""

    async def test_content_list_passes_policy_filter(self, tmp_path: Path) -> None:
        from dlightrag.core.ingestion.policy import IngestionPolicy

        vlm_response = json.dumps(
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Report"},
                    {"type": "text", "text": "Summary paragraph."},
                    {"type": "table", "html": "<table><tr><td>Data</td></tr></table>"},
                    {"type": "figure", "description": "Revenue chart."},
                ]
            }
        )

        parser = _make_parser(vlm_response)
        with _patch_renderer(parser):
            content_list, _ = await parser.parse(str(tmp_path / "test.pdf"), str(tmp_path / "out"))

        policy = IngestionPolicy()
        result = policy.apply(content_list)
        assert result.stats.indexed > 0
        assert result.stats.dropped_by_type == 0
