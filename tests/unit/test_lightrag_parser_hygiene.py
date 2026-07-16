# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's LightRAG parser hygiene boundary."""

from pathlib import Path

from dlightrag.core.ingestion.parser_hygiene import (
    apply_mineru_content_list_hygiene,
    filter_mineru_auxiliary_blocks,
    mineru_ir_builder_needs_auxiliary_filter,
    mineru_ir_builder_needs_drawing_alias_normalization,
    normalize_mineru_drawing_aliases,
)


def test_mineru_auxiliary_filter_preserves_semantic_and_multimodal_items() -> None:
    content_list = [
        {"type": "header", "text": "Running header"},
        {"type": "footer", "text": "Running footer"},
        {"type": "page_number", "text": "12"},
        {"type": "aside_text", "text": "Margin note"},
        {"type": "discarded_blocks", "text": "Discarded artifact"},
        {"type": "margin_note", "text": "Printer note"},
        {"type": "page_footnote", "text": "Page footnote"},
        {"type": "text", "text": "Body text"},
        {"type": "list", "list_items": ["one", "two"]},
        {"type": "code", "code_body": "print('x')"},
        {"type": "equation", "text": "$x=1$"},
        {"type": "table", "table_body": "<table><tr><td>A</td></tr></table>"},
        {"type": "image", "img_path": "images/figure.png"},
        {"type": "chart", "content": "chart data"},
    ]

    filtered = filter_mineru_auxiliary_blocks(content_list)

    assert [item["type"] for item in filtered] == [
        "aside_text",
        "margin_note",
        "page_footnote",
        "text",
        "list",
        "code",
        "equation",
        "table",
        "image",
        "chart",
    ]


def test_mineru_auxiliary_filter_can_drop_extended_page_notes(monkeypatch) -> None:
    monkeypatch.setenv("DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY", "extended")
    content_list = [
        {"type": "header", "text": "Running header"},
        {"type": "aside_text", "text": "Margin note"},
        {"type": "page_footnote", "text": "Page footnote"},
        {"type": "text", "text": "Body text"},
    ]

    filtered = filter_mineru_auxiliary_blocks(content_list)

    assert [item["type"] for item in filtered] == ["text"]


def test_mineru_ir_builder_patch_drops_auxiliary_page_furniture(tmp_path: Path) -> None:
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    apply_mineru_content_list_hygiene()
    assert mineru_ir_builder_needs_auxiliary_filter(MinerUIRBuilder) is False

    doc = MinerUIRBuilder()._normalize_content_list(
        [
            {"type": "header", "text": "Journal header", "page_idx": 0},
            {"type": "footer", "text": "Publisher footer", "page_idx": 0},
            {"type": "page_number", "text": "42", "page_idx": 0},
            {"type": "page_aside_text", "content": "Margin note", "page_idx": 0},
            {"type": "page_footnote", "text": "Author footnote", "page_idx": 0},
            {"type": "text", "text": "Main body survives", "page_idx": 0},
        ],
        tmp_path,
        document_name="sample.pdf",
    )

    content = "\n".join(block.content_template for block in doc.blocks)
    assert "Main body survives" in content
    assert "Journal header" not in content
    assert "Publisher footer" not in content
    assert "42" not in content
    assert "Margin note" in content
    assert "Author footnote" in content


# ---------------------------------------------------------------------------
# Drawing-alias normalization (MinerU ``chart`` → ``image``)
# ---------------------------------------------------------------------------


def test_normalize_drawing_aliases_remaps_chart_to_image() -> None:
    content_list = [
        {
            "type": "chart",
            "img_path": "images/fig1.jpg",
            "content": "",
            "chart_caption": ["Figure 1. Foreign Reserves"],
            "chart_footnote": ["Note: annual series."],
            "bbox": [1, 2, 3, 4],
            "page_idx": 0,
        },
        {"type": "text", "text": "Body"},
    ]

    chart, text = normalize_mineru_drawing_aliases(content_list)

    assert chart["type"] == "image"
    assert chart["img_path"] == "images/fig1.jpg"
    assert chart["image_caption"] == ["Figure 1. Foreign Reserves"]
    assert chart["image_footnote"] == ["Note: annual series."]
    assert "chart_caption" not in chart
    assert "chart_footnote" not in chart
    assert chart["bbox"] == [1, 2, 3, 4]
    assert chart["page_idx"] == 0
    assert text == {"type": "text", "text": "Body"}


def test_normalize_drawing_aliases_does_not_clobber_existing_image_fields() -> None:
    content_list = [
        {
            "type": "chart",
            "img_path": "images/fig.jpg",
            "image_caption": ["kept"],
            "chart_caption": ["ignored"],
        }
    ]

    (item,) = normalize_mineru_drawing_aliases(content_list)

    assert item["type"] == "image"
    assert item["image_caption"] == ["kept"]
    assert "chart_caption" not in item


def test_normalize_drawing_aliases_leaves_non_alias_items_unchanged() -> None:
    content_list = [
        {"type": "image", "img_path": "a.jpg"},
        {"type": "table", "table_body": "<table></table>"},
        {"type": "equation", "text": "$x$"},
        {"type": "text", "text": "hi"},
    ]

    assert normalize_mineru_drawing_aliases(content_list) == content_list


def test_drawing_alias_probe_detects_dropped_vs_handled_charts() -> None:
    class _Block:
        def __init__(self, content_template: str) -> None:
            self.content_template = content_template

    class _Doc:
        def __init__(self, blocks: list[_Block]) -> None:
            self.blocks = blocks

    class _DropsChartBuilder:
        def _normalize_content_list(self, content_list, raw_dir, *, document_name):
            return _Doc([_Block("plain text, chart dropped")])

    class _HandlesChartBuilder:
        def _normalize_content_list(self, content_list, raw_dir, *, document_name):
            return _Doc([_Block("{{IMG:im0}}")])

    assert mineru_ir_builder_needs_drawing_alias_normalization(_DropsChartBuilder) is True
    assert mineru_ir_builder_needs_drawing_alias_normalization(_HandlesChartBuilder) is False


def test_mineru_hygiene_routes_chart_through_drawing(tmp_path: Path) -> None:
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    apply_mineru_content_list_hygiene()  # idempotent

    doc = MinerUIRBuilder()._normalize_content_list(
        [
            {
                "type": "chart",
                "img_path": "images/fig1.jpg",
                "content": "",
                "chart_caption": ["Figure 1. Foreign Reserves"],
                "page_idx": 0,
            },
            {"type": "text", "text": "Body text survives", "page_idx": 0},
        ],
        tmp_path,
        document_name="sample.pdf",
    )

    content = "\n".join(block.content_template for block in doc.blocks)
    assert "{{IMG:" in content
    assert "Body text survives" in content
