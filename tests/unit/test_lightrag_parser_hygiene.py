# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's LightRAG parser hygiene boundary."""

from pathlib import Path

from dlightrag.core.ingestion.parser_hygiene import (
    apply_mineru_auxiliary_block_filter,
    filter_mineru_auxiliary_blocks,
    mineru_ir_builder_needs_auxiliary_filter,
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

    apply_mineru_auxiliary_block_filter()
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
