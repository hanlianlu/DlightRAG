# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's LightRAG parser hygiene boundary."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
    apply_mineru_content_list_hygiene,
    filter_mineru_auxiliary_blocks,
    normalize_mineru_drawing_aliases,
)


def _unpatched_normalize_content_list():
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    apply_mineru_content_list_hygiene()
    original = getattr(MinerUIRBuilder._normalize_content_list, "__wrapped__", None)
    assert original is not None, "MinerU hygiene patch did not install"
    return MinerUIRBuilder, original


def test_active_mineru_patch_fails_closed_when_upstream_contract_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incompatible_builder = ModuleType("lightrag.parser.external.mineru.ir_builder")
    monkeypatch.setitem(
        sys.modules,
        "lightrag.parser.external.mineru.ir_builder",
        incompatible_builder,
    )

    with pytest.raises(ImportError):
        apply_mineru_content_list_hygiene()


def test_mineru_auxiliary_filter_preserves_semantic_and_upstream_owned_items() -> None:
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
        "page_number",
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


def test_mineru_ir_builder_patch_drops_auxiliary_page_furniture(tmp_path: Path) -> None:
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    apply_mineru_content_list_hygiene()

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


def test_upstream_still_drops_charts_and_keeps_page_furniture(tmp_path: Path) -> None:
    """Drift alarm: drop the matching transform once this starts failing."""
    builder_cls, upstream = _unpatched_normalize_content_list()

    chart_doc = upstream(
        builder_cls(),
        [{"type": "chart", "img_path": "images/fig1.jpg", "chart_caption": ["Figure 1"]}],
        tmp_path,
        document_name="sample.pdf",
    )
    assert "{{IMG:" not in "\n".join(block.content_template for block in chart_doc.blocks)

    furniture_doc = upstream(
        builder_cls(),
        [
            {"type": "header", "text": "Journal header"},
            {"type": "text", "text": "Main body"},
        ],
        tmp_path,
        document_name="sample.pdf",
    )
    content = "\n".join(block.content_template for block in furniture_doc.blocks)
    assert "Journal header" in content
    assert "Main body" in content


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


# ---------------------------------------------------------------------------
# Payload-less media normalization (``table``/``equation`` with an image only)
# ---------------------------------------------------------------------------

# Verbatim MinerU output for a full-page line drawing: the table model claimed
# the page, every cell is empty, and the only surviving content is ``img_path``.
_PAYLOADLESS_TABLE = {
    "type": "table",
    "img_path": "images/3307e3fb8f0b213dbfe9e8dc0be195ca4c1e40482c95e5ceca61b036ee985c08.jpg",
    "table_caption": ["Plate 4"],
    "table_footnote": ["Note: untitled."],
    "table_body": "<table><tr><td></td><td></td></tr></table>",
    "bbox": [272, 0, 854, 998],
    "page_idx": 0,
}


def test_payloadless_media_routes_empty_table_with_image_to_drawing() -> None:
    from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
        normalize_mineru_payloadless_media,
    )

    converted, text = normalize_mineru_payloadless_media(
        [dict(_PAYLOADLESS_TABLE), {"type": "text", "text": "Body"}]
    )

    assert converted["type"] == "image"
    assert converted["img_path"] == _PAYLOADLESS_TABLE["img_path"]
    assert converted["image_caption"] == ["Plate 4"]
    assert converted["image_footnote"] == ["Note: untitled."]
    assert "table_caption" not in converted
    assert converted["bbox"] == [272, 0, 854, 998]
    assert converted["page_idx"] == 0
    assert text == {"type": "text", "text": "Body"}


def test_payloadless_media_keeps_tables_that_carry_visible_text() -> None:
    from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
        normalize_mineru_payloadless_media,
    )

    with_text = {
        "type": "table",
        "img_path": "images/table.jpg",
        "table_body": "<table><tr><td>Region</td><td>1.9</td></tr></table>",
    }
    rows_only = {"type": "table", "rows": [["Region", "1.9"]], "img_path": "images/t.jpg"}
    nbsp_only = {
        "type": "table",
        "img_path": "images/blank.jpg",
        "table_body": "<table><tr><td>&nbsp;</td></tr></table>",
    }

    converted = normalize_mineru_payloadless_media([with_text, rows_only, nbsp_only])

    assert converted[0] == with_text
    assert converted[1] == rows_only
    assert converted[2]["type"] == "image"


def test_payloadless_media_requires_a_materialized_image() -> None:
    from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
        normalize_mineru_payloadless_media,
    )

    no_image = {"type": "table", "table_body": "<table><tr><td></td></tr></table>"}
    blank_image = {"type": "equation", "img_path": "   ", "text": ""}

    assert normalize_mineru_payloadless_media([no_image, blank_image]) == [
        no_image,
        blank_image,
    ]


def test_payloadless_media_converts_empty_equation_items() -> None:
    from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
        normalize_mineru_payloadless_media,
    )

    empty = {"type": "equation", "img_path": "images/eq1.png", "text": "", "latex": ""}
    real = {"type": "equation", "img_path": "images/eq2.png", "text": "$x^2$"}

    converted = normalize_mineru_payloadless_media([empty, real])

    assert converted[0]["type"] == "image"
    assert converted[0]["img_path"] == "images/eq1.png"
    assert converted[1] == real


def test_mineru_hygiene_routes_payloadless_table_through_drawing(tmp_path: Path) -> None:
    from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

    apply_mineru_content_list_hygiene()  # idempotent

    doc = MinerUIRBuilder()._normalize_content_list(
        [dict(_PAYLOADLESS_TABLE), {"type": "text", "text": "Body text survives"}],
        tmp_path,
        document_name="plate.jpg",
    )

    content = "\n".join(block.content_template for block in doc.blocks)
    assert "{{IMG:" in content
    assert "{{TBL:" not in content
    assert "Body text survives" in content


def test_upstream_still_keeps_a_payloadless_table(tmp_path: Path) -> None:
    """Drift alarm: drop the matching transform once this starts failing."""

    builder_cls, upstream = _unpatched_normalize_content_list()

    doc = upstream(
        builder_cls(),
        [dict(_PAYLOADLESS_TABLE)],
        tmp_path,
        document_name="plate.jpg",
    )

    content = "\n".join(block.content_template for block in doc.blocks)
    assert "{{TBL:" in content
    assert "{{IMG:" not in content
