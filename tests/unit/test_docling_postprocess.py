# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Docling JSON post-processing module."""

from __future__ import annotations

import json
from pathlib import Path

from dlightrag.core.ingestion.docling_postprocess import (
    rebuild_text_items_from_docling_json,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_docling_json(tmp_path: Path, data: dict) -> Path:
    """Write a Docling JSON fixture to a temp file and return its path."""
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _minimal_doc(
    texts: list[dict] | None = None,
    groups: list[dict] | None = None,
    body_children: list[dict] | None = None,
) -> dict:
    """Build a minimal Docling JSON document.

    If body_children is None, auto-generate refs from texts/groups lengths.
    """
    texts = texts or []
    groups = groups or []
    if body_children is None:
        body_children = [{"$ref": f"texts/{i}"} for i in range(len(texts))]
        body_children += [{"$ref": f"groups/{i}"} for i in range(len(groups))]
    return {
        "body": {"children": body_children},
        "texts": texts,
        "groups": groups,
        "pictures": [],
        "tables": [],
    }


# ---------------------------------------------------------------------------
# TestHeadingMarkers
# ---------------------------------------------------------------------------


class TestHeadingMarkers:
    """section_header and title blocks get markdown heading prefixes."""

    def test_section_header_level_1(self, tmp_path: Path) -> None:
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Introduction",
                    "label": "section_header",
                    "level": 1,
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 12]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "# Introduction"

    def test_section_header_level_2(self, tmp_path: Path) -> None:
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Background",
                    "label": "section_header",
                    "level": 2,
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 10]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert result[0]["text"] == "## Background"

    def test_section_header_default_level(self, tmp_path: Path) -> None:
        """section_header without explicit level defaults to 1."""
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "NoLevel",
                    "label": "section_header",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 7]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert result[0]["text"] == "# NoLevel"

    def test_title_becomes_h1(self, tmp_path: Path) -> None:
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Document Title",
                    "label": "title",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 14]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert result[0]["text"] == "# Document Title"


# ---------------------------------------------------------------------------
# TestPageIdx
# ---------------------------------------------------------------------------


class TestPageIdx:
    """page_idx is sourced from prov[0].page_no."""

    def test_page_idx_from_prov(self, tmp_path: Path) -> None:
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Page zero",
                    "label": "paragraph",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 9]}],
                },
                {
                    "orig": "Page three",
                    "label": "paragraph",
                    "prov": [{"page_no": 3, "bbox": {}, "charspan": [0, 10]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 2
        assert result[0]["page_idx"] == 0
        assert result[1]["page_idx"] == 3

    def test_each_item_has_own_page_idx(self, tmp_path: Path) -> None:
        """Items are NOT merged; each preserves its individual page_idx."""
        doc = _minimal_doc(
            texts=[
                {
                    "orig": f"Block {i}",
                    "label": "paragraph",
                    "prov": [{"page_no": i, "bbox": {}, "charspan": [0, 7]}],
                }
                for i in range(4)
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 4
        for i, item in enumerate(result):
            assert item["page_idx"] == i


# ---------------------------------------------------------------------------
# TestFormula
# ---------------------------------------------------------------------------


class TestFormula:
    """Formula blocks become type=equation items."""

    def test_formula_type_equation(self, tmp_path: Path) -> None:
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "E = mc^2",
                    "label": "formula",
                    "prov": [{"page_no": 2, "bbox": {}, "charspan": [0, 8]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "equation"
        assert result[0]["text"] == "E = mc^2"
        assert result[0]["page_idx"] == 2

    def test_formula_among_text(self, tmp_path: Path) -> None:
        """A formula surrounded by paragraphs has correct types."""
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Before",
                    "label": "paragraph",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 6]}],
                },
                {
                    "orig": "x + y = z",
                    "label": "formula",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 9]}],
                },
                {
                    "orig": "After",
                    "label": "paragraph",
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 5]}],
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "equation"
        assert result[2]["type"] == "text"


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Graceful failure returns None so caller falls back."""

    def test_missing_file(self) -> None:
        assert rebuild_text_items_from_docling_json(Path("/nonexistent/path.json")) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not { valid json !!!", encoding="utf-8")
        assert rebuild_text_items_from_docling_json(p) is None

    def test_missing_body_key(self, tmp_path: Path) -> None:
        p = tmp_path / "nobody.json"
        p.write_text(json.dumps({"texts": [], "groups": []}), encoding="utf-8")
        assert rebuild_text_items_from_docling_json(p) is None

    def test_missing_prov_defaults_page_idx_zero(self, tmp_path: Path) -> None:
        """A text block with no prov gets page_idx=0."""
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "No provenance",
                    "label": "paragraph",
                },
            ],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert result[0]["page_idx"] == 0


# ---------------------------------------------------------------------------
# TestNestedGroups
# ---------------------------------------------------------------------------


class TestNestedGroups:
    """Blocks nested inside groups are flattened into the output."""

    def test_single_group(self, tmp_path: Path) -> None:
        """A body -> group -> text chain is resolved correctly."""
        doc = _minimal_doc(
            texts=[
                {
                    "orig": "Grouped paragraph",
                    "label": "paragraph",
                    "prov": [{"page_no": 1, "bbox": {}, "charspan": [0, 17]}],
                },
            ],
            groups=[
                {"children": [{"$ref": "texts/0"}]},
            ],
            body_children=[{"$ref": "groups/0"}],
        )
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Grouped paragraph"
        assert result[0]["page_idx"] == 1

    def test_nested_groups(self, tmp_path: Path) -> None:
        """group -> group -> text resolves recursively."""
        doc = {
            "body": {"children": [{"$ref": "groups/0"}]},
            "texts": [
                {
                    "orig": "Deeply nested",
                    "label": "paragraph",
                    "prov": [{"page_no": 5, "bbox": {}, "charspan": [0, 13]}],
                },
            ],
            "groups": [
                {"children": [{"$ref": "groups/1"}]},
                {"children": [{"$ref": "texts/0"}]},
            ],
            "pictures": [],
            "tables": [],
        }
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 1
        assert result[0]["text"] == "Deeply nested"
        assert result[0]["page_idx"] == 5

    def test_mixed_body_refs(self, tmp_path: Path) -> None:
        """Body with direct text refs and group refs produces correct order."""
        doc = {
            "body": {
                "children": [
                    {"$ref": "texts/0"},
                    {"$ref": "groups/0"},
                    {"$ref": "texts/2"},
                ],
            },
            "texts": [
                {
                    "orig": "First",
                    "label": "section_header",
                    "level": 1,
                    "prov": [{"page_no": 0, "bbox": {}, "charspan": [0, 5]}],
                },
                {
                    "orig": "Inside group",
                    "label": "paragraph",
                    "prov": [{"page_no": 1, "bbox": {}, "charspan": [0, 12]}],
                },
                {
                    "orig": "Last",
                    "label": "paragraph",
                    "prov": [{"page_no": 2, "bbox": {}, "charspan": [0, 4]}],
                },
            ],
            "groups": [
                {"children": [{"$ref": "texts/1"}]},
            ],
            "pictures": [],
            "tables": [],
        }
        result = rebuild_text_items_from_docling_json(_write_docling_json(tmp_path, doc))
        assert result is not None
        assert len(result) == 3
        assert result[0]["text"] == "# First"
        assert result[0]["page_idx"] == 0
        assert result[1]["text"] == "Inside group"
        assert result[1]["page_idx"] == 1
        assert result[2]["text"] == "Last"
        assert result[2]["page_idx"] == 2
