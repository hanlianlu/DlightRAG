# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal sidecar provenance resolution."""

import json
from pathlib import Path

import pytest

from dlightrag.core.sidecar_provenance import (
    block_ids_from_multimodal_item,
    is_multimodal_sidecar,
)


def _mm_sidecar(kind: str, item_id: str) -> dict[str, object]:
    return {"type": kind, "id": item_id, "refs": [{"type": kind, "id": item_id}]}


def _write_items(path: Path, root_key: str, items: dict[str, dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"version": "1.0", root_key: items}, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("sidecar", "expected"),
    [
        ({"type": "table"}, True),
        ({"type": "drawing"}, True),
        ({"type": "equation"}, True),
        ({"type": "block"}, False),
        ({}, False),
    ],
)
def test_is_multimodal_sidecar(sidecar: dict[str, object], expected: bool) -> None:
    assert is_multimodal_sidecar(sidecar) is expected


@pytest.mark.parametrize(
    ("kind", "filename", "root_key", "item_id"),
    [
        ("table", "doc.tables.json", "tables", "tb-doc-0001"),
        ("drawing", "doc.drawings.json", "drawings", "im-doc-0001"),
        ("equation", "doc.equations.json", "equations", "eq-doc-0001"),
    ],
)
def test_block_ids_from_multimodal_item_resolves_blockid(
    tmp_path: Path, kind: str, filename: str, root_key: str, item_id: str
) -> None:
    _write_items(tmp_path / filename, root_key, {item_id: {"id": item_id, "blockid": "block-7"}})

    assert block_ids_from_multimodal_item(tmp_path, _mm_sidecar(kind, item_id)) == ["block-7"]


def test_block_ids_from_multimodal_item_returns_empty_for_missing_item(tmp_path: Path) -> None:
    _write_items(tmp_path / "doc.tables.json", "tables", {"tb-1": {"blockid": "block-1"}})

    assert block_ids_from_multimodal_item(tmp_path, _mm_sidecar("table", "tb-absent")) == []


def test_block_ids_from_multimodal_item_ignores_non_multimodal_sidecar(tmp_path: Path) -> None:
    _write_items(tmp_path / "doc.tables.json", "tables", {"tb-1": {"blockid": "block-1"}})

    assert block_ids_from_multimodal_item(tmp_path, {"type": "block", "id": "block-1"}) == []


def test_block_ids_from_multimodal_item_tolerates_missing_or_bad_file(tmp_path: Path) -> None:
    # No file at all.
    assert block_ids_from_multimodal_item(tmp_path, _mm_sidecar("table", "tb-1")) == []

    # Unparseable file is skipped rather than raising.
    (tmp_path / "doc.tables.json").write_text("{ not json", encoding="utf-8")
    assert block_ids_from_multimodal_item(tmp_path, _mm_sidecar("table", "tb-1")) == []
