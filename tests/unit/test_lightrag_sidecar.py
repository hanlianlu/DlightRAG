# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG sidecar reference collection."""

from __future__ import annotations

import json

from dlightrag.core.ingestion.lightrag_sidecar import collect_sidecar_refs


def test_collects_drawing_table_equation_refs(tmp_path) -> None:
    artifact_dir = tmp_path / "doc"
    artifact_dir.mkdir()
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "drawings": {
                    "fig-1": {
                        "id": "fig-1",
                        "path": "sample.blocks.assets/fig.png",
                        "page": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "sample.tables.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "tables": {
                    "table-1": {
                        "id": "table-1",
                        "llm_analyze_result": {"status": "success"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "sample.equations.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "equations": {
                    "eq-1": {
                        "id": "eq-1",
                        "llm_analyze_result": {"status": "success"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    refs = collect_sidecar_refs(artifact_dir)

    assert [(r.sidecar_type, r.sidecar_id) for r in refs] == [
        ("drawing", "fig-1"),
        ("table", "table-1"),
        ("equation", "eq-1"),
    ]
    assert refs[0].asset_path == (artifact_dir / "sample.blocks.assets/fig.png").resolve()
    assert refs[0].page_index == 1


def test_collects_page_index_from_block_positions(tmp_path) -> None:
    artifact_dir = tmp_path / "doc"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text(
        json.dumps(
            {
                "type": "content",
                "blockid": "block-1",
                "positions": [{"type": "bbox", "anchor": 3, "range": [1, 2, 3, 4]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "fig-1": {
                        "id": "fig-1",
                        "blockid": "block-1",
                        "path": "sample.blocks.assets/fig.png",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    refs = collect_sidecar_refs(artifact_dir)

    assert refs[0].block_id == "block-1"
    assert refs[0].page_index == 3
    assert refs[0].bbox == {"page_index": 3, "range": [1, 2, 3, 4]}
