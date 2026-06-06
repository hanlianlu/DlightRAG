# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG sidecar drawing asset resolution."""

from __future__ import annotations

import json

from dlightrag.core.ingestion.lightrag_sidecar import collect_lightrag_drawing_assets


def test_collects_successful_drawing_assets_for_lightrag_mm_chunks(tmp_path) -> None:
    artifact_dir = tmp_path / "doc"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("", encoding="utf-8")
    assets_dir = artifact_dir / "sample.blocks.assets"
    assets_dir.mkdir()
    image_path = assets_dir / "fig.png"
    image_path.write_bytes(b"image")
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "ignored": {
                        "id": "ignored",
                        "path": "sample.blocks.assets/ignored.png",
                        "llm_analyze_result": {"status": "skipped"},
                    },
                    "fig-1": {
                        "id": "fig-1",
                        "img_path": "sample.blocks.assets/fig.png",
                        "llm_analyze_result": {
                            "status": "success",
                            "name": "Figure",
                            "type": "Illustration",
                            "description": "A figure.",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assets = collect_lightrag_drawing_assets(artifact_dir, doc_id="doc-abc")

    assert len(assets) == 1
    assert assets[0].chunk_id == "doc-abc-mm-drawing-001"
    assert assets[0].sidecar_id == "fig-1"
    assert assets[0].image_path == image_path.resolve()


def test_requires_blocks_anchor_to_match_lightrag_builder(tmp_path) -> None:
    artifact_dir = tmp_path / "doc"
    artifact_dir.mkdir()
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "fig-1": {
                        "id": "fig-1",
                        "path": "sample.blocks.assets/fig.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert collect_lightrag_drawing_assets(artifact_dir, doc_id="doc-abc") == []
