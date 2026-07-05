# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for browser-facing answer event presentation."""

from dlightrag.web.answer_events import _retrieved_image_cards


def test_retrieved_image_cards_uses_visual_chunk_route_without_inline_image_data() -> None:
    cards = _retrieved_image_cards(
        flat_contexts=[
            {
                "chunk_id": "doc-1-mm-drawing-001",
                "file_path": "/docs/hist.pdf",
                "_workspace": "default",
                "content": "[Image Name]hist_overall_architecture_diagram",
                "sidecar": {"type": "drawing"},
            }
        ],
        seen_img_ids=set(),
        workspace="default",
        default_workspace="default",
        existing_count=0,
    )

    assert cards == [
        {
            "chunk_id": "doc-1-mm-drawing-001",
            "url": "/web/images/default/doc-1-mm-drawing-001?size=full",
            "thumb_url": "/web/images/default/doc-1-mm-drawing-001?size=thumb",
            "label": "/docs/hist.pdf",
        }
    ]


def test_retrieved_image_cards_skips_text_chunk_without_image_data() -> None:
    cards = _retrieved_image_cards(
        flat_contexts=[
            {
                "chunk_id": "doc-1-chunk-001",
                "file_path": "/docs/hist.pdf",
                "_workspace": "default",
                "content": "plain text",
            }
        ],
        seen_img_ids=set(),
        workspace="default",
        default_workspace="default",
        existing_count=0,
    )

    assert cards == []
