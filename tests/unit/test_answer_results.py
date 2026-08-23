# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Answer results expose shared usage and Evidence summaries."""

from dlightrag.answer.runs.results import (
    project_answer_result,
    restore_answer_result,
    store_answer_result,
)


def test_usage_and_evidence_round_trip_on_every_projection() -> None:
    stored = store_answer_result(
        answer="Grounded answer.",
        contexts={
            "chunks": [{"chunk_id": "c1", "content": "fact", "metadata": {}}],
            "entities": [{"entity": "one"}],
            "relationships": [],
        },
        sources=[],
        answer_images=[],
        trace={"usage": {"usage_details": {"total_tokens": 12}}},
        image_descriptions=[],
    )

    assert stored["usage"] == {"usage_details": {"total_tokens": 12}}
    assert stored["evidence"] == {
        "chunks": 1,
        "entities": 1,
        "relationships": 0,
        "sources": 0,
    }
    restored = restore_answer_result(stored)
    projected = project_answer_result(stored)
    assert restored.usage == stored["usage"]
    assert restored.evidence == stored["evidence"]
    assert projected["usage"] == stored["usage"]
    assert projected["evidence"] == stored["evidence"]
