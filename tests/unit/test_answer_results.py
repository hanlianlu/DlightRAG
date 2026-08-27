# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Answer results expose shared usage and Evidence summaries."""

from dlightrag.application.answer_runs.results import (
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
        evidence_images=[],
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
    assert restored.artifact_outcome.status == "complete"
    assert projected["parts"] == [{"type": "markdown", "text": "Grounded answer."}]


def test_parts_derive_artifact_and_inline_evidence_placements() -> None:
    artifact = {
        "resource_id": "artifact-report",
        "role": "primary_report",
        "media_type": "text/markdown",
        "label": "Report",
        "filename": "report.md",
        "byte_size": 10,
        "digest": "a" * 64,
        "presentation": "markdown",
        "status": "available",
    }
    stored = {
        "answer": (
            "Intro. [View report](artifact:artifact-report) ![Inline chart](evidence:chart-1) End."
        ),
        "sources": [],
        "contexts": {},
        "evidence_images": [
            {
                "id": "chart-1",
                "chunk_id": "chunk-1",
                "workspace": "default",
                "source_ref": "1",
                "label": "Chart",
            }
        ],
        "artifacts": [artifact],
        "artifact_outcome": {"status": "complete", "issues": []},
    }

    projected = project_answer_result(
        stored,
        run_id="run-1",
        artifact_url_prefix="/answer",
    )

    assert [part["type"] for part in projected["parts"]] == [
        "markdown",
        "artifact",
        "markdown",
        "evidence_image",
        "markdown",
    ]
    assert projected["parts"][1]["artifact"]["role"] == "primary_report"
    assert projected["parts"][1]["artifact"]["data_url"].endswith(
        "/run-1/artifacts/artifact-report"
    )
    assert projected["parts"][3]["evidence_image"]["source_ref"] == "1"
