# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-context packing."""

from __future__ import annotations

from dlightrag.core.answer_context import AnswerContextPacker
from dlightrag.core.answer_images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_packer_skips_image_only_chunks_when_image_budget_is_exhausted() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "visual-1",
                "reference_id": "1",
                "file_path": "/docs/figures.pdf",
                "content": "",
                "image_data": _PNG_B64,
            },
            {
                "chunk_id": "text-1",
                "reference_id": "2",
                "file_path": "/docs/report.pdf",
                "content": "Revenue grew 15%.",
            },
        ],
        "entities": [{"entity_name": "Figure", "description": "Skipped", "source_id": "visual-1"}],
        "relationships": [],
    }
    budget = _budget(max_images=0)

    packed = AnswerContextPacker().pack(contexts, image_budget=budget)

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["text-1"]
    assert packed.image_blocks_by_chunk_id == {}
    assert packed.trace["answer_context_skipped_image_only_chunks"] == 1
    assert packed.contexts["entities"] == []


def test_packer_keeps_text_when_chunk_image_does_not_fit() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "mixed-1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "The chart shows revenue growth.",
                "image_data": _PNG_B64,
            },
        ],
        "entities": [{"entity_name": "Revenue", "description": "Growth", "source_id": "mixed-1"}],
        "relationships": [],
    }
    budget = _budget(max_images=0)

    packed = AnswerContextPacker().pack(contexts, image_budget=budget)

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["mixed-1"]
    assert packed.image_blocks_by_chunk_id == {}
    assert packed.contexts["chunks"][0]["content"] == "The chart shows revenue growth."
    assert packed.trace["answer_context_images_skipped"] == 1
    assert packed.contexts["entities"][0]["entity_name"] == "Revenue"


def test_packer_keeps_fitting_image_blocks_by_chunk_id() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "visual-1",
                "reference_id": "1",
                "file_path": "/docs/figures.pdf",
                "content": "",
                "image_data": _PNG_B64,
            },
        ],
        "entities": [],
        "relationships": [],
    }
    budget = _budget(max_images=1)

    packed = AnswerContextPacker().pack(contexts, image_budget=budget)

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["visual-1"]
    assert packed.image_blocks_by_chunk_id["visual-1"]["type"] == "image_url"
    assert packed.trace["answer_context_images_sent"] == 1


def test_packer_backfills_answer_context_after_skipping_image_only_chunks() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "visual-1",
                "reference_id": "1",
                "file_path": "/docs/figures.pdf",
                "content": "",
                "image_data": _PNG_B64,
            },
            {
                "chunk_id": "text-1",
                "reference_id": "2",
                "file_path": "/docs/report.pdf",
                "content": "First text candidate.",
            },
            {
                "chunk_id": "text-2",
                "reference_id": "3",
                "file_path": "/docs/report.pdf",
                "content": "Second text candidate.",
            },
        ],
        "entities": [],
        "relationships": [],
    }

    packed = AnswerContextPacker().pack(
        contexts,
        image_budget=_budget(max_images=0),
        context_top_k=2,
    )

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["text-1", "text-2"]
    assert packed.trace["answer_context_input_chunks"] == 3
    assert packed.trace["answer_context_candidate_chunks"] == 3
    assert packed.trace["answer_context_target_chunks"] == 2
    assert packed.trace["answer_context_chunks"] == 2


def _budget(*, max_images: int) -> AnswerImageBudget:
    return AnswerImageBudget(
        max_images=max_images,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )
