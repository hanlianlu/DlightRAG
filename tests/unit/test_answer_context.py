# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-context packing."""

from dlightrag.core.answer.context import AnswerContextPacker
from dlightrag.core.answer.images import AnswerImageBudget
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
    assert packed.trace["answer_context_images_skipped"] == 1
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


def test_composer_visual_does_not_evict_rag_visual() -> None:
    composer = {
        "chunk_id": "composer-visual",
        "reference_id": "composer_doc",
        "file_path": "upload.pdf",
        "content": "Uploaded figure",
        "image_data": _PNG_B64,
        "metadata": {"source_type": "web_attachment"},
    }
    rag = {
        "chunk_id": "rag-visual",
        "reference_id": "rag-doc",
        "file_path": "workspace.pdf",
        "content": "Workspace figure",
        "image_data": _PNG_B64,
        "metadata": {"source_type": "file"},
    }

    rag_only = AnswerContextPacker().pack(
        {"chunks": [rag], "entities": [], "relationships": []},
        image_budget=_budget(max_images=1),
        context_top_k=1,
    )
    with_composer = AnswerContextPacker().pack(
        {"chunks": [composer, rag], "entities": [], "relationships": []},
        image_budget=_budget(max_images=1),
        context_top_k=2,
    )

    rag_only_row = rag_only.contexts["chunks"][0]
    merged_rag_row = next(
        row for row in with_composer.contexts["chunks"] if row["chunk_id"] == "rag-visual"
    )
    assert merged_rag_row == rag_only_row
    assert "rag-visual" in with_composer.image_blocks_by_chunk_id
    assert "composer-visual" not in with_composer.image_blocks_by_chunk_id
    assert [row["chunk_id"] for row in with_composer.contexts["chunks"]] == [
        "composer-visual",
        "rag-visual",
    ]


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


def test_packer_drops_kg_items_without_matching_source_chunks() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "text-1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "First text candidate.",
            },
        ],
        "entities": [
            {"entity_name": "Revenue", "description": "Growth", "source_id": "text-1"},
            {"entity_name": "Unsourced", "description": "No provenance"},
            {"entity_name": "Skipped", "description": "Not packed", "source_id": "text-2"},
        ],
        "relationships": [
            {
                "src_id": "Revenue",
                "tgt_id": "Report",
                "description": "Appears in the packed chunk",
                "source_id": "text-1",
            },
            {
                "src_id": "Unsourced",
                "tgt_id": "Report",
                "description": "No provenance",
            },
            {
                "src_id": "Skipped",
                "tgt_id": "Report",
                "description": "Not packed",
                "source_id": "text-2",
            },
        ],
    }

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=0))

    assert [e["entity_name"] for e in packed.contexts["entities"]] == ["Revenue"]
    assert [r["src_id"] for r in packed.contexts["relationships"]] == ["Revenue"]


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
