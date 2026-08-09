# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-context packing with one shared image budget."""

from dlightrag.core.answer.context import AnswerContextPacker
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import RetrievalContexts

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _budget(*, max_images: int, max_total_bytes: int = 24_000_000) -> AnswerImageBudget:
    return AnswerImageBudget(
        max_images=max_images,
        max_total_bytes=max_total_bytes,
        max_bytes_per_image=3_000_000,
        max_pixels=40_000_000,
        max_px=1536,
        min_px=1024,
        quality=89,
        min_quality=79,
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

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=0))

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["text-1"]
    assert packed.image_blocks_by_context_key == {}
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

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=0))

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["mixed-1"]
    assert packed.image_blocks_by_context_key == {}
    assert packed.contexts["chunks"][0]["content"] == "The chart shows revenue growth."
    assert packed.contexts["chunks"][0]["_answer_image_sent"] is False
    assert packed.trace["answer_context_images_skipped"] == 1
    assert packed.contexts["entities"][0]["entity_name"] == "Revenue"


def test_packer_keeps_fitting_image_blocks_by_context_key() -> None:
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

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=1))

    assert [c["chunk_id"] for c in packed.contexts["chunks"]] == ["visual-1"]
    assert packed.image_blocks_by_context_key["visual-1"]["type"] == "image_url"
    assert packed.contexts["chunks"][0]["_answer_image_sent"] is True
    assert packed.trace["answer_context_images_sent"] == 1


def test_packer_uses_one_shared_budget_for_every_source() -> None:
    """Attachment and knowledge-base visuals draw from the same single budget."""
    attachment = {
        "chunk_id": "attachment-visual",
        "reference_id": "attachment_doc",
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

    packed = AnswerContextPacker().pack(
        {"chunks": [attachment, rag], "entities": [], "relationships": []},
        image_budget=_budget(max_images=1),
    )

    # A single image slot is shared, so exactly one visual is transported.
    assert packed.trace["answer_context_images_sent"] == 1
    assert packed.trace["answer_context_images_skipped"] == 1
    assert "answer_context_composer_images_sent" not in packed.trace
    assert "answer_context_rag_images_sent" not in packed.trace
    # Both chunks keep their text regardless of image budget.
    assert [row["chunk_id"] for row in packed.contexts["chunks"]] == [
        "attachment-visual",
        "rag-visual",
    ]


def test_federated_same_chunk_id_keeps_distinct_image_blocks() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "shared-hash",
                "reference_id": "1",
                "file_path": "/legal/report.pdf",
                "content": "Legal figure",
                "image_data": _PNG_B64,
                "_workspace": "legal",
            },
            {
                "chunk_id": "shared-hash",
                "reference_id": "2",
                "file_path": "/finance/report.pdf",
                "content": "Finance figure",
                "image_data": _PNG_B64,
                "_workspace": "finance",
            },
        ],
        "entities": [],
        "relationships": [],
    }

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=2))

    assert set(packed.image_blocks_by_context_key) == {
        "legal:shared-hash",
        "finance:shared-hash",
    }
    assert packed.trace["answer_context_images_sent"] == 2


def test_packer_filters_kg_to_included_chunk_sources() -> None:
    contexts: RetrievalContexts = {
        "chunks": [
            {
                "chunk_id": "kept",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Kept text.",
            },
        ],
        "entities": [
            {"entity_name": "Kept", "description": "in", "source_id": "kept"},
            {"entity_name": "Dropped", "description": "out", "source_id": "missing"},
        ],
        "relationships": [],
    }

    packed = AnswerContextPacker().pack(contexts, image_budget=_budget(max_images=1))

    assert [e["entity_name"] for e in packed.contexts["entities"]] == ["Kept"]
    assert packed.trace["answer_context_input_chunks"] == 1
    assert packed.trace["answer_context_chunks"] == 1
