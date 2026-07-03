# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client payload helpers."""

from __future__ import annotations

from dlightrag.citations.schemas import SourceReference
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.models.schemas import Reference


def test_project_contexts_for_client_strips_inline_images_and_adds_image_urls() -> None:
    from dlightrag.core.client_payloads import project_contexts_for_client

    contexts = {
        "chunks": [
            {
                "chunk_id": "image chunk/1",
                "reference_id": "1",
                "file_path": "/private/report.pdf",
                "content": "Figure evidence",
                "page_idx": 2,
                "bbox": {"page_index": 1, "range": [1, 2, 3, 4]},
                "image_mime_type": "image/png",
                "relevance_score": 0.87,
                "metadata": {"department": "finance"},
                "image_data": "base64-payload",
                "_workspace": "workspace a",
                "full_doc_id": "doc-internal",
                "score": 0.22,
                "rerank_score": 0.91,
                "distance": 0.78,
                "bm25_profile": "english",
                "sidecar": {"type": "drawing"},
                "sidecar_location": "file:///tmp/report.parsed",
            }
        ],
        "entities": [{"entity_name": "ACME"}],
    }

    public = project_contexts_for_client(contexts)

    chunk = public["chunks"][0]
    assert "image_data" not in chunk
    assert chunk["image_url"] == "/images/workspace%20a/image%20chunk%2F1?size=full"
    assert chunk["thumbnail_url"] == "/images/workspace%20a/image%20chunk%2F1?size=thumb"
    assert chunk["bbox"] == {"page_index": 1, "range": [1, 2, 3, 4]}
    assert chunk["metadata"] == {"department": "finance"}
    assert "full_doc_id" not in chunk
    assert "score" not in chunk
    assert "rerank_score" not in chunk
    assert "distance" not in chunk
    assert "bm25_profile" not in chunk
    assert "sidecar" not in chunk
    assert "sidecar_location" not in chunk
    assert public["entities"] == [{"entity_name": "ACME"}]
    assert "image_data" in contexts["chunks"][0]


def test_project_contexts_for_client_accepts_lightrag_id_alias() -> None:
    from dlightrag.core.client_payloads import project_contexts_for_client

    public = project_contexts_for_client({"chunks": [{"id": "c1", "content": "Evidence"}]})

    assert public["chunks"] == [
        {"chunk_id": "c1", "reference_id": "", "file_path": "", "content": "Evidence"}
    ]


def test_answer_payload_uses_public_contexts_and_existing_sources() -> None:
    from dlightrag.core.client_payloads import answer_payload

    result = RetrievalResult(
        answer="Answer [1-1].",
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "Evidence",
                    "image_data": "base64-payload",
                    "_workspace": "default",
                }
            ],
        },
        references=[Reference(id="1", title="report.pdf")],
        sources=[SourceReference(id="1", title="report.pdf", path="/private/report.pdf")],
        trace={"phase": "answer"},
        image_descriptions=["chart"],
        current_image_ids=["c1"],
    )

    payload = answer_payload(result)

    assert payload["answer"] == "Answer [1-1]."
    assert payload["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in payload["contexts"]["chunks"][0]
    assert payload["references"] == [{"id": "1", "title": "report.pdf"}]
    assert payload["sources"][0]["id"] == "1"
    assert payload["trace"] == {"phase": "answer"}
    assert payload["image_descriptions"] == ["chart"]
    assert payload["current_image_ids"] == ["c1"]
