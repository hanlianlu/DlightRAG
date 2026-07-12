# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client payload helpers."""

import logging

import pytest

from dlightrag.citations.schemas import ChunkSnippet, SourceReference
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.models.schemas import Reference


def _internal_source(*, chunks: list[ChunkSnippet] | None = None) -> SourceReference:
    return SourceReference(
        id="1",
        title="report.pdf",
        source_uri="local://default/report.pdf",
        workspace="default",
        document_id="doc-report",
        download_locator="/private/report.pdf",
        chunks=chunks,
    )


def test_project_source_payloads_resolves_and_hides_raw_locator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.core.client_payloads import project_source_payloads
    from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder

    source = SourceReference(
        id="1",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.core.client_payloads"):
        projected = project_source_payloads([source], resolver=SourceDownloadLinkBuilder())[0]

    assert projected.download_url == "/files/raw/doc-report?workspace=finance"
    assert source.download_locator not in projected.download_url
    payload = projected.model_dump()
    assert "download_locator" not in payload
    assert "workspace" not in payload
    record = next(
        record
        for record in caplog.records
        if record.message == "source_download_projection_outcome"
    )
    assert getattr(record, "outcome", None) == "resolved"
    assert getattr(record, "source_id", None) == "1"
    assert source.download_locator not in caplog.text


def test_project_source_payloads_rejects_invalid_locator_without_logging_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.core.client_payloads import (
        SourceDownloadInvariantError,
        project_source_payloads,
    )
    from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder

    source = SourceReference(
        id="unsafe\nsource",
        source_uri="bynder://asset/1",
        workspace="finance",
        document_id="",
        download_locator="file://secret-host/private/report.pdf",
    )

    with (
        caplog.at_level(logging.INFO, logger="dlightrag.core.client_payloads"),
        pytest.raises(SourceDownloadInvariantError, match=r"unsafe\\nsource"),
    ):
        project_source_payloads([source], resolver=SourceDownloadLinkBuilder())

    record = next(
        record
        for record in caplog.records
        if record.message == "source_download_projection_outcome"
    )
    assert getattr(record, "outcome", None) == "invalid"
    assert getattr(record, "source_id", None) == r"unsafe\nsource"
    assert source.download_locator not in caplog.text


def test_project_source_payloads_omits_link_without_download_permission() -> None:
    from dlightrag.core.client_payloads import project_source_payloads
    from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder

    source = SourceReference(
        id="1",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )

    projected = project_source_payloads(
        [source],
        resolver=SourceDownloadLinkBuilder(),
        downloadable_workspaces=set(),
    )[0]

    assert projected.download_url is None


def test_answer_payload_projects_transport_neutral_source_without_download_url() -> None:
    from dlightrag.core.client_payloads import answer_payload

    result = RetrievalResult(
        sources=[
            SourceReference(
                id="1",
                source_uri="s3://bucket/report.pdf",
                workspace="finance",
                document_id="doc-report",
                download_locator="s3://bucket/report.pdf",
            )
        ]
    )

    source = answer_payload(result)["sources"][0]

    assert source["source_uri"] == "s3://bucket/report.pdf"
    assert source["download_url"] is None
    assert {"workspace", "download_locator", "path", "url"}.isdisjoint(source)


def test_public_context_projection_strips_internal_source_metadata() -> None:
    from dlightrag.core.client_payloads import project_contexts_for_client

    contexts = {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "r1",
                "file_path": "/srv/dlightrag/inputs/finance/report.pdf",
                "content": "text",
                "metadata": {
                    "source_uri": "bynder://asset/1",
                    "source_download_locator": "https://cdn.example.com/assets/1.pdf",
                    "category": "research",
                },
            }
        ],
        "entities": [],
        "relationships": [],
    }

    projected = project_contexts_for_client(contexts)

    assert projected["chunks"][0]["metadata"] == {"category": "research"}
    assert projected["chunks"][0]["file_path"] == "report.pdf"


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


def test_project_contexts_for_client_adds_visual_chunk_urls_without_inline_image_data() -> None:
    from dlightrag.core.client_payloads import project_contexts_for_client

    public = project_contexts_for_client(
        {
            "chunks": [
                {
                    "chunk_id": "doc-1-mm-drawing-001",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "[Image Name]architecture",
                    "_workspace": "default",
                    "sidecar": {"type": "drawing"},
                },
                {
                    "chunk_id": "doc-1-chunk-001",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "plain text",
                    "_workspace": "default",
                },
            ]
        }
    )

    assert public["chunks"][0]["image_url"] == "/images/default/doc-1-mm-drawing-001?size=full"
    assert "image_url" not in public["chunks"][1]


def test_project_contexts_for_client_skips_chunks_without_public_id() -> None:
    from dlightrag.core.client_payloads import project_contexts_for_client

    public = project_contexts_for_client({"chunks": [{"content": "orphan"}]})

    assert public["chunks"] == []


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
        sources=[_internal_source()],
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
    assert payload["answer_images"] == []
    assert payload["answer_blocks"] == []
    assert payload["trace"] == {"phase": "answer"}
    assert payload["image_descriptions"] == ["chart"]
    assert payload["current_image_ids"] == ["c1"]


def test_answer_helpers_derive_visual_images_and_blocks() -> None:
    from dlightrag.core.answer_media import (
        answer_blocks_from_markdown,
        answer_images_from_sources,
    )

    sources = [
        _internal_source(
            chunks=[
                ChunkSnippet(
                    chunk_id="fig-1",
                    chunk_idx=1,
                    content="Figure evidence",
                    image_url="/images/default/fig-1?size=full",
                    thumbnail_url="/images/default/fig-1?size=thumb",
                )
            ]
        )
    ]

    images = answer_images_from_sources(sources)

    assert images == [
        {
            "id": "fig-1",
            "chunk_id": "fig-1",
            "source_ref": "1-1",
            "url": "/images/default/fig-1?size=full",
            "thumbnail_url": "/images/default/fig-1?size=thumb",
            "label": "report.pdf",
        }
    ]
    assert answer_blocks_from_markdown("Diagram below [1-1]. Details after.", images) == [
        {"type": "markdown", "text": "Diagram below [1-1]."},
        {"type": "image_ref", "image_id": "fig-1"},
        {"type": "markdown", "text": " Details after."},
    ]


def test_answer_payload_serializes_result_answer_images_and_blocks() -> None:
    from dlightrag.core.client_payloads import answer_payload

    result = RetrievalResult(
        answer="Diagram below [1-1].",
        answer_images=[
            {
                "id": "fig-1",
                "chunk_id": "fig-1",
                "source_ref": "1-1",
                "url": "/images/default/fig-1?size=full",
                "thumbnail_url": "/images/default/fig-1?size=thumb",
                "label": "report.pdf",
            }
        ],
        answer_blocks=[
            {"type": "markdown", "text": "Diagram below [1-1]."},
            {"type": "image_ref", "image_id": "fig-1"},
        ],
    )

    payload = answer_payload(result)

    assert payload["answer_images"] == result.answer_images
    assert payload["answer_blocks"] == result.answer_blocks
