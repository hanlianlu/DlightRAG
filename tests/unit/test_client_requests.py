# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client request projection."""

import pytest
from pydantic import ValidationError

from dlightrag.core.client_contracts import IngestDocument, IngestSpec
from dlightrag.core.client_requests import (
    ingest_kwargs_from_payload,
    ingest_spec_from_payload,
    query_kwargs_from_payload,
)


def test_ingest_spec_from_payload_preserves_s3_manifest_fields() -> None:
    spec = ingest_spec_from_payload(
        {
            "source_type": "s3",
            "bucket": "my-bucket",
            "s3_region": "eu-north-1",
            "metadata": {"source_system": "s3-prod"},
            "metadata_policy": "reject_unknown",
            "retain_source_file": True,
            "documents": [
                {
                    "key": "docs/a.pdf",
                    "title": "A",
                    "metadata": {"department": "Legal", "asset_id": "a"},
                }
            ],
        }
    )

    assert spec == IngestSpec(
        source_type="s3",
        bucket="my-bucket",
        s3_region="eu-north-1",
        metadata={"source_system": "s3-prod"},
        metadata_policy="reject_unknown",
        retain_source_file=True,
        documents=[
            IngestDocument(
                key="docs/a.pdf",
                title="A",
                metadata={"department": "Legal", "asset_id": "a"},
            )
        ],
    )


def test_ingest_spec_from_payload_preserves_url_identity_fields() -> None:
    spec = ingest_spec_from_payload(
        {
            "source_type": "url",
            "url": "https://cdn.example.com/download?id=asset-1&signature=secret",
            "filename": "asset.pdf",
            "source_uri": "bynder://asset/asset-1",
        }
    )

    assert spec == IngestSpec(
        source_type="url",
        url="https://cdn.example.com/download?id=asset-1&signature=secret",
        filename="asset.pdf",
        source_uri="bynder://asset/asset-1",
    )


def test_url_ingest_projects_download_uri_fields() -> None:
    spec = IngestSpec(
        source_type="url",
        urls=["https://fetch.example.com/a.pdf", "https://fetch.example.com/b.pdf"],
        source_uris=["cms://a", "cms://b"],
        download_uris=[
            "https://cdn.example.com/a.pdf",
            "https://cdn.example.com/b.pdf",
        ],
    )

    kwargs = ingest_kwargs_from_payload(spec)

    assert kwargs["download_uris"] == [
        "https://cdn.example.com/a.pdf",
        "https://cdn.example.com/b.pdf",
    ]


def test_url_ingest_preserves_explicit_empty_download_uri_for_canonical_validation() -> None:
    kwargs = ingest_kwargs_from_payload(
        {
            "source_type": "url",
            "url": "https://fetch.example.com/a.pdf",
            "download_uri": "",
        }
    )

    assert kwargs["download_uri"] == ""


def test_url_ingest_download_uri_cardinality_is_strict() -> None:
    with pytest.raises(ValidationError, match="download_uris"):
        IngestSpec(
            source_type="url",
            urls=["https://example.com/a.pdf", "https://example.com/b.pdf"],
            download_uris=["https://cdn.example.com/a.pdf"],
        )


def test_url_ingest_single_download_uri_requires_single_url() -> None:
    with pytest.raises(ValidationError, match="single url"):
        IngestSpec(
            source_type="url",
            urls=["https://example.com/a.pdf", "https://example.com/b.pdf"],
            download_uri="https://cdn.example.com/a.pdf",
        )


def test_url_ingest_download_uri_forms_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        IngestSpec(
            source_type="url",
            url="https://example.com/a.pdf",
            download_uri="https://cdn.example.com/a.pdf",
            download_uris=["https://cdn.example.com/a.pdf"],
        )


def test_url_manifest_projects_per_document_download_uri() -> None:
    spec = IngestSpec(
        source_type="url",
        documents=[
            IngestDocument(
                url="https://fetch.example.com/download?sig=secret",
                source_uri="cms://asset/a",
                download_uri="https://cdn.example.com/a.pdf",
            )
        ],
    )

    assert ingest_kwargs_from_payload(spec)["documents"] == [
        {
            "url": "https://fetch.example.com/download?sig=secret",
            "source_uri": "cms://asset/a",
            "download_uri": "https://cdn.example.com/a.pdf",
        }
    ]


def test_url_manifest_rejects_top_level_download_uri() -> None:
    with pytest.raises(ValidationError, match="documents.*mutually exclusive"):
        IngestSpec(
            source_type="url",
            documents=[IngestDocument(url="https://fetch.example.com/a.pdf")],
            download_uri="https://cdn.example.com/a.pdf",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "source_type": "local",
            "documents": [{"path": "a.pdf", "download_uri": "https://cdn.example.com/a.pdf"}],
        },
        {
            "source_type": "azure_blob",
            "container_name": "container",
            "documents": [{"key": "a.pdf", "download_uri": "azure://container/a.pdf"}],
        },
        {
            "source_type": "s3",
            "bucket": "bucket",
            "documents": [{"key": "a.pdf", "download_uri": "s3://bucket/a.pdf"}],
        },
        {
            "source_type": "local",
            "path": "a.pdf",
            "download_uri": "https://cdn.example.com/a.pdf",
        },
    ],
)
def test_non_url_ingest_rejects_download_uri_fields_before_manifest_returns(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="only valid for URL ingestion"):
        IngestSpec.model_validate(payload)


def test_query_kwargs_from_payload_forwards_bm25_query() -> None:
    kwargs = query_kwargs_from_payload({"bm25_query": "alpha beta"})

    assert kwargs["bm25_query"] == "alpha beta"


def test_query_kwargs_from_payload_omits_absent_bm25_query() -> None:
    kwargs = query_kwargs_from_payload({"query": "q"})

    assert "bm25_query" not in kwargs


def test_retrieve_request_accepts_bm25_query() -> None:
    from dlightrag.api.models import RetrieveRequest

    body = RetrieveRequest(query="q", bm25_query="alpha beta")

    assert body.bm25_query == "alpha beta"


def test_retrieve_request_rejects_overlong_bm25_query() -> None:
    import pytest
    from pydantic import ValidationError

    from dlightrag.api.models import RetrieveRequest

    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", bm25_query="x" * 2000)


def test_retrieve_input_accepts_bm25_query() -> None:
    from dlightrag.mcp.contracts import RetrieveInput

    args = RetrieveInput(query="q", bm25_query="alpha beta")

    assert args.bm25_query == "alpha beta"
