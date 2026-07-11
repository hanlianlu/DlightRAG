# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client request projection."""

from dlightrag.core.client_contracts import IngestDocument, IngestSpec
from dlightrag.core.client_requests import ingest_spec_from_payload, query_kwargs_from_payload


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
