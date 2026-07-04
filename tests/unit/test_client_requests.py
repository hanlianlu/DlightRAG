# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral client request projection."""

from dlightrag.core.client_contracts import IngestDocument, IngestSpec
from dlightrag.core.client_requests import ingest_spec_from_payload


def test_ingest_spec_from_payload_preserves_s3_manifest_fields() -> None:
    spec = ingest_spec_from_payload(
        {
            "source_type": "s3",
            "bucket": "my-bucket",
            "region": "eu-north-1",
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
        region="eu-north-1",
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
