# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for S3 source download URL signing."""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.sourcing.aws_s3 import generate_s3_presigned_url


class _FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], int]] = []

    def generate_presigned_url(
        self,
        client_method: str,
        *,
        Params: dict[str, Any],
        ExpiresIn: int,
    ) -> str:
        self.calls.append((client_method, Params, ExpiresIn))
        return "https://signed.example.com/report.pdf?sig=x"


class _FakeClientContext:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client

    async def __aenter__(self) -> _FakeS3Client:
        return self.client

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client
        self.create_client_calls: list[tuple[str, dict[str, Any]]] = []

    def create_client(self, service_name: str, **kwargs: Any) -> _FakeClientContext:
        self.create_client_calls.append((service_name, kwargs))
        return _FakeClientContext(self.client)


@pytest.mark.asyncio
async def test_generate_s3_presigned_url_uses_bucket_key_expiry_and_region() -> None:
    client = _FakeS3Client()
    session = _FakeSession(client)

    url = await generate_s3_presigned_url(
        raw_path="s3://my-bucket/reports/q1.pdf",
        expiry_seconds=900,
        region="eu-north-1",
        session=session,
    )

    assert url == "https://signed.example.com/report.pdf?sig=x"
    assert session.create_client_calls == [("s3", {"region_name": "eu-north-1"})]
    assert client.calls == [
        (
            "get_object",
            {"Bucket": "my-bucket", "Key": "reports/q1.pdf"},
            900,
        )
    ]


@pytest.mark.asyncio
async def test_generate_s3_presigned_url_omits_region_when_unspecified() -> None:
    client = _FakeS3Client()
    session = _FakeSession(client)

    await generate_s3_presigned_url(
        raw_path="s3://my-bucket/reports/q1.pdf",
        region=None,
        session=session,
    )

    assert session.create_client_calls == [("s3", {})]


@pytest.mark.asyncio
async def test_generate_s3_presigned_url_rejects_non_s3_path() -> None:
    with pytest.raises(ValueError, match="s3://"):
        await generate_s3_presigned_url(
            raw_path="azure://container/reports/q1.pdf",
            session=_FakeSession(_FakeS3Client()),
        )
