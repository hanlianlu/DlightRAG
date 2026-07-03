# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AWS S3 data source using aiobotocore (native async)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from inspect import isawaitable
from pathlib import Path
from typing import Any

from aiobotocore.session import AioSession
from botocore.exceptions import BotoCoreError, NoCredentialsError, PartialCredentialsError

from dlightrag.sourcing.base import AsyncDataSource
from dlightrag.sourcing.uri import parse_remote_uri

logger = logging.getLogger(__name__)


class S3PresignError(RuntimeError):
    """Base error for S3 source URL signing failures."""


class S3CredentialsUnavailable(S3PresignError):
    """Raised when no usable AWS credentials are available for URL signing."""


async def generate_s3_presigned_url(
    *,
    raw_path: str,
    expiry_seconds: int = 3600,
    region: str | None = None,
    session: Any | None = None,
) -> str:
    """Generate a time-limited S3 GET URL for an ``s3://bucket/key`` path."""
    source_type, parts = parse_remote_uri(raw_path)
    if source_type != "s3":
        raise ValueError(f"Expected s3:// path, got: {raw_path!r}")

    client_kwargs: dict[str, Any] = {}
    if region:
        client_kwargs["region_name"] = region

    s3_session = session or AioSession()
    try:
        async with s3_session.create_client("s3", **client_kwargs) as client:
            signed_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": parts["bucket"], "Key": parts["key"]},
                ExpiresIn=expiry_seconds,
            )
            if isawaitable(signed_url):
                signed_url = await signed_url
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise S3CredentialsUnavailable("S3 credentials not configured") from exc
    except BotoCoreError as exc:
        raise S3PresignError("S3 presigned URL generation failed") from exc

    return str(signed_url)


class S3DataSource(AsyncDataSource):
    """Async AWS S3 backend using aiobotocore."""

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._session = AioSession()
        self._client_kwargs: dict[str, Any] = {}
        if region:
            self._client_kwargs["region_name"] = region
        if aws_access_key_id:
            self._client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            self._client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        self._client: Any = None
        self._ctx: Any = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            ctx = self._session.create_client("s3", **self._client_kwargs)
            self._ctx = ctx
            self._client = await ctx.__aenter__()
        return self._client

    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
        """Stream object keys in the bucket."""
        client = await self._ensure_client()
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix or ""):
            for obj in page.get("Contents", []):
                yield str(obj["Key"])

    async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
        """Download an S3 object to a local parser input file."""
        client = await self._ensure_client()
        resp = await client.get_object(Bucket=self._bucket, Key=doc_id)
        body = resp["Body"]
        try:
            with destination.open("wb") as out:
                while chunk := await body.read(1024 * 1024):
                    out.write(chunk)
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                result = close()
                if isawaitable(result):
                    await result

    async def aclose(self) -> None:
        """Close the S3 session."""
        if self._client is not None:
            await self._ctx.__aexit__(None, None, None)
            self._client = None


__all__ = [
    "S3CredentialsUnavailable",
    "S3DataSource",
    "S3PresignError",
    "generate_s3_presigned_url",
]
