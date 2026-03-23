# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AWS S3 data source using aiobotocore (native async)."""

from __future__ import annotations

import logging
from typing import Any

from aiobotocore.session import AioSession

from dlightrag.sourcing.base import AsyncDataSource

logger = logging.getLogger(__name__)


class S3DataSource(AsyncDataSource):
    """Async AWS S3 backend using aiobotocore."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._session = AioSession()
        self._client_kwargs: dict[str, Any] = {"region_name": region}
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

    async def alist_documents(self, prefix: str | None = None) -> list[str]:
        """List object keys in the bucket (paginated, handles >1000 objects)."""
        client = await self._ensure_client()
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix or ""):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    async def aload_document(self, doc_id: str) -> bytes:
        """Download an S3 object as bytes."""
        client = await self._ensure_client()
        resp = await client.get_object(Bucket=self._bucket, Key=doc_id)
        return await resp["Body"].read()

    async def aclose(self) -> None:
        """Close the S3 session."""
        if self._client is not None:
            await self._ctx.__aexit__(None, None, None)
            self._client = None


__all__ = ["S3DataSource"]
