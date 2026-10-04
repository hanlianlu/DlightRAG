# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Mailbox over an S3-compatible bucket the deployment fills (ADR 0034).

How mail reaches the bucket is the deployment's business. Its one obligation is the layout: each
message whole, as one object, under ``<prefix>/<envelope recipient>/``. This adapter lists one
alias's prefix, reads at most a few of its newest messages, and never lists the bucket's root,
writes or deletes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aiobotocore.session import AioSession
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from dlightrag.engine.answer.agent_browser import (
    MAX_LISTED,
    AgentMailboxError,
    MailListing,
    MailObject,
)

#: S3 returns at most this many keys a page.
_PAGE_KEYS = 1000
#: The pages of keys one listing reads, which hold the most messages a listing may read.
_MAX_PAGES = MAX_LISTED // _PAGE_KEYS


class S3AgentMailbox:
    """Mail read from ``bucket`` on an S3-compatible endpoint, such as AWS S3 or Cloudflare R2."""

    def __init__(
        self,
        *,
        alias_domain: str,
        bucket: str,
        prefix: str,
        endpoint: str | None,
        region: str | None,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        self.alias_domain = alias_domain
        self._bucket = bucket
        self._prefix = f"{prefix}/" if prefix else ""
        self._session = AioSession()
        self._client_options: dict[str, Any] = {
            "endpoint_url": endpoint,
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "config": Config(
                connect_timeout=10,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        }
        # Unset, the SDK resolves the region as it does for any S3 client.
        if region is not None:
            self._client_options["region_name"] = region

    async def messages(
        self, address: str, *, since: datetime, limit: int, max_bytes: int
    ) -> MailListing:
        # A client lives for one call, so there is nothing left open to close. The library's own
        # error names the endpoint, so the code is taken here and raised outside the handler,
        # which leaves no error chained to it.
        code = ""
        try:
            async with self._session.create_client("s3", **self._client_options) as client:
                return await self._read(client, address, since, limit, max_bytes)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or type(exc).__name__)
        except BotoCoreError as exc:
            code = type(exc).__name__
        raise AgentMailboxError(code)

    async def _read(
        self, client: Any, address: str, since: datetime, limit: int, max_bytes: int
    ) -> MailListing:
        prefix = f"{self._prefix}{address}/"
        recent: list[tuple[datetime, str, int]] = []
        token: str | None = None
        for _ in range(_MAX_PAGES):
            listing = await client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
                **({} if token is None else {"ContinuationToken": token}),
            )
            recent.extend(
                (item["LastModified"], item["Key"], item["Size"])
                for item in listing.get("Contents", [])
                if item["LastModified"] >= since
            )
            if not listing.get("IsTruncated"):
                truncated = False
                break
            token = listing["NextContinuationToken"]
        else:
            truncated = True
        recent.sort(reverse=True)
        messages = []
        for modified, key, size in recent[:limit]:
            if size > max_bytes:
                messages.append(MailObject(modified, None))
                continue
            response = await client.get_object(Bucket=self._bucket, Key=key)
            async with response["Body"] as body:
                messages.append(MailObject(modified, await body.read()))
        return MailListing(tuple(messages), max(0, len(recent) - limit), truncated)


__all__ = ["S3AgentMailbox"]
