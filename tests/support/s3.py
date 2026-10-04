# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A loopback S3 endpoint: ListObjectsV2 and GetObject over a table of objects, with a request log.

A test points an S3 client at ``stub.endpoint`` with any credentials and observes what it asked
for: which prefixes it listed and which keys it fetched. The stub does not check a signature, it
serves one bucket in path style, and it answers each request on a connection of its own. Callers
keep the client off any proxy with ``bypass_proxies``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qs, quote, unquote, urlsplit
from xml.sax.saxutils import escape


@dataclass(frozen=True, slots=True)
class StoredObject:
    body: bytes
    modified: datetime
    """When the object was stored: its ``LastModified``, timezone-aware."""


@dataclass(frozen=True, slots=True)
class S3Request:
    method: str
    key: str
    """The object's key for a GET of one, else the empty string."""
    prefix: str | None
    """The prefix of a listing, else None."""


class S3Stub:
    """One bucket on loopback, answering a list of a prefix and a get of a key."""

    def __init__(
        self, bucket: str, objects: Mapping[str, StoredObject], *, page_keys: int, denied: bool
    ) -> None:
        self.bucket = bucket
        self.objects = dict(objects)
        self.endpoint = ""
        self.requests: list[S3Request] = []
        self._page_keys = page_keys
        self._denied = denied

    @property
    def listed(self) -> list[str]:
        """The prefix of every listing asked for."""
        return [request.prefix for request in self.requests if request.prefix is not None]

    @property
    def fetched(self) -> list[str]:
        """The key of every object fetched."""
        return [request.key for request in self.requests if request.key]

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
            method, target, _ = head.split("\r\n", 1)[0].split(" ", 2)
            url = urlsplit(target)
            query = {name: values[0] for name, values in parse_qs(url.query).items()}
            path = unquote(url.path).lstrip("/")
            bucket, _, key = path.partition("/")
            listing = not key
            self.requests.append(
                S3Request(method, key, query.get("prefix", "") if listing else None)
            )
            if self._denied or bucket != self.bucket:
                writer.write(_error(403, "AccessDenied", "Access Denied"))
            elif listing:
                writer.write(self._list(query))
            elif key in self.objects:
                stored = self.objects[key]
                writer.write(
                    _response(
                        200,
                        stored.body,
                        {"content-type": "message/rfc822", "etag": '"stub"'},
                    )
                )
            else:
                writer.write(_error(404, "NoSuchKey", "The specified key does not exist."))
            await writer.drain()
        except asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError:
            pass
        finally:
            writer.close()

    def _list(self, query: Mapping[str, str]) -> bytes:
        prefix = query.get("prefix", "")
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        start = int(query["continuation-token"]) if "continuation-token" in query else 0
        page = keys[start : start + self._page_keys]
        more = start + self._page_keys < len(keys)
        contents = "".join(
            f"<Contents><Key>{escape(quote(key, safe='/'))}</Key>"
            f"<LastModified>{self.objects[key].modified.strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
            f"</LastModified><ETag>&quot;stub&quot;</ETag><Size>{len(self.objects[key].body)}</Size>"
            "<StorageClass>STANDARD</StorageClass></Contents>"
            for key in page
        )
        token = (
            f"<NextContinuationToken>{start + self._page_keys}</NextContinuationToken>"
            if more
            else ""
        )
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>{self.bucket}</Name><Prefix>{escape(prefix)}</Prefix>"
            f"<KeyCount>{len(page)}</KeyCount><MaxKeys>{self._page_keys}</MaxKeys>"
            f"<EncodingType>url</EncodingType><IsTruncated>{str(more).lower()}</IsTruncated>"
            f"{token}{contents}</ListBucketResult>"
        )
        return _response(200, body.encode(), {"content-type": "application/xml"})


@asynccontextmanager
async def s3_stub(
    objects: Mapping[str, StoredObject],
    *,
    bucket: str = "mailbox",
    page_keys: int = 1000,
    denied: bool = False,
) -> AsyncIterator[S3Stub]:
    """Serve ``objects`` (by key) from ``bucket``, ``page_keys`` keys to a listing page.

    A stub that is ``denied`` answers every request with AccessDenied, as a bucket does a client
    whose credentials may not read it.
    """
    stub = S3Stub(bucket, objects, page_keys=page_keys, denied=denied)
    server = await asyncio.start_server(stub.serve, "127.0.0.1", 0)
    stub.endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        yield stub
    finally:
        server.close()
        await server.wait_closed()


def _response(status: int, body: bytes, headers: Mapping[str, str]) -> bytes:
    lines = [f"HTTP/1.1 {status} Stub", *(f"{name}: {value}" for name, value in headers.items())]
    lines += [f"content-length: {len(body)}", "connection: close", "", ""]
    return "\r\n".join(lines).encode("latin-1") + body


def _error(status: int, code: str, message: str) -> bytes:
    body = f"<?xml version='1.0'?><Error><Code>{code}</Code><Message>{message}</Message></Error>"
    return _response(status, body.encode(), {"content-type": "application/xml"})


__all__ = ["S3Request", "S3Stub", "StoredObject", "s3_stub"]
