# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""URL-backed data source for remote document ingestion."""

from __future__ import annotations

import fnmatch
import inspect
import ipaddress
import socket
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

from dlightrag.sourcing.base import AsyncDataSource

_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class URLDataSource(AsyncDataSource):
    """Download public HTTPS documents by URL.

    This adapter is intentionally small: it supports signed/public URLs for
    REST/MCP ingestion. Connectors that need auth headers or custom pagination
    should implement ``AsyncDataSource`` and use the SDK ``aingest_source`` API.
    """

    def __init__(
        self,
        *,
        urls: Sequence[str],
        filename: str | None = None,
        source_uri: str | None = None,
        source_uris: Sequence[str] | None = None,
        client: Any | None = None,
        timeout: float = 120.0,
        max_download_bytes: int = 100 * 1024 * 1024,
        allow_private_hosts: Sequence[str] | None = None,
    ) -> None:
        if not urls:
            raise ValueError("'url' or 'urls' is required for url ingestion")
        if filename is not None and len(urls) != 1:
            raise ValueError("'filename' can only be used with a single url")
        if source_uri is not None and len(urls) != 1:
            raise ValueError("'source_uri' can only be used with a single url")
        if source_uri is not None and source_uris is not None:
            raise ValueError("'source_uri' and 'source_uris' are mutually exclusive")
        if source_uris is not None and len(source_uris) != len(urls):
            raise ValueError("'source_uris' must match the number of urls")

        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._max_download_bytes = max(1, int(max_download_bytes))
        self._allow_private_hosts = _normalize_host_patterns(allow_private_hosts or ())
        self._url_by_key: dict[str, str] = {}
        self._source_uri_by_key: dict[str, str] = {}

        for index, raw_url in enumerate(urls):
            url = _validate_public_https_url(
                raw_url,
                resolve_host=self._owns_client,
                allow_private_hosts=self._allow_private_hosts,
            )
            key = _document_key_from_url(url, index=index, filename=filename)
            key = _dedupe_key(key, self._url_by_key)
            self._url_by_key[key] = url
            if source_uri is not None:
                stable_source_uri = source_uri
            elif source_uris is not None:
                stable_source_uri = source_uris[index]
            else:
                stable_source_uri = _default_source_uri_from_url(url)
            self._source_uri_by_key[key] = _validate_source_uri(stable_source_uri)

    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[str]:
        for key in self._url_by_key:
            if prefix is None or key.startswith(prefix):
                yield key

    async def amaterialize_document(self, doc_id: str, destination: Path) -> None:
        try:
            url = self._url_by_key[doc_id]
        except KeyError as exc:
            raise KeyError(f"unknown URL document id: {doc_id}") from exc

        client = self._ensure_client()
        current_url = url
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                async with client.stream("GET", current_url) as response:
                    status_code = int(getattr(response, "status_code", 200) or 200)
                    if status_code in _REDIRECT_STATUSES:
                        current_url = _redirect_target(
                            current_url,
                            response,
                            resolve_host=self._owns_client,
                            allow_private_hosts=self._allow_private_hosts,
                        )
                        continue

                    response.raise_for_status()
                    _validate_public_https_url(
                        str(getattr(response, "url", current_url)),
                        allow_private_hosts=self._allow_private_hosts,
                    )
                    await self._write_response(response, destination)
                    return
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        raise ValueError("url ingestion exceeded maximum redirects")

    def source_uri_for_key(self, key: str) -> str:
        try:
            return self._source_uri_by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown URL document id: {key}") from exc

    async def aclose(self) -> None:
        if not self._owns_client or self._client is None:
            return
        close = getattr(self._client, "aclose", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def _write_response(self, response: Any, destination: Path) -> None:
        written = 0
        with destination.open("wb") as out:
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                written += len(chunk)
                if written > self._max_download_bytes:
                    raise ValueError(
                        f"url ingest exceeds maximum size of {self._max_download_bytes} bytes"
                    )
                out.write(chunk)


def _validate_public_https_url(
    raw_url: str,
    *,
    resolve_host: bool = False,
    allow_private_hosts: frozenset[str] = frozenset(),
) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme.lower() != "https":
        raise ValueError("url ingestion only accepts https URLs")
    if not parsed.hostname:
        raise ValueError("url ingestion requires a hostname")
    if parsed.username or parsed.password:
        raise ValueError("url ingestion does not accept credentials in URLs")

    host = _normalize_host(parsed.hostname)
    if _host_allowed_private(host, allow_private_hosts):
        return raw_url
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("url ingestion requires a public host")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if resolve_host:
            _validate_resolved_public_host(host, parsed.port or 443)
        return raw_url
    if not ip.is_global:
        raise ValueError("url ingestion requires a public host")
    return raw_url


def _validate_resolved_public_host(host: str, port: int) -> None:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("url ingestion requires a resolvable public host") from exc
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        if not ipaddress.ip_address(str(sockaddr[0])).is_global:
            raise ValueError("url ingestion requires a public host")


def _redirect_target(
    current_url: str,
    response: Any,
    *,
    resolve_host: bool,
    allow_private_hosts: frozenset[str],
) -> str:
    headers = getattr(response, "headers", {}) or {}
    location = headers.get("location") or headers.get("Location")
    if not location:
        raise ValueError("url redirect is missing Location header")
    return _validate_public_https_url(
        urljoin(current_url, str(location)),
        resolve_host=resolve_host,
        allow_private_hosts=allow_private_hosts,
    )


def _normalize_host_patterns(values: Sequence[str]) -> frozenset[str]:
    return frozenset(_normalize_host(value) for value in values if value)


def _normalize_host(value: str) -> str:
    return value.lower().strip("[]").rstrip(".")


def _host_allowed_private(host: str, patterns: frozenset[str]) -> bool:
    return any(fnmatch.fnmatchcase(host, pattern) for pattern in patterns)


def _default_source_uri_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def _validate_source_uri(value: str) -> str:
    if not value or "\0" in value:
        raise ValueError("source_uri is invalid")
    return value


def _document_key_from_url(
    url: str,
    *,
    index: int,
    filename: str | None,
) -> str:
    if filename is not None:
        return _clean_filename(filename)

    parsed = urlparse(url)
    name = _clean_filename(unquote(PurePosixPath(parsed.path).name or f"document-{index + 1}"))
    if not Path(name).suffix:
        name = f"{name}.html"
    return name


def _clean_filename(value: str) -> str:
    candidate = value.replace("\\", "/")
    name = PurePosixPath(candidate).name
    if not name or name in {".", ".."} or "\0" in name:
        raise ValueError("url ingestion filename is invalid")
    return name


def _dedupe_key(key: str, existing: dict[str, str]) -> str:
    if key not in existing:
        return key
    path = Path(key)
    stem = path.stem or "document"
    suffix = path.suffix
    digest = 1
    while True:
        candidate = f"{stem}-{digest}{suffix}"
        if candidate not in existing:
            return candidate
        digest += 1


__all__ = ["URLDataSource"]
