# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""URL-backed data source for remote document ingestion."""

import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from dlightrag.engine.public_http import (
    download_public_http,
    validate_public_http_url,
)
from dlightrag.engine.rag.corpus.sources.base import AsyncDataSource, SourceDocument
from dlightrag.engine.rag.corpus.sources.source_contract import (
    implicit_https_download_uri,
    safe_source_filename,
    validate_download_uri,
    validate_source_uri,
)

logger = logging.getLogger(__name__)


class URLDataSource(AsyncDataSource):
    """Download bounded HTTP(S) documents through the shared public client."""

    def __init__(
        self,
        *,
        urls: Sequence[str] | None = None,
        documents: Sequence[SourceDocument] | None = None,
        filename: str | None = None,
        source_uri: str | None = None,
        source_uris: Sequence[str] | None = None,
        download_uri: str | None = None,
        download_uris: Sequence[str] | None = None,
        client: Any | None = None,
        timeout: float = 120.0,
        max_download_bytes: int = 100 * 1024 * 1024,
        allow_private_hosts: Sequence[str] | None = None,
    ) -> None:
        if documents is not None and any(
            value is not None
            for value in (
                urls,
                filename,
                source_uri,
                source_uris,
                download_uri,
                download_uris,
            )
        ):
            raise ValueError("'documents' is mutually exclusive with URL shortcut fields")
        if documents is None and not urls:
            raise ValueError("'url' or 'urls' is required for url ingestion")
        url_list = list(urls or [])
        if filename is not None and len(url_list) != 1:
            raise ValueError("'filename' can only be used with a single url")
        if source_uri is not None and len(url_list) != 1:
            raise ValueError("'source_uri' can only be used with a single url")
        if source_uri is not None and source_uris is not None:
            raise ValueError("'source_uri' and 'source_uris' are mutually exclusive")
        if source_uris is not None and len(source_uris) != len(url_list):
            raise ValueError("'source_uris' must match the number of urls")
        if download_uri is not None and download_uris is not None:
            raise ValueError("'download_uri' and 'download_uris' are mutually exclusive")
        if download_uri is not None and len(url_list) != 1:
            raise ValueError("'download_uri' can only be used with a single url")
        if download_uris is not None and len(download_uris) != len(url_list):
            raise ValueError("'download_uris' must match the number of urls")

        self._client = client
        self._timeout = timeout
        self._max_download_bytes = max(1, int(max_download_bytes))
        self._allow_private_hosts = tuple(allow_private_hosts or ())
        self._url_by_key: dict[str, str] = {}
        self._source_uri_by_key: dict[str, str] = {}
        self._download_uri_by_key: dict[str, str | None] = {}
        self._document_by_key: dict[str, SourceDocument] = {}

        if documents is not None:
            document_inputs = list(documents)
            if not document_inputs:
                raise ValueError("'documents' must contain at least one document")
        else:
            document_inputs = [
                SourceDocument(
                    key=raw_url,
                    source_uri=(
                        source_uri
                        if source_uri is not None
                        else source_uris[index]
                        if source_uris is not None
                        else None
                    ),
                    display_filename=filename,
                )
                for index, raw_url in enumerate(url_list)
            ]

        for index, document in enumerate(document_inputs):
            url = validate_public_http_url(
                document.key,
                resolve_host=True,
                allow_private_hosts=self._allow_private_hosts,
            )
            key = _document_key_from_url(url, index=index, filename=document.display_filename)
            key = _dedupe_key(key, self._url_by_key)
            self._url_by_key[key] = url
            stable_source_uri = validate_source_uri(
                document.source_uri or _default_source_uri_from_url(url)
            )
            self._source_uri_by_key[key] = stable_source_uri
            explicit_download_uri = (
                document.download_uri
                if document.download_uri is not None
                else download_uri
                if download_uri is not None
                else download_uris[index]
                if download_uris is not None
                else None
            )
            if explicit_download_uri is not None:
                resolved_download_uri = validate_download_uri(explicit_download_uri)
            else:
                try:
                    resolved_download_uri = implicit_https_download_uri(url)
                except ValueError:
                    resolved_download_uri = None
                if resolved_download_uri is None and ("?" in url or "#" in url):
                    logger.info(
                        "source_download_locator_outcome",
                        extra={
                            "outcome": "ephemeral",
                            "locator_kind": "https",
                            "source_filename": safe_source_filename(
                                document.display_filename or key
                            ),
                        },
                    )
            self._download_uri_by_key[key] = resolved_download_uri
            self._document_by_key[key] = SourceDocument(
                key=key,
                source_uri=stable_source_uri,
                download_uri=resolved_download_uri,
                display_filename=document.display_filename,
                title=document.title,
                author=document.author,
                metadata=document.metadata,
            )

    async def aiter_documents(self, prefix: str | None = None) -> AsyncIterator[SourceDocument]:
        for key, document in self._document_by_key.items():
            if prefix is None or key.startswith(prefix):
                yield document

    async def amaterialize_document(self, document: SourceDocument, destination: Path) -> None:
        try:
            url = self._url_by_key[document.key]
        except KeyError as exc:
            raise KeyError(f"unknown URL document id: {document.key}") from exc
        await download_public_http(
            url,
            destination,
            max_bytes=self._max_download_bytes,
            timeout=self._timeout,
            allow_private_hosts=self._allow_private_hosts,
            client=self._client,
        )

    def source_uri_for_key(self, key: str) -> str:
        try:
            return self._source_uri_by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown URL document id: {key}") from exc

    def download_uri_for_key(self, key: str) -> str | None:
        return self._download_uri_by_key[key]

    async def aclose(self) -> None:
        # Shared public HTTP owns default clients per request; injected clients
        # are caller-owned test/integration transports.
        return None


def _default_source_uri_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def _document_key_from_url(url: str, *, index: int, filename: str | None) -> str:
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
