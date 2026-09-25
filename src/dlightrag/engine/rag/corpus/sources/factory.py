# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Construct built-in sources using durable routing and current deployment policy."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlightrag.engine.rag.corpus.sources.base import AsyncDataSource, SourceDocument
from dlightrag.engine.rag.workspace.settings import RagSettings

if TYPE_CHECKING:
    from dlightrag.engine.rag.corpus.sources.url import URLDataSource


@dataclass(frozen=True, slots=True)
class SourceRetrievalOptions:
    """Non-secret S3 routing accepted when the document was ingested.

    An explicit null region delegates to the SDK. Missing historical metadata
    instead resolves the deployment's current default; the old value is unknown.
    Credentials and download safety policy are never snapshots.
    """

    s3_region: str | None

    def __post_init__(self) -> None:
        if self.s3_region is not None and (
            not isinstance(self.s3_region, str)
            or not self.s3_region.strip()
            or self.s3_region != self.s3_region.strip()
        ):
            raise ValueError("Source retrieval region must be a non-empty string or null")

    def payload(self) -> dict[str, str | None]:
        return {"s3_region": self.s3_region}


def resolve_source_options(
    source_type: str,
    stored: object,
    *,
    default_s3_region: str | None,
) -> SourceRetrievalOptions | None:
    """Read the exact internal contract, rejecting malformed or mismatched data."""
    if stored is None:
        return SourceRetrievalOptions(default_s3_region) if source_type == "s3" else None
    if not isinstance(stored, Mapping):
        raise ValueError("Source retrieval options must be an object")
    if source_type != "s3":
        if stored:
            raise ValueError("Source retrieval options do not match the download locator")
        return None
    if not stored:
        return SourceRetrievalOptions(default_s3_region)
    if set(stored) != {"s3_region"}:
        raise ValueError("S3 retrieval options require exactly s3_region")
    region = stored["s3_region"]
    if region is not None and not isinstance(region, str):
        raise ValueError("Source retrieval region must be a string or null")
    return SourceRetrievalOptions(region)


class RemoteSourceFactory:
    """One construction owner for first ingestion and failed-document replay.

    The factory owns deployment policy and credentials. Custom SDK adapters
    still enter aingest_source directly and remain outside its ownership.
    """

    def __init__(self, settings: RagSettings) -> None:
        self._settings = settings

    def url(
        self,
        *,
        urls: Sequence[str] | None = None,
        documents: Sequence[SourceDocument] | None = None,
        filename: str | None = None,
        source_uri: str | None = None,
        source_uris: Sequence[str] | None = None,
        download_uri: str | None = None,
        download_uris: Sequence[str] | None = None,
    ) -> URLDataSource:
        from dlightrag.engine.rag.corpus.sources.url import URLDataSource

        return URLDataSource(
            urls=urls,
            documents=documents,
            filename=filename,
            source_uri=source_uri,
            source_uris=source_uris,
            download_uri=download_uri,
            download_uris=download_uris,
            max_download_bytes=self._settings.url_ingest_max_bytes,
            allow_private_hosts=self._settings.url_ingest_private_host_allowlist,
        )

    def s3(self, bucket: str, options: SourceRetrievalOptions) -> AsyncDataSource:
        from dlightrag.engine.rag.corpus.sources.aws_s3 import S3DataSource

        return S3DataSource(bucket=bucket, region=options.s3_region)

    def azure(self, container: str) -> AsyncDataSource:
        from dlightrag.engine.rag.corpus.sources.azure_blob import AzureBlobDataSource

        return AzureBlobDataSource(
            connection_string=self._settings.blob_connection_string,
            container_name=container,
        )
