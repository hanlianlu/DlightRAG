# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral preparation of authorized source downloads."""

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from dlightrag.config import DlightragConfig
from dlightrag.core.ingestion.paths import lightrag_archived_source_path
from dlightrag.sourcing.aws_s3 import (
    S3CredentialsUnavailable,
    S3PresignError,
    generate_s3_presigned_url,
)
from dlightrag.sourcing.azure_blob import generate_azure_sas_url
from dlightrag.sourcing.source_contract import validate_download_uri
from dlightrag.storage.protocols import MetadataIndexProtocol
from dlightrag.utils import normalize_workspace


class SourceDownloadError(RuntimeError):
    """Base class for safe source-download preparation failures."""


class SourceDownloadInvalidError(SourceDownloadError):
    """Stored source metadata cannot produce a safe download."""


class SourceDownloadNotFoundError(SourceDownloadError):
    """The requested document or retained bytes do not exist."""


class SourceDownloadUnavailableError(SourceDownloadError):
    """A configured remote storage adapter cannot currently sign a download."""


@dataclass(frozen=True, slots=True)
class LocalDownloadTarget:
    """Contained local file ready for an HTTP adapter to stream."""

    path: Path
    media_type: str
    filename: str


@dataclass(frozen=True, slots=True)
class RedirectDownloadTarget:
    """Remote URL ready for an HTTP adapter to redirect to."""

    url: str


SourceDownloadTarget = LocalDownloadTarget | RedirectDownloadTarget


class SourceDownloadService:
    """Resolve one workspace-scoped metadata row into a download target."""

    def __init__(
        self,
        *,
        config: DlightragConfig,
        metadata_index: MetadataIndexProtocol,
        workspace: str,
    ) -> None:
        self._config = config
        self._metadata_index = metadata_index
        self._workspace = normalize_workspace(workspace)

    async def prepare(self, document_id: str) -> SourceDownloadTarget:
        """Prepare the exact metadata document without exposing its locator."""
        if (
            not isinstance(document_id, str)
            or not document_id.strip()
            or len(document_id) > 255
            or "\x00" in document_id
        ):
            raise SourceDownloadInvalidError("Invalid document ID")

        metadata = await self._metadata_index.get(document_id)
        if metadata is None:
            raise SourceDownloadNotFoundError("Source not found")
        locator = metadata.get("download_locator")
        if not isinstance(locator, str) or not locator:
            raise SourceDownloadInvalidError("Source download metadata is invalid")

        if "://" not in locator:
            return await self._prepare_local(document_id, locator)
        return await self._prepare_remote(locator)

    async def _prepare_local(
        self,
        document_id: str,
        locator: str,
    ) -> LocalDownloadTarget:
        path = Path(locator)
        if not path.is_absolute():
            raise SourceDownloadInvalidError("Source download metadata is invalid")

        workspace_root = (self._config.input_dir_path / self._workspace).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(workspace_root)
        except OSError, RuntimeError, ValueError:
            raise SourceDownloadInvalidError("Source download metadata is invalid") from None
        if not resolved.is_file():
            archived = lightrag_archived_source_path(resolved).resolve(strict=False)
            try:
                archived.relative_to(workspace_root)
            except ValueError:
                raise SourceDownloadInvalidError("Source download metadata is invalid") from None
            if not archived.is_file():
                raise SourceDownloadNotFoundError("Source not found")
            resolved = archived
            repaired = str(resolved)
            if not self._config.is_reader:
                await self._metadata_index.upsert(
                    document_id,
                    {"download_locator": repaired, "file_path": repaired},
                )

        media_type, _ = mimetypes.guess_type(str(resolved))
        return LocalDownloadTarget(
            path=resolved,
            media_type=media_type or "application/octet-stream",
            filename=resolved.name,
        )

    async def _prepare_remote(self, locator: str) -> RedirectDownloadTarget:
        try:
            canonical = validate_download_uri(locator)
        except ValueError as exc:
            raise SourceDownloadInvalidError("Source download metadata is invalid") from exc

        if canonical.startswith("https://"):
            return RedirectDownloadTarget(url=canonical)
        if canonical.startswith("azure://"):
            if not self._config.blob_connection_string:
                raise SourceDownloadUnavailableError("Azure blob storage not configured")
            try:
                url = generate_azure_sas_url(
                    connection_string=self._config.blob_connection_string,
                    raw_path=canonical,
                    expiry_seconds=self._config.azure_sas_expiry,
                )
            except ValueError as exc:
                raise SourceDownloadUnavailableError("Azure blob download signing failed") from exc
            return RedirectDownloadTarget(url=url)
        if canonical.startswith("s3://"):
            try:
                url = await generate_s3_presigned_url(
                    raw_path=canonical,
                    expiry_seconds=self._config.s3_presign_expiry,
                    region=self._config.s3_region,
                )
            except S3CredentialsUnavailable as exc:
                raise SourceDownloadUnavailableError("S3 credentials not configured") from exc
            except S3PresignError as exc:
                raise SourceDownloadUnavailableError("S3 presigned URL generation failed") from exc
            return RedirectDownloadTarget(url=url)
        raise SourceDownloadInvalidError("Source download metadata is invalid")


__all__ = [
    "LocalDownloadTarget",
    "RedirectDownloadTarget",
    "SourceDownloadError",
    "SourceDownloadInvalidError",
    "SourceDownloadNotFoundError",
    "SourceDownloadService",
    "SourceDownloadTarget",
    "SourceDownloadUnavailableError",
]
