# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Azure Blob Storage data source."""

from __future__ import annotations

import os
from datetime import UTC

from dlightrag.sourcing.base import AsyncDataSource, DataSource

try:
    from azure.storage.blob import generate_blob_sas
except ImportError:  # pragma: no cover
    generate_blob_sas = None  # type: ignore[assignment]


def _parse_connection_string(connection_string: str) -> dict[str, str]:
    """Parse Azure Storage connection string into key-value pairs."""
    pairs: dict[str, str] = {}
    for part in connection_string.split(";"):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            # AccountKey values contain '=' (base64), so only split on first '='
            pairs[key] = value
    # AccountKey may have been truncated — re-extract properly
    if "AccountKey" in connection_string:
        start = connection_string.index("AccountKey=") + len("AccountKey=")
        end = connection_string.find(";", start)
        pairs["AccountKey"] = connection_string[start:end] if end != -1 else connection_string[start:]
    return pairs


def generate_azure_sas_url(
    connection_string: str,
    raw_path: str,
    expiry_seconds: int = 3600,
) -> str:
    """Generate a time-limited Azure SAS signed URL.

    Parses account_name and account_key from the connection string.
    Converts azure://container/blob path into a signed HTTPS URL.

    Args:
        connection_string: Azure Storage connection string
            (contains AccountName and AccountKey).
        raw_path: Path in azure://container/blob format.
        expiry_seconds: SAS URL expiry in seconds (default: 1 hour).

    Returns:
        Signed HTTPS URL, e.g.:
        "https://myaccount.blob.core.windows.net/mycontainer/doc.pdf?sv=...&sig=..."

    Raises:
        ValueError: If raw_path is not in azure:// format or
            connection_string is missing required fields.
    """
    from datetime import datetime, timedelta

    from azure.storage.blob import BlobSasPermissions

    if not raw_path.startswith("azure://"):
        raise ValueError(f"Expected azure:// path, got: {raw_path!r}")

    stripped = raw_path.removeprefix("azure://")
    if "/" not in stripped or not stripped:
        raise ValueError(f"Invalid azure path (need container/blob): {raw_path!r}")

    container, blob = stripped.split("/", 1)
    if not container or not blob:
        raise ValueError(f"Invalid azure path (empty container or blob): {raw_path!r}")

    parsed = _parse_connection_string(connection_string)
    account_name = parsed.get("AccountName", "")
    account_key = parsed.get("AccountKey", "")
    if not account_name or not account_key:
        raise ValueError("Connection string missing AccountName or AccountKey")

    sas_token = generate_blob_sas(
        account_name=account_name,
        account_key=account_key,
        container_name=container,
        blob_name=blob,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(tz=UTC) + timedelta(seconds=expiry_seconds),
    )

    return f"https://{account_name}.blob.core.windows.net/{container}/{blob}?{sas_token}"


class AzureBlobDataSource(DataSource, AsyncDataSource):
    """Azure Blob Storage adapter (sync + async)."""

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str | None = None,
    ) -> None:
        from azure.storage.blob import BlobServiceClient, ContainerClient

        self.connection_string = connection_string or os.getenv("BLOB_CONNECTION_STRING", "")
        self.container_name = container_name or os.getenv("BLOB_CONTAINER_NAME", "")

        self.blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client: ContainerClient = self.blob_service.get_container_client(
            self.container_name
        )
        # Lazy async clients
        self._async_blob_service = None
        self._async_container_client = None

    def list_documents(self, prefix: str | None = None) -> list[str]:
        """List all blob names in container (sync)."""
        blobs = self.container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]

    def load_document(self, doc_id: str) -> bytes:
        """Download blob content as bytes (sync)."""
        blob_client = self.container_client.get_blob_client(doc_id)
        stream = blob_client.download_blob()
        return stream.readall()

    async def alist_documents(self, prefix: str | None = None) -> list[str]:
        """List all blob names in container (async)."""
        client = await self._get_async_container_client()
        blobs = client.list_blobs(name_starts_with=prefix)
        return [blob.name async for blob in blobs]

    async def _get_async_container_client(self):
        """Lazy init async container client."""
        if self._async_container_client is None:
            from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

            self._async_blob_service = AsyncBlobServiceClient.from_connection_string(
                self.connection_string
            )
            self._async_container_client = self._async_blob_service.get_container_client(
                self.container_name
            )
        return self._async_container_client

    async def aload_document(self, doc_id: str) -> bytes:
        """Async download blob content as bytes."""
        client = await self._get_async_container_client()
        blob_client = client.get_blob_client(doc_id)
        stream = await blob_client.download_blob()
        return await stream.readall()

    async def aupload_document(
        self,
        blob_name: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        """Async upload content to blob storage.

        Returns:
            Full blob URL in format: azure://{container}/{blob_name}
        """
        from azure.storage.blob import ContentSettings

        client = await self._get_async_container_client()
        blob_client = client.get_blob_client(blob_name)

        content_settings = None
        if content_type:
            content_settings = ContentSettings(content_type=content_type)

        await blob_client.upload_blob(
            content,
            overwrite=True,
            content_settings=content_settings,
        )

        return f"azure://{self.container_name}/{blob_name}"

    async def aclose(self) -> None:
        """Close async clients."""
        if self._async_container_client:
            await self._async_container_client.close()
            self._async_container_client = None
        if self._async_blob_service:
            await self._async_blob_service.close()
            self._async_blob_service = None


__all__ = ["AzureBlobDataSource", "generate_azure_sas_url"]
