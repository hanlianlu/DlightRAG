# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral client payload contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClientContractModel(BaseModel):
    """Base model for public client contracts."""

    model_config = ConfigDict(extra="forbid")


class ImageURL(ClientContractModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class TextContentBlock(ClientContractModel):
    type: Literal["text"]
    text: str


class ImageURLContentBlock(ClientContractModel):
    type: Literal["image_url"]
    image_url: ImageURL


type ContentBlock = Annotated[
    TextContentBlock | ImageURLContentBlock,
    Field(discriminator="type"),
]
type QueryImage = ImageURLContentBlock


class ConversationMessage(ClientContractModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlock]


type SourceType = Literal["local", "azure_blob", "s3", "url"]
type MetadataPolicy = Literal["validate", "reject_unknown", "store_only"]


class IngestDocument(ClientContractModel):
    """One explicitly listed document in an ingest manifest."""

    path: str | None = None
    key: str | None = None
    url: str | None = None
    filename: str | None = None
    source_uri: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None


class IngestSpec(ClientContractModel):
    """Transport-neutral ingest source specification shared by SDK, REST, and MCP."""

    source_type: SourceType
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    prefix: str | None = None
    bucket: str | None = None
    region: str | None = None
    key: str | None = None
    url: str | None = None
    urls: list[str] | None = None
    filename: str | None = None
    source_uri: str | None = None
    source_uris: list[str] | None = None
    documents: list[IngestDocument] | None = None
    retain_source_file: bool | None = None
    replace: bool | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> IngestSpec:
        if self.source_type == "local":
            if self.documents is not None:
                if self.path:
                    raise ValueError("'path' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "path", source_type=self.source_type)
                return self
            if not self.path:
                raise ValueError("'path' is required for local ingestion")
        elif self.source_type == "azure_blob":
            if not self.container_name:
                raise ValueError("'container_name' is required for azure_blob")
            if self.documents is not None:
                if self.blob_path or self.prefix is not None:
                    raise ValueError("'blob_path'/'prefix' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "key", source_type=self.source_type)
                return self
            if self.blob_path and self.prefix is not None:
                raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
        elif self.source_type == "s3":
            if not self.bucket:
                raise ValueError("'bucket' is required for s3")
            if self.documents is not None:
                if self.key or self.prefix is not None:
                    raise ValueError("'key'/'prefix' and 'documents' are mutually exclusive")
                _require_document_field(self.documents, "key", source_type=self.source_type)
                return self
            if not self.key and self.prefix is None:
                raise ValueError("'key' or 'prefix' is required for s3")
            if self.key and self.prefix is not None:
                raise ValueError("'key' and 'prefix' are mutually exclusive")
        elif self.source_type == "url":
            if self.documents is not None:
                if any(
                    value is not None
                    for value in (
                        self.url,
                        self.urls,
                        self.filename,
                        self.source_uri,
                        self.source_uris,
                    )
                ):
                    raise ValueError(
                        "'url'/'urls'/'filename'/'source_uri' and 'documents' are mutually exclusive"
                    )
                _require_document_field(self.documents, "url", source_type=self.source_type)
                return self
            url_count = _url_count(self.url, self.urls)
            if url_count == 0:
                raise ValueError("'url' or 'urls' is required for url ingestion")
            if self.url and self.urls is not None:
                raise ValueError("'url' and 'urls' are mutually exclusive")
            if self.filename and url_count != 1:
                raise ValueError("'filename' can only be used with a single url")
            if self.source_uri and self.source_uris is not None:
                raise ValueError("'source_uri' and 'source_uris' are mutually exclusive")
            if self.source_uri and url_count != 1:
                raise ValueError("'source_uri' can only be used with a single url")
            if self.source_uris is not None and len(self.source_uris) != url_count:
                raise ValueError("'source_uris' must match the number of urls")
        return self


class IngestPayload(IngestSpec):
    """Transport ingest request; workspace routes the request, not the source spec."""

    workspace: str | None = None


def _url_count(url: str | None, urls: list[str] | None) -> int:
    if url:
        return 1
    return len(urls or [])


def _require_document_field(
    documents: list[IngestDocument], field_name: Literal["path", "key", "url"], *, source_type: str
) -> None:
    if not documents:
        raise ValueError("'documents' must contain at least one document")
    for index, document in enumerate(documents):
        if not getattr(document, field_name):
            raise ValueError(
                f"'documents[{index}].{field_name}' is required for {source_type} ingestion"
            )


def model_dump_json_safe(value: Any) -> Any:
    """Return plain JSON-ready data from Pydantic models and containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_dump_json_safe(item) for key, item in value.items()}
    return value


def dump_optional_list(value: list[Any] | None) -> list[Any] | None:
    if value is None:
        return None
    return model_dump_json_safe(value)


__all__ = [
    "ClientContractModel",
    "ContentBlock",
    "ConversationMessage",
    "ImageURL",
    "ImageURLContentBlock",
    "IngestDocument",
    "IngestPayload",
    "IngestSpec",
    "MetadataPolicy",
    "QueryImage",
    "SourceType",
    "TextContentBlock",
    "dump_optional_list",
    "model_dump_json_safe",
]
