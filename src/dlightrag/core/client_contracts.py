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


class IngestPayload(ClientContractModel):
    """Transport-neutral ingest request fields shared by REST and MCP."""

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
    retain_source_file: bool | None = None
    replace: bool | None = None
    workspace: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> IngestPayload:
        if self.source_type == "local":
            if not self.path:
                raise ValueError("'path' is required for local ingestion")
        elif self.source_type == "azure_blob":
            if not self.container_name:
                raise ValueError("'container_name' is required for azure_blob")
            if self.blob_path and self.prefix is not None:
                raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
        elif self.source_type == "s3":
            if not self.bucket:
                raise ValueError("'bucket' is required for s3")
            if not self.key and self.prefix is None:
                raise ValueError("'key' or 'prefix' is required for s3")
            if self.key and self.prefix is not None:
                raise ValueError("'key' and 'prefix' are mutually exclusive")
        elif self.source_type == "url":
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


def _url_count(url: str | None, urls: list[str] | None) -> int:
    if url:
        return 1
    return len(urls or [])


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
    "IngestPayload",
    "MetadataPolicy",
    "QueryImage",
    "SourceType",
    "TextContentBlock",
    "dump_optional_list",
    "model_dump_json_safe",
]
