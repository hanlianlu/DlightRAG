# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral corpus contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

type SourceType = Literal["local", "azure_blob", "s3", "url"]
type VisualAssetSize = Literal["full", "thumb"]


class _RagContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class IngestDocument(_RagContractModel):
    """One explicitly listed document in a corpus-ingest manifest."""

    path: str | None = None
    key: str | None = None
    url: str | None = None
    filename: str | None = None
    source_uri: str | None = None
    download_uri: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None


class Reference(BaseModel):
    """One document-level reference returned by corpus retrieval."""

    id: str = Field(description="Reference id matching the corpus context")
    title: str = Field(description="Document title or filename")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> str:
        return str(value)


__all__ = ["IngestDocument", "Reference", "SourceType", "VisualAssetSize"]
