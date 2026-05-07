# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and response models for the DlightRAG REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

# ═══════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════


class MetadataFilterRequest(BaseModel):
    """Structured metadata filter for retrieval queries."""

    filename: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    rag_mode: str | None = None
    custom: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    source_type: Literal["local", "azure_blob", "s3"]
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    prefix: str | None = None
    query: str | None = None
    table: str | None = None
    bucket: str | None = None
    key: str | None = None
    replace: bool | None = None
    workspace: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> IngestRequest:
        if self.source_type == "local":
            if not self.path:
                raise ValueError("'path' is required for local ingestion")
        elif self.source_type == "azure_blob":
            if not self.container_name:
                raise ValueError("'container_name' is required for azure_blob")
        elif self.source_type == "s3":
            if not self.bucket:
                raise ValueError("'bucket' is required for s3")
            if not self.key:
                raise ValueError("'key' is required for s3")
        return self


class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    filters: MetadataFilterRequest | None = None


class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    stream: bool
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    multimodal_content: list[dict[str, Any]] | None = None
    conversation_history: list[dict[str, str]] | None = None
    query_images: list[str | dict[str, Any]] | None = None
    """User-attached images inlined into the answer LLM call as
    OpenAI ``image_url`` content blocks. Each item is either a URL/data
    URI string or a pre-built ``{"type":"image_url","image_url":{...}}``
    dict. Distinct from ``multimodal_content`` (which drives visual
    retrieval in unified mode); pass both if you want the same images
    used for both retrieval and answer reasoning."""

    @field_validator("multimodal_content")
    @classmethod
    def validate_image_count(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v and len(v) > 3:
            raise ValueError("Maximum 3 multimodal items per query")
        return v

    @field_validator("query_images")
    @classmethod
    def validate_query_image_count(
        cls, v: list[str | dict[str, Any]] | None
    ) -> list[str | dict[str, Any]] | None:
        if v and len(v) > 10:
            raise ValueError("Maximum 10 query_images per request")
        return v


class DeleteRequest(BaseModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    delete_source: bool = True
    workspace: str | None = None


class ResetRequest(BaseModel):
    """Request to reset a workspace."""

    workspace: str | None = None
    keep_files: bool = False
    dry_run: bool = False


class MetadataUpdateRequest(BaseModel):
    metadata: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════


class ErrorDetail(BaseModel):
    detail: str
    error_type: str  # "unavailable", "validation", "auth", "internal"
