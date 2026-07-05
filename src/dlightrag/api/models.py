# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and response models for the DlightRAG REST API."""

import datetime
from typing import Any, Literal

from pydantic import Field

from dlightrag.citations.schemas import SourceReference
from dlightrag.core.client_contracts import (
    ClientContractModel,
    ContentBlock,
    ConversationMessage,
    IngestPayload,
    QueryImage,
)

# ═══════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════


class MetadataFilterRequest(ClientContractModel):
    """Structured metadata filter for retrieval queries."""

    filename: str | None = None
    filename_stem: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    custom: dict[str, Any] | None = None


class IngestRequest(IngestPayload):
    pass


class RetrieveRequest(ClientContractModel):
    query: str
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    filters: MetadataFilterRequest | None = None
    multimodal_content: list[ContentBlock] | None = Field(default=None, max_length=3)
    query_images: list[QueryImage] | None = Field(default=None, max_length=3)
    session_id: str | None = None
    referenced_image_ids: list[str] | None = None


class AnswerRequest(ClientContractModel):
    query: str
    stream: bool = True
    top_k: int | None = None
    chunk_top_k: int | None = None
    answer_context_top_k: int | None = Field(default=None, ge=1)
    workspaces: list[str] | None = None
    filters: MetadataFilterRequest | None = None
    multimodal_content: list[ContentBlock] | None = Field(default=None, max_length=3)
    conversation_history: list[ConversationMessage] | None = None
    session_id: str | None = None
    referenced_image_ids: list[str] | None = None
    semantic_highlights: bool = False
    query_images: list[QueryImage] | None = Field(default=None, max_length=3)
    """User-attached images used for VLM semantic enhancement, direct visual
    retrieval, session image memory, and bounded answer-model image blocks."""


class DeleteRequest(ClientContractModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    workspace: str | None = None
    dry_run: bool = False


class WorkspaceCreateRequest(ClientContractModel):
    """Request to create an empty workspace."""

    workspace: str
    display_name: str | None = None


class ResetRequest(ClientContractModel):
    """Request to reset a workspace."""

    workspace: str | None = None
    keep_files: bool = False
    dry_run: bool = False


class MetadataUpdateRequest(ClientContractModel):
    metadata: dict[str, Any]
    mode: Literal["merge", "replace"] = "merge"
    metadata_policy: Literal["validate", "reject_unknown", "store_only"] | None = None


# ═══════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════


class ReferenceSummary(ClientContractModel):
    id: str
    title: str | None = None


class RetrievalResponse(ClientContractModel):
    answer: str | None = None
    contexts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    image_descriptions: list[str] = Field(default_factory=list)
    current_image_ids: list[str] = Field(default_factory=list)


class AnswerResponse(RetrievalResponse):
    references: list[ReferenceSummary] = Field(default_factory=list)
    answer_images: list[dict[str, Any]] = Field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)


class IngestJobStatusResponse(ClientContractModel):
    job_id: str
    workspace: str | None = None
    source_type: str | None = None
    status: str
    status_url: str | None = None
    request: dict[str, Any] | None = None
    total_items: int | None = None
    processed_items: int | None = None
    failed_items: int | None = None
    current_window: int | None = None
    errors: list[str] | None = None
    result: dict[str, Any] | None = None
    created_at: datetime.datetime | str | None = None
    updated_at: datetime.datetime | str | None = None
    started_at: datetime.datetime | str | None = None
    finished_at: datetime.datetime | str | None = None


class FileListResponse(ClientContractModel):
    files: list[Any]
    count: int
    workspace: str


class FailedFilesResponse(ClientContractModel):
    failed: list[Any]
    count: int
    workspace: str


class DeleteFilesResponse(ClientContractModel):
    results: list[dict[str, Any]]
    workspace: str


class WorkspaceRecord(ClientContractModel):
    workspace: str
    display_name: str
    embedding_model: str
    created_at: str | None = None
    updated_at: str | None = None


class WorkspacesResponse(ClientContractModel):
    workspaces: list[str]
    records: list[WorkspaceRecord]


class WorkspaceCreateResponse(ClientContractModel):
    workspace: str
    display_name: str
    created: bool


class WorkspaceDeleteResponse(ClientContractModel):
    workspace: str
    deleted: bool
    result: dict[str, Any]


class MetadataResponse(ClientContractModel):
    doc_id: str
    metadata: dict[str, Any]


class MetadataUpdateResponse(ClientContractModel):
    status: Literal["success"]
    doc_id: str


class ResetResponse(ClientContractModel):
    workspaces: dict[str, Any]
    total_errors: int


class HealthStorageResponse(ClientContractModel):
    vector: str
    graph: str
    kv: str


class HealthResponse(ClientContractModel):
    status: Literal["healthy", "degraded"]
    rag_initialized: bool
    crafted_by: str
    maintained_by: str
    storage: HealthStorageResponse
    warnings: list[str] | None = None
    postgres: str


class ErrorDetail(ClientContractModel):
    detail: str
    error_type: str  # "unavailable", "validation", "auth", "internal"
