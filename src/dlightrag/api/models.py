# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and response models for the DlightRAG REST API."""

import datetime
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from dlightrag.access import validate_query_workspace_selection
from dlightrag.answer.citations.schemas import SourceReferencePayload
from dlightrag.answer.client_contracts import (
    MAX_HISTORY_CONTENT_CHARS,
    MAX_HISTORY_MESSAGES,
    AnswerRequestContract,
    ClientContractModel,
    RetrieveRequestContract,
)
from dlightrag.runtime import AnswerRunPhase, AnswerRunStatus
from dlightrag.services.corpora import IngestSpec

# Maximum UTF-8 history payload plus query/workspace/JSON framing. Shared by the
# REST multipart parser and its receive-layer body cap.
ANSWER_REQUEST_PART_MAX_BYTES = MAX_HISTORY_MESSAGES * MAX_HISTORY_CONTENT_CHARS * 4 + 64 * 1024

# ═══════════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════════


class QueryWorkspaceSelection(ClientContractModel):
    """REST/MCP query workspace selector."""

    workspaces: list[str] | None = None
    all_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_workspace_selection(self) -> Self:
        validate_query_workspace_selection(
            all_workspaces=self.all_workspaces,
            workspaces=self.workspaces,
        )
        return self


class MetadataFilterRequest(ClientContractModel):
    """Structured metadata filter for retrieval queries."""

    filename: str | None = None
    file_extension: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date_from: datetime.datetime | None = None
    creation_date_to: datetime.datetime | None = None
    custom: dict[str, Any] | None = None


class IngestRequest(IngestSpec):
    workspace: str | None = None


class RetrieveRequest(QueryWorkspaceSelection, RetrieveRequestContract):
    filters: MetadataFilterRequest | None = None


class AnswerRequest(QueryWorkspaceSelection, AnswerRequestContract):
    filters: MetadataFilterRequest | None = None
    """Prior conversation turns supplied by the caller. Stateless: the client
    owns persistence and re-sends history each request; DlightRAG never stores
    it. Feeds the planner's standalone-query rewrite and answer generation."""


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


# ═══════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════


class ReferenceSummary(ClientContractModel):
    id: str
    title: str | None = None


class RetrievalResponse(ClientContractModel):
    contexts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sources: list[SourceReferencePayload] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    image_descriptions: list[str] = Field(default_factory=list)


class AnswerResponse(RetrievalResponse):
    answer: str | None = None
    references: list[ReferenceSummary] = Field(default_factory=list)
    answer_images: list[dict[str, Any]] = Field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)
    primary_report: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class AnswerRunDescriptor(ClientContractModel):
    """The 202 acceptance every answer request receives, replay included."""

    run_id: str
    status: AnswerRunStatus
    status_url: str
    events_url: str
    cancel_url: str


class AnswerRunStatusResponse(AnswerRunDescriptor):
    """Authoritative lifecycle state, plus the canonical result once it exists."""

    phase: AnswerRunPhase | None = None
    durable_progress_version: int = 0
    cancel_requested: bool = False
    result: AnswerResponse | None = None
    error_kind: str | None = None
    error_message: str | None = None
    created_at: datetime.datetime | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None


class IngestJobStatusResponse(ClientContractModel):
    # Job rows carry queue bookkeeping (lease_owner, lease_expires_at) that clients
    # must not see; ignoring extras drops it instead of failing response validation.
    model_config = ConfigDict(extra="ignore")

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
    errors_truncated: bool | None = None
    result: dict[str, Any] | None = None
    created_at: datetime.datetime | str | None = None
    updated_at: datetime.datetime | str | None = None
    started_at: datetime.datetime | str | None = None
    finished_at: datetime.datetime | str | None = None


class UploadIngestJobResponse(IngestJobStatusResponse):
    """Ingest job response for direct file uploads (adds the persisted path)."""

    uploaded_file: str | None = None
    filename: str | None = None


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


class SearchMetadataResponse(ClientContractModel):
    document_ids: list[str]
    count: int
    workspace: str


class MetadataUpdateResponse(ClientContractModel):
    status: Literal["success"]
    doc_id: str


class ResetResponse(ClientContractModel):
    workspaces: dict[str, Any]
    total_errors: int


class ErrorDetail(ClientContractModel):
    detail: str
    error_type: str  # "unavailable", "validation", "auth", "configuration", "internal"
    error_kind: str | None = None  # stable answer-image error kind, if applicable

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize omitting the optional error_kind unless a classification applies."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)
