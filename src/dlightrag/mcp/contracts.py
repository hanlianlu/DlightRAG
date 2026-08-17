# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic input contracts for DlightRAG MCP tools."""

from typing import Any, Self

from pydantic import Field, model_validator

from dlightrag.access import validate_query_workspace_selection
from dlightrag.answer.resources.images import MAX_QUERY_IMAGES
from dlightrag.core.client_contracts import (
    AnswerAttachmentLink,
    AnswerRequestContract,
    ClientContractModel,
    ConversationMessage,
    QueryImage,
    RetrieveRequestContract,
)
from dlightrag.services.corpora import IngestSpec


class MCPInput(ClientContractModel):
    pass


class QueryWorkspaceSelection(ClientContractModel):
    workspaces: list[str] | None = None
    all_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_workspace_selection(self) -> Self:
        validate_query_workspace_selection(
            all_workspaces=self.all_workspaces,
            workspaces=self.workspaces,
        )
        return self


class RetrieveInput(QueryWorkspaceSelection, RetrieveRequestContract):
    filters: dict[str, Any] | None = None
    query_images: list[QueryImage] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=list,
        max_length=MAX_QUERY_IMAGES,
    )


class AnswerInput(QueryWorkspaceSelection, AnswerRequestContract):
    filters: dict[str, Any] | None = None
    attachments: list[AnswerAttachmentLink] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=list,
    )
    idempotency_key: str | None = Field(default=None, max_length=255)


class AnswerRunInput(MCPInput):
    """One owned answer run addressed by the id the answer tool returned."""

    run_id: str = Field(min_length=1, max_length=64)


class IngestInput(IngestSpec):
    workspace: str | None = None


class IngestJobStatusInput(MCPInput):
    job_id: str


class CreateWorkspaceInput(MCPInput):
    workspace: str
    display_name: str | None = None


class DeleteWorkspaceInput(MCPInput):
    workspace: str
    keep_files: bool = False
    dry_run: bool = False


class ListFilesInput(MCPInput):
    workspace: str | None = None


class DeleteFilesInput(MCPInput):
    filenames: list[str] | None = None
    file_paths: list[str] | None = None
    workspace: str | None = None
    dry_run: bool = False


__all__ = [
    "AnswerInput",
    "AnswerRunInput",
    "ConversationMessage",
    "CreateWorkspaceInput",
    "DeleteFilesInput",
    "DeleteWorkspaceInput",
    "IngestInput",
    "IngestJobStatusInput",
    "ListFilesInput",
    "RetrieveInput",
]
