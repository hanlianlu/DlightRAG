# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic input contracts for DlightRAG MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dlightrag.core.client_contracts import (
    ClientContractModel,
    ConversationMessage,
    IngestPayload,
    MetadataPolicy,
    QueryImage,
    SourceType,
)


class MCPInput(ClientContractModel):
    pass


class RetrieveInput(MCPInput):
    query: str
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
    filters: dict[str, Any] | None = None
    query_images: list[QueryImage] = Field(default_factory=list, max_length=3)
    session_id: str | None = None
    referenced_image_ids: list[str] = Field(default_factory=list)


class AnswerInput(RetrieveInput):
    chunk_top_k: int | None = None
    answer_context_top_k: int | None = None
    conversation_history: list[ConversationMessage] | None = None
    semantic_highlights: bool = False


class IngestInput(IngestPayload):
    pass


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
    "CreateWorkspaceInput",
    "DeleteFilesInput",
    "DeleteWorkspaceInput",
    "IngestInput",
    "IngestJobStatusInput",
    "ListFilesInput",
    "MetadataPolicy",
    "QueryImage",
    "RetrieveInput",
    "SourceType",
]
