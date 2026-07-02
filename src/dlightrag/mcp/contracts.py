# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic input contracts for DlightRAG MCP tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dlightrag.core.client_contracts import ClientContractModel, ConversationMessage, QueryImage

MetadataPolicy = Literal["validate", "reject_unknown", "store_only"]
SourceType = Literal["local", "azure_blob", "s3"]


class MCPInput(ClientContractModel):
    pass


class RetrieveInput(MCPInput):
    query: str
    top_k: int | None = None
    workspaces: list[str] | None = None
    filters: dict[str, Any] | None = None
    query_images: list[QueryImage] = Field(default_factory=list, max_length=3)
    session_id: str | None = None
    referenced_image_ids: list[str] = Field(default_factory=list)


class AnswerInput(RetrieveInput):
    chunk_top_k: int | None = None
    answer_context_top_k: int | None = None
    conversation_history: list[ConversationMessage] | None = None


class IngestInput(MCPInput):
    source_type: SourceType
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    bucket: str | None = None
    key: str | None = None
    prefix: str | None = None
    replace: bool | None = None
    workspace: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_policy: MetadataPolicy | None = None
    wait: bool | None = None

    @model_validator(mode="after")
    def validate_selectors(self) -> IngestInput:
        if self.source_type == "local" and not self.path:
            raise ValueError("'path' is required for local ingestion")
        if self.source_type == "azure_blob" and not self.container_name:
            raise ValueError("'container_name' is required for azure_blob")
        if self.source_type == "azure_blob" and self.blob_path and self.prefix is not None:
            raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
        if self.source_type == "s3" and not self.bucket:
            raise ValueError("'bucket' is required for s3")
        if self.source_type == "s3" and not self.key and self.prefix is None:
            raise ValueError("'key' or 'prefix' is required for s3")
        if self.source_type == "s3" and self.key and self.prefix is not None:
            raise ValueError("'key' and 'prefix' are mutually exclusive")
        return self


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


__all__ = [
    "AnswerInput",
    "CreateWorkspaceInput",
    "DeleteFilesInput",
    "DeleteWorkspaceInput",
    "IngestInput",
    "IngestJobStatusInput",
    "ListFilesInput",
    "QueryImage",
    "RetrieveInput",
]
