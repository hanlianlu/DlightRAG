# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser contracts for the Files panel and ingest polling."""

from typing import Literal

from dlightrag.application.answer_runs.client_contracts import ClientContractModel


class WebFileItem(ClientContractModel):
    file_name: str
    file_path: str


class WebIngestStatus(ClientContractModel):
    busy: bool = False
    message: str = ""
    progress_percent: int | None = None
    current_batch: int | None = None
    total_batches: int | None = None
    documents: int | None = None
    pending_enqueues: int = 0


class WebFilePanelSnapshot(ClientContractModel):
    workspace: str
    files: list[WebFileItem]
    ingest: WebIngestStatus
    next_cursor: str | None = None


class WebUploadReceipt(ClientContractModel):
    workspace: str
    file_count: int
    queued: bool
    ingest: WebIngestStatus


class WebFailedFileItem(ClientContractModel):
    document_id: str
    file_name: str
    error: str
    updated_at: str


type WebFailedRecoveryStatus = Literal["queued", "running", "succeeded", "partial", "failed"]


class WebFailedRecoveryJob(ClientContractModel):
    job_id: str
    workspace: str
    status: WebFailedRecoveryStatus
    retried: int = 0
    succeeded: int = 0
    failed: int = 0


class WebFailedFilesPage(ClientContractModel):
    workspace: str
    failed: list[WebFailedFileItem]
    next_cursor: str | None = None
    active_recovery: WebFailedRecoveryJob | None = None


__all__ = [
    "WebFailedFileItem",
    "WebFailedFilesPage",
    "WebFailedRecoveryJob",
    "WebFailedRecoveryStatus",
    "WebFileItem",
    "WebFilePanelSnapshot",
    "WebIngestStatus",
    "WebUploadReceipt",
]
