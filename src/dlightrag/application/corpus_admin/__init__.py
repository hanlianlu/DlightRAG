# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus Administration use case and caller contracts."""

from dlightrag.engine.ai.telemetry import safe_log_text
from dlightrag.engine.rag.corpus.contracts import SourceType, VisualAssetSize
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.corpus.sources.url import validate_public_web_url
from dlightrag.engine.rag.workspace.workspaces import (
    normalize_workspace,
    normalize_workspace_ids,
)

from .errors import (
    LocalDownloadTarget,
    MetadataValidationError,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
    UnsafeUploadNameError,
    UploadTooLargeError,
)
from .service import (
    CorpusAdmin,
    CorpusAdminSettings,
    CorpusIngestError,
    CorpusResetResult,
    FilePanelSnapshot,
    FilePanelStore,
    IngestJobs,
    IngestSpec,
    SourceDownloadFactory,
    SourceDownloadPreparer,
    UploadReader,
    ingest_kwargs_from_spec,
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
    safe_upload_basename,
    validate_workspace_name,
)

__all__ = [
    "CorpusAdmin",
    "CorpusAdminSettings",
    "CorpusIngestError",
    "CorpusResetResult",
    "FilePanelSnapshot",
    "FilePanelStore",
    "IngestJobs",
    "IngestSpec",
    "LocalDownloadTarget",
    "MetadataValidationError",
    "RedirectDownloadTarget",
    "SourceDownloadFactory",
    "SourceDownloadInvalidError",
    "SourceDownloadNotFoundError",
    "SourceDownloadPreparer",
    "SourceDownloadTarget",
    "SourceDownloadUnavailableError",
    "UnsafeUploadNameError",
    "UploadReader",
    "UploadTooLargeError",
    "ingest_kwargs_from_spec",
    "ingest_spec_from_payload",
    "managed_local_ingest_documents",
    "managed_local_ingest_path",
    "normalize_workspace",
    "normalize_workspace_ids",
    "safe_log_text",
    "safe_source_filename",
    "safe_upload_basename",
    "SourceType",
    "validate_public_web_url",
    "validate_workspace_name",
    "VisualAssetSize",
]
