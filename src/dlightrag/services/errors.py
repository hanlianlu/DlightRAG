# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors and outcomes projected by typed application services.

Corpus-owned identities are re-exported, not translated: transports and RAG
helpers observe the same exception and download-target objects.
"""

from dlightrag.rag.ingestion.uploads import UploadTooLargeError
from dlightrag.rag.ports import CorpusUnavailableError
from dlightrag.rag.retrieval.metadata_fields import MetadataValidationError
from dlightrag.rag.source_download import (
    LocalDownloadTarget,
    RedirectDownloadTarget,
    SourceDownloadInvalidError,
    SourceDownloadNotFoundError,
    SourceDownloadTarget,
    SourceDownloadUnavailableError,
)


class StorageSchemaError(RuntimeError):
    """Durable storage schema is incompatible with this revision."""


class UnsafeUploadNameError(ValueError):
    """An upload filename is unsafe or not a single basename."""


__all__ = [
    "CorpusUnavailableError",
    "LocalDownloadTarget",
    "MetadataValidationError",
    "RedirectDownloadTarget",
    "SourceDownloadInvalidError",
    "SourceDownloadNotFoundError",
    "SourceDownloadTarget",
    "SourceDownloadUnavailableError",
    "StorageSchemaError",
    "UnsafeUploadNameError",
    "UploadTooLargeError",
]
