# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product error and outcome types the transports use without corpus internals.

RAG-core raises its own pool, port, and storage exceptions, and resolves source
downloads into its own target types. The service layer translates them into
these product types at the boundary, so REST, MCP, and Web depend on typed
application services only.
"""

from dataclasses import dataclass
from pathlib import Path


class CorpusUnavailableError(RuntimeError):
    """A corpus workspace cannot be built or its pool is closed."""


class StorageSchemaError(RuntimeError):
    """Durable storage schema is incompatible with this revision."""


class MetadataValidationError(ValueError):
    """A metadata filter is invalid."""


class UnsafeUploadNameError(ValueError):
    """An upload filename is unsafe or not a single basename."""


class UploadTooLargeError(ValueError):
    """A streamed upload exceeded its byte cap."""


class SourceDownloadInvalidError(ValueError):
    """Stored source metadata cannot produce a safe download."""


class SourceDownloadNotFoundError(RuntimeError):
    """The requested document or retained bytes do not exist."""


class SourceDownloadUnavailableError(RuntimeError):
    """A configured remote storage adapter cannot currently sign a download."""


@dataclass(frozen=True, slots=True)
class LocalDownloadTarget:
    """Contained local file ready for an HTTP adapter to stream."""

    path: Path
    media_type: str
    filename: str


@dataclass(frozen=True, slots=True)
class RedirectDownloadTarget:
    """Remote URL ready for an HTTP adapter to redirect to."""

    url: str


SourceDownloadTarget = LocalDownloadTarget | RedirectDownloadTarget


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
