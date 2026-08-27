# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus Administration caller errors and download outcomes."""

from dataclasses import dataclass
from pathlib import Path


class UnsafeUploadNameError(ValueError):
    """An upload filename is unsafe or not a single basename."""


class UploadTooLargeError(ValueError):
    """A streamed upload exceeded its configured byte cap."""


class MetadataValidationError(ValueError):
    """Caller-supplied document metadata is invalid."""


class SourceDownloadInvalidError(ValueError):
    """Stored source metadata cannot produce a safe download."""


class SourceDownloadNotFoundError(RuntimeError):
    """The requested document or retained bytes do not exist."""


class SourceDownloadUnavailableError(RuntimeError):
    """A remote source adapter cannot currently sign a download."""


@dataclass(frozen=True, slots=True)
class LocalDownloadTarget:
    """Contained local file ready for a transport to stream."""

    path: Path
    media_type: str
    filename: str


@dataclass(frozen=True, slots=True)
class RedirectDownloadTarget:
    """Remote URL ready for a transport to redirect to."""

    url: str


type SourceDownloadTarget = LocalDownloadTarget | RedirectDownloadTarget


__all__ = [
    "LocalDownloadTarget",
    "MetadataValidationError",
    "RedirectDownloadTarget",
    "SourceDownloadInvalidError",
    "SourceDownloadNotFoundError",
    "SourceDownloadTarget",
    "SourceDownloadUnavailableError",
    "UnsafeUploadNameError",
    "UploadTooLargeError",
]
