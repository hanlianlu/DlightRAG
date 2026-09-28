# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus Administration caller errors and download outcomes."""

from dataclasses import dataclass
from pathlib import Path

from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
)


class UnsafeUploadNameError(ApplicationInputError):
    """An upload filename is unsafe or not a single basename."""


class CorpusMutationInputError(ApplicationInputError):
    """A Corpus Mutation request the caller must change before it can be accepted."""


class LocalIngestPathError(ApplicationInputError):
    """A caller-supplied local ingest path is absolute or escapes its workspace input root."""


class WorkspaceNameError(ApplicationInputError):
    """A user-facing workspace name is empty, too long, or has forbidden characters."""


class WorkspaceExistsError(ApplicationConflictError):
    """A workspace with this canonical identity is already registered."""


class UploadTooLargeError(ApplicationInputError):
    """A streamed upload exceeded its configured byte cap."""


class MetadataValidationError(ApplicationInputError):
    """Caller-supplied document metadata is invalid."""


class SourceDownloadInvalidError(ValueError):
    """Stored source metadata cannot produce a safe download."""


class SourceDownloadNotFoundError(RuntimeError):
    """The requested document or retained bytes do not exist."""


class SourceDownloadUnavailableError(ApplicationUnavailableError):
    """A remote source adapter cannot currently sign a download."""


class CorpusMutationUnavailableError(ApplicationUnavailableError):
    """This deployment cannot accept corpus writes, because it is a read-only replica.

    A `reader` process registers no corpus-mutation executor, so accepting a write would
    stage bytes for a Run that can never execute and then fail it late; direct corpus
    writes such as workspace creation are refused the same way. The refusal happens
    before anything changes and names the remedy: send ``request`` to a `writer`.
    """

    def __init__(self, *, request: str) -> None:
        super().__init__(
            "This deployment is a read-only replica of the knowledge base: it accepts no "
            f"corpus writes. Send {request} to a writer."
        )


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
    "WorkspaceExistsError",
]
