# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded file-panel pages and opaque continuation cursors."""

import datetime
from dataclasses import dataclass
from typing import Any

from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

FILE_PANEL_PAGE_DEFAULT_LIMIT = 50
FILE_PANEL_PAGE_MAX_LIMIT = 100


class FilePanelCursorError(ValueError):
    """An opaque file-panel page cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class FilePanelCursor:
    """The complete mixed-direction ordering key for one workspace status view."""

    workspace: str
    updated_at: datetime.datetime | None
    doc_id: str
    view: str = "processed"

    def __post_init__(self) -> None:
        canonical_workspace = require_canonical_workspace_id(self.workspace)
        if canonical_workspace != self.workspace:
            raise ValueError("file-panel cursor workspace must be canonical")
        if self.updated_at is not None:
            if not isinstance(self.updated_at, datetime.datetime):
                raise ValueError("file-panel cursor timestamp must be a datetime or null")
            if self.updated_at.tzinfo is not None or self.updated_at.utcoffset() is not None:
                raise ValueError("file-panel cursor timestamp must not include a timezone")
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("file-panel cursor document id must be non-empty")
        if len(self.doc_id) > 255:
            raise ValueError("file-panel cursor document id exceeds the storage bound")
        if not isinstance(self.view, str) or self.view not in {"processed", "failed"}:
            raise ValueError("file-panel cursor view is invalid")


@dataclass(frozen=True, slots=True)
class FilePanelPageRequest:
    """One hard-bounded recent or older file-status page request."""

    limit: int = FILE_PANEL_PAGE_DEFAULT_LIMIT
    cursor: FilePanelCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("file-panel page limit must be an integer")
        if not 1 <= self.limit <= FILE_PANEL_PAGE_MAX_LIMIT:
            raise ValueError(
                f"file-panel page limit must be between 1 and {FILE_PANEL_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, FilePanelCursor):
            raise ValueError("file-panel cursor is invalid")


@dataclass(frozen=True, slots=True)
class ProcessedFileRow:
    """One processed document plus its private page-order facts."""

    doc_id: str
    file_path: str
    updated_at: datetime.datetime | None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("processed file document id must be non-empty")
        if self.updated_at is not None and (
            not isinstance(self.updated_at, datetime.datetime)
            or self.updated_at.tzinfo is not None
            or self.updated_at.utcoffset() is not None
        ):
            raise ValueError("processed file timestamp must be a naive datetime or null")

    def presentation(self) -> dict[str, Any]:
        """Return the transport-neutral fields callers may present."""
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "status": "processed",
            "updated_at": (
                self.updated_at.isoformat(timespec="microseconds")
                if self.updated_at is not None
                else ""
            ),
        }


@dataclass(frozen=True, slots=True)
class FilePanelRowPage:
    """Bounded processed-file result, including the physical fetch size."""

    items: tuple[ProcessedFileRow, ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class FailedFileRow:
    """One failed document plus its private page-order facts."""

    doc_id: str
    file_path: str
    error: str
    updated_at: datetime.datetime | None

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("failed file document id must be non-empty")
        if self.updated_at is not None and (
            not isinstance(self.updated_at, datetime.datetime)
            or self.updated_at.tzinfo is not None
            or self.updated_at.utcoffset() is not None
        ):
            raise ValueError("failed file timestamp must be a naive datetime or null")

    def presentation(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "error": self.error,
            "updated_at": (
                self.updated_at.isoformat(timespec="microseconds")
                if self.updated_at is not None
                else ""
            ),
        }


@dataclass(frozen=True, slots=True)
class FailedFileRowPage:
    """Bounded failed-file result, including the physical fetch size."""

    items: tuple[FailedFileRow, ...]
    has_more: bool
    fetched_rows: int


class FilePanelCursorCodec:
    """Encode file ordering facts as a signed, opaque, workspace-bound token."""

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="file-panel",
            scope="file-panel",
            fields_by_version={
                1: {"doc_id", "updated_at", "workspace"},
                2: {"doc_id", "updated_at", "view", "workspace"},
            },
            current_version=2,
        )

    def encode(self, cursor: FilePanelCursor) -> str:
        return self._envelope.encode(
            {
                "doc_id": cursor.doc_id,
                "updated_at": _canonical_timestamp(cursor.updated_at),
                "view": cursor.view,
                "workspace": cursor.workspace,
            }
        )

    def decode(self, token: str) -> FilePanelCursor:
        try:
            decoded = self._envelope.decode(token)
            version = decoded["v"]
            view = "processed" if version == 1 else decoded["view"]
            doc_id = decoded["doc_id"]
            workspace = decoded["workspace"]
            timestamp_value = decoded["updated_at"]
            if not isinstance(doc_id, str) or not isinstance(workspace, str):
                raise ValueError
            if timestamp_value is None:
                updated_at = None
            elif isinstance(timestamp_value, str):
                updated_at = datetime.datetime.fromisoformat(timestamp_value)
                if _canonical_timestamp(updated_at) != timestamp_value:
                    raise ValueError
            else:
                raise ValueError
            return FilePanelCursor(
                workspace=workspace,
                updated_at=updated_at,
                doc_id=doc_id,
                view=view,
            )
        except ValueError as exc:
            raise FilePanelCursorError("invalid file-panel page cursor") from exc


def _canonical_timestamp(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None or value.utcoffset() is not None:
        raise ValueError("file-panel cursor timestamp must not include a timezone")
    return value.isoformat(timespec="microseconds")


__all__ = [
    "FILE_PANEL_PAGE_DEFAULT_LIMIT",
    "FILE_PANEL_PAGE_MAX_LIMIT",
    "FailedFileRow",
    "FailedFileRowPage",
    "FilePanelCursor",
    "FilePanelCursorCodec",
    "FilePanelCursorError",
    "FilePanelPageRequest",
    "FilePanelRowPage",
    "ProcessedFileRow",
]
