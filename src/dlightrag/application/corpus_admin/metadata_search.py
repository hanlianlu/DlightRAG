# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded metadata-search pages and opaque continuation cursors."""

from dataclasses import dataclass
from typing import Literal, cast

from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

METADATA_SEARCH_PAGE_DEFAULT_LIMIT = 50
METADATA_SEARCH_PAGE_MAX_LIMIT = 100
MetadataSearchFilenameMode = Literal["exact", "contains"]
_FILENAME_MODES = frozenset({"exact", "contains"})


class MetadataSearchCursorError(ValueError):
    """An opaque metadata-search page cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class MetadataSearchCursor:
    """The complete document-id ordering key for one workspace metadata page."""

    workspace: str
    after_doc_id: str
    mode: MetadataSearchFilenameMode

    def __post_init__(self) -> None:
        canonical_workspace = require_canonical_workspace_id(self.workspace)
        if canonical_workspace != self.workspace:
            raise ValueError("metadata-search cursor workspace must be canonical")
        if not isinstance(self.after_doc_id, str) or not self.after_doc_id:
            raise ValueError("metadata-search cursor document id must be non-empty")
        if len(self.after_doc_id) > 255:
            raise ValueError("metadata-search cursor document id exceeds the storage bound")
        if self.mode not in _FILENAME_MODES:
            raise ValueError("metadata-search cursor filename mode is invalid")


@dataclass(frozen=True, slots=True)
class MetadataSearchPageRequest:
    """One hard-bounded metadata-match page request."""

    limit: int = METADATA_SEARCH_PAGE_DEFAULT_LIMIT
    cursor: MetadataSearchCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("metadata-search page limit must be an integer")
        if not 1 <= self.limit <= METADATA_SEARCH_PAGE_MAX_LIMIT:
            raise ValueError(
                f"metadata-search page limit must be between 1 and {METADATA_SEARCH_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, MetadataSearchCursor):
            raise ValueError("metadata-search cursor is invalid")


@dataclass(frozen=True, slots=True)
class MetadataMatchRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    document_ids: tuple[str, ...]
    has_more: bool
    fetched_rows: int
    mode: MetadataSearchFilenameMode


@dataclass(frozen=True, slots=True)
class MetadataSearchPage:
    """Application page of matching document ids plus a typed continuation."""

    document_ids: tuple[str, ...]
    next_cursor: MetadataSearchCursor | None
    fetched_rows: int


class MetadataSearchCursorCodec:
    """Encode metadata-search ordering facts as a signed, workspace-bound token."""

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="metadata-match",
            scope="metadata-match",
            fields_by_version={1: {"after_doc_id", "mode", "workspace"}},
            current_version=1,
        )

    def encode(self, cursor: MetadataSearchCursor) -> str:
        return self._envelope.encode(
            {
                "after_doc_id": cursor.after_doc_id,
                "mode": cursor.mode,
                "workspace": cursor.workspace,
            }
        )

    def decode(self, token: str) -> MetadataSearchCursor:
        try:
            decoded = self._envelope.decode(token)
            after_doc_id = decoded["after_doc_id"]
            workspace = decoded["workspace"]
            mode = decoded["mode"]
            if not isinstance(after_doc_id, str) or not isinstance(workspace, str):
                raise ValueError
            if not isinstance(mode, str) or mode not in _FILENAME_MODES:
                raise ValueError
            return MetadataSearchCursor(
                workspace=workspace,
                after_doc_id=after_doc_id,
                mode=cast(MetadataSearchFilenameMode, mode),
            )
        except ValueError as exc:
            raise MetadataSearchCursorError("invalid metadata-search page cursor") from exc


__all__ = [
    "METADATA_SEARCH_PAGE_DEFAULT_LIMIT",
    "METADATA_SEARCH_PAGE_MAX_LIMIT",
    "MetadataMatchRowPage",
    "MetadataSearchCursor",
    "MetadataSearchCursorCodec",
    "MetadataSearchCursorError",
    "MetadataSearchFilenameMode",
    "MetadataSearchPage",
    "MetadataSearchPageRequest",
]
