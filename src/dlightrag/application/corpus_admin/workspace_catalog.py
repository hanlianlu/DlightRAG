# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded workspace-catalog pages and opaque continuation cursors."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dlightrag.application.access import WorkspaceRecord
from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT = 50
WORKSPACE_CATALOG_PAGE_MAX_LIMIT = 100


class WorkspaceCatalogCursorError(ValueError):
    """An opaque workspace-catalog page cursor is malformed or fails checks."""


@dataclass(frozen=True, slots=True)
class WorkspaceCatalogCursor:
    """The complete ascending ordering key for one workspace-catalog page."""

    after_workspace: str

    def __post_init__(self) -> None:
        canonical_workspace = require_canonical_workspace_id(self.after_workspace)
        if canonical_workspace != self.after_workspace:
            raise ValueError("workspace-catalog cursor workspace must be canonical")


@dataclass(frozen=True, slots=True)
class WorkspaceCatalogPageRequest:
    """One hard-bounded ascending workspace-catalog page request."""

    limit: int = WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT
    cursor: WorkspaceCatalogCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("workspace-catalog page limit must be an integer")
        if not 1 <= self.limit <= WORKSPACE_CATALOG_PAGE_MAX_LIMIT:
            raise ValueError(
                "workspace-catalog page limit must be between 1 and "
                f"{WORKSPACE_CATALOG_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, WorkspaceCatalogCursor):
            raise ValueError("workspace-catalog cursor is invalid")


@dataclass(frozen=True, slots=True)
class WorkspaceCatalogRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    items: tuple[Mapping[str, Any], ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class WorkspaceCatalogPage:
    """Application page of workspace rows plus a typed continuation."""

    items: tuple[WorkspaceRecord, ...]
    next_cursor: WorkspaceCatalogCursor | None
    fetched_rows: int


class WorkspaceCatalogCursorCodec:
    """Encode workspace ordering facts as a signed, opaque continuation token.

    The cursor carries no authorization state: every page re-runs the caller's
    access gate over the returned rows, exactly like the full-catalog reads.
    """

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="workspace-catalog",
            scope="workspace-catalog",
            fields_by_version={1: {"after_workspace"}},
            current_version=1,
        )

    def encode(self, cursor: WorkspaceCatalogCursor) -> str:
        return self._envelope.encode({"after_workspace": cursor.after_workspace})

    def decode(self, token: str) -> WorkspaceCatalogCursor:
        try:
            decoded = self._envelope.decode(token)
            after_workspace = decoded["after_workspace"]
            if not isinstance(after_workspace, str):
                raise ValueError
            return WorkspaceCatalogCursor(after_workspace=after_workspace)
        except ValueError as exc:
            raise WorkspaceCatalogCursorError("invalid workspace-catalog page cursor") from exc


__all__ = [
    "WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT",
    "WORKSPACE_CATALOG_PAGE_MAX_LIMIT",
    "WorkspaceCatalogCursor",
    "WorkspaceCatalogCursorCodec",
    "WorkspaceCatalogCursorError",
    "WorkspaceCatalogPage",
    "WorkspaceCatalogPageRequest",
    "WorkspaceCatalogRowPage",
]
