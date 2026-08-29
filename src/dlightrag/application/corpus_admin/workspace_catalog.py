# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded workspace-catalog pages and opaque continuation cursors."""

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dlightrag.application.access import WorkspaceRecord
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT = 50
WORKSPACE_CATALOG_PAGE_MAX_LIMIT = 100
_CURSOR_MAC_BYTES = 16
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


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

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: WorkspaceCatalogCursor) -> str:
        payload = _canonical_json(
            {
                "after_workspace": cursor.after_workspace,
                "scope": "workspace-catalog",
                "v": 1,
            }
        )
        mac = _cursor_mac(self._secret, payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> WorkspaceCatalogCursor:
        try:
            encoded, encoded_mac = token.split(".")
            if not encoded or not encoded_mac:
                raise ValueError
            payload = _base64url_decode(encoded)
            supplied_mac = _base64url_decode(encoded_mac)
            expected_mac = _cursor_mac(self._secret, payload)
            if len(supplied_mac) != _CURSOR_MAC_BYTES or not hmac.compare_digest(
                supplied_mac, expected_mac
            ):
                raise ValueError
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or set(decoded) != {
                "after_workspace",
                "scope",
                "v",
            }:
                raise ValueError
            if _canonical_json(decoded) != payload:
                raise ValueError
            if (
                type(decoded["v"]) is not int
                or decoded["v"] != 1
                or decoded["scope"] != "workspace-catalog"
            ):
                raise ValueError
            after_workspace = decoded["after_workspace"]
            if not isinstance(after_workspace, str):
                raise ValueError
            return WorkspaceCatalogCursor(after_workspace=after_workspace)
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkspaceCatalogCursorError("invalid workspace-catalog page cursor") from exc


def _cursor_mac(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, b"workspace-catalog\0" + payload, hashlib.sha256).digest()[
        :_CURSOR_MAC_BYTES
    ]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not value or any(character not in _BASE64URL_CHARACTERS for character in value):
        raise ValueError("invalid base64url")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded


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
