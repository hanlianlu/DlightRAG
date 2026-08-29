# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded file-panel pages and opaque continuation cursors."""

import base64
import binascii
import datetime
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

FILE_PANEL_PAGE_DEFAULT_LIMIT = 50
FILE_PANEL_PAGE_MAX_LIMIT = 100
_CURSOR_MAC_BYTES = 16
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class FilePanelCursorError(ValueError):
    """An opaque file-panel page cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class FilePanelCursor:
    """The complete mixed-direction ordering key for one workspace page."""

    workspace: str
    updated_at: datetime.datetime | None
    doc_id: str

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


@dataclass(frozen=True, slots=True)
class FilePanelPageRequest:
    """One hard-bounded recent or older processed-file page request."""

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
    """Bounded persistence result, including the measured physical fetch size."""

    items: tuple[ProcessedFileRow, ...]
    has_more: bool
    fetched_rows: int


class FilePanelCursorCodec:
    """Encode file ordering facts as a signed, opaque, workspace-bound token."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: FilePanelCursor) -> str:
        payload = _canonical_json(
            {
                "doc_id": cursor.doc_id,
                "scope": "file-panel",
                "updated_at": _canonical_timestamp(cursor.updated_at),
                "v": 1,
                "workspace": cursor.workspace,
            }
        )
        mac = _cursor_mac(self._secret, payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> FilePanelCursor:
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
                "doc_id",
                "scope",
                "updated_at",
                "v",
                "workspace",
            }:
                raise ValueError
            if _canonical_json(decoded) != payload:
                raise ValueError
            if (
                type(decoded["v"]) is not int
                or decoded["v"] != 1
                or decoded["scope"] != "file-panel"
            ):
                raise ValueError
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
            )
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise FilePanelCursorError("invalid file-panel page cursor") from exc


def _cursor_mac(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, b"file-panel\0" + payload, hashlib.sha256).digest()[:_CURSOR_MAC_BYTES]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_timestamp(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None or value.utcoffset() is not None:
        raise ValueError("file-panel cursor timestamp must not include a timezone")
    return value.isoformat(timespec="microseconds")


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
    "FILE_PANEL_PAGE_DEFAULT_LIMIT",
    "FILE_PANEL_PAGE_MAX_LIMIT",
    "FilePanelCursor",
    "FilePanelCursorCodec",
    "FilePanelCursorError",
    "FilePanelPageRequest",
    "FilePanelRowPage",
    "ProcessedFileRow",
]
