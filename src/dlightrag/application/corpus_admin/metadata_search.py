# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded metadata-search pages and opaque continuation cursors."""

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

METADATA_SEARCH_PAGE_DEFAULT_LIMIT = 50
METADATA_SEARCH_PAGE_MAX_LIMIT = 100
MetadataSearchFilenameMode = Literal["exact", "contains"]
_CURSOR_MAC_BYTES = 16
_FILENAME_MODES = frozenset({"exact", "contains"})
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


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

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: MetadataSearchCursor) -> str:
        payload = _canonical_json(
            {
                "after_doc_id": cursor.after_doc_id,
                "mode": cursor.mode,
                "scope": "metadata-match",
                "v": 1,
                "workspace": cursor.workspace,
            }
        )
        mac = _cursor_mac(self._secret, payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> MetadataSearchCursor:
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
                "after_doc_id",
                "mode",
                "scope",
                "v",
                "workspace",
            }:
                raise ValueError
            if _canonical_json(decoded) != payload:
                raise ValueError
            if (
                type(decoded["v"]) is not int
                or decoded["v"] != 1
                or decoded["scope"] != "metadata-match"
            ):
                raise ValueError
            after_doc_id = decoded["after_doc_id"]
            workspace = decoded["workspace"]
            mode = decoded["mode"]
            if not isinstance(after_doc_id, str) or not isinstance(workspace, str):
                raise ValueError
            if mode not in _FILENAME_MODES:
                raise ValueError
            return MetadataSearchCursor(
                workspace=workspace,
                after_doc_id=after_doc_id,
                mode=mode,
            )
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise MetadataSearchCursorError("invalid metadata-search page cursor") from exc


def _cursor_mac(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, b"metadata-match\0" + payload, hashlib.sha256).digest()[
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
