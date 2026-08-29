# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded Profile Memory listing pages and opaque continuation cursors."""

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
from uuid import UUID

from dlightrag_memory import MemoryRecord

MEMORY_LIST_PAGE_DEFAULT_LIMIT = 50
MEMORY_LIST_PAGE_MAX_LIMIT = 100
_CURSOR_MAC_BYTES = 16
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class MemoryListCursorError(ValueError):
    """An opaque memory-list page cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class MemoryListCursor:
    """The complete newest-first ordering key for one owner's memory page.

    The cursor deliberately carries no owner: ownership is a mandatory query
    predicate derived from authentication on every page, so a cursor can only
    ever page through rows the requester already owns.
    """

    updated_at: datetime.datetime
    memory_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.updated_at, datetime.datetime):
            raise ValueError("memory-list cursor timestamp must be a datetime")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("memory-list cursor timestamp must include a timezone")
        if not isinstance(self.memory_id, UUID):
            raise ValueError("memory-list cursor memory id must be a UUID")


@dataclass(frozen=True, slots=True)
class MemoryListPageRequest:
    """One hard-bounded newest-first memory-list page request."""

    limit: int = MEMORY_LIST_PAGE_DEFAULT_LIMIT
    cursor: MemoryListCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("memory-list page limit must be an integer")
        if not 1 <= self.limit <= MEMORY_LIST_PAGE_MAX_LIMIT:
            raise ValueError(
                f"memory-list page limit must be between 1 and {MEMORY_LIST_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, MemoryListCursor):
            raise ValueError("memory-list cursor is invalid")


@dataclass(frozen=True, slots=True)
class MemoryListPage:
    """Application page of active memory records plus a typed continuation."""

    records: tuple[MemoryRecord, ...]
    next_cursor: MemoryListCursor | None


class MemoryListCursorCodec:
    """Encode memory ordering facts as a signed, opaque, owner-free token."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: MemoryListCursor) -> str:
        payload = _canonical_json(
            {
                "memory_id": str(cursor.memory_id),
                "scope": "memory-list",
                "updated_at": _canonical_timestamp(cursor.updated_at),
                "v": 1,
            }
        )
        mac = _cursor_mac(self._secret, payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> MemoryListCursor:
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
                "memory_id",
                "scope",
                "updated_at",
                "v",
            }:
                raise ValueError
            if _canonical_json(decoded) != payload:
                raise ValueError
            if (
                type(decoded["v"]) is not int
                or decoded["v"] != 1
                or decoded["scope"] != "memory-list"
            ):
                raise ValueError
            memory_id_text = decoded["memory_id"]
            timestamp_text = decoded["updated_at"]
            if not isinstance(memory_id_text, str) or not isinstance(timestamp_text, str):
                raise ValueError
            memory_id = UUID(memory_id_text)
            if str(memory_id) != memory_id_text:
                raise ValueError
            updated_at = datetime.datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            if _canonical_timestamp(updated_at) != timestamp_text:
                raise ValueError
            return MemoryListCursor(updated_at=updated_at, memory_id=memory_id)
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise MemoryListCursorError("invalid memory-list page cursor") from exc


def _cursor_mac(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, b"memory-list\0" + payload, hashlib.sha256).digest()[:_CURSOR_MAC_BYTES]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory-list cursor timestamp must include a timezone")
    return value.astimezone(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
    "MEMORY_LIST_PAGE_DEFAULT_LIMIT",
    "MEMORY_LIST_PAGE_MAX_LIMIT",
    "MemoryListCursor",
    "MemoryListCursorCodec",
    "MemoryListCursorError",
    "MemoryListPage",
    "MemoryListPageRequest",
]
