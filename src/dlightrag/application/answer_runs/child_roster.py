# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded child-roster pages and opaque continuation cursors."""

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

CHILD_ROSTER_PAGE_DEFAULT_LIMIT = 50
CHILD_ROSTER_PAGE_MAX_LIMIT = 100
_CURSOR_MAC_BYTES = 16
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class ChildRosterCursorError(ValueError):
    """An opaque child-roster page cursor is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class ChildRosterCursor:
    """The complete newest-first ordering key for one run's child roster."""

    run_id: UUID
    created_at: datetime.datetime
    child_session_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID):
            raise ValueError("child-roster cursor run id must be a UUID")
        if not isinstance(self.child_session_id, UUID):
            raise ValueError("child-roster cursor child session id must be a UUID")
        if not isinstance(self.created_at, datetime.datetime):
            raise ValueError("child-roster cursor timestamp must be a datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("child-roster cursor timestamp must include a timezone")


@dataclass(frozen=True, slots=True)
class ChildRosterPageRequest:
    """One hard-bounded newest-first child-roster page request."""

    limit: int = CHILD_ROSTER_PAGE_DEFAULT_LIMIT
    cursor: ChildRosterCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("child-roster page limit must be an integer")
        if not 1 <= self.limit <= CHILD_ROSTER_PAGE_MAX_LIMIT:
            raise ValueError(
                f"child-roster page limit must be between 1 and {CHILD_ROSTER_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ChildRosterCursor):
            raise ValueError("child-roster cursor is invalid")


@dataclass(frozen=True, slots=True)
class ChildRosterRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    children: tuple[Mapping[str, Any], ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ChildRosterPage:
    """Application page of child rows plus a typed continuation."""

    children: tuple[Mapping[str, Any], ...]
    next_cursor: ChildRosterCursor | None
    fetched_rows: int


class ChildRosterCursorCodec:
    """Encode roster ordering facts as a signed, opaque, run-bound token."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    def encode(self, cursor: ChildRosterCursor) -> str:
        payload = _canonical_json(
            {
                "child_session_id": str(cursor.child_session_id),
                "created_at": _canonical_timestamp(cursor.created_at),
                "run_id": str(cursor.run_id),
                "scope": "child-roster",
                "v": 1,
            }
        )
        mac = _cursor_mac(self._secret, payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> ChildRosterCursor:
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
                "child_session_id",
                "created_at",
                "run_id",
                "scope",
                "v",
            }:
                raise ValueError
            if _canonical_json(decoded) != payload:
                raise ValueError
            if (
                type(decoded["v"]) is not int
                or decoded["v"] != 1
                or decoded["scope"] != "child-roster"
            ):
                raise ValueError
            run_id_text = decoded["run_id"]
            child_session_text = decoded["child_session_id"]
            timestamp_text = decoded["created_at"]
            if not isinstance(run_id_text, str) or not isinstance(child_session_text, str):
                raise ValueError
            if not isinstance(timestamp_text, str):
                raise ValueError
            run_id = UUID(run_id_text)
            if str(run_id) != run_id_text:
                raise ValueError
            child_session_id = UUID(child_session_text)
            if str(child_session_id) != child_session_text:
                raise ValueError
            created_at = datetime.datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            if _canonical_timestamp(created_at) != timestamp_text:
                raise ValueError
            return ChildRosterCursor(
                run_id=run_id,
                created_at=created_at,
                child_session_id=child_session_id,
            )
        except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ChildRosterCursorError("invalid child-roster page cursor") from exc


def _cursor_mac(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, b"child-roster\0" + payload, hashlib.sha256).digest()[
        :_CURSOR_MAC_BYTES
    ]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("child-roster cursor timestamp must include a timezone")
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
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
]
