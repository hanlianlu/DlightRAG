# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded Profile Memory listing pages and opaque continuation cursors."""

import datetime
from dataclasses import dataclass
from uuid import UUID

from dlightrag_memory import MemoryRecord

from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope

MEMORY_LIST_PAGE_DEFAULT_LIMIT = 50
MEMORY_LIST_PAGE_MAX_LIMIT = 100


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

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="memory-list",
            scope="memory-list",
            fields_by_version={1: {"memory_id", "updated_at"}},
            current_version=1,
        )

    def encode(self, cursor: MemoryListCursor) -> str:
        return self._envelope.encode(
            {
                "memory_id": str(cursor.memory_id),
                "updated_at": _canonical_timestamp(cursor.updated_at),
            }
        )

    def decode(self, token: str) -> MemoryListCursor:
        try:
            decoded = self._envelope.decode(token)
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
        except ValueError as exc:
            raise MemoryListCursorError("invalid memory-list page cursor") from exc


def _canonical_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory-list cursor timestamp must include a timezone")
    return value.astimezone(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "MEMORY_LIST_PAGE_DEFAULT_LIMIT",
    "MEMORY_LIST_PAGE_MAX_LIMIT",
    "MemoryListCursor",
    "MemoryListCursorCodec",
    "MemoryListCursorError",
    "MemoryListPage",
    "MemoryListPageRequest",
]
