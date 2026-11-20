# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded child-roster pages and opaque continuation cursors."""

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from dlightrag.application.errors import ApplicationInputError
from dlightrag.application.opaque_cursor import (
    OpaqueCursorEnvelope,
    format_utc_timestamp,
    parse_canonical_uuid,
    parse_utc_timestamp,
)
from dlightrag.engine.answer.feedback import Judgement

CHILD_ROSTER_PAGE_DEFAULT_LIMIT = 50
CHILD_ROSTER_PAGE_MAX_LIMIT = 100


class ChildRosterCursorError(ApplicationInputError):
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
    """Encode roster ordering facts as a sealed, opaque, run-bound token."""

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="child-roster",
            scope="child-roster",
            fields={"child_session_id", "created_at", "run_id"},
        )

    def encode(self, cursor: ChildRosterCursor) -> str:
        return self._envelope.encode(
            {
                "child_session_id": str(cursor.child_session_id),
                "created_at": format_utc_timestamp(cursor.created_at),
                "run_id": str(cursor.run_id),
            }
        )

    def decode(self, token: str) -> ChildRosterCursor:
        try:
            decoded = self._envelope.decode(token)
            return ChildRosterCursor(
                run_id=parse_canonical_uuid(decoded["run_id"]),
                created_at=parse_utc_timestamp(decoded["created_at"]),
                child_session_id=parse_canonical_uuid(decoded["child_session_id"]),
            )
        except ValueError as exc:
            raise ChildRosterCursorError("invalid child-roster page cursor") from exc


def _gated(text: str, mask_reader_text: Callable[[str], str] | None) -> str:
    """One reader-facing string, replaced where it reproduces a protected document's text."""
    return text if mask_reader_text is None else mask_reader_text(text)


def child_result_lineage(
    row: Mapping[str, Any], *, mask_reader_text: Callable[[str], str] | None = None
) -> dict[str, Any] | None:
    """Project attributable result/evidence handles without private host state.

    ``summary`` is the Child's own conclusion, which can quote the documents it read; it is
    the reader's text, so a surface that renders this passes its content gate for it.
    """
    host_state = row.get("host_state")
    payload = host_state.get("terminal_outcome") if isinstance(host_state, Mapping) else None
    if not isinstance(payload, Mapping):
        summary = row.get("summary")
        handles = row.get("result_handles")
        if summary in {None, ""} and not handles:
            return None
        return {
            "status": str(row.get("status") or ""),
            "summary": _gated(str(summary or ""), mask_reader_text),
            "handles": _string_handles(handles),
            "operation_id": (str(row["operation_id"]) if row.get("operation_id") else None),
        }
    return {
        "status": str(payload.get("status") or row.get("status") or ""),
        "summary": _gated(
            str(payload.get("summary") or row.get("summary") or ""), mask_reader_text
        ),
        "handles": _string_handles(payload.get("handles")),
        "operation_id": (
            str(payload["operation_id"])
            if payload.get("operation_id")
            else (str(row["operation_id"]) if row.get("operation_id") else None)
        ),
    }


def stamp(value: Any) -> str | None:
    """Format one stored timestamp as the ISO-8601 UTC text every public record carries."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat()).replace("+00:00", "Z")
    text = str(value).strip()
    return text or None


def public_child_status(
    row: Mapping[str, Any], *, mask_reader_text: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """Return the Web/REST child projection without private runtime envelopes.

    ``started_at`` is when the Child's current Operation began and ``finished_at``
    when it settled, so a client can show how long it has run or took. A Child
    whose row carries no Operation falls back to its own creation and update time.
    ``result_sequence`` names the message the Child answered with, in its own Session,
    and ``feedback`` is the owner's judgement of that message: each is null until
    there is one. ``summary`` is the Child's conclusion, so it passes the reader's
    content gate like any other text a reader is handed.
    """
    lineage = child_result_lineage(row, mask_reader_text=mask_reader_text)
    usage = row.get("usage")
    status = str(row.get("status") or "")
    has_operation = row.get("operation_created_at") is not None
    judgement: Judgement | None = row.get("feedback")
    summary = row.get("summary")
    if summary in {None, ""}:
        summary = None if lineage is None else lineage.get("summary") or None
    elif mask_reader_text is not None:
        summary = mask_reader_text(str(summary))
    return {
        "child_session_id": str(row.get("child_session_id") or ""),
        "status": status,
        "objective": row.get("objective"),
        "model_role": row.get("model_role"),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "operation_id": row.get("operation_id"),
        "operation_sequence": row.get("operation_sequence"),
        "operation_status": row.get("operation_status"),
        "cancellation_origin": row.get("cancellation_origin"),
        "summary": summary,
        "result_handles": list(lineage["handles"]) if lineage is not None else [],
        "started_at": stamp(
            row.get("operation_created_at") if has_operation else row.get("created_at")
        ),
        "finished_at": (
            None
            if status == "running"
            else stamp(row.get("operation_updated_at") if has_operation else row.get("updated_at"))
        ),
        "pending_questions": int(row.get("pending_questions") or 0),
        "result_sequence": row.get("result_sequence"),
        "feedback": judgement.payload() if judgement is not None else None,
    }


def _string_handles(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


__all__ = [
    "CHILD_ROSTER_PAGE_DEFAULT_LIMIT",
    "CHILD_ROSTER_PAGE_MAX_LIMIT",
    "ChildRosterCursor",
    "ChildRosterCursorCodec",
    "ChildRosterCursorError",
    "ChildRosterPage",
    "ChildRosterPageRequest",
    "ChildRosterRowPage",
    "child_result_lineage",
    "public_child_status",
    "stamp",
]
