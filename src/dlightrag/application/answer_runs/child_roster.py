# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded child-roster pages and opaque continuation cursors."""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from dlightrag.application.opaque_cursor import OpaqueCursorEnvelope

CHILD_ROSTER_PAGE_DEFAULT_LIMIT = 50
CHILD_ROSTER_PAGE_MAX_LIMIT = 100


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

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="child-roster",
            scope="child-roster",
            fields_by_version={1: {"child_session_id", "created_at", "run_id"}},
            current_version=1,
        )

    def encode(self, cursor: ChildRosterCursor) -> str:
        return self._envelope.encode(
            {
                "child_session_id": str(cursor.child_session_id),
                "created_at": _canonical_timestamp(cursor.created_at),
                "run_id": str(cursor.run_id),
            }
        )

    def decode(self, token: str) -> ChildRosterCursor:
        try:
            decoded = self._envelope.decode(token)
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
        except ValueError as exc:
            raise ChildRosterCursorError("invalid child-roster page cursor") from exc


def _canonical_timestamp(value: datetime.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("child-roster cursor timestamp must include a timezone")
    return value.astimezone(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def child_result_lineage(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project attributable result/evidence handles without private host state."""
    host_state = row.get("host_state")
    payload = host_state.get("terminal_outcome") if isinstance(host_state, Mapping) else None
    if not isinstance(payload, Mapping):
        summary = row.get("summary")
        handles = row.get("result_handles")
        if summary in {None, ""} and not handles:
            return None
        return {
            "status": str(row.get("status") or ""),
            "summary": str(summary or ""),
            "handles": _string_handles(handles),
            "operation_id": (str(row["operation_id"]) if row.get("operation_id") else None),
        }
    return {
        "status": str(payload.get("status") or row.get("status") or ""),
        "summary": str(payload.get("summary") or row.get("summary") or ""),
        "handles": _string_handles(payload.get("handles")),
        "operation_id": (
            str(payload["operation_id"])
            if payload.get("operation_id")
            else (str(row["operation_id"]) if row.get("operation_id") else None)
        ),
    }


def public_child_status(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Web/REST/MCP child projection without private runtime envelopes."""
    lineage = child_result_lineage(row)
    usage = row.get("usage")
    return {
        "child_session_id": str(row.get("child_session_id") or ""),
        "status": str(row.get("status") or ""),
        "objective": row.get("objective"),
        "model_role": row.get("model_role"),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "operation_id": row.get("operation_id"),
        "operation_sequence": row.get("operation_sequence"),
        "operation_status": row.get("operation_status"),
        "cancellation_origin": row.get("cancellation_origin"),
        "summary": (
            row.get("summary")
            if row.get("summary") not in {None, ""}
            else (None if lineage is None else lineage.get("summary") or None)
        ),
        "result_handles": list(lineage["handles"]) if lineage is not None else [],
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
]
