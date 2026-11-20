# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opaque child-roster cursor and page request contracts."""

import datetime
from typing import Any, cast
from uuid import UUID

import pytest

from dlightrag.application.answer_runs.child_roster import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterCursorError,
    ChildRosterPageRequest,
    child_result_lineage,
    public_child_status,
)
from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex

_SECRET = b"child-roster-test-secret" * 2
_RUN_ID = UUID("0199a0a0-0000-7000-8000-000000000077")
_CHILD_ID = UUID("0199a0a0-0000-7000-8000-000000000088")
_TS = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=datetime.UTC)


def _cursor() -> ChildRosterCursor:
    return ChildRosterCursor(run_id=_RUN_ID, created_at=_TS, child_session_id=_CHILD_ID)


def _payload_fields() -> dict[str, Any]:
    return {
        "child_session_id": str(_CHILD_ID),
        "created_at": "2026-03-04T05:06:07.123456Z",
        "run_id": str(_RUN_ID),
    }


def test_codec_round_trips_one_canonical_cursor() -> None:
    codec = ChildRosterCursorCodec(_SECRET)

    token = codec.encode(_cursor())
    assert codec.decode(token) == _cursor()


def test_codec_rejects_tampered_or_malformed_tokens() -> None:
    codec = ChildRosterCursorCodec(_SECRET)
    token = codec.encode(_cursor())

    with pytest.raises(ChildRosterCursorError):
        codec.decode(token + "x")
    with pytest.raises(ChildRosterCursorError):
        codec.decode(token[:-4] + ("AAAA" if token[-4] != "AAAA" else "BBBB"))
    with pytest.raises(ChildRosterCursorError):
        codec.decode("garbage.without-dot")
    with pytest.raises(ChildRosterCursorError):
        codec.decode("")
    other = ChildRosterCursorCodec(b"different-secret" * 2)
    with pytest.raises(ChildRosterCursorError):
        other.decode(token)


def test_cursor_requires_aware_timestamp_and_uuid_fields() -> None:
    with pytest.raises(ValueError, match="timezone"):
        ChildRosterCursor(
            run_id=_RUN_ID,
            created_at=datetime.datetime(2026, 3, 4, 5, 6, 7),
            child_session_id=_CHILD_ID,
        )
    with pytest.raises(ValueError, match="run id"):
        ChildRosterCursor(
            run_id=cast(Any, "not-a-uuid"),
            created_at=_TS,
            child_session_id=_CHILD_ID,
        )
    with pytest.raises(ValueError, match="child session id"):
        ChildRosterCursor(
            run_id=_RUN_ID,
            created_at=_TS,
            child_session_id=cast(Any, "not-a-uuid"),
        )


def test_page_request_enforces_bounds_and_cursor_type() -> None:
    assert ChildRosterPageRequest().limit == CHILD_ROSTER_PAGE_DEFAULT_LIMIT
    assert ChildRosterPageRequest(limit=1).limit == 1
    assert ChildRosterPageRequest(limit=CHILD_ROSTER_PAGE_MAX_LIMIT).limit == (
        CHILD_ROSTER_PAGE_MAX_LIMIT
    )
    for limit in (0, CHILD_ROSTER_PAGE_MAX_LIMIT + 1, True, "50"):
        with pytest.raises(ValueError, match="limit"):
            ChildRosterPageRequest(limit=cast(Any, limit))
    with pytest.raises(ValueError, match="cursor"):
        ChildRosterPageRequest(cursor=cast(Any, "opaque-token"))


_PROTECTED_PARAGRAPH = (
    "The protected method reconciles the ledger in two passes: first it groups every "
    "receipt by its audit window, then it compares the grouped totals against the "
    "settlement extract before writing a single adjustment."
)


def _protected_index() -> ProtectedTextIndex:
    """A one-document fingerprint index, standing in for the process's Skill protection."""
    return ProtectedTextIndex.build(
        [("charts", "references/method.md", _PROTECTED_PARAGRAPH.encode("utf-8"))]
    )


def test_public_child_status_gates_the_childs_summary() -> None:
    """The projection is pure: a caller that renders it hands over its reader's gate."""
    row = {
        "child_session_id": "child-1",
        "status": "succeeded",
        "summary": _PROTECTED_PARAGRAPH,
        "result_handles": [],
    }

    assert public_child_status(row)["summary"] == _PROTECTED_PARAGRAPH

    masked = public_child_status(row, mask_reader_text=_protected_index().mask)
    assert masked["summary"] == MASK_PLACEHOLDER


def test_child_result_lineage_gates_the_terminal_outcome_summary() -> None:
    """A Child's conclusion is reader text; its handles are facts and stay."""
    row = {
        "child_session_id": "child-1",
        "status": "succeeded",
        "host_state": {
            "terminal_outcome": {
                "status": "succeeded",
                "summary": _PROTECTED_PARAGRAPH,
                "handles": ["ev-1"],
                "operation_id": "op-1",
            }
        },
    }

    lineage = child_result_lineage(row, mask_reader_text=_protected_index().mask)

    assert lineage is not None
    assert lineage["summary"] == MASK_PLACEHOLDER
    assert lineage["handles"] == ["ev-1"]
