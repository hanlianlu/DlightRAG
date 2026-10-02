# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opaque child-roster cursor and page request contracts."""

import datetime
from typing import Any, cast
from uuid import UUID

import pytest

from dlightrag.application.answer_runs import (
    CHILD_ROSTER_PAGE_DEFAULT_LIMIT,
    CHILD_ROSTER_PAGE_MAX_LIMIT,
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterCursorError,
    ChildRosterPageRequest,
)

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


def test_codec_rejects_noncanonical_uuids_and_timestamps() -> None:
    """Defense in depth: a sealed token is trusted for its shape, never its values."""
    codec = ChildRosterCursorCodec(_SECRET)
    seal = codec._envelope.encode

    upper_run = dict(_payload_fields(), run_id=str(_RUN_ID).upper())
    with pytest.raises(ChildRosterCursorError):
        codec.decode(seal(upper_run))

    offset_timestamp = dict(_payload_fields(), created_at="2026-03-04T05:06:07.123456+00:00")
    with pytest.raises(ChildRosterCursorError):
        codec.decode(seal(offset_timestamp))

    naive_timestamp = dict(_payload_fields(), created_at="2026-03-04T05:06:07.123456")
    with pytest.raises(ChildRosterCursorError):
        codec.decode(seal(naive_timestamp))


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
