# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Memory Write checklist and auto-recall caps."""

import pytest

from dlightrag.answer.errors import MemoryWriteRejectedError
from dlightrag.answer.memory import (
    MEMORY_BODY_LIMIT,
    MEMORY_RECALL_LIMIT,
    MemoryProvenance,
    MemoryRecord,
    MemoryWrite,
    evaluate_memory_write,
    render_auto_recall,
    reserved_auto_recall_text,
    select_auto_recall,
    standing_memory_for_acceptance,
)


def _write(**overrides: object) -> MemoryWrite:
    payload: dict[str, object] = {
        "owner_id": "owner",
        "kind": "preference",
        "body": "Do not use email.",
        "confidence": 0.9,
        "provenance": MemoryProvenance(run_id="run-1", session_id="sess-1"),
    }
    payload.update(overrides)
    return MemoryWrite(**payload)  # type: ignore[arg-type]


def test_remember_passes() -> None:
    evaluate_memory_write(_write())


def test_owner_eligibility_is_root_policy() -> None:
    """The package never judges auth_mode; the root gate does."""
    from dlightrag.answer.memory import memory_owner_allowed

    assert memory_owner_allowed("jwt")
    assert not memory_owner_allowed("none")
    assert not memory_owner_allowed("simple")


def test_empty_body_and_citation_markers_are_rejected() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="   "))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="See [1] for the filing."))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="x" * (MEMORY_BODY_LIMIT + 1)))


def test_provenance_is_enforced() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(provenance=MemoryProvenance(run_id="")))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(provenance=MemoryProvenance(run_id="run-1")))


def test_forget_requires_a_target() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(action="forget", body="", supersedes_id=None))
    evaluate_memory_write(_write(action="forget", body="", supersedes_id="mem-1"))


def test_auto_recall_keeps_newest_active_within_caps() -> None:
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 1, 1, tzinfo=UTC)
    records = tuple(
        MemoryRecord(
            owner_id="o",
            memory_id=str(index),
            kind="preference" if index % 2 == 0 else "fact",
            body=f"item {index}",
            confidence=1.0,
            provenance=MemoryProvenance(run_id="r"),
            status="superseded" if index == 0 else "active",
            updated_at=start + timedelta(minutes=index),
        )
        for index in range(20)
    )
    chosen = select_auto_recall(tuple(reversed(records)))
    assert len(chosen) == MEMORY_RECALL_LIMIT
    assert all(record.status == "active" for record in chosen)
    assert chosen[0].memory_id == "19"
    text = render_auto_recall(chosen)
    assert "not evidence" in text
    assert "[1]" not in text
    assert render_auto_recall(()) == ""


def test_acceptance_reserves_full_recall_only_for_jwt() -> None:
    reserved = reserved_auto_recall_text()
    assert reserved.count("- (") == MEMORY_RECALL_LIMIT
    assert MEMORY_BODY_LIMIT * MEMORY_RECALL_LIMIT <= len(reserved)
    assert standing_memory_for_acceptance("jwt") == reserved
    assert standing_memory_for_acceptance("none") == ""
    assert standing_memory_for_acceptance("simple") == ""
