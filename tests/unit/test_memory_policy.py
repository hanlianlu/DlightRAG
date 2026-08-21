# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Memory Write checklist and standing-block bounds."""

import pytest

from dlightrag.answer.errors import MemoryWriteRejectedError
from dlightrag.answer.memory import (
    MEMORY_BODY_LIMIT,
    RECALL_CHAR_BUDGET,
    MemoryProvenance,
    MemoryWrite,
    evaluate_memory_write,
    render_auto_recall,
    reserved_auto_recall_text,
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


def test_render_auto_recall_keeps_the_framing() -> None:
    from dlightrag.answer.memory import MemoryRecord

    records = (
        MemoryRecord(
            owner_id="o",
            memory_id="1",
            kind="preference",
            body="No email.",
            confidence=1.0,
            provenance=MemoryProvenance(run_id="r"),
        ),
    )
    text = render_auto_recall(records)
    assert "not citable" in text
    assert "the current request takes priority" in text
    assert "[1]" not in text
    assert render_auto_recall(()) == ""


def test_acceptance_reserves_full_recall_only_for_jwt() -> None:
    reserved = reserved_auto_recall_text()
    # The façade caps packed bodies at RECALL_CHAR_BUDGET (4000) -> at most
    # eight 500-char records; the reservation renders exactly that worst case,
    # including header and per-record prefixes.
    assert reserved.count("- (") == 8
    assert len(reserved) >= RECALL_CHAR_BUDGET
    assert standing_memory_for_acceptance("jwt") == reserved
    assert standing_memory_for_acceptance("none") == ""
    assert standing_memory_for_acceptance("simple") == ""
