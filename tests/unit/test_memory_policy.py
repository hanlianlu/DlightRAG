# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Memory Write checklist and auto-recall caps."""

import pytest

from dlightrag.answer.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag.answer.memory import (
    MEMORY_ACTIVE_LIMIT,
    MEMORY_BODY_LIMIT,
    MEMORY_RECALL_LIMIT,
    MEMORY_WRITES_PER_HOUR,
    MemoryProvenance,
    MemoryRecord,
    MemoryWrite,
    evaluate_memory_write,
    render_auto_recall,
    select_auto_recall,
)


def _write(**overrides: object) -> MemoryWrite:
    payload: dict[str, object] = {
        "owner_id": "owner",
        "auth_mode": "jwt",
        "kind": "preference",
        "body": "Do not use email.",
        "confidence": 0.9,
        "provenance": MemoryProvenance(run_id="run-1"),
    }
    payload.update(overrides)
    return MemoryWrite(**payload)  # type: ignore[arg-type]


def test_jwt_remember_passes() -> None:
    evaluate_memory_write(_write())


def test_none_owner_cannot_write() -> None:
    with pytest.raises(MemoryUnavailableError):
        evaluate_memory_write(_write(auth_mode="none"))


def test_simple_owner_cannot_write() -> None:
    with pytest.raises(MemoryUnavailableError):
        evaluate_memory_write(_write(auth_mode="simple"))


def test_empty_body_and_citation_markers_are_rejected() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="   "))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="See [1] for the filing."))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(body="x" * (MEMORY_BODY_LIMIT + 1)))


def test_quota_and_provenance_are_enforced() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(active_count=MEMORY_ACTIVE_LIMIT))
    evaluate_memory_write(_write(active_count=MEMORY_ACTIVE_LIMIT, supersedes_id="mem-old"))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(writes_last_hour=MEMORY_WRITES_PER_HOUR))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(provenance=MemoryProvenance(run_id="")))


def test_forget_requires_a_target() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_write(_write(action="forget", body="", supersedes_id=None))
    evaluate_memory_write(_write(action="forget", body="", supersedes_id="mem-1"))


def test_auto_recall_keeps_newest_active_within_caps() -> None:
    records = tuple(
        MemoryRecord(
            owner_id="o",
            memory_id=str(index),
            kind="preference" if index % 2 == 0 else "fact",
            body=f"item {index}",
            confidence=1.0,
            provenance=MemoryProvenance(run_id="r"),
            status="superseded" if index == 0 else "active",
        )
        for index in range(20)
    )
    chosen = select_auto_recall(records)
    assert len(chosen) == MEMORY_RECALL_LIMIT
    assert all(record.status == "active" for record in chosen)
    text = render_auto_recall(chosen)
    assert "not evidence" in text
    assert "[1]" not in text
    assert render_auto_recall(()) == ""
