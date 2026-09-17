# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The handle list one compaction summary carries forward."""

from dlightrag.engine.answer.continuation_handles import (
    MAX_SPILL_HANDLES,
    compose_durable_handles,
    spill_handle,
)
from dlightrag.engine.runtime.workspace import CommittedSpillRecord


def _spill(
    resource_id: str, *, size_bytes: int = 4_096, intent_id: str = "i"
) -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest="a" * 64,
        size_bytes=size_bytes,
        session_id="session",
        intent_id=intent_id,
    )


def test_spill_handle_spells_out_the_call_the_next_turn_must_make() -> None:
    """The receipt that taught the call is inside the covered prefix by then.

    A cursor is deliberately absent rather than rendered as ``0``: the Tool takes
    a string cursor and an omitted one already starts at the first line, so a
    handle that showed an integer would invite a rejected call.
    """
    handle = spill_handle(_spill("spill_web_search_ab12", size_bytes=924_133))

    assert handle == (
        "[spill] spill_web_search_ab12 (924133 bytes) — "
        're-read with read(resource_id="spill_web_search_ab12")'
    )
    assert "cursor=0" not in handle


def test_newest_spills_precede_evidence_and_evidence_survives_them() -> None:
    handles = compose_durable_handles(
        spills=[_spill("spill_new"), _spill("spill_old")],
        evidence_handles=["[1] report.pdf"],
    )

    assert handles == [
        spill_handle(_spill("spill_new")),
        spill_handle(_spill("spill_old")),
        "[1] report.pdf",
    ]


def test_spills_claim_a_bounded_share_and_never_evict_evidence_entirely() -> None:
    """A share, not priority: a retrieval-heavy Run keeps its citation handles."""
    spills = [_spill(f"spill_{index:03d}") for index in range(MAX_SPILL_HANDLES + 5)]

    handles = compose_durable_handles(spills=spills, evidence_handles=["[1] report.pdf"])

    assert len(handles) == MAX_SPILL_HANDLES + 1
    assert handles[:MAX_SPILL_HANDLES] == [
        spill_handle(spill) for spill in spills[:MAX_SPILL_HANDLES]
    ]
    assert handles[-1] == "[1] report.pdf"


def test_either_class_alone_still_composes() -> None:
    assert compose_durable_handles(spills=[], evidence_handles=[]) == []
    assert compose_durable_handles(spills=[], evidence_handles=["[1] memo"]) == ["[1] memo"]
    assert compose_durable_handles(spills=[_spill("spill_a")], evidence_handles=[]) == [
        spill_handle(_spill("spill_a"))
    ]
