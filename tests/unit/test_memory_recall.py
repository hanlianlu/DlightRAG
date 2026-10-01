# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Frozen recall cases: RRF fusion, the packing prior, and the character budget.

Recall over stored records runs against PostgreSQL in tests/integration/test_memory_pg.py.
"""

from datetime import UTC, datetime

from dlightrag_memory.fusion import rrf_fuse
from dlightrag_memory.memory import _packing_prior, _truncate_to_budget
from dlightrag_memory.models import MemoryProvenance, MemoryRecord


def _record(*, memory_id: str, kind: str, body: str) -> MemoryRecord:
    return MemoryRecord(
        owner_id="alpha",
        memory_id=memory_id,
        kind=kind,  # type: ignore[arg-type]
        body=body,
        provenance=MemoryProvenance(
            origin_kind="answer_run",
            origin_id="r",
            run_id="r",
            session_id="s",
        ),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_rrf_fusion_is_rank_based_and_deterministic() -> None:
    scores = rrf_fuse([["a", "b"], ["b", "a"]], k=60)

    assert scores == {"a": 1 / 61 + 1 / 62, "b": 1 / 61 + 1 / 62}
    assert scores["a"] == scores["b"]


def test_rrf_rewards_consensus_rank() -> None:
    scores = rrf_fuse([["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c"]], k=60)

    assert scores["a"] > scores["b"] > scores["c"]


def test_packing_prior_keeps_one_of_each_kind_first() -> None:
    records = [
        _record(memory_id="f1", kind="fact", body="fact one"),
        _record(memory_id="f2", kind="fact", body="fact two"),
        _record(memory_id="p1", kind="preference", body="pref one"),
    ]

    kept = _packing_prior(records)

    assert kept[0].memory_id == "p1"
    assert kept[1].memory_id in {"f1", "f2"}


def test_char_budget_truncates_after_the_header() -> None:
    records = [_record(memory_id=str(index), kind="fact", body="x" * 100) for index in range(3)]

    kept = _truncate_to_budget(records, budget=300)

    assert len(kept) == 1  # header (160) + 100 fits; the second would exceed 300
