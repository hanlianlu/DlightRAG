# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Frozen recall cases: RRF fusion, exact-first, packing prior, budgets."""

from dlightrag_memory.fusion import rrf_fuse
from dlightrag_memory.memory import Memory, _packing_prior, _truncate_to_budget
from dlightrag_memory.models import MemoryProvenance, MemoryRecord
from dlightrag_memory.store import InMemoryMemoryStore


def _record(
    *,
    memory_id: str,
    kind: str,
    body: str,
    updated_minute: int = 0,
) -> MemoryRecord:
    from datetime import UTC, datetime, timedelta

    return MemoryRecord(
        owner_id="alpha",
        memory_id=memory_id,
        kind=kind,  # type: ignore[arg-type]
        body=body,
        confidence=1.0,
        provenance=MemoryProvenance(run_id="r", session_id="s"),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=updated_minute),
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


async def test_recall_pins_exact_matches_first_and_orders_chronologically() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    for record in (
        _record(memory_id="old", kind="fact", body="deploy to staging", updated_minute=1),
        _record(memory_id="new", kind="fact", body="deploy to staging", updated_minute=2),
    ):
        await store.insert(record)

    result = await memory.recall(owner_id="alpha", query="deploy to staging")

    # Two exact matches: chronological ascending (old before new).
    assert [record.memory_id for record in result.records] == ["old", "new"]


async def test_recall_returns_empty_on_no_match() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    await store.insert(_record(memory_id="1", kind="fact", body="trains are fast"))

    result = await memory.recall(owner_id="alpha", query="zzz-nothing")

    assert result.records == ()
    assert result.candidates == ()
    assert result.strategy == "query_search"


async def test_recall_timeout_falls_back_to_recent_active_records(monkeypatch) -> None:
    import asyncio

    store = InMemoryMemoryStore()
    memory = Memory(store)
    await store.insert(_record(memory_id="recent", kind="fact", body="recent profile"))

    async def timeout(**_kwargs):
        await asyncio.sleep(0.02)
        return ()

    monkeypatch.setattr(store, "search_candidates", timeout)
    monkeypatch.setattr("dlightrag_memory.memory._SEARCH_DEADLINE_SECONDS", 0.001)

    result = await memory.recall(owner_id="alpha", query="anything")

    assert [record.memory_id for record in result.records] == ["recent"]
    assert result.strategy == "recent_fallback"
    assert result.degraded == ("search_timeout",)


async def test_recall_budget_caps_top_k() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    for index in range(15):
        await store.insert(
            _record(memory_id=str(index), kind="fact", body=f"project alpha item {index}")
        )

    result = await memory.recall(owner_id="alpha", query="project alpha item", top_k=5)

    assert len(result.records) <= 5
