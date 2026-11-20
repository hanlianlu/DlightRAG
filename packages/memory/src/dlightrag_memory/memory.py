# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The host-neutral Profile Memory façade."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from dlightrag_memory.fusion import rrf_fuse
from dlightrag_memory.models import (
    STANDING_KINDS,
    MemoryKind,
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryOriginKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRunChange,
)
from dlightrag_memory.policy import (
    RECALL_CHAR_BUDGET,
    RECALL_LINE_OVERHEAD,
    RECALL_QUOTE_OVERHEAD,
    RECALL_TIME_FORMAT,
    RECALL_TOP_K,
    RELEVANT_CHAR_BUDGET,
    STANDING_FETCH_LIMIT,
    evaluate_memory_operation,
)
from dlightrag_memory.ports import SearchCandidate
from dlightrag_memory.recall import recall_recency
from dlightrag_memory.store import MemoryStore, OperationGuard

# A lookup pages past what recall already shows for at most this many pages.
_LOOKUP_RECENT_PAGES = 5


@dataclass(frozen=True, slots=True)
class RecallResult:
    """What one recall shows: records relevant to the query, then standing ones.

    ``relevant`` holds records of any kind the query's search found; it is
    filled first. ``standing`` holds the newest preferences and experiences
    that fit the rest of the budget and are not already relevant, and
    ``standing_omitted`` counts the standing records shown nowhere.
    ``recent`` is filled only by ``Memory.lookup``: the newest other records,
    listed so an agent can correct one recall did not show.
    """

    relevant: tuple[MemoryRecord, ...] = ()
    standing: tuple[MemoryRecord, ...] = ()
    recent: tuple[MemoryRecord, ...] = ()
    standing_omitted: int = 0

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        """What recall injects: the standing records, then the relevant ones."""
        return (*self.standing, *self.relevant)

    @property
    def content_chars(self) -> int:
        """The characters of memory a recall carries: every body, and every quote behind one."""
        return sum(len(record.body) + len(record.quote or "") for record in self.records)


def recall_time(record: MemoryRecord) -> str | None:
    """When the record was last written, to the hour in UTC (``YYYY-MM-DD HH:00``).

    ``updated_at``, else ``created_at``; an aware time is converted to UTC and a naive one
    is taken to be UTC already. None when the record has neither.
    """
    moment = record.updated_at or record.created_at
    if moment is None:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.strftime(RECALL_TIME_FORMAT)


def recall_label(record: MemoryRecord) -> str:
    """``(kind, time)``: what every renderer tells a model about a record besides its body.

    The time is ``recall_time``; a record with none carries its kind alone.
    """
    written = recall_time(record)
    if written is None:
        return f"({record.kind})"
    return f"({record.kind}, {written})"


def recall_line(record: MemoryRecord, *, with_id: bool = False) -> str:
    """One record as every renderer lists it: ``- [id ](kind, time) body``.

    Without an id, the framing around the body fits inside ``RECALL_LINE_OVERHEAD``
    with its line break, which is what recall's budget charges for each line.
    """
    memory_id = f"{record.memory_id} " if with_id else ""
    return f"- {memory_id}{recall_label(record)} {record.body}"


def recall_quote_line(record: MemoryRecord) -> str | None:
    """The line under a record that has the owner's words behind it, or None when it has none.

    ``  owner said: "…"``, the words as stored and in the language the owner wrote them. Its
    framing fits inside ``RECALL_QUOTE_OVERHEAD`` with its line break, which is what recall's
    budget charges for it.
    """
    if record.quote is None:
        return None
    return f'  owner said: "{record.quote}"'


def recall_record_cost(record: MemoryRecord) -> int:
    """What one record spends of a recall's budget: the lines it renders as, framing included."""
    cost = len(record.body) + RECALL_LINE_OVERHEAD
    if record.quote is not None:
        cost += len(record.quote) + RECALL_QUOTE_OVERHEAD
    return cost


def recall_cost(records: Sequence[MemoryRecord]) -> int:
    """The budget a set of records spends: each rendered line's text plus its framing."""
    return sum(recall_record_cost(record) for record in records)


class Memory:
    """Cross-conversation owner Profile Memory behind one deep mutation seam."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def apply(
        self,
        operation: MemoryOperation,
        *,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        """Validate and atomically settle one idempotent operation."""
        evaluate_memory_operation(operation)
        return await self._store.apply_operation(operation, guard=guard)

    async def remember(
        self,
        *,
        owner_id: str,
        kind: MemoryKind,
        body: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        supersedes_id: str | None = None,
        mutation_scope: str | None = None,
        mutation_limit: int | None = None,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        return await self.apply(
            MemoryOperation(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                action="remember",
                provenance=provenance,
                kind=kind,
                body=body,
                supersedes_id=supersedes_id,
                mutation_scope=mutation_scope,
                mutation_limit=mutation_limit,
            ),
            guard=guard,
        )

    async def forget(
        self,
        *,
        owner_id: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        memory_id: str | None = None,
        body: str | None = None,
        mutation_scope: str | None = None,
        mutation_limit: int | None = None,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        return await self.apply(
            MemoryOperation(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                action="forget",
                provenance=provenance,
                memory_id=memory_id,
                body=body or "",
                mutation_scope=mutation_scope,
                mutation_limit=mutation_limit,
            ),
            guard=guard,
        )

    async def undo(
        self,
        *,
        owner_id: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        change_id: str | None = None,
        memory_id: str | None = None,
        body: str | None = None,
        mutation_scope: str | None = None,
        mutation_limit: int | None = None,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        """Compensate one change, named directly or through the active record it produced."""
        return await self.apply(
            MemoryOperation(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                action="undo",
                provenance=provenance,
                target_change_id=change_id,
                memory_id=memory_id,
                body=body or "",
                mutation_scope=mutation_scope,
                mutation_limit=mutation_limit,
            ),
            guard=guard,
        )

    async def fuse(
        self,
        *,
        owner_id: str,
        memory_ids: Sequence[str],
        body: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        kind: MemoryKind | None = None,
        mutation_scope: str | None = None,
        mutation_limit: int | None = None,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        """Replace several active records with one supplied sentence, atomically."""
        return await self.apply(
            MemoryOperation(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                action="fuse",
                provenance=provenance,
                kind=kind,
                body=body,
                source_ids=tuple(memory_ids),
                mutation_scope=mutation_scope,
                mutation_limit=mutation_limit,
            ),
            guard=guard,
        )

    async def clear(
        self,
        *,
        owner_id: str,
        guard: OperationGuard | None = None,
    ) -> int:
        """Physically erase one owner's complete Profile Memory schema state."""
        return await self._store.clear_owner(owner_id=owner_id, guard=guard)

    async def count_active(self, *, owner_id: str) -> int:
        return await self._store.count_active(owner_id=owner_id)

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        """One record of any status, or None when the owner has none with that id."""
        return await self._store.get(owner_id=owner_id, memory_id=memory_id)

    async def browse(
        self,
        *,
        owner_id: str,
        cursor: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        return await self._store.list_active_page(owner_id=owner_id, after=cursor, limit=limit)

    async def search(
        self, *, owner_id: str, query: str, limit: int = RECALL_TOP_K
    ) -> tuple[MemoryRecord, ...]:
        """Active records of every kind relevant to ``query``, best first."""
        candidates = await self._store.search(owner_id=owner_id, query=query, limit=limit)
        return tuple(_fused(candidates)[:limit])

    async def session_records(
        self, *, owner_id: str, session_id: str, origin_kind: MemoryOriginKind
    ) -> tuple[MemoryRecord, ...]:
        """The active records one origin wrote for one Session, newest first."""
        return await self._store.list_session_records(
            owner_id=owner_id, session_id=session_id, origin_kind=origin_kind
        )

    async def run_changes(self, *, owner_id: str, run_id: str) -> tuple[MemoryRunChange, ...]:
        """The records one Run's writers left in the profile that are still active.

        Both writers count: the Agent's own tool calls in the Run, and the Session Memory
        Pass that the Run scheduled. A record since forgotten, corrected or merged away is
        not one of them, and neither is one an undo restored.
        """
        return await self._store.list_run_changes(owner_id=owner_id, run_id=run_id)

    async def recall(self, *, owner_id: str, query: str) -> RecallResult:
        """Records relevant to ``query``, then the standing ones the rest of the budget holds.

        The relevant section is filled first, from a search over every kind,
        under ``RECALL_TOP_K`` and ``RELEVANT_CHAR_BUDGET``: it is the one way
        back for a standing record newer ones crowded out, and the only way a
        fact reaches a request. The standing section of preferences and
        experiences, newest first, takes whatever budget the relevant section
        left and skips any record it already shows. Time never scores
        relevance: it orders each section oldest first, so the latest reads last.
        """
        candidates, (standing_pool, standing_total) = await asyncio.gather(
            self._store.search(owner_id=owner_id, query=query, limit=RECALL_TOP_K),
            self._store.list_standing(owner_id=owner_id, limit=STANDING_FETCH_LIMIT),
        )
        relevant = _within(_fused(candidates)[:RECALL_TOP_K], RELEVANT_CHAR_BUDGET)
        shown = {record.memory_id for record in relevant}
        standing = _within(
            [record for record in standing_pool if record.memory_id not in shown],
            RECALL_CHAR_BUDGET - recall_cost(relevant),
        )
        standing_relevant = sum(1 for record in relevant if record.kind in STANDING_KINDS)
        return RecallResult(
            relevant=_chronological(relevant),
            standing=_chronological(standing),
            standing_omitted=max(0, standing_total - len(standing) - standing_relevant),
        )

    async def lookup(self, *, owner_id: str, query: str) -> RecallResult:
        """Recall for an agent that may correct what it finds.

        Recall needs evidence before it returns a record as relevant, so it can
        miss the very record a correction replaces ("I moved to Munich" shares
        nothing with "Lives in Berlin."). A lookup also lists the newest other
        records, ids included.
        """
        recalled = await self.recall(owner_id=owner_id, query=query)
        shown = {record.memory_id for record in recalled.records}
        others: list[MemoryRecord] = []
        cursor: tuple[datetime, str] | None = None
        # Page past everything recall already shows, so newer shown records
        # cannot crowd the record a correction replaces out of the list.
        for _ in range(_LOOKUP_RECENT_PAGES):
            page, cursor = await self.browse(owner_id=owner_id, cursor=cursor, limit=100)
            others.extend(record for record in page if record.memory_id not in shown)
            if len(others) >= RECALL_TOP_K or cursor is None:
                break
        return replace(recalled, recent=_chronological(others[:RECALL_TOP_K]))

    async def purge_superseded(self, *, older_than: datetime) -> int:
        return await self._store.purge_superseded(older_than=older_than)


def _fused(candidates: Sequence[SearchCandidate]) -> list[MemoryRecord]:
    """Relevant records, best first by rank-only RRF across the legs."""
    rankings: dict[str, list[str]] = {}
    records: dict[str, MemoryRecord] = {}
    for candidate in candidates:
        ranking = rankings.setdefault(candidate.leg, [])
        if candidate.record.memory_id not in ranking:
            ranking.append(candidate.record.memory_id)
        records[candidate.record.memory_id] = candidate.record
    scores = rrf_fuse(list(rankings.values()))
    return [records[memory_id] for memory_id in sorted(scores, key=lambda m: (-scores[m], m))]


def _within(records: Sequence[MemoryRecord], budget: int) -> list[MemoryRecord]:
    """First fit: keep each record, in priority order, whose rendered lines still fit."""
    kept: list[MemoryRecord] = []
    for record in records:
        cost = recall_record_cost(record)
        if cost <= budget:
            kept.append(record)
            budget -= cost
    return kept


def _chronological(records: list[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    return tuple(sorted(records, key=lambda record: (recall_recency(record), record.memory_id)))


__all__ = [
    "Memory",
    "RecallResult",
    "recall_cost",
    "recall_label",
    "recall_line",
    "recall_quote_line",
    "recall_time",
]
