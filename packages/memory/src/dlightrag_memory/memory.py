# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The host-neutral Profile Memory façade."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from dlightrag_memory.fusion import rrf_fuse
from dlightrag_memory.models import (
    MemoryKind,
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
)
from dlightrag_memory.policy import (
    RECALL_CHAR_BUDGET,
    RECALL_TOP_K,
    evaluate_memory_operation,
)
from dlightrag_memory.ports import SearchCandidate
from dlightrag_memory.recall import recall_recency
from dlightrag_memory.store import MemoryStore, OperationGuard

_logger = logging.getLogger(__name__)
_SEARCH_DEADLINE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class RecallResult:
    """The owner's standing preferences and the facts relevant to one query."""

    preferences: tuple[MemoryRecord, ...] = ()
    facts: tuple[MemoryRecord, ...] = ()

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return (*self.preferences, *self.facts)

    @property
    def content_chars(self) -> int:
        return sum(len(record.body) for record in self.records)


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
        change_id: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        return await self.apply(
            MemoryOperation(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                action="undo",
                provenance=provenance,
                target_change_id=change_id,
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

    async def browse(
        self,
        *,
        owner_id: str,
        cursor: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        return await self._store.list_active_page(owner_id=owner_id, after=cursor, limit=limit)

    async def recall(self, *, owner_id: str, query: str) -> RecallResult:
        """Standing preferences, then the facts relevant to ``query``.

        Preferences say how the owner wants every answer, so the newest
        ``RECALL_TOP_K`` stand whatever the query. A fact is recalled only on
        evidence that it bears on the query (``MemoryStore.search_facts``), and
        relevant facts rank by rank-only RRF across the legs. Preferences claim
        the character budget first. Time never scores: it orders each section
        oldest first, so the latest record reads last.
        """
        preferences = await self._store.list_preferences(owner_id=owner_id, limit=RECALL_TOP_K)
        try:
            candidates = await asyncio.wait_for(
                self._store.search_facts(owner_id=owner_id, query=query, limit=RECALL_TOP_K),
                timeout=_SEARCH_DEADLINE_SECONDS,
            )
        except TimeoutError:
            _logger.warning(
                "Profile Memory fact search exceeded %ss; recalling preferences only",
                _SEARCH_DEADLINE_SECONDS,
            )
            candidates = ()
        kept_preferences = _within(preferences, RECALL_CHAR_BUDGET)
        remaining = RECALL_CHAR_BUDGET - sum(len(record.body) for record in kept_preferences)
        kept_facts = _within(_fused(candidates)[:RECALL_TOP_K], remaining)
        return RecallResult(
            preferences=_chronological(kept_preferences),
            facts=_chronological(kept_facts),
        )

    async def purge_superseded(self, *, older_than: datetime) -> int:
        return await self._store.purge_superseded(older_than=older_than)


def _fused(candidates: Sequence[SearchCandidate]) -> list[MemoryRecord]:
    """Relevant facts, best first by rank-only RRF across the legs."""
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
    """Keep records in priority order while their bodies fit ``budget``."""
    kept: list[MemoryRecord] = []
    for record in records:
        if len(record.body) <= budget:
            kept.append(record)
            budget -= len(record.body)
    return kept


def _chronological(records: list[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    return tuple(sorted(records, key=lambda record: (recall_recency(record), record.memory_id)))


__all__ = ["Memory", "RecallResult"]
