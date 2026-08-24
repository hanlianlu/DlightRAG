# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The host-neutral Profile Memory façade."""

from __future__ import annotations

import asyncio
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

_SEARCH_DEADLINE_SECONDS = 2.0
_HEADER_CHARS = 160


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Structured query-aware recall result."""

    records: tuple[MemoryRecord, ...]
    strategy: str
    candidates: tuple[SearchCandidate, ...] = ()
    degraded: tuple[str, ...] = ()
    content_chars: int = 0

    @property
    def skipped(self) -> bool:
        return bool(self.degraded)


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

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        return await self._store.list_active(owner_id=owner_id)

    async def browse(
        self,
        *,
        owner_id: str,
        cursor: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        return await self._store.list_active_page(owner_id=owner_id, after=cursor, limit=limit)

    async def recall(
        self,
        *,
        owner_id: str,
        query: str,
        top_k: int = RECALL_TOP_K,
        char_budget: int = RECALL_CHAR_BUDGET,
    ) -> RecallResult:
        cap = max(1, min(int(top_k), 100))
        budget = max(_HEADER_CHARS, int(char_budget))
        try:
            candidates = await asyncio.wait_for(
                self._store.search_candidates(owner_id=owner_id, query=query, limit=cap),
                timeout=_SEARCH_DEADLINE_SECONDS,
            )
        except TimeoutError:
            recent = list(await self._store.list_active(owner_id=owner_id))[:cap]
            recent = _truncate_to_budget(recent, budget=budget)
            recent.sort(key=lambda record: (recall_recency(record), record.memory_id))
            return RecallResult(
                records=tuple(recent),
                strategy="recent_fallback",
                degraded=("search_timeout",),
                content_chars=sum(len(record.body) for record in recent),
            )

        exact_ids: list[str] = []
        fused_ids: dict[str, float] = {}
        rankings: list[list[str]] = []
        for leg in ("exact", "sparse", "dense"):
            ranking: list[str] = []
            for candidate in candidates:
                if candidate.leg != leg or candidate.record.memory_id in ranking:
                    continue
                ranking.append(candidate.record.memory_id)
            if leg == "exact":
                exact_ids = ranking
            else:
                rankings.append(ranking)
        fused_ids.update(rrf_fuse(rankings))

        by_id = {candidate.record.memory_id: candidate.record for candidate in candidates}
        exact_records = [by_id[memory_id] for memory_id in exact_ids if memory_id in by_id]
        ranked_records = [
            by_id[memory_id]
            for memory_id, _ in sorted(
                (
                    (memory_id, score)
                    for memory_id, score in fused_ids.items()
                    if memory_id not in exact_ids
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if memory_id in by_id
        ]
        ordered = _packing_prior([*exact_records, *ranked_records])[:cap]
        ordered = _truncate_to_budget(ordered, budget=budget)
        exact_set = set(exact_ids)
        exact_block = [record for record in ordered if record.memory_id in exact_set]
        rest = [record for record in ordered if record.memory_id not in exact_set]
        exact_block.sort(key=lambda record: (recall_recency(record), record.memory_id))
        rest.sort(key=lambda record: (recall_recency(record), record.memory_id))
        ordered = [*exact_block, *rest]
        return RecallResult(
            records=tuple(ordered),
            strategy="query_search",
            candidates=tuple(candidates),
            content_chars=sum(len(record.body) for record in ordered),
        )

    async def purge_superseded(self, *, older_than: datetime) -> int:
        return await self._store.purge_superseded(older_than=older_than)


def _packing_prior(records: list[MemoryRecord]) -> list[MemoryRecord]:
    if not records:
        return records
    first_preference = next((record for record in records if record.kind == "preference"), None)
    first_fact = next((record for record in records if record.kind == "fact"), None)
    if first_preference is None or first_fact is None:
        return records
    kept = [first_preference, first_fact]
    kept.extend(
        record
        for record in records
        if record.memory_id not in {first_preference.memory_id, first_fact.memory_id}
    )
    return kept


def _truncate_to_budget(records: list[MemoryRecord], *, budget: int) -> list[MemoryRecord]:
    kept: list[MemoryRecord] = []
    used = _HEADER_CHARS
    for record in records:
        cost = len(record.body)
        if used + cost > budget and kept:
            break
        kept.append(record)
        used += cost
    return kept


__all__ = ["Memory", "RecallResult"]
