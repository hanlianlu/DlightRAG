# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deep model/owner-facing Memory façade.

One interface for every cross-conversation Profile operation. The host owns
identity, eligibility, and context placement; this module owns the checklist,
atomic write commit, structured query recall, and lifecycle policy. Every
method returns structured records — never prompt fragments.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from dlightrag_memory.fusion import rrf_fuse
from dlightrag_memory.models import MemoryProvenance, MemoryRecord, MemoryWrite
from dlightrag_memory.policy import RECALL_CHAR_BUDGET, RECALL_TOP_K
from dlightrag_memory.ports import SearchCandidate
from dlightrag_memory.recall import recall_recency
from dlightrag_memory.store import MemoryStore, commit_memory_write

_MANAGEMENT_PROVENANCE = MemoryProvenance(run_id="management", session_id="management")
_SEARCH_DEADLINE_SECONDS = 2.0
_HEADER_CHARS = 160


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Structured recall outcome: selected records plus provenance of choice.

    ``records`` is the final chronologically-ordered selection; ``candidates``
    carries the fused candidates; ``degraded`` reports skipped capability
    (search timeout or a missing dense leg).
    """

    records: tuple[MemoryRecord, ...]
    strategy: str
    candidates: tuple[SearchCandidate, ...] = ()
    degraded: tuple[str, ...] = ()
    content_chars: int = 0

    @property
    def skipped(self) -> bool:
        """True when recall degraded below a full result set."""
        return bool(self.degraded)


class Memory:
    """Cross-conversation Owner Profile Memory bound to one storage adapter."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def remember(
        self,
        *,
        owner_id: str,
        kind: str,
        body: str,
        confidence: float,
        provenance: MemoryProvenance,
        supersedes_id: str | None = None,
    ) -> MemoryRecord | None:
        """Store one preference or fact, optionally superseding an older record."""
        return await commit_memory_write(
            self._store,
            MemoryWrite(
                owner_id=owner_id,
                kind=kind,  # type: ignore[arg-type]
                body=body,
                confidence=confidence,
                provenance=provenance,
                action="remember",
                supersedes_id=supersedes_id,
            ),
        )

    async def forget(
        self,
        *,
        owner_id: str,
        memory_id: str | None = None,
        body: str | None = None,
        all: bool = False,
        provenance: MemoryProvenance | None = None,
    ) -> None:
        """Hard-delete one record by id, every record with an exact body, or all.

        The three selectors are mutually exclusive: one of ``memory_id``,
        ``body``, or ``all`` must be given.
        """
        selectors = sum(1 for value in (memory_id, body, all) if value)
        if selectors != 1:
            raise ValueError("forget needs exactly one of memory_id, body, or all")
        if all:
            await self._store.forget_all(owner_id=owner_id)
            return
        await commit_memory_write(
            self._store,
            MemoryWrite(
                owner_id=owner_id,
                kind="preference",
                body=body or "",
                confidence=1.0,
                provenance=provenance or _MANAGEMENT_PROVENANCE,
                action="forget",
                supersedes_id=memory_id,
            ),
        )

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        """Return one owner's active records, newest first."""
        return await self._store.list_active(owner_id=owner_id)

    async def browse(
        self,
        *,
        owner_id: str,
        cursor: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        """Page one owner's active records; returns the next keyset cursor."""
        return await self._store.list_active_page(owner_id=owner_id, after=cursor, limit=limit)

    async def recall(
        self,
        *,
        owner_id: str,
        query: str,
        top_k: int = RECALL_TOP_K,
        char_budget: int = RECALL_CHAR_BUDGET,
    ) -> RecallResult:
        """Query-aware recall: RRF-fused leg rankings, exact-first, chronological.

        Time never enters the score: it orders presentation and breaks ties.
        Exact-leg matches pin to the front; the rest follow in RRF order. An
        empty candidate set yields an empty result — no threshold, no fallback.
        """
        cap = max(1, min(int(top_k), 100))
        budget = max(_HEADER_CHARS, int(char_budget))
        try:
            candidates = await asyncio.wait_for(
                self._store.search_candidates(owner_id=owner_id, query=query, limit=cap),
                timeout=_SEARCH_DEADLINE_SECONDS,
            )
        except TimeoutError:
            return RecallResult(records=(), strategy="query_search", degraded=("search_timeout",))

        # Per-leg ordered rankings of memory ids (candidates arrive in rank
        # order; each ranking is unique per id so RRF never double-counts).
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
        for memory_id, score in rrf_fuse(rankings).items():
            fused_ids[memory_id] = score

        by_id = {candidate.record.memory_id: candidate.record for candidate in candidates}
        exact_records = [by_id[memory_id] for memory_id in exact_ids if memory_id in by_id]
        fused_ids = {mid: score for mid, score in fused_ids.items() if mid not in exact_ids}
        ranked_records = [
            by_id[memory_id]
            for memory_id, _ in sorted(fused_ids.items(), key=lambda item: item[1], reverse=True)
            if memory_id in by_id
        ]
        ordered: list[MemoryRecord] = [*exact_records, *ranked_records]

        # Packing prior: keep at least one preference and one fact when present.
        ordered = _packing_prior(ordered)

        # Top-k, then the char budget, then presentation: exact matches stay
        # pinned first (chronological among themselves), the rest follow
        # chronologically. Time never enters the score.
        ordered = ordered[:cap]
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
        """Delete superseded history past the retention floor."""
        return await self._store.purge_superseded(older_than=older_than)


def _packing_prior(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Ensure one preference and one fact survive truncation when present."""
    if not records:
        return records
    first_preference = next((record for record in records if record.kind == "preference"), None)
    first_fact = next((record for record in records if record.kind == "fact"), None)
    if first_preference is None or first_fact is None:
        return records
    kept = [first_preference, first_fact]
    for record in records:
        if record.memory_id not in {first_preference.memory_id, first_fact.memory_id}:
            kept.append(record)
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
