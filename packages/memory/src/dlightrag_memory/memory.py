# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deep model/owner-facing Memory façade.

One interface for every cross-conversation Profile operation. The host owns
identity, eligibility, and context placement; this module owns the checklist,
atomic write commit, structured recall, and lifecycle policy. Every method
returns structured records — never prompt fragments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from dlightrag_memory.models import MemoryProvenance, MemoryRecord, MemoryWrite
from dlightrag_memory.ports import MemorySearch, SearchCandidate
from dlightrag_memory.recall import select_auto_recall
from dlightrag_memory.store import MemoryStore, commit_memory_write

_MANAGEMENT_PROVENANCE = MemoryProvenance(run_id="management", session_id="management")


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Structured recall outcome: selected records plus provenance of choice.

    ``strategy`` names the path that produced the records (query search or the
    recency window); ``candidates`` carries score/leg provenance when search
    ran; ``degraded`` is set when a configured leg was unavailable.
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
    """Cross-conversation Owner Profile Memory bound to one storage adapter.

    ``search`` is the optional P4 candidate surface; without it, recall falls
    back to the recency window. ``browse`` pages one owner's active records
    with keyset cursors independent of retrieval.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        search: MemorySearch | None = None,
    ) -> None:
        self._store = store
        self._search = search

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
            removed = await self._store.forget_all(owner_id=owner_id)
            if removed == 0:
                return
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

    async def recall(self, *, owner_id: str, query: str, limit: int = 12) -> RecallResult:
        """Query-aware recall when the store can search; recency fallback otherwise.

        Candidates are deduplicated by record id keeping the best score. P4
        wires this into host context assembly and benchmarks the fusion.
        """
        cap = max(1, min(int(limit), 100))
        if self._search is not None:
            candidates = await self._search.search_candidates(
                owner_id=owner_id, query=query, limit=cap
            )
            records = tuple(candidate.record for candidate in candidates[:cap])
            return RecallResult(
                records=records,
                strategy="query_search",
                candidates=tuple(candidates[:cap]),
                content_chars=sum(len(record.body) for record in records),
            )
        records = select_auto_recall(await self.list_active(owner_id=owner_id))[:cap]
        return RecallResult(
            records=records,
            strategy="recency_window",
            content_chars=sum(len(record.body) for record in records),
        )

    async def purge_superseded(self, *, older_than: datetime) -> int:
        """Delete superseded history past the retention floor."""
        return await self._store.purge_superseded(older_than=older_than)


__all__ = ["Memory", "RecallResult"]
