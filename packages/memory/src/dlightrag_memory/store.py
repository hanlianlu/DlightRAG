# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Memory Record persistence and the closed write commit."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import MemoryProposal, MemoryRecord, MemoryWrite
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.policy import MEMORY_SUPERSEDE_RETENTION_DAYS, evaluate_memory_write
from dlightrag_memory.ports import SearchCandidate
from dlightrag_memory.recall import recall_recency


class MemoryStore(Protocol):
    """Persist and search Memory Records. Callers enforce eligibility.

    ``search_candidates`` returns leg-tagged candidates in per-leg rank order
    (no cross-leg dedupe): the façade fuses the rankings with RRF. Row methods
    are owner-scoped; purge and clear-all are fleet/owner retention operations.
    """

    async def insert(self, record: MemoryRecord) -> None: ...

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None: ...

    async def forget(self, *, owner_id: str, memory_id: str) -> bool: ...

    async def forget_matching(self, *, owner_id: str, body: str) -> int: ...

    async def forget_all(self, *, owner_id: str) -> int: ...

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None: ...

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]: ...

    async def search_candidates(
        self, *, owner_id: str, query: str, limit: int
    ) -> tuple[SearchCandidate, ...]: ...

    async def list_active_page(
        self,
        *,
        owner_id: str,
        after: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]: ...

    async def purge_superseded(self, *, older_than: datetime) -> int: ...


class InMemoryMemoryStore:
    """Process-local store with the durable store's owner isolation."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], MemoryRecord] = {}

    async def insert(self, record: MemoryRecord) -> None:
        key = (record.owner_id, record.memory_id)
        current = self._rows.get(key)
        if current is not None and current != record:
            raise ValueError("memory id already exists with different content")
        self._rows[key] = record

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None:
        if new.owner_id != owner_id:
            raise ValueError("supersede cannot change owner")
        current = self._rows.get((owner_id, old_id))
        if current is None or current.status != "active":
            raise KeyError(old_id)
        now = datetime.now(UTC)
        self._rows[(owner_id, old_id)] = MemoryRecord(
            owner_id=current.owner_id,
            memory_id=current.memory_id,
            kind=current.kind,
            body=current.body,
            confidence=current.confidence,
            provenance=current.provenance,
            status="superseded",
            supersedes_id=current.supersedes_id,
            created_at=current.created_at,
            updated_at=now,
        )
        self._rows[(new.owner_id, new.memory_id)] = new

    async def forget(self, *, owner_id: str, memory_id: str) -> bool:
        key = (owner_id, memory_id)
        current = self._rows.get(key)
        if current is None or current.status == "forgotten":
            return False
        self._rows[key] = replace(current, status="forgotten", updated_at=datetime.now(UTC))
        return True

    async def forget_matching(self, *, owner_id: str, body: str) -> int:
        target = body.strip()
        victims = [
            key
            for key, record in self._rows.items()
            if record.owner_id == owner_id
            and record.status != "forgotten"
            and record.body.strip() == target
        ]
        now = datetime.now(UTC)
        for key in victims:
            self._rows[key] = replace(self._rows[key], status="forgotten", updated_at=now)
        return len(victims)

    async def forget_all(self, *, owner_id: str) -> int:
        victims = [
            key
            for key, record in self._rows.items()
            if record.owner_id == owner_id and record.status != "forgotten"
        ]
        now = datetime.now(UTC)
        for key in victims:
            self._rows[key] = replace(self._rows[key], status="forgotten", updated_at=now)
        return len(victims)

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        return self._rows.get((owner_id, memory_id))

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        rows = [
            record
            for record in self._rows.values()
            if record.owner_id == owner_id and record.status == "active"
        ]
        rows.sort(key=recall_recency, reverse=True)
        return tuple(rows)

    async def search_candidates(
        self, *, owner_id: str, query: str, limit: int
    ) -> tuple[SearchCandidate, ...]:
        """Naive two-leg rankings: exact equality, then token-overlap."""
        cap = max(1, min(int(limit), 100))
        active = [
            record
            for record in self._rows.values()
            if record.owner_id == owner_id and record.status == "active"
        ]
        key = normalized_body(query)
        exact = sorted(
            (record for record in active if normalized_body(record.body) == key),
            key=recall_recency,
            reverse=True,
        )
        query_terms = set(key.split())
        scored: list[tuple[float, MemoryRecord]] = []
        for record in active:
            if normalized_body(record.body) == key:
                continue
            terms = set(normalized_body(record.body).split())
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            substring = 0.25 if key and key in normalized_body(record.body) else 0.0
            score = overlap + substring
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], recall_recency(item[1])), reverse=True)
        candidates = [
            SearchCandidate(record=record, leg="exact", score=2.0) for record in exact[:cap]
        ]
        candidates.extend(
            SearchCandidate(record=record, leg="sparse", score=score)
            for score, record in scored[: max(0, cap - len(candidates))]
        )
        return tuple(candidates)

    async def list_active_page(
        self,
        *,
        owner_id: str,
        after: tuple[datetime, str] | None = None,
        limit: int = 50,
    ) -> tuple[tuple[MemoryRecord, ...], tuple[datetime, str] | None]:
        cap = max(1, min(int(limit), 100))
        rows = [
            record
            for record in self._rows.values()
            if record.owner_id == owner_id and record.status == "active"
        ]
        rows.sort(key=lambda record: (record.updated_at, record.memory_id), reverse=True)
        if after is not None:
            rows = [
                record
                for record in rows
                if (record.updated_at, record.memory_id) < (after[0], after[1])
            ]
        page = tuple(rows[:cap])
        if len(rows) <= cap:
            return page, None
        last = rows[cap - 1]
        cursor_time = last.updated_at or last.created_at or datetime.min.replace(tzinfo=UTC)
        return page, (cursor_time, last.memory_id)

    async def purge_superseded(self, *, older_than: datetime) -> int:
        victims = [
            key
            for key, record in self._rows.items()
            if record.status == "superseded"
            and record.updated_at is not None
            and record.updated_at < older_than
        ]
        for key in victims:
            del self._rows[key]
        return len(victims)


async def commit_memory_proposal(
    store: MemoryStore, proposal: MemoryProposal
) -> MemoryRecord | None:
    """Commit one validated proposal idempotently, leaving tombstones on forget."""
    write = proposal.write
    evaluate_memory_write(write)
    if write.action == "forget":
        if (write.supersedes_id or "").strip():
            await store.forget(owner_id=write.owner_id, memory_id=write.supersedes_id or "")
            return None
        await store.forget_matching(owner_id=write.owner_id, body=write.body)
        return None
    memory_id = str(
        uuid5(NAMESPACE_URL, f"dlightrag-memory:{write.owner_id}:{proposal.proposal_id}")
    )
    existing = await store.get(owner_id=write.owner_id, memory_id=memory_id)
    if existing is not None:
        if _same_proposal(existing, write):
            return existing
        raise MemoryWriteRejectedError("Memory proposal id was reused with different content.")
    record = MemoryRecord(
        owner_id=write.owner_id,
        memory_id=memory_id,
        kind=write.kind,
        body=write.body.strip(),
        confidence=write.confidence,
        provenance=write.provenance,
        status="active",
        supersedes_id=write.supersedes_id,
        created_at=proposal.proposed_at,
        updated_at=proposal.proposed_at,
    )
    if (write.supersedes_id or "").strip():
        try:
            await store.supersede(
                owner_id=write.owner_id, old_id=write.supersedes_id or "", new=record
            )
        except KeyError:
            replay = await store.get(owner_id=write.owner_id, memory_id=memory_id)
            if replay is not None and _same_proposal(replay, write):
                return replay
            raise MemoryWriteRejectedError("No matching memory to replace.") from None
    else:
        try:
            await store.insert(record)
        except ValueError:
            replay = await store.get(owner_id=write.owner_id, memory_id=memory_id)
            if replay is not None and _same_proposal(replay, write):
                return replay
            raise MemoryWriteRejectedError(
                "Memory proposal id was reused with different content."
            ) from None
    return record


async def commit_memory_write(store: MemoryStore, write: MemoryWrite) -> MemoryRecord | None:
    """Compatibility-free convenience for hosts without an external proposal id."""
    now = datetime.now(UTC)
    proposal = MemoryProposal(
        proposal_id=str(uuid5(NAMESPACE_URL, f"{write.owner_id}:{write}:{now.isoformat()}")),
        write=write,
        proposed_at=now,
    )
    return await commit_memory_proposal(store, proposal)


def _same_proposal(record: MemoryRecord, write: MemoryWrite) -> bool:
    return (
        record.owner_id == write.owner_id
        and record.kind == write.kind
        and record.body == write.body.strip()
        and record.confidence == write.confidence
        and record.provenance == write.provenance
        and record.supersedes_id == write.supersedes_id
    )


def default_purge_cutoff(days: int = MEMORY_SUPERSEDE_RETENTION_DAYS) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


__all__ = [
    "InMemoryMemoryStore",
    "MemoryStore",
    "commit_memory_proposal",
    "commit_memory_write",
    "default_purge_cutoff",
]
