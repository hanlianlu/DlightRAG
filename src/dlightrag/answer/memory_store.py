# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Memory Record storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from dlightrag.answer.errors import MemoryWriteRejectedError
from dlightrag.answer.memory import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryRecord,
    MemoryWrite,
    evaluate_memory_write,
    select_auto_recall,
)


def _recency(record: MemoryRecord) -> datetime:
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


class AnswerMemoryStore(Protocol):
    """Persist Memory Records. Callers enforce JWT; every method is owner-scoped."""

    async def count_active(self, *, owner_id: str) -> int: ...

    async def count_writes_since(self, *, owner_id: str, since: datetime) -> int: ...

    async def insert(self, record: MemoryRecord) -> None: ...

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None: ...

    async def forget(self, *, owner_id: str, memory_id: str) -> bool: ...

    async def forget_matching(self, *, owner_id: str, body: str) -> int: ...

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None: ...

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]: ...

    async def list_for_recall(self, *, owner_id: str) -> tuple[MemoryRecord, ...]: ...

    async def purge_superseded(self, *, older_than: datetime) -> int: ...

    async def prune_write_log(self, *, older_than: datetime) -> int: ...


class InMemoryAnswerMemoryStore:
    """Process-local store with the durable store's owner isolation."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], MemoryRecord] = {}
        self._writes: list[tuple[str, datetime]] = []

    def _log(self, owner_id: str) -> None:
        self._writes.append((owner_id, datetime.now(UTC)))

    async def count_active(self, *, owner_id: str) -> int:
        return sum(
            1
            for record in self._rows.values()
            if record.owner_id == owner_id and record.status == "active"
        )

    async def count_writes_since(self, *, owner_id: str, since: datetime) -> int:
        return sum(1 for owner, stamp in self._writes if owner == owner_id and stamp >= since)

    async def insert(self, record: MemoryRecord) -> None:
        self._rows[(record.owner_id, record.memory_id)] = record
        self._log(record.owner_id)

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
        self._log(owner_id)

    async def forget(self, *, owner_id: str, memory_id: str) -> bool:
        removed = self._rows.pop((owner_id, memory_id), None) is not None
        if removed:
            self._log(owner_id)
        return removed

    async def forget_matching(self, *, owner_id: str, body: str) -> int:
        target = body.strip()
        victims = [
            key
            for key, record in self._rows.items()
            if record.owner_id == owner_id and record.body.strip() == target
        ]
        for key in victims:
            del self._rows[key]
        if victims:
            self._log(owner_id)
        return len(victims)

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None:
        return self._rows.get((owner_id, memory_id))

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        rows = [
            record
            for record in self._rows.values()
            if record.owner_id == owner_id and record.status == "active"
        ]
        rows.sort(key=_recency, reverse=True)
        return tuple(rows)

    async def list_for_recall(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        return select_auto_recall(await self.list_active(owner_id=owner_id))

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

    async def prune_write_log(self, *, older_than: datetime) -> int:
        kept = [(owner, stamp) for owner, stamp in self._writes if stamp >= older_than]
        removed = len(self._writes) - len(kept)
        self._writes = kept
        return removed


async def commit_memory_write(store: AnswerMemoryStore, write: MemoryWrite) -> MemoryRecord | None:
    """Run the checklist, then insert, supersede, or hard-delete."""
    since = datetime.now(UTC) - timedelta(hours=1)
    filled = MemoryWrite(
        owner_id=write.owner_id,
        auth_mode=write.auth_mode,
        kind=write.kind,
        body=write.body,
        confidence=write.confidence,
        provenance=write.provenance,
        action=write.action,
        supersedes_id=write.supersedes_id,
        active_count=await store.count_active(owner_id=write.owner_id),
        writes_last_hour=await store.count_writes_since(owner_id=write.owner_id, since=since),
    )
    evaluate_memory_write(filled)
    if write.action == "forget":
        if (write.supersedes_id or "").strip():
            removed = await store.forget(
                owner_id=write.owner_id, memory_id=write.supersedes_id or ""
            )
            if not removed:
                raise MemoryWriteRejectedError("No matching memory to forget.")
            return None
        deleted = await store.forget_matching(owner_id=write.owner_id, body=write.body)
        if deleted == 0:
            raise MemoryWriteRejectedError("No matching memory to forget.")
        return None
    now = datetime.now(UTC)
    record = MemoryRecord(
        owner_id=write.owner_id,
        memory_id=str(uuid4()),
        kind=write.kind,
        body=write.body.strip(),
        confidence=write.confidence,
        provenance=write.provenance,
        status="active",
        supersedes_id=write.supersedes_id,
        created_at=now,
        updated_at=now,
    )
    if (write.supersedes_id or "").strip():
        try:
            await store.supersede(
                owner_id=write.owner_id, old_id=write.supersedes_id or "", new=record
            )
        except KeyError:
            raise MemoryWriteRejectedError("No matching memory to replace.") from None
    else:
        await store.insert(record)
    return record


def default_purge_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(days=MEMORY_SUPERSEDE_RETENTION_DAYS)


def write_log_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(hours=2)


__all__ = [
    "AnswerMemoryStore",
    "InMemoryAnswerMemoryStore",
    "commit_memory_write",
    "default_purge_cutoff",
    "write_log_cutoff",
]
