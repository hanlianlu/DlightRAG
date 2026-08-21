# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Memory Record persistence and the closed write commit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import MemoryRecord, MemoryWrite
from dlightrag_memory.policy import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    evaluate_memory_write,
)


def _recency(record: MemoryRecord) -> datetime:
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


class MemoryStore(Protocol):
    """Persist Memory Records. Callers enforce JWT.

    Row methods are owner-scoped; purge is fleet-wide retention.
    """

    async def insert(self, record: MemoryRecord) -> None: ...

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None: ...

    async def forget(self, *, owner_id: str, memory_id: str) -> bool: ...

    async def forget_matching(self, *, owner_id: str, body: str) -> int: ...

    async def get(self, *, owner_id: str, memory_id: str) -> MemoryRecord | None: ...

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]: ...

    async def purge_superseded(self, *, older_than: datetime) -> int: ...


class InMemoryMemoryStore:
    """Process-local store with the durable store's owner isolation."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], MemoryRecord] = {}

    async def insert(self, record: MemoryRecord) -> None:
        self._rows[(record.owner_id, record.memory_id)] = record

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
        return self._rows.pop((owner_id, memory_id), None) is not None

    async def forget_matching(self, *, owner_id: str, body: str) -> int:
        target = body.strip()
        victims = [
            key
            for key, record in self._rows.items()
            if record.owner_id == owner_id and record.body.strip() == target
        ]
        for key in victims:
            del self._rows[key]
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


async def commit_memory_write(store: MemoryStore, write: MemoryWrite) -> MemoryRecord | None:
    """Run the checklist, then insert, supersede, or hard-delete."""
    filled = MemoryWrite(
        owner_id=write.owner_id,
        auth_mode=write.auth_mode,
        kind=write.kind,
        body=write.body,
        confidence=write.confidence,
        provenance=write.provenance,
        action=write.action,
        supersedes_id=write.supersedes_id,
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


def default_purge_cutoff(days: int = MEMORY_SUPERSEDE_RETENTION_DAYS) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


__all__ = [
    "InMemoryMemoryStore",
    "MemoryStore",
    "commit_memory_write",
    "default_purge_cutoff",
]
