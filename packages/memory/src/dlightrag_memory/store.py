# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Profile Memory persistence and atomic operation settlement."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.models import (
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryRecord,
)
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.policy import MEMORY_SUPERSEDE_RETENTION_DAYS
from dlightrag_memory.ports import SearchCandidate
from dlightrag_memory.recall import recall_recency

# The adapter passes its opaque transaction context (or ``None``) so a host can
# validate an external capability in the same atomic settlement without the
# package importing or knowing that capability's schema.
OperationGuard = Callable[[object | None], Awaitable[None]]


class MemoryStore(Protocol):
    """Deep storage seam: atomic mutations plus owner-scoped recall reads."""

    async def apply_operation(
        self,
        operation: MemoryOperation,
        *,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt: ...

    async def clear_owner(
        self,
        *,
        owner_id: str,
        guard: OperationGuard | None = None,
    ) -> int: ...

    async def count_active(self, *, owner_id: str) -> int: ...

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


@dataclass(frozen=True, slots=True)
class _StoredOperation:
    fingerprint: str
    receipt: MemoryOperationReceipt
    before_records: tuple[MemoryRecord, ...] = ()


class InMemoryMemoryStore:
    """Process-local adapter with the durable adapter's exact operation semantics."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], MemoryRecord] = {}
        self._operations: dict[tuple[str, str], _StoredOperation] = {}
        self._undone_by: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def apply_operation(
        self,
        operation: MemoryOperation,
        *,
        guard: OperationGuard | None = None,
    ) -> MemoryOperationReceipt:
        change_id = operation_change_id(operation)
        fingerprint = operation_fingerprint(operation)
        key = (operation.owner_id, change_id)
        async with self._lock:
            if guard is not None:
                await guard(None)
            replay = self._operations.get(key)
            if replay is not None:
                if replay.fingerprint != fingerprint:
                    raise MemoryWriteRejectedError(
                        "Memory idempotency key was reused with different input."
                    )
                return replay.receipt

            candidate = dict(self._rows)
            receipt, before_records, undone_target = self._apply_to_rows(
                candidate, operation=operation, change_id=change_id
            )
            if receipt.changed and operation.mutation_scope is not None:
                used = sum(
                    1
                    for (owner_id, _), stored in self._operations.items()
                    if owner_id == operation.owner_id
                    and stored.receipt.outcome == "changed"
                    and _receipt_scope(stored.receipt) == operation.mutation_scope
                )
                if used >= int(operation.mutation_limit or 0):
                    raise MemoryWriteRejectedError(
                        "This Answer Run reached its Memory mutation limit."
                    )

            self._rows = candidate
            self._operations[key] = _StoredOperation(
                fingerprint=fingerprint,
                receipt=receipt,
                before_records=before_records,
            )
            if undone_target is not None and receipt.changed:
                self._undone_by[(operation.owner_id, undone_target)] = change_id
            return receipt

    def _apply_to_rows(
        self,
        rows: dict[tuple[str, str], MemoryRecord],
        *,
        operation: MemoryOperation,
        change_id: str,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...], str | None]:
        now = datetime.now(UTC)
        if operation.action == "remember":
            return self._remember_rows(rows, operation=operation, change_id=change_id, now=now)
        if operation.action == "forget":
            return self._forget_rows(rows, operation=operation, change_id=change_id, now=now)
        return self._undo_rows(rows, operation=operation, change_id=change_id, now=now)

    def _remember_rows(
        self,
        rows: dict[tuple[str, str], MemoryRecord],
        *,
        operation: MemoryOperation,
        change_id: str,
        now: datetime,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...], None]:
        body = operation.body.strip()
        duplicate = next(
            (
                record
                for record in rows.values()
                if record.owner_id == operation.owner_id
                and record.status == "active"
                and normalized_body(record.body) == normalized_body(body)
            ),
            None,
        )
        if duplicate is not None:
            if operation.supersedes_id and duplicate.memory_id != operation.supersedes_id:
                return (
                    _receipt(
                        operation,
                        change_id,
                        "conflict",
                        memory_ids=(duplicate.memory_id,),
                        kind=duplicate.kind,
                        body=duplicate.body,
                        now=now,
                    ),
                    (),
                    None,
                )
            return (
                _receipt(
                    operation,
                    change_id,
                    "unchanged",
                    memory_ids=(duplicate.memory_id,),
                    kind=duplicate.kind,
                    body=duplicate.body,
                    now=now,
                ),
                (),
                None,
            )

        before: tuple[MemoryRecord, ...] = ()
        if operation.supersedes_id:
            old_key = (operation.owner_id, operation.supersedes_id)
            old = rows.get(old_key)
            if old is None or old.status != "active":
                return (
                    _receipt(operation, change_id, "conflict", body=body, now=now),
                    (),
                    None,
                )
            before = (old,)
            rows[old_key] = replace(old, status="superseded", updated_at=now)

        memory_id = operation_record_id(operation.owner_id, change_id)
        record = MemoryRecord(
            owner_id=operation.owner_id,
            memory_id=memory_id,
            kind=operation.kind or "fact",
            body=body,
            provenance=operation.provenance,
            status="active",
            supersedes_id=operation.supersedes_id,
            created_at=now,
            updated_at=now,
        )
        rows[(operation.owner_id, memory_id)] = record
        return (
            _receipt(
                operation,
                change_id,
                "changed",
                memory_ids=(memory_id,),
                kind=record.kind,
                body=body,
                now=now,
            ),
            before,
            None,
        )

    def _forget_rows(
        self,
        rows: dict[tuple[str, str], MemoryRecord],
        *,
        operation: MemoryOperation,
        change_id: str,
        now: datetime,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...], None]:
        if operation.memory_id:
            row = rows.get((operation.owner_id, operation.memory_id))
            matches = [row] if row is not None and row.status == "active" else []
        else:
            target = normalized_body(operation.body)
            matches = [
                record
                for record in rows.values()
                if record.owner_id == operation.owner_id
                and record.status == "active"
                and normalized_body(record.body) == target
            ]
        if not matches:
            return (_receipt(operation, change_id, "unchanged", now=now), (), None)
        for record in matches:
            rows[(record.owner_id, record.memory_id)] = replace(
                record, status="forgotten", updated_at=now
            )
        first = matches[0]
        return (
            _receipt(
                operation,
                change_id,
                "changed",
                memory_ids=tuple(record.memory_id for record in matches),
                kind=first.kind,
                body=first.body,
                now=now,
            ),
            tuple(matches),
            None,
        )

    def _undo_rows(
        self,
        rows: dict[tuple[str, str], MemoryRecord],
        *,
        operation: MemoryOperation,
        change_id: str,
        now: datetime,
    ) -> tuple[MemoryOperationReceipt, tuple[MemoryRecord, ...], str | None]:
        target_id = operation.target_change_id or ""
        target = self._operations.get((operation.owner_id, target_id))
        if (
            target is None
            or not target.receipt.changed
            or target.receipt.action == "undo"
            or (operation.owner_id, target_id) in self._undone_by
        ):
            return (
                _receipt(
                    operation,
                    change_id,
                    "conflict",
                    target_change_id=target_id,
                    now=now,
                ),
                (),
                None,
            )

        target_receipt = target.receipt
        if target_receipt.action == "remember":
            current_id = target_receipt.memory_id or ""
            current = rows.get((operation.owner_id, current_id))
            if current is None or current.status != "active":
                return (
                    _receipt(
                        operation,
                        change_id,
                        "conflict",
                        target_change_id=target_id,
                        now=now,
                    ),
                    (),
                    None,
                )
            if target_receipt.supersedes_id and target.before_records:
                old = target.before_records[0]
                rows[(operation.owner_id, current_id)] = replace(
                    current, status="superseded", updated_at=now
                )
                restored_id = operation_record_id(operation.owner_id, change_id)
                restored_record = replace(
                    old,
                    memory_id=restored_id,
                    provenance=operation.provenance,
                    status="active",
                    supersedes_id=current_id,
                    created_at=now,
                    updated_at=now,
                )
                rows[(operation.owner_id, restored_id)] = restored_record
                return (
                    _receipt(
                        operation,
                        change_id,
                        "changed",
                        memory_ids=(restored_id,),
                        kind=restored_record.kind,
                        body=restored_record.body,
                        supersedes_id=current_id,
                        target_change_id=target_id,
                        now=now,
                    ),
                    (current,),
                    target_id,
                )
            rows[(operation.owner_id, current_id)] = replace(
                current, status="forgotten", updated_at=now
            )
            return (
                _receipt(
                    operation,
                    change_id,
                    "changed",
                    memory_ids=(current_id,),
                    kind=current.kind,
                    body=current.body,
                    target_change_id=target_id,
                    now=now,
                ),
                (current,),
                target_id,
            )

        restored: list[MemoryRecord] = []
        for index, old in enumerate(target.before_records):
            current = rows.get((operation.owner_id, old.memory_id))
            if current is None or current.status != "forgotten":
                return (
                    _receipt(
                        operation,
                        change_id,
                        "conflict",
                        target_change_id=target_id,
                        now=now,
                    ),
                    (),
                    None,
                )
            if any(
                record.owner_id == operation.owner_id
                and record.status == "active"
                and normalized_body(record.body) == normalized_body(old.body)
                for record in rows.values()
            ):
                return (
                    _receipt(
                        operation,
                        change_id,
                        "conflict",
                        target_change_id=target_id,
                        now=now,
                    ),
                    (),
                    None,
                )
            restored_id = operation_record_id(operation.owner_id, change_id, index=index)
            record = replace(
                old,
                memory_id=restored_id,
                provenance=operation.provenance,
                status="active",
                supersedes_id=old.memory_id,
                created_at=now,
                updated_at=now,
            )
            rows[(operation.owner_id, restored_id)] = record
            restored.append(record)
        first = restored[0] if restored else None
        return (
            _receipt(
                operation,
                change_id,
                "changed",
                memory_ids=tuple(record.memory_id for record in restored),
                kind=None if first is None else first.kind,
                body="" if first is None else first.body,
                target_change_id=target_id,
                now=now,
            ),
            target.before_records,
            target_id,
        )

    # Raw row methods remain adapter test/bootstrap helpers. Product mutations use apply_operation.
    async def insert(self, record: MemoryRecord) -> None:
        key = (record.owner_id, record.memory_id)
        current = self._rows.get(key)
        if current is not None and current != record:
            raise ValueError("memory id already exists with different content")
        self._rows[key] = record

    async def supersede(self, *, owner_id: str, old_id: str, new: MemoryRecord) -> None:
        current = self._rows.get((owner_id, old_id))
        if new.owner_id != owner_id:
            raise ValueError("supersede cannot change owner")
        if current is None or current.status != "active":
            raise KeyError(old_id)
        self._rows[(owner_id, old_id)] = replace(
            current, status="superseded", updated_at=datetime.now(UTC)
        )
        self._rows[(new.owner_id, new.memory_id)] = new

    async def forget(self, *, owner_id: str, memory_id: str) -> bool:
        key = (owner_id, memory_id)
        current = self._rows.get(key)
        if current is None or current.status != "active":
            return False
        self._rows[key] = replace(current, status="forgotten", updated_at=datetime.now(UTC))
        return True

    async def clear_owner(
        self,
        *,
        owner_id: str,
        guard: OperationGuard | None = None,
    ) -> int:
        async with self._lock:
            if guard is not None:
                await guard(None)
            row_keys = [key for key in self._rows if key[0] == owner_id]
            operation_keys = [key for key in self._operations if key[0] == owner_id]
            for key in row_keys:
                del self._rows[key]
            for key in operation_keys:
                del self._operations[key]
            for key in [key for key in self._undone_by if key[0] == owner_id]:
                del self._undone_by[key]
            return len(row_keys)

    async def count_active(self, *, owner_id: str) -> int:
        return sum(
            record.owner_id == owner_id and record.status == "active"
            for record in self._rows.values()
        )

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
        cap = max(1, min(int(limit), 100))
        active = list(await self.list_active(owner_id=owner_id))
        key = normalized_body(query)
        exact = [record for record in active if normalized_body(record.body) == key]
        query_terms = set(key.split())
        scored: list[tuple[float, MemoryRecord]] = []
        for record in active:
            if normalized_body(record.body) == key:
                continue
            terms = set(normalized_body(record.body).split())
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            substring = 0.25 if key and key in normalized_body(record.body) else 0.0
            if (score := overlap + substring) > 0:
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
        rows = list(await self.list_active(owner_id=owner_id))
        rows.sort(key=lambda record: (_cursor_time(record), record.memory_id), reverse=True)
        if after is not None:
            rows = [
                record
                for record in rows
                if (_cursor_time(record), record.memory_id) < (after[0], after[1])
            ]
        page = tuple(rows[:cap])
        if len(rows) <= cap:
            return page, None
        last = rows[cap - 1]
        return page, (_cursor_time(last), last.memory_id)

    async def purge_superseded(self, *, older_than: datetime) -> int:
        async with self._lock:
            row_keys = [
                key
                for key, record in self._rows.items()
                if record.status != "active"
                and record.updated_at is not None
                and record.updated_at < older_than
            ]
            operation_keys = [
                key
                for key, stored in self._operations.items()
                if stored.receipt.created_at is not None and stored.receipt.created_at < older_than
            ]
            for key in row_keys:
                del self._rows[key]
            for key in operation_keys:
                del self._operations[key]
                self._undone_by.pop(key, None)
            return len(row_keys) + len(operation_keys)


def operation_change_id(operation: MemoryOperation) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"dlightrag-memory-operation:{operation.owner_id}:{operation.idempotency_key}",
        )
    )


def operation_record_id(owner_id: str, change_id: str, *, index: int = 0) -> str:
    return str(uuid5(NAMESPACE_URL, f"dlightrag-memory-record:{owner_id}:{change_id}:{index}"))


def operation_fingerprint(operation: MemoryOperation) -> str:
    payload = {
        "action": operation.action,
        "body": operation.body.strip(),
        "kind": operation.kind,
        "memory_id": operation.memory_id,
        "mutation_limit": operation.mutation_limit,
        "mutation_scope": operation.mutation_scope,
        "origin_id": operation.provenance.origin_id,
        "origin_kind": operation.provenance.origin_kind,
        "owner_id": operation.owner_id,
        "run_id": operation.provenance.run_id,
        "session_id": operation.provenance.session_id,
        "supersedes_id": operation.supersedes_id,
        "target_change_id": operation.target_change_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _receipt(
    operation: MemoryOperation,
    change_id: str,
    outcome: str,
    *,
    memory_ids: tuple[str, ...] = (),
    kind: str | None = None,
    body: str = "",
    supersedes_id: str | None = None,
    target_change_id: str | None = None,
    now: datetime,
) -> MemoryOperationReceipt:
    receipt = MemoryOperationReceipt(
        change_id=change_id,
        action=operation.action,
        outcome=outcome,  # type: ignore[arg-type]
        memory_ids=memory_ids,
        provenance=operation.provenance,
        kind=kind,  # type: ignore[arg-type]
        body=body,
        supersedes_id=supersedes_id if supersedes_id is not None else operation.supersedes_id,
        target_change_id=(
            target_change_id if target_change_id is not None else operation.target_change_id
        ),
        mutation_scope=operation.mutation_scope,
        created_at=now,
    )
    return receipt


def _receipt_scope(receipt: MemoryOperationReceipt) -> str | None:
    return receipt.mutation_scope


def _cursor_time(record: MemoryRecord) -> datetime:
    return record.updated_at or record.created_at or datetime.min.replace(tzinfo=UTC)


def default_purge_cutoff(days: int = MEMORY_SUPERSEDE_RETENTION_DAYS) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


__all__ = [
    "InMemoryMemoryStore",
    "MemoryStore",
    "OperationGuard",
    "default_purge_cutoff",
    "operation_change_id",
    "operation_fingerprint",
    "operation_record_id",
]
