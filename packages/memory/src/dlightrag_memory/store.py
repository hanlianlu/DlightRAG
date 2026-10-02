# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral Profile Memory persistence and atomic operation settlement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from dlightrag_memory.models import (
    MemoryOperation,
    MemoryOperationReceipt,
    MemoryRecord,
)
from dlightrag_memory.policy import MEMORY_SUPERSEDE_RETENTION_DAYS
from dlightrag_memory.ports import SearchCandidate

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

    async def list_preferences(self, *, owner_id: str, limit: int) -> tuple[MemoryRecord, ...]: ...

    async def search_facts(
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


def operation_receipt(
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


def default_purge_cutoff(days: int = MEMORY_SUPERSEDE_RETENTION_DAYS) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


__all__ = [
    "MemoryStore",
    "OperationGuard",
    "default_purge_cutoff",
    "operation_change_id",
    "operation_fingerprint",
    "operation_receipt",
    "operation_record_id",
]
