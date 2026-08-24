# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Profile Memory records, operations, and receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryKind = Literal["preference", "fact"]
MemoryStatus = Literal["active", "superseded", "forgotten"]
MemoryOriginKind = Literal["answer_run", "management", "mcp", "undo"]
MemoryOperationAction = Literal["remember", "forget", "undo"]
MemoryOperationOutcome = Literal["changed", "unchanged", "conflict"]


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """Trusted host-bound source of one Memory operation."""

    origin_kind: MemoryOriginKind
    origin_id: str
    run_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One owner-scoped, non-citable remembered preference or fact."""

    owner_id: str
    memory_id: str
    kind: MemoryKind
    body: str
    provenance: MemoryProvenance
    status: MemoryStatus = "active"
    supersedes_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryOperation:
    """One canonical, idempotent mutation request at the storage seam."""

    owner_id: str
    idempotency_key: str
    action: MemoryOperationAction
    provenance: MemoryProvenance
    kind: MemoryKind | None = None
    body: str = ""
    memory_id: str | None = None
    supersedes_id: str | None = None
    target_change_id: str | None = None
    mutation_scope: str | None = None
    mutation_limit: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryOperationReceipt:
    """Replay-stable result of one settled Memory operation."""

    change_id: str
    action: MemoryOperationAction
    outcome: MemoryOperationOutcome
    memory_ids: tuple[str, ...]
    provenance: MemoryProvenance
    kind: MemoryKind | None = None
    body: str = ""
    supersedes_id: str | None = None
    target_change_id: str | None = None
    mutation_scope: str | None = None
    created_at: datetime | None = None

    @property
    def memory_id(self) -> str | None:
        """The primary affected record, when the operation has one."""
        return self.memory_ids[0] if self.memory_ids else None

    @property
    def changed(self) -> bool:
        return self.outcome == "changed"


__all__ = [
    "MemoryKind",
    "MemoryOperation",
    "MemoryOperationAction",
    "MemoryOperationOutcome",
    "MemoryOperationReceipt",
    "MemoryOriginKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
]
