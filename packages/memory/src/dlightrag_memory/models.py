# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Profile Memory records, operations, and receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryKind = Literal["preference", "fact", "experience"]
MemoryStatus = Literal["active", "superseded", "forgotten"]
MemoryOriginKind = Literal["answer_run", "management", "mcp", "undo", "session_pass"]
MemoryOperationAction = Literal["remember", "forget", "undo", "fuse"]
MemoryOperationOutcome = Literal["changed", "unchanged", "conflict"]
#: How a record a Run left came to be: a new statement, a correction of an earlier one, or a merge.
MemoryRunChangeAction = Literal["add", "update", "fuse"]

#: Every kind, in one place: storage checks, policy and tool schemas derive from it.
MEMORY_KINDS: tuple[MemoryKind, ...] = ("preference", "fact", "experience")
#: Kinds that apply to every request and so stand in recall whenever they fit;
#: a ``fact`` reaches a request only through relevance.
STANDING_KINDS: tuple[MemoryKind, ...] = ("preference", "experience")
MEMORY_ORIGIN_KINDS: tuple[MemoryOriginKind, ...] = (
    "answer_run",
    "management",
    "mcp",
    "undo",
    "session_pass",
)
MEMORY_OPERATION_ACTIONS: tuple[MemoryOperationAction, ...] = ("remember", "forget", "undo", "fuse")


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
    #: A short verbatim excerpt of what the owner said, the words this record's current form
    #: rests on; None where the owner did not say it, or the writer gave none.
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRunChange:
    """One active record a Run's writer left, with the change that made it.

    ``change_id`` is what undoes it; ``action`` says whether it added a statement, corrected
    an earlier one (``update``) or replaced several (``fuse``).
    """

    record: MemoryRecord
    change_id: str
    action: MemoryRunChangeAction


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
    #: The records one ``fuse`` replaces, in the caller's order; empty otherwise.
    source_ids: tuple[str, ...] = ()
    #: The owner's own words behind the record a ``remember`` or ``fuse`` writes (see
    #: ``MemoryRecord.quote``); the policy admits one only where it can stand.
    quote: str | None = None


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
    "MEMORY_KINDS",
    "MEMORY_OPERATION_ACTIONS",
    "MEMORY_ORIGIN_KINDS",
    "STANDING_KINDS",
    "MemoryKind",
    "MemoryOperation",
    "MemoryOperationAction",
    "MemoryOperationOutcome",
    "MemoryOperationReceipt",
    "MemoryOriginKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryRunChange",
    "MemoryRunChangeAction",
    "MemoryStatus",
]
