# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Memory Record and Memory Write shapes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryKind = Literal["preference", "fact"]
MemoryStatus = Literal["active", "superseded", "forgotten"]


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """Where a Memory Write learned this body."""

    run_id: str
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One owner-scoped, non-citable remembered preference or fact."""

    owner_id: str
    memory_id: str
    kind: MemoryKind
    body: str
    confidence: float
    provenance: MemoryProvenance
    status: MemoryStatus = "active"
    supersedes_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    """One remember or forget attempt before the store sees it.

    The package never judges caller identity: eligibility is host policy and
    the host raises its own unavailable error before reaching the façade.
    """

    owner_id: str
    kind: MemoryKind
    body: str
    confidence: float
    provenance: MemoryProvenance
    action: Literal["remember", "forget"] = "remember"
    supersedes_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """A validated formation decision before any storage mutation."""

    proposal_id: str
    write: MemoryWrite
    proposed_at: datetime


__all__ = [
    "MemoryKind",
    "MemoryProposal",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryWrite",
]
