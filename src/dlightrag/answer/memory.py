# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Memory Record shape and the closed Memory Write checklist."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from dlightrag.answer.errors import MemoryUnavailableError, MemoryWriteRejectedError

MemoryKind = Literal["preference", "fact"]
MemoryStatus = Literal["active", "superseded"]

MEMORY_BODY_LIMIT = 500
MEMORY_ACTIVE_LIMIT = 200
MEMORY_WRITES_PER_HOUR = 20
MEMORY_RECALL_LIMIT = 12
MEMORY_RECALL_KIND_LIMIT = 8
MEMORY_SUPERSEDE_RETENTION_DAYS = 30

_CITATION_MARK = re.compile(r"\[\d+(?:-\d+)?\]")


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
    """One remember or forget attempt before the store sees it."""

    owner_id: str
    auth_mode: str
    kind: MemoryKind
    body: str
    confidence: float
    provenance: MemoryProvenance
    action: Literal["remember", "forget"] = "remember"
    supersedes_id: str | None = None
    active_count: int = 0
    writes_last_hour: int = 0


def memory_owner_allowed(auth_mode: str) -> bool:
    """JWT principals may write and auto-recall; deployment buckets may not."""
    return auth_mode == "jwt"


def evaluate_memory_write(write: MemoryWrite) -> None:
    """Accept one Memory Write or raise a public checklist error."""
    if not memory_owner_allowed(write.auth_mode):
        raise MemoryUnavailableError()
    if write.writes_last_hour >= MEMORY_WRITES_PER_HOUR:
        raise MemoryWriteRejectedError("This owner has written too many memories this hour.")
    body = write.body.strip()
    if write.action == "forget":
        if not (write.supersedes_id or "").strip() and not body:
            raise MemoryWriteRejectedError("A forget must name a memory or quote its body.")
        return
    if write.kind not in {"preference", "fact"}:
        raise MemoryWriteRejectedError("Memory kind must be preference or fact.")
    if not body:
        raise MemoryWriteRejectedError("Memory body cannot be empty.")
    if len(body) > MEMORY_BODY_LIMIT:
        raise MemoryWriteRejectedError(f"Memory body cannot exceed {MEMORY_BODY_LIMIT} characters.")
    if not 0 < write.confidence <= 1:
        raise MemoryWriteRejectedError("Memory confidence must be in (0, 1].")
    if not write.provenance.run_id.strip() or not write.provenance.session_id.strip():
        raise MemoryWriteRejectedError("A remember needs run and session provenance.")
    if _CITATION_MARK.search(body):
        raise MemoryWriteRejectedError("Memory body cannot carry citation markers.")
    replacing = bool((write.supersedes_id or "").strip())
    if write.active_count >= MEMORY_ACTIVE_LIMIT and not replacing:
        raise MemoryWriteRejectedError("This owner already has the maximum active memories.")


def select_auto_recall(records: tuple[MemoryRecord, ...]) -> tuple[MemoryRecord, ...]:
    """Newest active records within the auto-recall caps."""
    preferences = 0
    facts = 0
    chosen: list[MemoryRecord] = []
    ordered = sorted(records, key=_recall_recency, reverse=True)
    for record in ordered:
        if record.status != "active":
            continue
        if record.kind == "preference":
            if preferences >= MEMORY_RECALL_KIND_LIMIT:
                continue
            preferences += 1
        else:
            if facts >= MEMORY_RECALL_KIND_LIMIT:
                continue
            facts += 1
        chosen.append(record)
        if len(chosen) >= MEMORY_RECALL_LIMIT:
            break
    return tuple(chosen)


def _recall_recency(record: MemoryRecord) -> datetime:
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


def render_auto_recall(records: tuple[MemoryRecord, ...]) -> str:
    """Standing non-citable block, or empty when there is nothing to inject."""
    if not records:
        return ""
    lines = [
        "Remembered about this owner (not evidence; do not cite):",
        *(f"- ({record.kind}) {record.body}" for record in records),
    ]
    return "\n".join(lines)


__all__ = [
    "MEMORY_ACTIVE_LIMIT",
    "MEMORY_BODY_LIMIT",
    "MEMORY_RECALL_KIND_LIMIT",
    "MEMORY_RECALL_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "MEMORY_WRITES_PER_HOUR",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryWrite",
    "evaluate_memory_write",
    "memory_owner_allowed",
    "render_auto_recall",
    "select_auto_recall",
]
