# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Standing auto-recall selection and rendering.

Selection is recency-windowed by design today; query-aware recall replaces it
later without changing the package's storage semantics. The rendering is a
plain non-citable block — the host decides where it sits in the model context.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dlightrag_memory.models import MemoryProvenance, MemoryRecord
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_RECALL_KIND_LIMIT,
    MEMORY_RECALL_LIMIT,
    memory_owner_allowed,
)


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


def reserved_auto_recall_text() -> str:
    """Worst-case standing block one JWT accept must leave room for."""
    body = "x" * MEMORY_BODY_LIMIT
    records = tuple(
        MemoryRecord(
            owner_id="reserve",
            memory_id=f"{index:02d}",
            kind="preference" if index < MEMORY_RECALL_KIND_LIMIT else "fact",
            body=body,
            confidence=1.0,
            provenance=MemoryProvenance(run_id="reserve", session_id="reserve"),
        )
        for index in range(MEMORY_RECALL_LIMIT)
    )
    return render_auto_recall(records)


def standing_memory_for_acceptance(auth_mode: str) -> str:
    """Reserve full auto-recall at accept so execute cannot overflow after 202."""
    if not memory_owner_allowed(auth_mode):
        return ""
    return reserved_auto_recall_text()


__all__ = [
    "render_auto_recall",
    "reserved_auto_recall_text",
    "select_auto_recall",
    "standing_memory_for_acceptance",
]
