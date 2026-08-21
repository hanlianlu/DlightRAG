# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Active-record selection for standing recall.

Selection is recency-windowed by design today; query-aware recall replaces it
later without changing the package's storage semantics. Rendering stays with
the host: this module returns records, never prompt fragments.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dlightrag_memory.models import MemoryRecord
from dlightrag_memory.policy import MEMORY_RECALL_KIND_LIMIT, MEMORY_RECALL_LIMIT


def select_auto_recall(records: tuple[MemoryRecord, ...]) -> tuple[MemoryRecord, ...]:
    """Newest active records within the auto-recall caps."""
    preferences = 0
    facts = 0
    chosen: list[MemoryRecord] = []
    ordered = sorted(records, key=recall_recency, reverse=True)
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


def recall_recency(record: MemoryRecord) -> datetime:
    """One record's standing-ordering timestamp."""
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


__all__ = ["recall_recency", "select_auto_recall"]
