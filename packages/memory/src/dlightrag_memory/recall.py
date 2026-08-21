# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Recency ordering for record tie-breaks.

Query relevance decides the candidate set (fusion in ``fusion.py``); time is
never a score component — it only orders presentation and breaks ties, the
same design MemMachine uses.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dlightrag_memory.models import MemoryRecord


def recall_recency(record: MemoryRecord) -> datetime:
    """One record's standing-ordering timestamp."""
    if record.updated_at is not None:
        return record.updated_at
    if record.created_at is not None:
        return record.created_at
    return datetime.min.replace(tzinfo=UTC)


__all__ = ["recall_recency"]
