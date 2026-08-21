# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-side Memory policy and context rendering.

The canonical Memory shapes, checklist, recall selection, and storage contract
live in the independently installable ``dlightrag_memory`` package; this
module re-exports them for Answer callers and keeps the root-owned concerns:
owner eligibility policy and rendering the non-citable standing block.
"""

from dlightrag_memory import (
    MEMORY_BODY_LIMIT,
    MEMORY_RECALL_KIND_LIMIT,
    MEMORY_RECALL_LIMIT,
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
    MemoryWrite,
    evaluate_memory_write,
    select_auto_recall,
)
from dlightrag_memory.errors import MemoryUnavailableError


def memory_owner_allowed(auth_mode: str) -> bool:
    """JWT principals may write and auto-recall; deployment buckets may not.

    Eligibility is root product policy, not package behaviour.
    """
    return auth_mode == "jwt"


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


def apply_standing_memory(prompt: str, memory_text: str) -> str:
    """Append the non-citable recall block, or return the prompt unchanged."""
    if not memory_text:
        return prompt
    return f"{prompt}\n\n{memory_text}"


__all__ = [
    "MEMORY_BODY_LIMIT",
    "MEMORY_RECALL_KIND_LIMIT",
    "MEMORY_RECALL_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryUnavailableError",
    "MemoryWrite",
    "apply_standing_memory",
    "evaluate_memory_write",
    "memory_owner_allowed",
    "render_auto_recall",
    "reserved_auto_recall_text",
    "select_auto_recall",
    "standing_memory_for_acceptance",
]
