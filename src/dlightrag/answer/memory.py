# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-side Memory Record façade.

The canonical Memory shapes, checklist, recall selection, and storage contract
live in the independently installable ``dlightrag_memory`` package; this module
re-exports them for Answer callers and keeps the one root-owned context
operation: appending the non-citable standing block to a prompt.
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
    memory_owner_allowed,
    render_auto_recall,
    reserved_auto_recall_text,
    select_auto_recall,
    standing_memory_for_acceptance,
)


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
    "MemoryWrite",
    "apply_standing_memory",
    "evaluate_memory_write",
    "memory_owner_allowed",
    "render_auto_recall",
    "reserved_auto_recall_text",
    "select_auto_recall",
    "standing_memory_for_acceptance",
]
