# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cross-conversation Owner Profile Memory.

The package owns Memory Record shape, the closed write checklist, standing
recall, and the storage-neutral persistence contract. Hosts own identity,
eligibility, and where recalled text sits in the model context; recalled
records are never Evidence and never citable.
"""

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag_memory.memory import Memory
from dlightrag_memory.models import (
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
    MemoryWrite,
)
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_RECALL_KIND_LIMIT,
    MEMORY_RECALL_LIMIT,
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    evaluate_memory_write,
    memory_owner_allowed,
)
from dlightrag_memory.recall import (
    render_auto_recall,
    reserved_auto_recall_text,
    select_auto_recall,
    standing_memory_for_acceptance,
)
from dlightrag_memory.store import (
    InMemoryMemoryStore,
    MemoryStore,
    commit_memory_write,
    default_purge_cutoff,
)

__all__ = [
    "InMemoryMemoryStore",
    "MEMORY_BODY_LIMIT",
    "MEMORY_RECALL_KIND_LIMIT",
    "MEMORY_RECALL_LIMIT",
    "MEMORY_SUPERSEDE_RETENTION_DAYS",
    "Memory",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryStore",
    "MemoryUnavailableError",
    "MemoryWrite",
    "MemoryWriteRejectedError",
    "commit_memory_write",
    "default_purge_cutoff",
    "evaluate_memory_write",
    "memory_owner_allowed",
    "render_auto_recall",
    "reserved_auto_recall_text",
    "select_auto_recall",
    "standing_memory_for_acceptance",
]
