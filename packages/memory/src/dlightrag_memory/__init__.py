# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cross-conversation Owner Profile Memory.

The package owns Memory Record shape, the closed write checklist, structured
recall, and the storage-neutral persistence contract. Hosts own identity,
eligibility, and where recalled text sits in the model context; recalled
records are never Evidence and never citable.
"""

__version__ = "1.9.2"

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag_memory.memory import Memory, RecallResult
from dlightrag_memory.models import (
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
    MemoryWrite,
)
from dlightrag_memory.normalize import normalized_body
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    RECALL_CHAR_BUDGET,
    RECALL_TOP_K,
    evaluate_memory_write,
)
from dlightrag_memory.ports import (
    NullEmbedder,
    SearchCandidate,
    SearchLeg,
    TextEmbedder,
    Vector,
)
from dlightrag_memory.recall import recall_recency
from dlightrag_memory.store import (
    InMemoryMemoryStore,
    MemoryStore,
    commit_memory_write,
    default_purge_cutoff,
)

__all__ = [
    "InMemoryMemoryStore",
    "MEMORY_BODY_LIMIT",
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
    "NullEmbedder",
    "RECALL_CHAR_BUDGET",
    "RECALL_TOP_K",
    "RecallResult",
    "SearchCandidate",
    "SearchLeg",
    "TextEmbedder",
    "Vector",
    "commit_memory_write",
    "default_purge_cutoff",
    "evaluate_memory_write",
    "normalized_body",
    "recall_recency",
    "__version__",
]
