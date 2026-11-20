# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Narrow host interface for independent Profile Memory."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag-memory")
except Exception:
    __version__ = "0.0.0"

from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.memory import (
    Memory,
    RecallResult,
    recall_cost,
    recall_line,
    recall_quote_line,
)
from dlightrag_memory.models import (
    MEMORY_KINDS,
    STANDING_KINDS,
    MemoryKind,
    MemoryOperation,
    MemoryOperationAction,
    MemoryOperationOutcome,
    MemoryOperationReceipt,
    MemoryOriginKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRunChange,
    MemoryRunChangeAction,
    MemoryStatus,
)
from dlightrag_memory.store import MemoryStore

__all__ = [
    "MEMORY_KINDS",
    "STANDING_KINDS",
    "Memory",
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
    "MemoryStore",
    "MemoryWriteRejectedError",
    "RecallResult",
    "recall_cost",
    "recall_line",
    "recall_quote_line",
    "__version__",
]
