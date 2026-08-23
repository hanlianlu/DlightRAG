# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Narrow host interface for independent Profile Memory.

Conversation episodes, Agent working state, Evidence, Skills, ranking helpers,
policy constants, and storage implementations are intentionally not exported
from this package root. Hosts opt into adapters through their defining modules.
"""

__version__ = "3.0.0"

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from dlightrag_memory.memory import Memory, RecallResult
from dlightrag_memory.models import (
    MemoryKind,
    MemoryProposal,
    MemoryProvenance,
    MemoryRecord,
    MemoryStatus,
    MemoryWrite,
)
from dlightrag_memory.store import MemoryStore

__all__ = [
    "Memory",
    "MemoryKind",
    "MemoryProposal",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryStore",
    "MemoryUnavailableError",
    "MemoryWrite",
    "MemoryWriteRejectedError",
    "RecallResult",
    "__version__",
]
