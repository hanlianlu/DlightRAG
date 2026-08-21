# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-side Memory storage re-export.

The storage protocol, in-memory adapter, write commit, and purge cutoff live in
``dlightrag_memory``. The PostgreSQL adapter in ``dlightrag.adapters.postgres``
implements the same protocol.
"""

from dlightrag_memory import (
    InMemoryMemoryStore,
    MemoryStore,
    commit_memory_write,
    default_purge_cutoff,
)

# Back-compat aliases for Answer-era names.
AnswerMemoryStore = MemoryStore
InMemoryAnswerMemoryStore = InMemoryMemoryStore


__all__ = [
    "AnswerMemoryStore",
    "InMemoryAnswerMemoryStore",
    "InMemoryMemoryStore",
    "MemoryStore",
    "commit_memory_write",
    "default_purge_cutoff",
]
