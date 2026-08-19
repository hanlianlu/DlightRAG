# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner management API for Memory Records."""

from dlightrag.answer.errors import MemoryUnavailableError
from dlightrag.answer.memory import (
    MemoryProvenance,
    MemoryRecord,
    MemoryWrite,
    memory_owner_allowed,
)
from dlightrag.answer.memory_store import (
    AnswerMemoryStore,
    commit_memory_write,
    default_purge_cutoff,
    write_log_cutoff,
)


class MemoryService:
    """Owner list/forget plus fleet purge of expired superseded rows."""

    def __init__(self, store: AnswerMemoryStore) -> None:
        self._store = store

    async def list_active(self, *, owner_id: str, auth_mode: str) -> tuple[MemoryRecord, ...]:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        return await self._store.list_active(owner_id=owner_id)

    async def forget(self, *, owner_id: str, auth_mode: str, memory_id: str) -> None:
        await commit_memory_write(
            self._store,
            MemoryWrite(
                owner_id=owner_id,
                auth_mode=auth_mode,
                kind="preference",
                body="",
                confidence=1.0,
                provenance=MemoryProvenance(run_id="management", session_id="management"),
                action="forget",
                supersedes_id=memory_id,
            ),
        )

    async def purge_expired(self) -> int:
        removed = await self._store.purge_superseded(older_than=default_purge_cutoff())
        await self._store.prune_write_log(older_than=write_log_cutoff())
        return removed


__all__ = ["MemoryService"]
