# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner management API for Memory Records."""

from dlightrag.answer.errors import MemoryUnavailableError
from dlightrag.answer.memory import (
    MemoryProvenance,
    MemoryRecord,
    MemoryWrite,
    memory_owner_allowed,
)
from dlightrag.answer.memory_store import AnswerMemoryStore, commit_memory_write


class MemoryService:
    """List and forget Memory Records for one authenticated owner."""

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


__all__ = ["MemoryService"]
