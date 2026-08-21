# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner management API for Memory Records."""

from typing import Any

from dlightrag_memory import Memory, default_purge_cutoff

from dlightrag.answer.errors import MemoryUnavailableError
from dlightrag.answer.memory import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryRecord,
    memory_owner_allowed,
)


class MemoryService:
    """Owner list/forget plus fleet purge of expired superseded rows."""

    def __init__(
        self,
        store: Any,
        *,
        superseded_retention_days: int = MEMORY_SUPERSEDE_RETENTION_DAYS,
    ) -> None:
        self._memory = Memory(store)
        self._retention_days = superseded_retention_days

    async def list_active(self, *, owner_id: str, auth_mode: str) -> tuple[MemoryRecord, ...]:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        return await self._memory.list_active(owner_id=owner_id)

    async def forget(
        self, *, owner_id: str, auth_mode: str, memory_id: str, body: str | None = None
    ) -> None:
        await self._memory.forget(
            owner_id=owner_id,
            auth_mode=auth_mode,
            memory_id=memory_id,
            body=body,
        )

    async def purge_expired(self) -> int:
        return await self._memory.purge_superseded(
            older_than=default_purge_cutoff(days=self._retention_days)
        )


__all__ = ["MemoryService"]
