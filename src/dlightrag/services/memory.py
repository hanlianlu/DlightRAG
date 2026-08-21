# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner management API for Memory Records and enablement settings."""

from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag_memory import Memory, default_purge_cutoff

from dlightrag.answer.errors import MemoryUnavailableError
from dlightrag.answer.memory import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryRecord,
    memory_owner_allowed,
)


class MemorySettingsStore(Protocol):
    """Root-owned owner enablement flags."""

    async def enabled(self, *, owner_id: str) -> bool: ...

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> None: ...


class NoopMemorySettingsStore:
    """In-process composition without durable settings: always enabled."""

    async def enabled(self, *, owner_id: str) -> bool:
        return True

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> None:
        raise RuntimeError("memory settings are not durable in this composition")


class InMemoryMemorySettingsStore:
    """Process-local enablement flags for tests and standalone compositions."""

    def __init__(self) -> None:
        self._flags: dict[str, bool] = {}

    async def enabled(self, *, owner_id: str) -> bool:
        return self._flags.get(owner_id, True)

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> None:
        self._flags[owner_id] = enabled


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """One owner's cross-conversation Memory settings."""

    enabled: bool
    active_count: int


class MemoryService:
    """Owner list/forget/settings plus fleet purge of expired superseded rows."""

    def __init__(
        self,
        store: Any,
        *,
        settings_store: MemorySettingsStore | None = None,
        superseded_retention_days: int = MEMORY_SUPERSEDE_RETENTION_DAYS,
    ) -> None:
        self._memory = Memory(store)
        self._settings = settings_store or NoopMemorySettingsStore()
        self._retention_days = superseded_retention_days

    async def list_active(self, *, owner_id: str, auth_mode: str) -> tuple[MemoryRecord, ...]:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        return await self._memory.list_active(owner_id=owner_id)

    async def forget(
        self, *, owner_id: str, auth_mode: str, memory_id: str, body: str | None = None
    ) -> None:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        await self._memory.forget(owner_id=owner_id, memory_id=memory_id, body=body)

    async def settings(self, *, owner_id: str, auth_mode: str) -> MemorySettings:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        enabled = await self._settings.enabled(owner_id=owner_id)
        active = await self._memory.list_active(owner_id=owner_id)
        return MemorySettings(enabled=enabled, active_count=len(active))

    async def set_enabled(self, *, owner_id: str, auth_mode: str, enabled: bool) -> None:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        await self._settings.set_enabled(owner_id=owner_id, enabled=enabled)

    async def clear(self, *, owner_id: str, auth_mode: str) -> None:
        """Idempotently hard-delete every record; enablement is untouched."""
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()
        await self._memory.forget(owner_id=owner_id, all=True)

    async def recall_enabled(self, *, owner_id: str) -> bool:
        """Whether answer injection may use this owner's memory."""
        return await self._settings.enabled(owner_id=owner_id)

    async def purge_expired(self) -> int:
        return await self._memory.purge_superseded(
            older_than=default_purge_cutoff(days=self._retention_days)
        )


__all__ = [
    "InMemoryMemorySettingsStore",
    "MemoryService",
    "MemorySettings",
    "MemorySettingsStore",
    "NoopMemorySettingsStore",
]
