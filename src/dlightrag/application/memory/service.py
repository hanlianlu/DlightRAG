# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner Profile Memory capability gate and product operations."""

from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag_memory import (
    Memory,
    MemoryKind,
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
)
from dlightrag_memory.store import default_purge_cutoff

from dlightrag.answer.memory import (
    MEMORY_SUPERSEDE_RETENTION_DAYS,
    MemoryCapability,
    memory_owner_allowed,
)
from dlightrag.application.answer_runs.errors import MemoryDisabledError, MemoryUnavailableError


class MemorySettingsStore(Protocol):
    async def state(self, *, owner_id: str) -> MemoryCapability: ...

    async def state_in_settlement(
        self, *, owner_id: str, settlement: object | None
    ) -> MemoryCapability: ...

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> MemoryCapability: ...

    async def bump_epoch(self, *, owner_id: str) -> MemoryCapability: ...


class NoopMemorySettingsStore:
    """In-process composition without a mutable product control plane."""

    async def state(self, *, owner_id: str) -> MemoryCapability:
        return MemoryCapability(enabled=True, epoch=0)

    async def state_in_settlement(
        self, *, owner_id: str, settlement: object | None
    ) -> MemoryCapability:
        return await self.state(owner_id=owner_id)

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> MemoryCapability:
        raise RuntimeError("memory settings are not durable in this composition")

    async def bump_epoch(self, *, owner_id: str) -> MemoryCapability:
        return MemoryCapability(enabled=True, epoch=0)


class InMemoryMemorySettingsStore:
    """Process-local settings adapter with deactivation epoch semantics."""

    def __init__(self) -> None:
        self._states: dict[str, MemoryCapability] = {}

    async def state(self, *, owner_id: str) -> MemoryCapability:
        return self._states.get(owner_id, MemoryCapability(enabled=True, epoch=0))

    async def state_in_settlement(
        self, *, owner_id: str, settlement: object | None
    ) -> MemoryCapability:
        return await self.state(owner_id=owner_id)

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> MemoryCapability:
        current = await self.state(owner_id=owner_id)
        epoch = current.epoch + int(current.enabled and not enabled)
        updated = MemoryCapability(enabled=enabled, epoch=epoch)
        self._states[owner_id] = updated
        return updated

    async def bump_epoch(self, *, owner_id: str) -> MemoryCapability:
        current = await self.state(owner_id=owner_id)
        updated = MemoryCapability(enabled=current.enabled, epoch=current.epoch + 1)
        self._states[owner_id] = updated
        return updated


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Settings projection; count is absent while the capability is disabled."""

    enabled: bool
    epoch: int
    active_count: int | None


class MemoryService:
    """The root-owned hard capability gate over the host-neutral Memory module."""

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

    async def capability(self, *, owner_id: str) -> MemoryCapability:
        return await self._settings.state(owner_id=owner_id)

    async def execution_capability(self, *, owner_id: str) -> tuple[bool, int]:
        state = await self.capability(owner_id=owner_id)
        return state.enabled, state.epoch

    async def capability_current(
        self,
        *,
        owner_id: str,
        epoch: int,
        settlement: object | None = None,
    ) -> bool:
        state = await self._settings.state_in_settlement(
            owner_id=owner_id,
            settlement=settlement,
        )
        return state.enabled and state.epoch == epoch

    async def list_active(self, *, owner_id: str, auth_mode: str) -> tuple[MemoryRecord, ...]:
        await self._require_enabled(owner_id=owner_id, auth_mode=auth_mode)
        return await self._memory.list_active(owner_id=owner_id)

    async def remember(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        kind: MemoryKind,
        body: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
        supersedes_id: str | None = None,
    ) -> MemoryOperationReceipt:
        await self._require_enabled(owner_id=owner_id, auth_mode=auth_mode)
        return await self._memory.remember(
            owner_id=owner_id,
            kind=kind,
            body=body,
            provenance=provenance,
            idempotency_key=idempotency_key,
            supersedes_id=supersedes_id,
            guard=lambda settlement: self._guard_enabled(
                owner_id=owner_id,
                auth_mode=auth_mode,
                settlement=settlement,
            ),
        )

    async def forget(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        memory_id: str | None,
        body: str | None = None,
        provenance: MemoryProvenance,
        idempotency_key: str,
    ) -> MemoryOperationReceipt:
        await self._require_enabled(owner_id=owner_id, auth_mode=auth_mode)
        return await self._memory.forget(
            owner_id=owner_id,
            memory_id=memory_id,
            body=body,
            provenance=provenance,
            idempotency_key=idempotency_key,
            guard=lambda settlement: self._guard_enabled(
                owner_id=owner_id,
                auth_mode=auth_mode,
                settlement=settlement,
            ),
        )

    async def undo(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        change_id: str,
        provenance: MemoryProvenance,
        idempotency_key: str,
    ) -> MemoryOperationReceipt:
        await self._require_enabled(owner_id=owner_id, auth_mode=auth_mode)
        return await self._memory.undo(
            owner_id=owner_id,
            change_id=change_id,
            provenance=provenance,
            idempotency_key=idempotency_key,
            guard=lambda settlement: self._guard_enabled(
                owner_id=owner_id,
                auth_mode=auth_mode,
                settlement=settlement,
            ),
        )

    async def settings(self, *, owner_id: str, auth_mode: str) -> MemorySettings:
        self._require_owner(auth_mode)
        state = await self.capability(owner_id=owner_id)
        count = await self._memory.count_active(owner_id=owner_id) if state.enabled else None
        return MemorySettings(enabled=state.enabled, epoch=state.epoch, active_count=count)

    async def set_enabled(self, *, owner_id: str, auth_mode: str, enabled: bool) -> MemorySettings:
        self._require_owner(auth_mode)
        state = await self._settings.set_enabled(owner_id=owner_id, enabled=enabled)
        count = await self._memory.count_active(owner_id=owner_id) if state.enabled else None
        return MemorySettings(enabled=state.enabled, epoch=state.epoch, active_count=count)

    async def clear(self, *, owner_id: str, auth_mode: str) -> int:
        """Physically clear enabled Profile Memory and invalidate active runs."""
        await self._require_enabled(owner_id=owner_id, auth_mode=auth_mode)
        # Invalidate every active run before erasing package state. A failed erase
        # is safely retryable; the inverse order could let a stale run repopulate.
        await self._settings.bump_epoch(owner_id=owner_id)
        return await self._memory.clear(
            owner_id=owner_id,
            guard=lambda settlement: self._guard_enabled(
                owner_id=owner_id,
                auth_mode=auth_mode,
                settlement=settlement,
            ),
        )

    async def recall_enabled(self, *, owner_id: str) -> bool:
        return (await self.capability(owner_id=owner_id)).enabled

    async def purge_expired(self) -> int:
        return await self._memory.purge_superseded(
            older_than=default_purge_cutoff(days=self._retention_days)
        )

    async def _require_enabled(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        settlement: object | None = None,
    ) -> MemoryCapability:
        self._require_owner(auth_mode)
        state = await self._settings.state_in_settlement(
            owner_id=owner_id,
            settlement=settlement,
        )
        if not state.enabled:
            raise MemoryDisabledError()
        return state

    async def _guard_enabled(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        settlement: object | None,
    ) -> None:
        """Recheck activation inside the store's atomic settlement."""
        await self._require_enabled(
            owner_id=owner_id,
            auth_mode=auth_mode,
            settlement=settlement,
        )

    @staticmethod
    def _require_owner(auth_mode: str) -> None:
        if not memory_owner_allowed(auth_mode):
            raise MemoryUnavailableError()


__all__ = [
    "InMemoryMemorySettingsStore",
    "MemoryCapability",
    "MemoryService",
    "MemorySettings",
    "MemorySettingsStore",
    "NoopMemorySettingsStore",
]
