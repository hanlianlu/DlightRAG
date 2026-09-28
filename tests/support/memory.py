# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A process-local Profile Memory settings store for tests."""

from dlightrag.engine.answer.memory import MemoryCapability


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
