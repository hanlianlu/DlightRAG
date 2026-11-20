# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Turning Profile Memory on restarts the owner's Sessions: from off only, and before it is on.

A Session Memory Pass reads what a Session holds past its watermark, so text written while an
owner's memory was off would be read the moment it came back on, unless the owner's Sessions
start afresh at that moment (ADR 0045). The restart is the Run store's; what is pinned here is
when the service asks for it. What the store's statement does is in
tests/integration/test_session_memory_pass_pg.py.
"""

from __future__ import annotations

import pytest

from dlightrag.application.memory.service import MemoryService
from dlightrag.engine.answer.memory import MemoryCapability


class _Settings:
    """The switch as its table keeps it: an owner with nothing stored is on, going off bumps the epoch."""

    def __init__(self, events: list[str], *, enabled: bool = True) -> None:
        self.events = events
        self.enabled = enabled
        self.epoch = 0 if enabled else 1

    async def state(self, *, owner_id: str) -> MemoryCapability:
        return MemoryCapability(enabled=self.enabled, epoch=self.epoch)

    async def state_in_settlement(
        self, *, owner_id: str, settlement: object | None
    ) -> MemoryCapability:
        return await self.state(owner_id=owner_id)

    async def set_enabled(self, *, owner_id: str, enabled: bool) -> MemoryCapability:
        self.events.append(f"set {owner_id} {'on' if enabled else 'off'}")
        if self.enabled and not enabled:
            self.epoch += 1
        self.enabled = enabled
        return await self.state(owner_id=owner_id)

    async def bump_epoch(self, *, owner_id: str) -> MemoryCapability:
        self.epoch += 1
        return await self.state(owner_id=owner_id)


class _Sessions:
    """The Run store's restart, recording when it was asked for and failing on request."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.failure: Exception | None = None

    async def restart_session_memory(self, *, owner_id: str) -> None:
        self.events.append(f"restart {owner_id}")
        if self.failure is not None:
            raise self.failure


class _Records:
    async def count_active(self, *, owner_id: str) -> int:
        return 3


def _service(*, enabled: bool) -> tuple[MemoryService, _Settings, _Sessions, list[str]]:
    events: list[str] = []
    settings, sessions = _Settings(events, enabled=enabled), _Sessions(events)
    service = MemoryService(
        _Records(),
        memory_list_cursor_secret=b"switch-test",
        settings_store=settings,
        sessions=sessions,
    )
    return service, settings, sessions, events


async def test_turning_memory_on_after_it_was_off_restarts_the_owners_sessions_first() -> None:
    service, settings, _, events = _service(enabled=False)

    result = await service.set_enabled(owner_id="alpha", enabled=True)

    # The restart comes before the switch flips, so memory is never on without it; it names only
    # the owner whose switch it was, and it does not touch the epoch the off set.
    assert events == ["restart alpha", "set alpha on"]
    assert (result.enabled, result.epoch, result.active_count) == (True, 1, 3)
    assert settings.enabled


async def test_turning_memory_on_while_it_is_on_restarts_nothing() -> None:
    service, _, _, events = _service(enabled=True)

    result = await service.set_enabled(owner_id="alpha", enabled=True)

    assert events == ["set alpha on"]
    assert result.enabled and result.epoch == 0


async def test_turning_memory_off_restarts_nothing() -> None:
    service, _, _, events = _service(enabled=True)

    result = await service.set_enabled(owner_id="alpha", enabled=False)

    assert events == ["set alpha off"]
    assert not result.enabled and result.active_count is None


async def test_each_off_to_on_restarts_again_and_an_on_to_on_between_does_not() -> None:
    service, _, _, events = _service(enabled=True)

    for enabled in (False, True, True, False, True):
        await service.set_enabled(owner_id="alpha", enabled=enabled)

    assert events == [
        "set alpha off",
        "restart alpha",
        "set alpha on",
        "set alpha on",
        "set alpha off",
        "restart alpha",
        "set alpha on",
    ]


async def test_a_restart_that_fails_leaves_memory_off_and_a_retry_does_it_all_again() -> None:
    service, settings, sessions, events = _service(enabled=False)
    sessions.failure = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        await service.set_enabled(owner_id="alpha", enabled=True)

    # Memory is not on without the restart: the switch never flipped.
    assert events == ["restart alpha"]
    assert not settings.enabled

    sessions.failure = None
    result = await service.set_enabled(owner_id="alpha", enabled=True)

    assert events == ["restart alpha", "restart alpha", "set alpha on"]
    assert result.enabled
