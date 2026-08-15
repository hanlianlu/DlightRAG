# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for storage-neutral corpus backend adapters."""

from contextlib import asynccontextmanager

from dlightrag_rag.ports import WorkspaceCorpusBackend


class _CoordinationFake:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    @asynccontextmanager
    async def workspace_initialization(self):
        self._events.append("initialization:enter")
        try:
            yield
        finally:
            self._events.append("initialization:exit")

    @asynccontextmanager
    async def pipeline_recovery(self):
        self._events.append("recovery:enter")
        try:
            yield
        finally:
            self._events.append("recovery:exit")


class _MaintenanceFake:
    async def initialize(self, *, validate_only: bool = False) -> None:
        return None

    async def clean_orphan_rows(self, workspace: str, *, dry_run: bool) -> int:
        return 0

    async def delete_workspace_record(self, workspace: str) -> bool:
        return False

    async def list_workspace_records(self) -> tuple[dict[str, object], ...]:
        return ()

    async def register_workspace(
        self,
        *,
        workspace: str,
        display_name: str,
        embedding_model: str,
    ) -> None:
        return None


async def test_coordination_contexts_bound_the_owned_operation() -> None:
    events: list[str] = []
    backend = WorkspaceCorpusBackend(
        coordination=_CoordinationFake(events),
        maintenance=_MaintenanceFake(),
    )

    async with backend.coordination.workspace_initialization():
        events.append("initialize")
    async with backend.coordination.pipeline_recovery():
        events.append("recover")

    assert events == [
        "initialization:enter",
        "initialize",
        "initialization:exit",
        "recovery:enter",
        "recover",
        "recovery:exit",
    ]


async def test_maintenance_fake_uses_owner_facing_values() -> None:
    maintenance = _MaintenanceFake()

    await maintenance.initialize(validate_only=True)
    assert await maintenance.clean_orphan_rows("research", dry_run=True) == 0
    assert await maintenance.delete_workspace_record("research") is False
    assert await maintenance.list_workspace_records() == ()
    await maintenance.register_workspace(
        workspace="research",
        display_name="Research",
        embedding_model="embedding-model",
    )
