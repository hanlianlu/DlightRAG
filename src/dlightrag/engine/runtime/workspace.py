# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound workspace epoch, inventory, and committed-spill port."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import nsmallest
from typing import Literal, Protocol

from dlightrag.engine.runtime.settlements import InventoryPathRecord


@dataclass(frozen=True, slots=True)
class CommittedSpillRecord:
    """One committed spill digest for volume recovery."""

    resource_id: str
    content_digest: str
    size_bytes: int
    session_id: str
    intent_id: str


@dataclass(frozen=True, slots=True)
class HandoffCommit:
    """CAS moved workspace_epoch to the destination fencing generation."""

    workspace_epoch: int


@dataclass(frozen=True, slots=True)
class HandoffConflict:
    """The expected workspace_epoch no longer matches the stored value."""

    expected_epoch: int | None
    current_epoch: int | None


@dataclass(frozen=True, slots=True)
class HandoffLeaseLost:
    """The caller no longer holds the live lease."""


type HandoffResult = HandoffCommit | HandoffConflict | HandoffLeaseLost
type InventoryReplaceResult = Literal["committed", "lease_lost"]


class WorkspaceStore(Protocol):
    """Fenced workspace metadata. Handoff never advances durable progress."""

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult: ...

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]: ...

    async def replace_inventory(
        self, records: Sequence[InventoryPathRecord]
    ) -> InventoryReplaceResult: ...

    async def register_spill(self, spill: CommittedSpillRecord) -> InventoryReplaceResult: ...

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]: ...

    async def clear_spills(self) -> InventoryReplaceResult: ...


class InMemoryWorkspaceStore:
    """Process-local workspace store for unit tests."""

    def __init__(
        self,
        *,
        workspace_epoch: int | None = None,
        live: bool = True,
        progress_version: int = 0,
    ) -> None:
        self.workspace_epoch = workspace_epoch
        self.live = live
        self.progress_version = progress_version
        self.inventory: list[InventoryPathRecord] = []
        self.spills: list[CommittedSpillRecord] = []

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult:
        if not self.live:
            return HandoffLeaseLost()
        if self.workspace_epoch != expected_epoch:
            return HandoffConflict(
                expected_epoch=expected_epoch, current_epoch=self.workspace_epoch
            )
        if destination_epoch < 1:
            raise ValueError("destination epoch must be positive")
        self.workspace_epoch = destination_epoch
        self.inventory = list(inventory)
        return HandoffCommit(workspace_epoch=destination_epoch)

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]:
        return tuple(self.inventory)

    async def replace_inventory(
        self, records: Sequence[InventoryPathRecord]
    ) -> InventoryReplaceResult:
        if not self.live:
            return "lease_lost"
        self.inventory = list(records)
        return "committed"

    async def register_spill(self, spill: CommittedSpillRecord) -> InventoryReplaceResult:
        if not self.live:
            return "lease_lost"
        self.spills = [item for item in self.spills if item.resource_id != spill.resource_id]
        self.spills.append(spill)
        return "committed"

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        _validate_spill_page_limit(limit)
        matching = (
            spill
            for spill in self.spills
            if after_resource_id is None or spill.resource_id > after_resource_id
        )
        return tuple(nsmallest(limit, matching, key=lambda spill: spill.resource_id))

    async def clear_spills(self) -> InventoryReplaceResult:
        if not self.live:
            return "lease_lost"
        self.spills = []
        return "committed"


_MAX_SPILL_PAGE_LIMIT = 1_000


def _validate_spill_page_limit(limit: int) -> None:
    if limit < 1 or limit > _MAX_SPILL_PAGE_LIMIT:
        raise ValueError(f"spill page limit must be between 1 and {_MAX_SPILL_PAGE_LIMIT}")


__all__ = [
    "CommittedSpillRecord",
    "HandoffCommit",
    "HandoffConflict",
    "HandoffLeaseLost",
    "HandoffResult",
    "InMemoryWorkspaceStore",
    "InventoryReplaceResult",
    "WorkspaceStore",
]
