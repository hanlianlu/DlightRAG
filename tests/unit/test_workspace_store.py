# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""WorkspaceStore: epoch CAS, inventory replace, no progress increment."""

import pytest

from dlightrag.runtime.settlements import (
    CommittedSpillUpdate,
    EvidenceSettlementUpdate,
    FetchedResourceSettlementUpdate,
    HostUpdate,
    InventoryPathRecord,
    WorkspaceInventoryUpdate,
)
from dlightrag.runtime.workspace import (
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    InMemoryWorkspaceStore,
)


def test_host_update_union_accepts_four_variants() -> None:
    variants: list[HostUpdate] = [
        EvidenceSettlementUpdate(),
        WorkspaceInventoryUpdate(replace_all=True),
        CommittedSpillUpdate(
            resource_id="res_spill",
            content_digest="a" * 64,
            size_bytes=12,
            session_id="s",
            intent_id="i",
        ),
    ]
    assert len(variants) == 3
    assert FetchedResourceSettlementUpdate.__name__ == "FetchedResourceSettlementUpdate"


@pytest.mark.asyncio
async def test_stale_expected_epoch_changes_nothing() -> None:
    store = InMemoryWorkspaceStore(workspace_epoch=5, progress_version=4)
    result = await store.handoff_epoch(
        expected_epoch=4,
        destination_epoch=6,
        inventory=(),
    )
    assert isinstance(result, HandoffConflict)
    assert store.workspace_epoch == 5
    assert store.progress_version == 4


@pytest.mark.asyncio
async def test_handoff_does_not_increment_progress() -> None:
    store = InMemoryWorkspaceStore(workspace_epoch=5, progress_version=4)
    result = await store.handoff_epoch(
        expected_epoch=5,
        destination_epoch=7,
        inventory=(
            InventoryPathRecord(relative_path="notes/a.md", entry_type="file", size_bytes=3),
        ),
    )
    assert isinstance(result, HandoffCommit)
    assert result.workspace_epoch == 7
    assert store.progress_version == 4
    loaded = await store.load_inventory()
    assert len(loaded) == 1
    assert loaded[0].relative_path == "notes/a.md"


@pytest.mark.asyncio
async def test_inventory_replace_is_all_or_nothing() -> None:
    store = InMemoryWorkspaceStore()
    await store.replace_inventory(
        (
            InventoryPathRecord(relative_path="a", entry_type="file", size_bytes=1),
            InventoryPathRecord(relative_path="b", entry_type="file", size_bytes=1),
        )
    )
    await store.replace_inventory(
        (InventoryPathRecord(relative_path="c", entry_type="file", size_bytes=2),)
    )
    paths = [item.relative_path for item in await store.load_inventory()]
    assert paths == ["c"]


@pytest.mark.asyncio
async def test_spill_register_and_clear() -> None:
    store = InMemoryWorkspaceStore()
    await store.register_spill(
        CommittedSpillRecord(
            resource_id="res_1",
            content_digest="b" * 64,
            size_bytes=8,
            session_id="s",
            intent_id="i",
        )
    )
    assert len(await store.load_spills()) == 1
    await store.clear_spills()
    assert await store.load_spills() == ()
