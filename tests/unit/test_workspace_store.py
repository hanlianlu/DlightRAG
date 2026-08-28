# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""WorkspaceStore: epoch CAS, inventory replace, no progress increment."""

import pytest

from dlightrag.engine.runtime.settlements import (
    CommittedSpillUpdate,
    EffectHostUpdate,
    InventoryPathRecord,
    WorkspaceInventoryUpdate,
)
from dlightrag.engine.runtime.workspace import (
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    InMemoryWorkspaceStore,
)


def test_host_update_aggregates_spill_and_inventory() -> None:
    update = EffectHostUpdate(
        committed_outputs=(
            CommittedSpillUpdate(
                resource_id="res_spill",
                content_digest="a" * 64,
                size_bytes=12,
                session_id="s",
                intent_id="i",
            ),
        ),
        workspace_inventory=WorkspaceInventoryUpdate(replace_all=True),
    )
    assert update.committed_outputs[0].resource_id == "res_spill"
    assert update.workspace_inventory is not None
    assert update.workspace_inventory.replace_all is True


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


def _spill(resource_id: str) -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest="b" * 64,
        size_bytes=8,
        session_id="s",
        intent_id="i",
    )


@pytest.mark.asyncio
async def test_spill_pages_are_ordered_and_use_an_exclusive_cursor() -> None:
    store = InMemoryWorkspaceStore()
    for resource_id in ("res_3", "res_1", "res_4", "res_2"):
        await store.register_spill(_spill(resource_id))

    first = await store.load_spills_page(after_resource_id=None, limit=2)
    second = await store.load_spills_page(after_resource_id="res_2", limit=2)
    empty = await store.load_spills_page(after_resource_id="res_4", limit=2)

    assert [spill.resource_id for spill in first] == ["res_1", "res_2"]
    assert [spill.resource_id for spill in second] == ["res_3", "res_4"]
    assert empty == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1_001])
async def test_spill_page_rejects_invalid_limit(limit: int) -> None:
    store = InMemoryWorkspaceStore()
    with pytest.raises(ValueError, match="spill page limit"):
        await store.load_spills_page(after_resource_id=None, limit=limit)


@pytest.mark.asyncio
async def test_spill_register_and_clear() -> None:
    store = InMemoryWorkspaceStore()
    await store.register_spill(_spill("res_1"))
    assert len(await store.load_spills_page(after_resource_id=None, limit=1)) == 1
    await store.clear_spills()
    assert await store.load_spills_page(after_resource_id=None, limit=1) == ()
