# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""WorkspaceStore: epoch CAS without progress, spill reads, the PostgreSQL inventory write."""

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


def _spill(resource_id: str, *, intent_id: str = "i") -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest="b" * 64,
        size_bytes=8,
        session_id="s",
        intent_id=intent_id,
    )


@pytest.mark.asyncio
async def test_spill_pages_are_ordered_and_use_an_exclusive_cursor() -> None:
    store = InMemoryWorkspaceStore()
    store.spills = [_spill(resource_id) for resource_id in ("res_3", "res_1", "res_4", "res_2")]

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
async def test_recent_spills_are_newest_first_and_bounded() -> None:
    """The producing effect intent is the only monotone marker a spill carries.

    A spill resource id is a random handle and the row records no settlement
    time, so newest-first comes from the UUIDv7 intent, not from the page cursor
    the epoch-copy recovery read uses. The ids sort against the intents on
    purpose: fixtures where both keys agree cannot tell the two orders apart.
    """
    store = InMemoryWorkspaceStore()
    store.spills = [
        _spill(resource_id, intent_id=_intent(ordinal))
        for resource_id, ordinal in (("res_z", 1), ("res_m", 2), ("res_a", 3))
    ]

    newest = await store.load_recent_spills(limit=2)

    assert [spill.resource_id for spill in newest] == ["res_a", "res_m"]


def _intent(ordinal: int) -> str:
    return f"01930000-0000-7000-8000-00000000000{ordinal}"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1_001])
async def test_recent_spills_reject_invalid_limit(limit: int) -> None:
    store = InMemoryWorkspaceStore()
    with pytest.raises(ValueError, match="spill page limit"):
        await store.load_recent_spills(limit=limit)


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> str:
        self.statements.append((query, args))
        return "OK"


def _file(path: str, size: int = 1) -> InventoryPathRecord:
    return InventoryPathRecord(relative_path=path, entry_type="file", size_bytes=size)


@pytest.mark.asyncio
async def test_pg_inventory_rescan_is_two_statements_whatever_the_file_count() -> None:
    """A rescan runs under the Run lock, so its write must not grow per file."""
    import uuid

    from dlightrag.adapters.postgres.answer.workspace import write_inventory

    conn = _RecordingConnection()
    records = tuple(_file(f"src/file_{index}.py", index) for index in range(2_000))

    await write_inventory(conn, "owner", uuid.uuid4(), upserts=records, replace_all=True)

    assert len(conn.statements) == 2
    (removal, removal_args), (upsert, upsert_args) = conn.statements
    assert "NOT EXISTS" in removal and "unnest($3::text[])" in removal
    assert removal_args[2] == [record.relative_path for record in records]
    assert "FROM unnest(" in upsert and "IS DISTINCT FROM" in upsert
    assert upsert_args[2] == [record.relative_path for record in records]


@pytest.mark.asyncio
async def test_pg_inventory_delta_deletes_then_keeps_the_last_observation() -> None:
    import uuid

    from dlightrag.adapters.postgres.answer.workspace import write_inventory

    conn = _RecordingConnection()

    await write_inventory(
        conn,
        "owner",
        uuid.uuid4(),
        upserts=(_file("a.md", 1), _file("a.md", 7), _file("b.md", 2)),
        deletes=("gone.md", "a.md"),
    )

    (removal, removal_args), (_upsert, upsert_args) = conn.statements
    assert "relative_path = ANY($3::text[])" in removal
    assert removal_args[2] == ["gone.md", "a.md"]
    assert upsert_args[2] == ["a.md", "b.md"]
    assert upsert_args[5] == [7, 2]
