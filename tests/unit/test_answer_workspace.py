# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Epoch workspace bind creates a rooted environment and can copy a prior epoch."""

import hashlib
from pathlib import Path

import pytest

from dlightrag.engine.answer import workspace as workspace_module
from dlightrag.engine.answer.workspace import (
    WorkspaceIntegrityError,
    bind_run_workspace,
    copy_epoch_verified,
    epoch_paths,
    write_spill_file,
)
from dlightrag.engine.runtime.workspace import CommittedSpillRecord, InMemoryWorkspaceStore


class RecordingWorkspaceStore(InMemoryWorkspaceStore):
    def __init__(self) -> None:
        super().__init__()
        self.page_calls: list[tuple[str | None, int]] = []
        self.returned_page_sizes: list[int] = []
        self.returned_resource_ids: list[str] = []

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        page = await super().load_spills_page(after_resource_id=after_resource_id, limit=limit)
        self.page_calls.append((after_resource_id, limit))
        self.returned_page_sizes.append(len(page))
        self.returned_resource_ids.extend(spill.resource_id for spill in page)
        return page


class BrokenPageWorkspaceStore(InMemoryWorkspaceStore):
    def __init__(self, page: tuple[CommittedSpillRecord, ...]) -> None:
        super().__init__()
        self.page = page

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        return self.page


class FailingPageWorkspaceStore(InMemoryWorkspaceStore):
    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        raise RuntimeError("page fetch failed")


def _spill(resource_id: str, content: bytes) -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        session_id="session",
        intent_id="intent",
    )


async def _seed_spills(
    root: Path, store: InMemoryWorkspaceStore, resource_ids: list[str]
) -> dict[str, bytes]:
    _, spill_dir = epoch_paths(root, 1)
    spill_dir.mkdir(parents=True, exist_ok=True)
    contents: dict[str, bytes] = {}
    for resource_id in resource_ids:
        content = f"content for {resource_id}".encode()
        contents[resource_id] = content
        (spill_dir / f"{resource_id}.txt").write_bytes(content)
        await store.register_spill(_spill(resource_id, content))
    return contents


def _assert_no_temp_recovery_tree(root: Path, destination: int) -> None:
    epochs = root / "epochs"
    assert not (epochs / str(destination)).exists()
    assert list(epochs.glob(f".tmp-{destination}-*")) == []


@pytest.mark.asyncio
async def test_first_bind_creates_workspace_and_handoffs(tmp_path: Path) -> None:
    store = InMemoryWorkspaceStore()
    bound = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-1",
        fencing_epoch=3,
        recorded_epoch=None,
        store=store,
    )
    assert bound.epoch == 3
    assert (bound.workspace / "artifacts").is_dir()
    assert store.workspace_epoch == 3
    write_spill_file(bound.spill_dir, "res_1", "overflow")
    assert (bound.spill_dir / "res_1.txt").read_text(encoding="utf-8") == "overflow"


@pytest.mark.asyncio
async def test_recover_copies_prior_epoch(tmp_path: Path) -> None:
    store = InMemoryWorkspaceStore(workspace_epoch=1)
    first = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-1",
        fencing_epoch=1,
        recorded_epoch=1,
        store=store,
    )
    (first.workspace / "notes.txt").write_text("keep", encoding="utf-8")
    recovered = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-1",
        fencing_epoch=2,
        recorded_epoch=1,
        store=store,
    )
    assert recovered.epoch == 2
    assert (recovered.workspace / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert store.workspace_epoch == 2
    assert first.workspace.exists() is False or recovered.workspace != first.workspace


@pytest.mark.asyncio
async def test_recover_rejects_a_symlink_as_integrity_error(tmp_path: Path) -> None:
    store = InMemoryWorkspaceStore(workspace_epoch=1)
    first = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-2",
        fencing_epoch=1,
        recorded_epoch=1,
        store=store,
    )
    (first.workspace / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(WorkspaceIntegrityError):
        await bind_run_workspace(
            workspace_root=tmp_path,
            owner_id="owner",
            run_id="run-2",
            fencing_epoch=2,
            recorded_epoch=1,
            store=store,
        )


@pytest.mark.asyncio
async def test_recovery_pages_spills_in_keyset_order_without_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_module, "_SPILL_RECOVERY_PAGE_SIZE", 2)
    root = tmp_path / "run"
    store = RecordingWorkspaceStore()
    resource_ids = [f"spill-{index:02d}" for index in reversed(range(7))]
    contents = await _seed_spills(root, store, resource_ids)

    await copy_epoch_verified(root, 1, 2, store)

    _, destination_spills = epoch_paths(root, 2)
    for resource_id, expected in contents.items():
        copied = (destination_spills / f"{resource_id}.txt").read_bytes()
        assert copied == expected
        assert hashlib.sha256(copied).hexdigest() == hashlib.sha256(expected).hexdigest()
    assert store.page_calls == [
        (None, 2),
        ("spill-01", 2),
        ("spill-03", 2),
        ("spill-05", 2),
    ]
    assert store.returned_page_sizes == [2, 2, 2, 1]
    assert max(store.returned_page_sizes) == 2
    assert store.returned_resource_ids == sorted(contents)
    assert len(store.returned_resource_ids) == len(set(store.returned_resource_ids))


@pytest.mark.asyncio
async def test_recovery_exact_page_multiple_fetches_one_final_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_module, "_SPILL_RECOVERY_PAGE_SIZE", 2)
    root = tmp_path / "run"
    store = RecordingWorkspaceStore()
    await _seed_spills(root, store, ["spill-03", "spill-01", "spill-02", "spill-00"])

    await copy_epoch_verified(root, 1, 2, store)

    assert store.returned_page_sizes == [2, 2, 0]
    assert store.page_calls[-1] == ("spill-03", 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource_ids",
    [("spill-02", "spill-01"), ("spill-01", "spill-01")],
    ids=["non-monotonic", "duplicate"],
)
async def test_recovery_rejects_broken_spill_page_and_removes_temp_tree(
    tmp_path: Path, resource_ids: tuple[str, str]
) -> None:
    root = tmp_path / "run"
    records = tuple(_spill(resource_id, resource_id.encode()) for resource_id in resource_ids)
    store = BrokenPageWorkspaceStore(records)
    _, source_spills = epoch_paths(root, 1)
    source_spills.mkdir(parents=True)
    for record in records:
        (source_spills / f"{record.resource_id}.txt").write_bytes(record.resource_id.encode())

    with pytest.raises(WorkspaceIntegrityError, match="strictly ordered"):
        await copy_epoch_verified(root, 1, 2, store)

    _assert_no_temp_recovery_tree(root, 2)


@pytest.mark.asyncio
async def test_recovery_page_failure_removes_temp_tree(tmp_path: Path) -> None:
    root = tmp_path / "run"

    with pytest.raises(RuntimeError, match="page fetch failed"):
        await copy_epoch_verified(root, 1, 2, FailingPageWorkspaceStore())

    _assert_no_temp_recovery_tree(root, 2)


@pytest.mark.asyncio
async def test_recovery_rejects_corrupt_spill_and_removes_temp_tree(tmp_path: Path) -> None:
    root = tmp_path / "run"
    store = InMemoryWorkspaceStore()
    await _seed_spills(root, store, ["spill-01"])
    _, source_spills = epoch_paths(root, 1)
    (source_spills / "spill-01.txt").write_text("corrupt", encoding="utf-8")

    with pytest.raises(WorkspaceIntegrityError, match="failed digest check"):
        await copy_epoch_verified(root, 1, 2, store)

    _assert_no_temp_recovery_tree(root, 2)
