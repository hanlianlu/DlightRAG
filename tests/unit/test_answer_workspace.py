# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Epoch workspace bind creates a rooted environment and can copy a prior epoch."""

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.answer import workspace as workspace_module
from dlightrag.engine.answer.continuation_handles import MAX_CARRIED_RUN_NOTE_BYTES
from dlightrag.engine.answer.workspace import (
    AgentWorkspaceReclaimer,
    WorkspaceIntegrityError,
    active_epoch_workspace,
    agent_workspace_reclaimer,
    bind_run_workspace,
    carry_run_notes,
    copy_epoch_verified,
    epoch_paths,
    owner_shard,
    reclaim_discovered_run_root,
    reclaim_run_workspace,
    run_root,
    write_spill_file,
)
from dlightrag.engine.runtime.records import DeletedRun
from dlightrag.engine.runtime.settlements import InventoryPathRecord
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
async def test_first_bind_reclaims_only_exact_older_claim_local_temps(tmp_path: Path) -> None:
    root = run_root(tmp_path, "owner", "run-temps")
    epochs = root / "epochs"
    epochs.mkdir(parents=True)
    stale_names = [
        ".tmp-1-00000000000000000000000000000000",
        ".tmp-2-abcdefabcdefabcdefabcdefabcdefab",
    ]
    for name in stale_names:
        nested = epochs / name / "nested" / "deeper"
        nested.mkdir(parents=True)
        (nested / "orphan.txt").write_text("remove", encoding="utf-8")

    preserved_names = [
        ".tmp-3-11111111111111111111111111111111",
        ".tmp-4-22222222222222222222222222222222",
        ".tmp-0-33333333333333333333333333333333",
        ".tmp-01-44444444444444444444444444444444",
        ".tmp-1-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ".tmp-1-short",
        ".tmp-other",
        ".unrelated",
    ]
    # A numbered epoch below this attempt is residue: the Run's row records no epoch,
    # so an interrupted bind left it, and fencing generations only grow.
    residue_name = "1"
    for name in preserved_names:
        preserved = epochs / name
        preserved.mkdir()
        (preserved / "keep.txt").write_text("keep", encoding="utf-8")

    residue = epochs / residue_name
    residue.mkdir()
    (residue / "keep.txt").write_text("residue", encoding="utf-8")

    other_root = run_root(tmp_path, "owner", "other-run")
    other_temp = other_root / "epochs" / ".tmp-1-55555555555555555555555555555555"
    other_temp.mkdir(parents=True)
    (other_temp / "outside-claim.txt").write_text("keep", encoding="utf-8")

    await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-temps",
        fencing_epoch=3,
        recorded_epoch=None,
        store=InMemoryWorkspaceStore(),
    )

    assert all(not (epochs / name).exists() for name in stale_names)
    assert not (epochs / residue_name).exists()
    assert all(
        (epochs / name / "keep.txt").read_text(encoding="utf-8") == "keep"
        for name in preserved_names
    )
    assert other_temp.is_dir()
    assert (other_temp / "outside-claim.txt").read_text(encoding="utf-8") == "keep"


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
    stale_temp = (
        run_root(tmp_path, "owner", "run-1") / "epochs" / ".tmp-1-66666666666666666666666666666666"
    )
    (stale_temp / "nested").mkdir(parents=True)
    (stale_temp / "nested" / "orphan.txt").write_text("remove", encoding="utf-8")
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
    assert not stale_temp.exists()
    assert first.workspace.exists() is False or recovered.workspace != first.workspace


@pytest.mark.asyncio
async def test_stale_temp_symlinks_are_unlinked_without_touching_their_targets(
    tmp_path: Path,
) -> None:
    root = run_root(tmp_path, "owner", "run-symlink-temp")
    epochs = root / "epochs"
    epochs.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    stale_link = epochs / ".tmp-1-77777777777777777777777777777777"
    stale_link.symlink_to(outside, target_is_directory=True)
    stale_tree = epochs / ".tmp-2-88888888888888888888888888888888"
    stale_tree.mkdir()
    (stale_tree / "outside-link").symlink_to(outside, target_is_directory=True)

    await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-symlink-temp",
        fencing_epoch=3,
        recorded_epoch=None,
        store=InMemoryWorkspaceStore(),
    )

    assert not stale_link.exists()
    assert not stale_link.is_symlink()
    assert not stale_tree.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_stale_temp_cleanup_is_best_effort_per_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = run_root(tmp_path, "owner", "run-cleanup-failure")
    epochs = root / "epochs"
    failed = epochs / ".tmp-1-99999999999999999999999999999999"
    removable = epochs / ".tmp-2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    (failed / "nested").mkdir(parents=True)
    removable.mkdir(parents=True)
    outside = tmp_path / "outside-cleanup-failure"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    real_rmtree = shutil.rmtree

    def fail_one_tree(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == failed:
            raise PermissionError("simulated undeletable temp")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(workspace_module.shutil, "rmtree", fail_one_tree)
    with caplog.at_level(logging.WARNING, logger=workspace_module.__name__):
        bound = await bind_run_workspace(
            workspace_root=tmp_path,
            owner_id="owner",
            run_id="run-cleanup-failure",
            fencing_epoch=3,
            recorded_epoch=None,
            store=InMemoryWorkspaceStore(),
        )

    assert bound.workspace.is_dir()
    assert failed.is_dir()
    assert not removable.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert "Failed to reclaim stale epoch-copy temp" in caplog.text


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


@pytest.mark.asyncio
async def test_handoff_records_the_copied_epochs_observation(tmp_path: Path) -> None:
    """A verified copy is an observation, not an unknown.

    The Run Note set is a filter over the Workspace Inventory, so a handoff that
    replaced the table with nothing left every note unnamed after a crash-recovery
    claim — the bytes copied, the names gone — while the copy itself had just
    proven every path, size, and digest it was throwing away.
    """
    store = InMemoryWorkspaceStore()
    first = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-notes",
        fencing_epoch=1,
        recorded_epoch=None,
        store=store,
    )
    payload = "decided: keep the spill handles\n"
    note = first.workspace / "notes" / "plan.md"
    note.parent.mkdir()
    note.write_text(payload, encoding="utf-8")

    recovered = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id="run-notes",
        fencing_epoch=2,
        recorded_epoch=1,
        store=store,
    )

    assert recovered.epoch == 2
    observed = await store.load_inventory()
    assert [(item.relative_path, item.entry_type, item.size_bytes) for item in observed] == [
        ("notes/plan.md", "file", len(payload.encode("utf-8")))
    ]
    assert observed[0].content_digest == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # The retiring epoch is gone and the note is not: the observation describes the copy.
    assert not (note).exists()
    assert (recovered.workspace / "notes" / "plan.md").read_text(encoding="utf-8") == payload


def test_reclaim_removes_a_run_root_is_idempotent_and_does_not_follow_symlinks(
    tmp_path: Path,
) -> None:
    owner = "owner"
    run_id = str(uuid.uuid4())
    root = run_root(tmp_path, owner, run_id)
    inside = root / "epochs" / "1" / "workspace"
    inside.mkdir(parents=True)
    (inside / "note.txt").write_text("gone", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("keep", encoding="utf-8")
    (inside / "escape").symlink_to(outside)

    reclaim_run_workspace(tmp_path, owner, run_id)
    assert not root.exists()
    assert outside.read_text(encoding="utf-8") == "keep"

    reclaim_run_workspace(tmp_path, owner, run_id)
    assert not root.exists()
    assert outside.read_text(encoding="utf-8") == "keep"


def test_reclaim_unlinks_a_run_root_symlink_without_following_it(tmp_path: Path) -> None:
    owner = "owner"
    run_id = str(uuid.uuid4())
    expected = run_root(tmp_path, owner, run_id)
    expected.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("keep", encoding="utf-8")
    expected.symlink_to(outside)

    reclaim_run_workspace(tmp_path, owner, run_id)

    assert not expected.exists()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "keep"


def test_reclaim_refuses_a_path_that_is_not_the_expected_run_root(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceIntegrityError, match="not a run root"):
        reclaim_run_workspace(tmp_path, "owner", "../not-a-run")
    with pytest.raises(WorkspaceIntegrityError, match="not a run root"):
        reclaim_run_workspace(tmp_path, "owner", "not-a-uuid")


@pytest.mark.asyncio
async def test_orphan_sweep_deletes_a_directory_with_no_run_row_and_keeps_one_whose_row_exists(
    tmp_path: Path,
) -> None:
    owner = "owner"
    live_id = str(uuid.uuid4())
    dead_id = str(uuid.uuid4())
    live = run_root(tmp_path, owner, live_id)
    dead = run_root(tmp_path, owner, dead_id)
    for path in (live, dead):
        path.mkdir(parents=True)
        (path / "marker.txt").write_text("x", encoding="utf-8")

    class _Store:
        async def get_run_global(self, *, run_id: str) -> object | None:
            if run_id == live_id:
                return object()
            return None

    removed = await AgentWorkspaceReclaimer(tmp_path, page_size=1).sweep_orphans(_Store())

    assert removed == 1
    assert live.exists()
    assert not dead.exists()


@pytest.mark.asyncio
async def test_reclaim_skips_non_answer_runs(tmp_path: Path) -> None:
    owner = "owner"
    run_id = str(uuid.uuid4())
    root = run_root(tmp_path, owner, run_id)
    root.mkdir(parents=True)
    (root / "keep.txt").write_text("x", encoding="utf-8")
    reclaimer = AgentWorkspaceReclaimer(tmp_path)
    await reclaimer.reclaim((DeletedRun(owner_id=owner, run_id=run_id, run_kind="retrieval"),))
    assert root.exists()


def test_a_configured_root_still_builds_a_reclaimer_when_execution_is_disabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    reclaimer = agent_workspace_reclaimer(
        execution_environment="disabled",
        workspace_root=str(root),
    )
    assert reclaimer is not None
    assert reclaimer._root == root.resolve()
    assert not root.exists()


def test_no_workspace_root_produces_no_reclaimer() -> None:
    assert (
        agent_workspace_reclaimer(
            execution_environment="disabled",
            workspace_root=None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_the_production_reclaimer_removes_an_answer_runs_bytes(tmp_path: Path) -> None:
    """The claim is that a pruned Run's workspace is gone, not that a fake was asked.

    `reclaim` is the collaborator the retention prune drives, so it is the thing
    that has to delete: a version of it that only skipped non-answer runs would
    satisfy every coordinator test while leaving every working tree behind.
    """
    owner = "owner"
    answer_id = str(uuid.uuid4())
    retrieval_id = str(uuid.uuid4())
    for run_id in (answer_id, retrieval_id):
        root = run_root(tmp_path, owner, run_id)
        root.mkdir(parents=True)
        (root / "note.txt").write_text("x", encoding="utf-8")

    await AgentWorkspaceReclaimer(tmp_path).reclaim(
        (
            DeletedRun(owner_id=owner, run_id=answer_id, run_kind="answer"),
            DeletedRun(owner_id=owner, run_id=retrieval_id, run_kind="retrieval"),
        )
    )

    assert not run_root(tmp_path, owner, answer_id).exists()
    assert run_root(tmp_path, owner, retrieval_id).exists()


@pytest.mark.asyncio
async def test_a_failed_reclaim_is_retried_by_the_next_orphan_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree that survived one failure must not survive the pass that follows it."""
    owner = "owner"
    run_id = str(uuid.uuid4())
    root = run_root(tmp_path, owner, run_id)
    root.mkdir(parents=True)
    (root / "note.txt").write_text("x", encoding="utf-8")
    real_rmtree = shutil.rmtree

    def fail_this_tree(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == root:
            raise PermissionError("simulated undeletable workspace")
        real_rmtree(path, *args, **kwargs)

    class _Store:
        async def get_run_global(self, *, run_id: str) -> object | None:
            return None

    reclaimer = AgentWorkspaceReclaimer(tmp_path)
    monkeypatch.setattr(workspace_module.shutil, "rmtree", fail_this_tree)
    await reclaimer.reclaim((DeletedRun(owner_id=owner, run_id=run_id, run_kind="answer"),))
    assert root.exists()

    monkeypatch.setattr(workspace_module.shutil, "rmtree", real_rmtree)
    assert await reclaimer.sweep_orphans(_Store()) == 1
    assert not root.exists()


@pytest.mark.asyncio
async def test_the_sweep_walks_every_page_of_orphans(tmp_path: Path) -> None:
    """A bounded page that stops after the first would leave the rest forever."""
    owner = "owner"
    run_ids = [str(uuid.uuid4()) for _ in range(3)]
    for run_id in run_ids:
        root = run_root(tmp_path, owner, run_id)
        root.mkdir(parents=True)
        (root / "note.txt").write_text("x", encoding="utf-8")

    class _Store:
        async def get_run_global(self, *, run_id: str) -> object | None:
            return None

    removed = await AgentWorkspaceReclaimer(tmp_path, page_size=1).sweep_orphans(_Store())

    assert removed == 3
    assert all(not run_root(tmp_path, owner, run_id).exists() for run_id in run_ids)


def test_reclaim_refuses_a_path_outside_the_shard_layout(tmp_path: Path) -> None:
    """The discovered-path jail is the check that keeps a sweep inside the root."""
    outside = tmp_path / "outside"
    run_id = str(uuid.uuid4())
    outside.mkdir()
    (outside / run_id).mkdir()

    with pytest.raises(WorkspaceIntegrityError, match="not a run root"):
        reclaim_discovered_run_root(tmp_path, outside / run_id)
    with pytest.raises(WorkspaceIntegrityError, match="not a run root"):
        reclaim_discovered_run_root(tmp_path, tmp_path / "zz" / run_id)
    assert (outside / run_id).exists()


def test_reclaim_never_follows_a_symlinked_shard(tmp_path: Path) -> None:
    """A shard is two hex characters of path; following one would leave the root."""
    owner = "owner"
    run_id = str(uuid.uuid4())
    outside = tmp_path / "outside" / owner_shard(owner)
    (outside / run_id).mkdir(parents=True)
    (outside / run_id / "keep.txt").write_text("keep", encoding="utf-8")
    shard = tmp_path / owner_shard(owner)
    shard.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceIntegrityError, match="not a run root"):
        reclaim_run_workspace(tmp_path, owner, run_id)

    assert (outside / run_id / "keep.txt").read_text(encoding="utf-8") == "keep"


def _note(
    relative_path: str, content: bytes, *, digest: str | None = ""
) -> tuple[InventoryPathRecord, bytes]:
    recorded = hashlib.sha256(content).hexdigest() if digest == "" else digest
    return (
        InventoryPathRecord(
            relative_path=relative_path,
            entry_type="file",
            size_bytes=len(content),
            content_digest=recorded,
        ),
        content,
    )


def _parent_workspace(tmp_path: Path, owner: str, run_id: str, epoch: int = 1) -> Path:
    root = run_root(tmp_path, owner, run_id)
    workspace, _ = epoch_paths(root, epoch)
    workspace.mkdir(parents=True)
    return workspace


def test_carry_run_notes_copies_bytes_and_records_the_destination_digest(tmp_path: Path) -> None:
    parent = _parent_workspace(tmp_path, "owner", str(uuid.uuid4()))
    record, content = _note("notes/plan.md", b"the numbers are 4 and 9")
    (parent / "notes").mkdir()
    (parent / "notes" / "plan.md").write_bytes(content)
    dest = tmp_path / "child"
    dest.mkdir()

    copied = carry_run_notes(source_workspace=parent, destination_workspace=dest, notes=(record,))

    assert (dest / "notes" / "plan.md").read_bytes() == content
    assert copied == (
        InventoryPathRecord(
            relative_path="notes/plan.md",
            entry_type="file",
            size_bytes=len(content),
            content_digest=hashlib.sha256(content).hexdigest(),
        ),
    )


@pytest.mark.asyncio
async def test_a_chain_of_two_carries_accumulates_the_file_itself(
    tmp_path: Path,
) -> None:
    """The unit that accumulates is the file, and each hop is a real bind.

    This pins the bind/Inventory chain: three production binds with real Inventory and
    digest checks. That the middle hop is a *Fast execute* is pinned separately by
    `tests/unit/test_answer_executor.py::test_a_fast_continuation_binds_the_parent_note_into_its_own_epoch`.

    A chain that carried the inherited note but dropped the one the first
    continuation wrote would pass a test that only exercised the copy helper, so
    this drives two bindings and reads each hop's own Inventory back.
    """
    from dlightrag.engine.answer.continuation_handles import select_carried_run_notes

    owner = "owner"
    parent_id = "01930000-0000-7000-8000-0000000000a1"
    child_id = "01930000-0000-7000-8000-0000000000a2"
    grandchild_id = "01930000-0000-7000-8000-0000000000a3"
    store = InMemoryWorkspaceStore()
    parent = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=parent_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=store,
    )
    (parent.workspace / "notes").mkdir()
    (parent.workspace / "notes" / "plan.md").write_text("inherited", encoding="utf-8")
    await store.replace_inventory(_note_records(parent.workspace, ("notes/plan.md",)))

    child_store = InMemoryWorkspaceStore()
    child = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=child_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=child_store,
        carried_notes=select_carried_run_notes(await store.load_inventory()),
        carry_source=parent.workspace,
    )
    # The continuation writes a note of its own, observed the way a write settlement does.
    (child.workspace / "notes" / "findings.md").write_text("written", encoding="utf-8")
    await child_store.replace_inventory(
        (
            *await child_store.load_inventory(),
            *_note_records(child.workspace, ("notes/findings.md",)),
        )
    )

    grandchild_store = InMemoryWorkspaceStore()
    grandchild = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=grandchild_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=grandchild_store,
        carried_notes=select_carried_run_notes(await child_store.load_inventory()),
        carry_source=child.workspace,
    )

    assert (grandchild.workspace / "notes" / "plan.md").read_text(encoding="utf-8") == "inherited"
    assert (grandchild.workspace / "notes" / "findings.md").read_text(encoding="utf-8") == "written"
    assert [item.relative_path for item in await grandchild_store.load_inventory()] == [
        "notes/findings.md",
        "notes/plan.md",
    ]


@pytest.mark.asyncio
async def test_the_bind_chain_carries_the_note_through_a_middle_hop(tmp_path: Path) -> None:
    """Research writes a note, Fast binds it without writing, Research reads it back.

    Fast composes no tools, so it cannot rewrite the file; it only holds the
    Inventory naming it so the next Research turn can carry it again. A hop that
    bound an empty workspace would pass the two-Research chain and fail this one.
    """
    from dlightrag.engine.answer.continuation_handles import select_carried_run_notes

    owner = "owner"
    parent_id = "01930000-0000-7000-8000-0000000000b1"
    fast_id = "01930000-0000-7000-8000-0000000000b2"
    grandchild_id = "01930000-0000-7000-8000-0000000000b3"
    parent_store = InMemoryWorkspaceStore()
    parent = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=parent_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=parent_store,
    )
    (parent.workspace / "notes").mkdir()
    payload = b"the error was ECONNRESET on shard 4"
    (parent.workspace / "notes" / "plan.md").write_bytes(payload)
    await parent_store.replace_inventory(_note_records(parent.workspace, ("notes/plan.md",)))
    parent_digest = hashlib.sha256(payload).hexdigest()

    fast_store = InMemoryWorkspaceStore()
    fast = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=fast_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=fast_store,
        carried_notes=select_carried_run_notes(await parent_store.load_inventory()),
        carry_source=parent.workspace,
    )
    fast_inventory = await fast_store.load_inventory()
    assert [item.relative_path for item in fast_inventory] == ["notes/plan.md"]
    assert fast_inventory[0].content_digest == parent_digest
    assert (fast.workspace / "notes" / "plan.md").read_bytes() == payload

    grandchild_store = InMemoryWorkspaceStore()
    grandchild = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner,
        run_id=grandchild_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=grandchild_store,
        carried_notes=select_carried_run_notes(fast_inventory),
        carry_source=fast.workspace,
    )
    grandchild_inventory = await grandchild_store.load_inventory()
    assert [item.relative_path for item in grandchild_inventory] == ["notes/plan.md"]
    assert grandchild_inventory[0].content_digest == parent_digest
    assert (grandchild.workspace / "notes" / "plan.md").read_bytes() == payload


def _note_records(workspace: Path, relative_paths: tuple[str, ...]) -> list[InventoryPathRecord]:
    records: list[InventoryPathRecord] = []
    for relative_path in relative_paths:
        data = (workspace / relative_path).read_bytes()
        records.append(
            InventoryPathRecord(
                relative_path=relative_path,
                entry_type="file",
                size_bytes=len(data),
                content_digest=hashlib.sha256(data).hexdigest(),
            )
        )
    return records


@pytest.mark.asyncio
async def test_first_bind_hands_off_the_carried_notes_as_the_new_inventory(
    tmp_path: Path,
) -> None:
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    parent = _parent_workspace(tmp_path, "owner", parent_id)
    record, content = _note("notes/plan.md", b"after compaction")
    (parent / "notes").mkdir()
    (parent / "notes" / "plan.md").write_bytes(content)
    store = InMemoryWorkspaceStore()

    bound = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id=child_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=store,
        carried_notes=(record,),
        carry_source=parent,
    )

    assert (bound.workspace / "notes" / "plan.md").read_bytes() == content
    assert store.inventory[0].relative_path == "notes/plan.md"
    assert store.inventory[0].content_digest == hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_recovery_bind_does_not_copy_from_the_parent_again(tmp_path: Path) -> None:
    """A recovered Run may have written notes of its own; the parent must not replace them."""
    child_id = str(uuid.uuid4())
    first = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id=child_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=InMemoryWorkspaceStore(),
    )
    (first.workspace / "notes").mkdir()
    (first.workspace / "notes" / "own.md").write_text("mine", encoding="utf-8")
    parent = _parent_workspace(tmp_path, "owner", str(uuid.uuid4()))
    record, content = _note("notes/plan.md", b"parent")
    (parent / "notes").mkdir()
    (parent / "notes" / "plan.md").write_bytes(content)

    recovered = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id="owner",
        run_id=child_id,
        fencing_epoch=2,
        recorded_epoch=1,
        store=InMemoryWorkspaceStore(workspace_epoch=1),
        carried_notes=(record,),
        carry_source=parent,
    )

    assert (recovered.workspace / "notes" / "own.md").read_text(encoding="utf-8") == "mine"
    assert not (recovered.workspace / "notes" / "plan.md").exists()


def test_active_epoch_workspace_is_the_highest_numbered_live_tree(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    root = run_root(tmp_path, "owner", run_id)
    older, _ = epoch_paths(root, 1)
    newer, _ = epoch_paths(root, 4)
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "stale.txt").write_text("old", encoding="utf-8")
    (newer / "live.txt").write_text("new", encoding="utf-8")

    assert active_epoch_workspace(root) == newer
    assert active_epoch_workspace(tmp_path / "missing") is None


def test_a_note_over_the_byte_ceiling_is_not_selected() -> None:
    from dlightrag.engine.answer.continuation_handles import select_carried_run_notes

    huge = InventoryPathRecord(
        relative_path="notes/dump.md",
        entry_type="file",
        size_bytes=MAX_CARRIED_RUN_NOTE_BYTES + 1,
    )
    small = InventoryPathRecord(relative_path="notes/plan.md", entry_type="file", size_bytes=12)
    # Path order puts the dump first; the ceiling stops selection rather than skipping ahead.
    assert select_carried_run_notes((huge, small)) == ()


def test_carry_refuses_a_size_mismatch_and_a_directory(tmp_path: Path) -> None:
    """A registration that describes something other than a regular file is refused.

    Size is the one check a `bash`-written note has (its observation carries no
    digest), so a mismatch there is the only thing standing between a stale
    registration and carrying the wrong bytes.
    """
    parent = tmp_path / "parent"
    (parent / "notes" / "dir.md").mkdir(parents=True)
    (parent / "notes" / "plan.md").write_text("short", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()

    oversized = InventoryPathRecord(
        relative_path="notes/plan.md",
        entry_type="file",
        size_bytes=999,
        content_digest=hashlib.sha256(b"short").hexdigest(),
    )
    with pytest.raises(WorkspaceIntegrityError, match="size check"):
        carry_run_notes(source_workspace=parent, destination_workspace=child, notes=(oversized,))
    assert not (child / "notes").exists()

    not_a_file = InventoryPathRecord(relative_path="notes/dir.md", entry_type="file", size_bytes=0)
    with pytest.raises(WorkspaceIntegrityError, match="not a regular file"):
        carry_run_notes(source_workspace=parent, destination_workspace=child, notes=(not_a_file,))
    assert not (child / "notes").exists()


@pytest.mark.asyncio
async def test_the_audit_reports_orphans_and_deletes_nothing(tmp_path: Path) -> None:
    """The audit is the operator's look at the same fact the sweep acts on.

    It must never delete: a root left by an earlier configuration whose path this
    deployment does not own has to be countable before anything is asked to remove
    it. Both directories therefore still exist after the pass.
    """
    from dlightrag.engine.answer.workspace import audit_run_workspaces

    owner = "owner"
    live_id = str(uuid.uuid4())
    dead_id = str(uuid.uuid4())
    for run_id in (live_id, dead_id):
        root = run_root(tmp_path, owner, run_id)
        root.mkdir(parents=True)
        (root / "marker.txt").write_text("x", encoding="utf-8")

    class _Store:
        async def get_run_global(self, *, run_id: str) -> object | None:
            return object() if run_id == live_id else None

    report = await audit_run_workspaces(
        workspace_root=tmp_path, store=_Store(), page_size=1, sample=5
    )

    assert report.roots == 2
    assert report.unreadable == 0
    assert report.orphans == (f"{owner_shard(owner)}/{dead_id}",)
    # Nothing was deleted, and the audit said so by leaving both roots in place.
    assert run_root(tmp_path, owner, live_id).exists()
    assert run_root(tmp_path, owner, dead_id).exists()


def test_the_root_rule_follows_a_named_path_whatever_the_execution_mode(
    tmp_path: Path,
) -> None:
    """Reclamation and auditing must reach a named root an earlier config left.

    An unnamed root is the default path, and only an enabled configuration owns it;
    disabled does not invent it.
    """
    from dlightrag.engine.answer.workspace import (
        agent_workspace_reclaimer,
        resolve_workspace_root,
    )

    named = str(tmp_path / "dlightrag-named-root")
    assert resolve_workspace_root(execution_environment="disabled", workspace_root=named) == Path(
        named
    )
    assert resolve_workspace_root(execution_environment="disabled", workspace_root=None) is None
    default = resolve_workspace_root(execution_environment="trust", workspace_root=None)
    assert default is not None and default.is_absolute()
    assert (
        agent_workspace_reclaimer(execution_environment="disabled", workspace_root=named)
        is not None
    )
    assert agent_workspace_reclaimer(execution_environment="disabled", workspace_root=None) is None
