# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session workspace head: what a commit carries, and what a seed restores (S1.1)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from dlightrag.engine.answer.agentspace import (
    AgentspaceIntegrityError,
    AgentspaceSeed,
    AgentspaceUnavailable,
    archive_carried_agentspace,
    bind_run_agentspace,
    carried_manifest_digest,
    owner_tag,
    prune_session_head_trees,
    reclaim_owner_session_heads,
    reclaim_session_heads,
    seed_epoch_from_head,
    session_heads_root,
    stage_session_head,
)


def _write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "run" / "epochs" / "1" / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_a_commit_carries_everything_but_scratch_and_notes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write(workspace, "model.py", b"model")
    _write(workspace, "artifacts/report.html", b"<html></html>")
    _write(workspace, "tmp/scratch.txt", b"scratch")
    _write(workspace, "notes/plan.md", b"note")

    staged = stage_session_head(workspace=workspace, heads_root=tmp_path / "heads", run_id="run-1")

    assert _files(staged.workspace) == ["artifacts/report.html", "model.py"]
    assert len(staged.manifest_digest) == 64
    assert [file.relative_path for file in staged.files] == [
        "artifacts/report.html",
        "model.py",
    ]
    # Staging the same tree again names the same content, so a retried commit is idempotent,
    # and a run that changed nothing can see that without copying the tree first.
    again = stage_session_head(workspace=workspace, heads_root=tmp_path / "heads", run_id="run-1")
    assert again.manifest_digest == staged.manifest_digest
    assert carried_manifest_digest(workspace) == staged.manifest_digest


def test_a_seed_copies_the_committed_tree_and_checks_its_digest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write(workspace, "model.py", b"model")
    _write(workspace, "artifacts/report.html", b"<html></html>")
    _write(workspace, "tmp/scratch.txt", b"scratch")
    staged = stage_session_head(workspace=workspace, heads_root=tmp_path / "heads", run_id="run-1")
    destination = tmp_path / "new-epoch"
    destination.mkdir()

    inventory = seed_epoch_from_head(
        AgentspaceSeed(
            head_run_id="run-1",
            manifest_digest=staged.manifest_digest,
            workspace=staged.workspace,
        ),
        destination,
    )

    assert _files(destination) == ["artifacts/report.html", "model.py"]
    assert {item.relative_path for item in inventory} == {"artifacts/report.html", "model.py"}

    # A head whose bytes moved under its digest is refused, not carried as if it were whole.
    (staged.workspace / "model.py").write_bytes(b"tampered")
    destination.mkdir(exist_ok=True)
    with pytest.raises(AgentspaceIntegrityError):
        seed_epoch_from_head(
            AgentspaceSeed(
                head_run_id="run-1",
                manifest_digest=staged.manifest_digest,
                workspace=staged.workspace,
            ),
            tmp_path / "other-epoch",
        )


def test_a_missing_head_is_an_integrity_refusal(tmp_path: Path) -> None:
    with pytest.raises(AgentspaceIntegrityError):
        seed_epoch_from_head(
            AgentspaceSeed(
                head_run_id="run-gone",
                manifest_digest="0" * 64,
                workspace=tmp_path / "missing",
            ),
            tmp_path / "epoch",
        )


async def test_a_fresh_epoch_is_seeded_from_the_head_before_the_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write(workspace, "model.py", b"model")
    _write(workspace, "tmp/scratch.txt", b"scratch")
    staged = stage_session_head(workspace=workspace, heads_root=tmp_path / "heads", run_id="run-1")

    bound = await bind_run_agentspace(
        workspace_root=tmp_path,
        owner_id="owner-1",
        run_id="run-2",
        fencing_epoch=1,
        recorded_epoch=None,
        store=None,
        seed=AgentspaceSeed(
            head_run_id="run-1",
            manifest_digest=staged.manifest_digest,
            workspace=staged.workspace,
        ),
    )

    assert (bound.workspace / "model.py").read_bytes() == b"model"
    assert not (bound.workspace / "tmp/scratch.txt").exists()
    assert bound.seeded_head_run_id == "run-1"


def _head_tree(heads_root: Path, run_id: str, marker: str) -> Path:
    workspace = heads_root / run_id / "workspace"
    _write(workspace, marker, b"x")
    return workspace


def test_a_fresh_claim_prunes_every_head_but_the_committed_one(tmp_path: Path) -> None:
    heads_root = tmp_path / "heads"
    current = _head_tree(heads_root, "run-current", "model.py")
    stale = _head_tree(heads_root, "run-stale", "old.py")
    fresh_staging = heads_root / ".staging-run-x-abc"
    fresh_staging.mkdir()
    old_staging = heads_root / ".staging-run-y-def"
    old_staging.mkdir()
    old_time = 1_000_000_000
    for path in (old_staging,):
        os.utime(path, (old_time, old_time))

    removed = prune_session_head_trees(heads_root, keep_run_ids={"run-current"})

    assert current.is_dir()
    assert not stale.exists()
    assert fresh_staging.is_dir()
    assert not old_staging.exists()
    assert removed >= 2


def test_reclaiming_one_session_leaves_other_sessions_and_owners_alone(
    tmp_path: Path,
) -> None:
    root = tmp_path
    first = session_heads_root(root, "owner-a", "session-1")
    second = session_heads_root(root, "owner-a", "session-2")
    other_owner = session_heads_root(root, "owner-b", "session-1")
    for heads in (first, second, other_owner):
        (heads / "run-1" / "workspace").mkdir(parents=True)

    reclaim_session_heads(root, "owner-a", "session-1")
    assert not first.exists()
    assert second.is_dir()
    assert other_owner.is_dir()

    reclaim_owner_session_heads(root, "owner-a")
    assert not second.exists()
    assert other_owner.is_dir()
    assert owner_tag("owner-a") != owner_tag("owner-b")


def test_a_seeded_claim_requires_room_for_the_head_beside_its_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _write(workspace, "model.py", b"model")
    staged = stage_session_head(workspace=workspace, heads_root=tmp_path / "heads", run_id="run-1")
    seed = AgentspaceSeed(
        head_run_id="run-1", manifest_digest=staged.manifest_digest, workspace=staged.workspace
    )
    free = {"bytes": 0}

    class _Usage:
        def __init__(self, available: int) -> None:
            self.f_bavail = available
            self.f_frsize = 1

    monkeypatch.setattr(
        "dlightrag.engine.answer.agentspace.os.statvfs",
        lambda _path: _Usage(free["bytes"]),
    )

    from dlightrag.engine.agent.environment.errors import (
        WORKSPACE_MAX_BYTES,
    )

    free["bytes"] = WORKSPACE_MAX_BYTES + 6  # the epoch plus the 5-byte head
    bound = asyncio.run(
        bind_run_agentspace(
            workspace_root=tmp_path,
            owner_id="owner-1",
            run_id="run-2",
            fencing_epoch=1,
            recorded_epoch=None,
            store=None,
            seed=seed,
        )
    )
    assert (bound.workspace / "model.py").read_bytes() == b"model"

    free["bytes"] = WORKSPACE_MAX_BYTES - 1
    with pytest.raises(AgentspaceUnavailable):
        asyncio.run(
            bind_run_agentspace(
                workspace_root=tmp_path,
                owner_id="owner-1",
                run_id="run-3",
                fencing_epoch=1,
                recorded_epoch=None,
                store=None,
                seed=seed,
            )
        )


def test_a_completed_goal_archives_its_carried_workspace_reproducibly(tmp_path: Path) -> None:
    import gzip
    import io
    import tarfile

    workspace = _workspace(tmp_path)
    _write(workspace, "scripts/model.py", b"model = {}\n")
    _write(workspace, "artifacts/report.html", b"<html></html>")
    _write(workspace, "tmp/scratch.txt", b"scratch")
    _write(workspace, "notes/plan.md", b"note")

    archive = archive_carried_agentspace(workspace)

    assert archive is not None
    assert archive == archive_carried_agentspace(workspace), (
        "the same tree archives to one byte string"
    )
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive)), mode="r:") as tar:
        names = sorted(member.name for member in tar.getmembers())
        assert names == ["artifacts/report.html", "scripts/model.py"]
        member = tar.extractfile("scripts/model.py")
        assert member is not None
        assert member.read() == b"model = {}\n"


def test_a_workspace_with_nothing_carried_has_no_archive(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write(workspace, "tmp/scratch.txt", b"scratch")
    assert archive_carried_agentspace(workspace) is None


def test_unchanged_snapshot_bytes_are_stored_once_and_linked(tmp_path: Path) -> None:
    first = _workspace(tmp_path / "run-one")
    _write(first, "scripts/model.py", b"model = {}\n")
    _write(first, "artifacts/facts.md", b"facts")
    one = stage_session_head(workspace=first, heads_root=tmp_path / "heads", run_id="run-1")

    second = _workspace(tmp_path / "run-two")
    _write(second, "scripts/model.py", b"model = {}\n")
    _write(second, "artifacts/facts.md", b"facts")
    _write(second, "artifacts/report.html", b"<html></html>")
    two = stage_session_head(workspace=second, heads_root=tmp_path / "heads", run_id="run-2")

    blobs = [path for path in (tmp_path / "heads" / ".blobs").rglob("*") if path.is_file()]
    assert len(blobs) == 3, "one blob per distinct content, not per snapshot file"
    shared_one = one.workspace / "scripts/model.py"
    shared_two = two.workspace / "scripts/model.py"
    assert shared_one.read_bytes() == shared_two.read_bytes()
    assert shared_one.stat().st_ino == shared_two.stat().st_ino, "the second phase links the bytes"


def test_pruning_a_snapshot_reclaims_bytes_no_retained_tree_uses(tmp_path: Path) -> None:
    heads_root = tmp_path / "heads"
    first = _workspace(tmp_path / "run-one")
    _write(first, "case.json", b"first draft")
    stage_session_head(workspace=first, heads_root=heads_root, run_id="run-1")
    second = _workspace(tmp_path / "run-two")
    _write(second, "case.json", b"second draft")
    stage_session_head(workspace=second, heads_root=heads_root, run_id="run-2")

    prune_session_head_trees(heads_root, keep_run_ids={"run-2"})

    remaining = sorted(
        path.read_bytes() for path in (heads_root / ".blobs").rglob("*") if path.is_file()
    )
    assert remaining == [b"second draft"], "only the retained snapshot's bytes are kept"
    assert (heads_root / "run-2" / "workspace" / "case.json").read_bytes() == b"second draft"


def test_a_snapshot_tree_from_before_the_blob_store_is_never_mined_for_bytes(
    tmp_path: Path,
) -> None:
    heads_root = tmp_path / "heads"
    legacy = heads_root / "run-legacy" / "workspace"
    _write(legacy, "model.py", b"legacy")
    blob = heads_root / ".blobs" / "aa" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"must survive")

    prune_session_head_trees(heads_root, keep_run_ids={"run-legacy"})

    assert blob.read_bytes() == b"must survive"
