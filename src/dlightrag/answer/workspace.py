# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Workspace layout and epoch handoff for local trusted execution."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from dlightrag_agent.environment import LocalExecutionEnvironment

from dlightrag.runtime.workspace import HandoffCommit, WorkspaceStore


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """One claimed run's epoch directories and rooted environment."""

    epoch: int
    workspace: Path
    spill_dir: Path
    environment: LocalExecutionEnvironment


def owner_shard(owner_id: str) -> str:
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:2]


def run_root(workspace_root: Path, owner_id: str, run_id: str) -> Path:
    return workspace_root / owner_shard(owner_id) / run_id


def epoch_paths(root: Path, epoch: int) -> tuple[Path, Path]:
    base = root / "epochs" / str(epoch)
    return base / "workspace", base / "internal" / "tool-results"


async def bind_run_workspace(
    *,
    workspace_root: Path,
    owner_id: str,
    run_id: str,
    fencing_epoch: int,
    recorded_epoch: int | None,
    store: WorkspaceStore | None,
) -> RunWorkspace:
    """Create or recover the active epoch and return a rooted environment."""
    root = run_root(workspace_root, owner_id, run_id)
    source_epoch = recorded_epoch
    destination = fencing_epoch
    if source_epoch is None:
        workspace, spill = epoch_paths(root, destination)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts").mkdir(exist_ok=True)
        (workspace / "tmp").mkdir(exist_ok=True)
        spill.mkdir(parents=True, exist_ok=True)
        if store is not None:
            await store.handoff_epoch(
                expected_epoch=None, destination_epoch=destination, inventory=()
            )
        return RunWorkspace(
            epoch=destination,
            workspace=workspace,
            spill_dir=spill,
            environment=LocalExecutionEnvironment(workspace),
        )
    if source_epoch != destination:
        await _copy_epoch(root, source_epoch, destination, store)
        if store is not None:
            result = await store.handoff_epoch(
                expected_epoch=source_epoch, destination_epoch=destination, inventory=()
            )
            if not isinstance(result, HandoffCommit):
                raise RuntimeError("workspace epoch handoff failed")
        _retire_epoch(root, source_epoch)
    workspace, spill = epoch_paths(root, destination)
    workspace.mkdir(parents=True, exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    return RunWorkspace(
        epoch=destination,
        workspace=workspace,
        spill_dir=spill,
        environment=LocalExecutionEnvironment(workspace),
    )


async def _copy_epoch(
    root: Path, source_epoch: int, destination: int, store: WorkspaceStore | None
) -> None:
    source_ws, source_spill = epoch_paths(root, source_epoch)
    dest_ws, dest_spill = epoch_paths(root, destination)
    if dest_ws.parent.exists():
        shutil.rmtree(dest_ws.parent)
    dest_ws.parent.mkdir(parents=True)
    if source_ws.exists():
        shutil.copytree(source_ws, dest_ws, symlinks=False)
    else:
        dest_ws.mkdir(parents=True)
    dest_spill.mkdir(parents=True, exist_ok=True)
    if store is not None:
        for spill in await store.load_spills():
            src = source_spill / f"{spill.resource_id}.txt"
            if src.is_file():
                shutil.copy2(src, dest_spill / src.name)


def _retire_epoch(root: Path, epoch: int) -> None:
    stale = root / "epochs" / str(epoch)
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)


def write_spill_file(spill_dir: Path, resource_id: str, text: str) -> Path:
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / f"{resource_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "RunWorkspace",
    "bind_run_workspace",
    "epoch_paths",
    "owner_shard",
    "run_root",
    "write_spill_file",
]
