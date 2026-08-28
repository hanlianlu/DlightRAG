# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Workspace layout and epoch handoff for configured execution adapters."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from dlightrag.engine.agent.environment import (
    ExecutionEnvironment,
    ExecutionEnvironmentAdapter,
    TrustExecutionAdapter,
)
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.output import OutputStage
from dlightrag.engine.runtime.workspace import (
    CommittedSpillRecord,
    HandoffCommit,
    WorkspaceStore,
)


class WorkspaceRecoveryFailed(RuntimeError):
    """Source changed during copy or there is not enough headroom. Retryable."""


class WorkspaceIntegrityError(RuntimeError):
    """Unsupported entries or a stable source/destination digest mismatch."""


logger = logging.getLogger(__name__)

_SPILL_RECOVERY_PAGE_SIZE = 128
_EPOCH_COPY_TEMP_NAME = re.compile(r"\.tmp-([1-9][0-9]*)-[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """One claimed run's epoch directories and rooted environment."""

    epoch: int
    workspace: Path
    spill_dir: Path
    environment: ExecutionEnvironment


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
    execution_adapter: ExecutionEnvironmentAdapter | None = None,
) -> RunWorkspace:
    """Create or recover the active epoch and return a rooted environment."""
    root = run_root(workspace_root, owner_id, run_id)
    adapter = execution_adapter or TrustExecutionAdapter()
    source_epoch = recorded_epoch
    destination = fencing_epoch
    # This is deliberately claim-local, not a startup sweep: bind's caller already owns
    # this run's current fenced claim before stale epoch-copy trees may be reclaimed.
    _cleanup_stale_epoch_copy_temps(root, current_epoch=destination)
    if source_epoch is None:
        workspace, spill = _prepare_epoch_dirs(root, destination)
        if store is not None:
            await store.handoff_epoch(
                expected_epoch=None, destination_epoch=destination, inventory=()
            )
        _cleanup_stale_epoch_copy_temps(root, current_epoch=destination)
        return RunWorkspace(
            epoch=destination,
            workspace=workspace,
            spill_dir=spill,
            environment=adapter.create(workspace),
        )
    if source_epoch != destination:
        await copy_epoch_verified(root, source_epoch, destination, store)
        if store is not None:
            result = await store.handoff_epoch(
                expected_epoch=source_epoch, destination_epoch=destination, inventory=()
            )
            if not isinstance(result, HandoffCommit):
                raise WorkspaceRecoveryFailed("workspace epoch handoff failed")
        _retire_epoch(root, source_epoch)
        # Narrow the race in which the fenced-out worker creates its unique temp tree
        # after the pre-copy cleanup. A later creation remains safe but may survive.
        _cleanup_stale_epoch_copy_temps(root, current_epoch=destination)
    workspace, spill = epoch_paths(root, destination)
    workspace.mkdir(parents=True, exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    return RunWorkspace(
        epoch=destination,
        workspace=workspace,
        spill_dir=spill,
        environment=adapter.create(workspace),
    )


async def copy_epoch_verified(
    root: Path, source_epoch: int, destination: int, store: WorkspaceStore | None
) -> None:
    """Copy a stable source; the one fenced claimed run keeps spills immutable while paging."""
    source_ws, source_spill = epoch_paths(root, source_epoch)
    dest_parent = root / "epochs" / str(destination)
    temp_parent = root / "epochs" / f".tmp-{destination}-{uuid.uuid4().hex}"
    try:
        manifest_a = _workspace_manifest(source_ws) if source_ws.exists() else {}
        temp_ws = temp_parent / "workspace"
        temp_spill = temp_parent / "internal" / "tool-results"
        temp_ws.mkdir(parents=True)
        temp_spill.mkdir(parents=True)
        if source_ws.exists():
            _copy_tree_regular_files(source_ws, temp_ws)
        manifest_b = _workspace_manifest(source_ws) if source_ws.exists() else {}
        if manifest_a != manifest_b:
            raise WorkspaceRecoveryFailed("workspace source changed during copy")
        if _workspace_manifest(temp_ws) != manifest_a:
            raise WorkspaceIntegrityError("copied workspace does not match the source manifest")
        if store is not None:
            await _copy_committed_spills_paged(source_spill, temp_spill, store)
        if dest_parent.exists():
            shutil.rmtree(dest_parent)
        temp_parent.rename(dest_parent)
    except WorkspaceRecoveryFailed, WorkspaceIntegrityError:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise WorkspaceRecoveryFailed(str(exc)) from exc
    except BaseException:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise


class FileOutputStage(OutputStage):
    """Append-only staging file promoted atomically to a committed spill."""

    def __init__(self, spill_dir: Path, resource_id: str) -> None:
        spill_dir.mkdir(parents=True, exist_ok=True)
        self._resource_id = resource_id
        self._temporary = spill_dir / f".{resource_id}.staging"
        self._committed = spill_dir / f"{resource_id}.txt"
        self._file: BinaryIO | None = self._temporary.open("xb")
        self._digest = hashlib.sha256()
        self._size_bytes = 0

    def append(self, data: bytes) -> None:
        if self._file is None:
            raise RuntimeError("output stage is closed")
        self._file.write(data)
        self._digest.update(data)
        self._size_bytes += len(data)

    async def commit(self) -> CommittedOutput:
        if self._file is None:
            raise RuntimeError("output stage is closed")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None
        self._temporary.replace(self._committed)
        return CommittedOutput(
            resource_id=self._resource_id,
            content_digest=self._digest.hexdigest(),
            size_bytes=self._size_bytes,
        )

    def discard(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._temporary.unlink(missing_ok=True)


def write_spill_file(spill_dir: Path, resource_id: str, text: str) -> Path:
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / f"{resource_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def spill_receipt(resource_id: str, text: str) -> CommittedOutput:
    data = text.encode("utf-8")
    return CommittedOutput(
        resource_id=resource_id,
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _cleanup_stale_epoch_copy_temps(root: Path, *, current_epoch: int) -> None:
    """Best-effort reclaim of older copy temps for one currently claimed run.

    Fencing epochs strictly increase, so a prior worker cannot hand off after the current
    claim. Its uniquely named temp tree is never committed workspace state. Exact-name
    symlinks are unlinked rather than traversed, and each failed removal is isolated so
    an undeletable orphan cannot make otherwise valid recovery unavailable.
    """
    epochs = root / "epochs"
    try:
        entries = tuple(epochs.iterdir())
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Failed to inspect claim-local epoch-copy temps in %s", epochs, exc_info=True
        )
        return

    for entry in entries:
        match = _EPOCH_COPY_TEMP_NAME.fullmatch(entry.name)
        if match is None:
            continue
        try:
            destination_epoch = int(match.group(1))
        except ValueError:
            # An integer too large for Python's conversion limit was not system-created.
            continue
        if destination_epoch >= current_epoch:
            continue
        try:
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                entry.unlink()
            elif stat.S_ISDIR(mode):
                # shutil.rmtree unlinks nested symlinks instead of following them.
                shutil.rmtree(entry)
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to reclaim stale epoch-copy temp %s", entry, exc_info=True)


def _prepare_epoch_dirs(root: Path, epoch: int) -> tuple[Path, Path]:
    workspace, spill = epoch_paths(root, epoch)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)
    (workspace / "tmp").mkdir(exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    return workspace, spill


def _workspace_manifest(root: Path) -> dict[str, tuple[str, int, str]]:
    if not root.exists():
        return {}
    manifest: dict[str, tuple[str, int, str]] = {}
    for current, dirnames, filenames in os.walk(root):
        for name in list(dirnames):
            path = Path(current) / name
            if path.is_symlink():
                raise WorkspaceIntegrityError("workspace contains a symbolic link")
        for name in filenames:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                raise WorkspaceIntegrityError("workspace contains a special or linked file")
            rel = str(path.relative_to(root))
            data = path.read_bytes()
            manifest[rel] = ("file", len(data), hashlib.sha256(data).hexdigest())
    return manifest


def _copy_tree_regular_files(source: Path, dest: Path) -> None:
    for current, dirnames, filenames in os.walk(source):
        rel_dir = Path(current).relative_to(source)
        target_dir = dest / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in dirnames:
            if (Path(current) / name).is_symlink():
                raise WorkspaceIntegrityError("workspace contains a symbolic link")
        for name in filenames:
            src = Path(current) / name
            if src.is_symlink() or not src.is_file():
                raise WorkspaceIntegrityError("workspace contains a special or linked file")
            shutil.copy2(src, target_dir / name)


async def _copy_committed_spills_paged(
    source_dir: Path, dest_dir: Path, store: WorkspaceStore
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cursor: str | None = None
    while True:
        page = await store.load_spills_page(
            after_resource_id=cursor, limit=_SPILL_RECOVERY_PAGE_SIZE
        )
        if len(page) > _SPILL_RECOVERY_PAGE_SIZE:
            raise WorkspaceIntegrityError("committed spill page exceeded the requested limit")
        if not page:
            return
        for spill in page:
            if cursor is not None and spill.resource_id <= cursor:
                raise WorkspaceIntegrityError(
                    "committed spill pages are not strictly ordered by resource_id"
                )
            _copy_committed_spill(source_dir, dest_dir, spill)
            cursor = spill.resource_id
        if len(page) < _SPILL_RECOVERY_PAGE_SIZE:
            return


def _copy_committed_spill(source_dir: Path, dest_dir: Path, spill: CommittedSpillRecord) -> None:
    name = f"{spill.resource_id}.txt"
    src = source_dir / name
    if not src.is_file():
        raise WorkspaceIntegrityError(f"committed spill {spill.resource_id} is missing")
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spill.content_digest or len(data) != spill.size_bytes:
        raise WorkspaceIntegrityError(f"committed spill {spill.resource_id} failed digest check")
    dest = dest_dir / name
    dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise WorkspaceIntegrityError(f"copied spill {spill.resource_id} does not match")


def _retire_epoch(root: Path, epoch: int) -> None:
    stale = root / "epochs" / str(epoch)
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)


__all__ = [
    "RunWorkspace",
    "WorkspaceIntegrityError",
    "WorkspaceRecoveryFailed",
    "bind_run_workspace",
    "copy_epoch_verified",
    "epoch_paths",
    "owner_shard",
    "run_root",
    "spill_receipt",
    "write_spill_file",
]
