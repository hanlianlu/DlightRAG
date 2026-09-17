# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Workspace layout and epoch handoff for configured execution adapters."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
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
from dlightrag.engine.answer.continuation_handles import RUN_NOTE_DIRECTORY, is_run_note
from dlightrag.engine.answer.execution_settings import default_local_workspace_root
from dlightrag.engine.runtime.records import DeletedRun, parse_run_id
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.store import RunExistenceReader
from dlightrag.engine.runtime.workspace import (
    CommittedSpillRecord,
    HandoffCommit,
    WorkspaceStore,
)


class WorkspaceRecoveryFailed(RuntimeError):
    """Source changed during copy or there is not enough headroom. Retryable."""


class WorkspaceIntegrityError(RuntimeError):
    """Unsupported entries or a stable source/destination digest mismatch."""


class WorkspaceUnavailableError(RuntimeError):
    """The parent Run's Agent Workspace is gone; a continuation cannot carry from it."""


logger = logging.getLogger(__name__)

_SPILL_RECOVERY_PAGE_SIZE = 128
_ORPHAN_SWEEP_PAGE_SIZE = 128
_EPOCH_COPY_TEMP_NAME = re.compile(r"\.tmp-([1-9][0-9]*)-[0-9a-f]{32}")
_SHARD_NAME = re.compile(r"^[0-9a-f]{2}$")


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
    carried_notes: Sequence[InventoryPathRecord] = (),
    carry_source: Path | None = None,
) -> RunWorkspace:
    """Create or recover the active epoch and return a rooted environment.

    A continuation's first bind copies the parent's registered Run Notes into this
    epoch *before* the handoff records the inventory, so a crash cannot leave the
    files on disk with an empty observation. Recovery copies the whole epoch and
    must not copy from the parent again: this Run may have written notes of its own.
    """
    root = run_root(workspace_root, owner_id, run_id)
    adapter = execution_adapter or TrustExecutionAdapter()
    source_epoch = recorded_epoch
    destination = fencing_epoch
    # This is deliberately claim-local, not a startup sweep: bind's caller already owns
    # this run's current fenced claim before stale epoch-copy trees may be reclaimed.
    _cleanup_stale_epoch_copy_temps(root, current_epoch=destination)
    if source_epoch is None:
        # Nothing is recorded for this Run, so no numbered epoch below this attempt
        # is authoritative: an interrupted bind (crash or a lost claim between the
        # copy and the handoff) left one behind, and fencing epochs only grow.
        _discard_unrecorded_epochs(root, below=destination)
        workspace, spill = _prepare_epoch_dirs(root, destination)
        inventory: tuple[InventoryPathRecord, ...] = ()
        if carried_notes:
            if carry_source is None:
                raise WorkspaceUnavailableError(
                    "The parent Run's Agent Workspace is gone. "
                    "Continue from a Run whose workspace still exists."
                )
            inventory = carry_run_notes(
                source_workspace=carry_source,
                destination_workspace=workspace,
                notes=carried_notes,
            )
        if store is not None:
            committed = await store.handoff_epoch(
                expected_epoch=None, destination_epoch=destination, inventory=inventory
            )
            if not isinstance(committed, HandoffCommit):
                # A fenced-out worker must not compose a request that claims notes are
                # in a workspace this Run does not own.
                _discard_unrecorded_epochs(root, below=destination + 1)
                raise WorkspaceRecoveryFailed("workspace epoch handoff failed")
        _cleanup_stale_epoch_copy_temps(root, current_epoch=destination)
        return RunWorkspace(
            epoch=destination,
            workspace=workspace,
            spill_dir=spill,
            environment=adapter.create(workspace),
        )
    if source_epoch != destination:
        observed = await copy_epoch_verified(root, source_epoch, destination, store)
        if store is not None:
            # The copy was just verified byte-identical to the source, so the new
            # epoch's observation is known rather than unknown: discarding it here
            # left every reader of the inventory — a Run Note set is exactly that —
            # empty until the Run happened to write again.
            result = await store.handoff_epoch(
                expected_epoch=source_epoch,
                destination_epoch=destination,
                inventory=observed,
            )
            if not isinstance(result, HandoffCommit):
                raise WorkspaceRecoveryFailed("workspace epoch handoff failed")
        _retire_epoch(root, source_epoch)
        _discard_unrecorded_epochs(root, below=destination, keep=destination)
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
) -> tuple[InventoryPathRecord, ...]:
    """Copy a stable source; the one fenced claimed run keeps spills immutable while paging.

    Returns the verified observation of the copied workspace, which is the source
    manifest the copy had to match: the caller records it as the new epoch's
    Workspace Inventory, so a reader after a recovery sees what the Run holds
    rather than an empty table.
    """
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
    return _inventory_observation(manifest_a)


def _inventory_observation(
    manifest: Mapping[str, tuple[str, int, str]],
) -> tuple[InventoryPathRecord, ...]:
    """Return a copied epoch's observation from the manifest that verified it.

    ``mode`` stays unobserved here: the manifest proves type, size, and content,
    and the caller of a copy needs those, not the permission bits.
    """
    return tuple(
        InventoryPathRecord(
            relative_path=relative_path,
            entry_type=entry_type,
            size_bytes=size_bytes,
            content_digest=digest,
        )
        for relative_path, (entry_type, size_bytes, digest) in sorted(manifest.items())
    )


def active_epoch_workspace(root: Path) -> Path | None:
    """Return the highest numbered epoch's workspace directory, if it is a real tree.

    Retired epochs are deleted after a successful handoff, so the highest number is
    the live tree. A missing root is a continuation's explicit refusal, not an empty
    carry: silently copying nothing is the behaviour this slice removes.
    """
    for component in (root, root / "epochs"):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            return None
    newest: int | None = None
    try:
        entries = tuple((root / "epochs").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        epoch = int(entry.name)
        if epoch < 1:
            continue
        try:
            entry_mode = entry.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            continue
        if stat.S_ISLNK(entry_mode) or not stat.S_ISDIR(entry_mode):
            continue
        if newest is None or epoch > newest:
            newest = epoch
    if newest is None:
        return None
    workspace, _ = epoch_paths(root, newest)
    try:
        workspace_mode = workspace.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if stat.S_ISLNK(workspace_mode) or not stat.S_ISDIR(workspace_mode):
        return None
    return workspace


def carry_run_notes(
    *,
    source_workspace: Path,
    destination_workspace: Path,
    notes: Sequence[InventoryPathRecord],
) -> tuple[InventoryPathRecord, ...]:
    """Copy the parent's registered Run Notes into this epoch. All or nothing.

    The Inventory is the authority on which paths are notes; this function never
    walks the source tree to discover extras. A missing file, a digest mismatch,
    a symlink, or a non-file is a typed refusal and leaves the destination's notes
    directory untouched. The copy is a framework write, so the destination records
    always carry a digest even when the parent observation did not.
    """
    if not notes:
        return ()
    try:
        source_mode = source_workspace.lstat().st_mode
    except FileNotFoundError as exc:
        raise WorkspaceUnavailableError(
            "The parent Run's Agent Workspace is gone. "
            "Continue from a Run whose workspace still exists."
        ) from exc
    except OSError as exc:
        raise WorkspaceUnavailableError(
            "The parent Run's Agent Workspace is gone. "
            "Continue from a Run whose workspace still exists."
        ) from exc
    if stat.S_ISLNK(source_mode) or not stat.S_ISDIR(source_mode):
        raise WorkspaceIntegrityError("parent workspace is not a directory")
    staging = destination_workspace / f".carry-notes-{uuid.uuid4().hex}"
    try:
        copied: list[InventoryPathRecord] = []
        for record in notes:
            if not is_run_note(record.relative_path):
                raise WorkspaceIntegrityError("carried path is not a Run Note")
            source_file = _require_regular_file_inside(source_workspace, record.relative_path)
            data = source_file.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if len(data) != record.size_bytes:
                raise WorkspaceIntegrityError(
                    f"carried run note {record.relative_path} failed size check"
                )
            if record.content_digest is not None and digest != record.content_digest:
                raise WorkspaceIntegrityError(
                    f"carried run note {record.relative_path} failed digest check"
                )
            staged = staging / record.relative_path
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            copied.append(
                InventoryPathRecord(
                    relative_path=record.relative_path,
                    entry_type="file",
                    size_bytes=len(data),
                    content_digest=digest,
                )
            )
        _replace_notes_directory(destination_workspace, staging / RUN_NOTE_DIRECTORY)
    except WorkspaceIntegrityError, WorkspaceUnavailableError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise WorkspaceIntegrityError(str(exc)) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging, ignore_errors=True)
    return tuple(copied)


def _require_regular_file_inside(root: Path, relative_path: str) -> Path:
    """Resolve one Inventory path under root without following any symlink."""
    parts = [part for part in relative_path.split("/") if part]
    if not parts:
        raise WorkspaceIntegrityError("carried path is not a Run Note")
    current = root
    mode = 0
    for part in parts:
        if part in {".", ".."}:
            raise WorkspaceIntegrityError("carried path is not a Run Note")
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise WorkspaceIntegrityError(f"carried run note {relative_path} is missing") from exc
        if stat.S_ISLNK(mode):
            raise WorkspaceIntegrityError(f"carried run note {relative_path} is a symbolic link")
    if not stat.S_ISREG(mode):
        raise WorkspaceIntegrityError(f"carried run note {relative_path} is not a regular file")
    return current


def _replace_notes_directory(destination_workspace: Path, staged_notes: Path) -> None:
    """Swap the staged notes tree into place. A symlink at the reserved path is refused."""
    if not staged_notes.exists():
        raise WorkspaceIntegrityError("carried notes directory was not staged")
    final_notes = destination_workspace / RUN_NOTE_DIRECTORY
    try:
        mode = final_notes.lstat().st_mode
    except FileNotFoundError:
        staged_notes.rename(final_notes)
        return
    if stat.S_ISLNK(mode):
        raise WorkspaceIntegrityError("notes directory is a symbolic link")
    if not stat.S_ISDIR(mode):
        raise WorkspaceIntegrityError("notes path is not a directory")
    backup = destination_workspace / f".notes-old-{uuid.uuid4().hex}"
    final_notes.rename(backup)
    try:
        staged_notes.rename(final_notes)
    except OSError:
        backup.rename(final_notes)
        raise
    shutil.rmtree(backup, ignore_errors=True)


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


def _discard_unrecorded_epochs(root: Path, *, below: int, keep: int | None = None) -> None:
    """Remove numbered epochs no record names, below one fencing generation.

    Fencing epochs strictly increase, so every numbered directory below the current
    attempt belongs to a claim that has already lost; the one the caller is keeping
    is the epoch the Run's own row records.
    """
    epochs = root / "epochs"
    try:
        entries = tuple(epochs.iterdir())
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Failed to inspect epochs in %s", epochs, exc_info=True)
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            numbered = int(entry.name)
        except ValueError:
            continue
        if numbered >= below or numbered == keep:
            continue
        try:
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                entry.unlink()
            elif stat.S_ISDIR(mode):
                shutil.rmtree(entry)
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to discard unrecorded epoch %s", entry, exc_info=True)


def _prepare_epoch_dirs(root: Path, epoch: int) -> tuple[Path, Path]:
    workspace, spill = epoch_paths(root, epoch)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)
    (workspace / "tmp").mkdir(exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    _discard_carry_staging(workspace)
    return workspace, spill


def _discard_carry_staging(workspace: Path) -> None:
    """Remove a staging tree an interrupted carry left behind.

    The swap that installs carried notes is atomic, so anything still named for a
    staging pass is residue: a recovery copy would otherwise record it as content.
    """
    for entry in tuple(workspace.iterdir()):
        if not entry.name.startswith(".carry-notes-"):
            continue
        try:
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                entry.unlink()
            elif stat.S_ISDIR(mode):
                shutil.rmtree(entry)
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to discard carry staging %s", entry, exc_info=True)


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


class AgentWorkspaceReclaimer:
    """Deletes per-Run Agent Workspace roots after their rows are gone."""

    def __init__(self, workspace_root: Path, *, page_size: int = _ORPHAN_SWEEP_PAGE_SIZE) -> None:
        if not workspace_root.is_absolute():
            raise ValueError("workspace root must be an absolute path")
        self._root = workspace_root
        self._page_size = max(1, int(page_size))

    async def reclaim(self, runs: Sequence[DeletedRun]) -> None:
        for run in runs:
            if run.run_kind != "answer":
                continue
            try:
                # A retention batch is bounded but a working tree is not small: the
                # walk must not run on the coordinator's event loop.
                await asyncio.to_thread(reclaim_run_workspace, self._root, run.owner_id, run.run_id)
            except FileNotFoundError:
                continue
            except Exception:
                logger.warning(
                    "Failed to reclaim Agent Workspace for run %s",
                    run.run_id,
                    extra={"run_id": run.run_id, "owner_id": run.owner_id},
                    exc_info=True,
                )

    async def sweep_orphans(self, store: RunExistenceReader) -> int:
        removed = 0
        after: tuple[str, str] | None = None
        while True:
            page = await asyncio.to_thread(
                list_run_workspace_roots,
                self._root,
                after=after,
                limit=self._page_size,
            )
            if not page:
                return removed
            for path in page:
                after = (path.parent.name, path.name)
                try:
                    record = await store.get_run_global(run_id=path.name)
                except Exception:
                    logger.warning(
                        "Failed to read run %s during Agent Workspace orphan sweep",
                        path.name,
                        exc_info=True,
                    )
                    continue
                if record is not None:
                    # A live claim owns its epoch directory, and a live claim has a row.
                    continue
                try:
                    await asyncio.to_thread(reclaim_discovered_run_root, self._root, path)
                    removed += 1
                except FileNotFoundError:
                    continue
                except Exception:
                    logger.warning(
                        "Failed to reclaim orphan Agent Workspace at %s",
                        path,
                        exc_info=True,
                    )
            if len(page) < self._page_size:
                return removed


def agent_workspace_reclaimer(
    *,
    execution_environment: str,
    workspace_root: str | None,
) -> AgentWorkspaceReclaimer | None:
    """Return a reclaimer when a workspace root is configured.

    Deletion creates nothing, so a deployment that turns execution off still
    reclaims the trees earlier enabled runs left behind. No configured root
    means none: disabled does not invent the default path, because that path
    was never this process's workspace.
    """
    if execution_environment not in {"disabled", "trust", "sandbox"}:
        raise ValueError(f"unknown agent execution mode: {execution_environment}")
    raw = (workspace_root or "").strip()
    if not raw or raw in {"null", "None"}:
        if execution_environment == "disabled":
            return None
        root = default_local_workspace_root()
    else:
        root = Path(raw).expanduser()
        if not root.is_absolute():
            raise ValueError("agent.workspace_root must be an absolute path")
        root = root.resolve()
    return AgentWorkspaceReclaimer(root)


def reclaim_run_workspace(workspace_root: Path, owner_id: str, run_id: str) -> None:
    """Remove one Run's workspace root. Missing is success; a bad path is refused."""
    _remove_run_root(_expected_run_root(workspace_root, owner_id, run_id))


def reclaim_discovered_run_root(workspace_root: Path, path: Path) -> None:
    """Remove a listed run root. The path must be the expected shard/run layout."""
    _require_discovered_run_root(workspace_root, path)
    _remove_run_root(path)


def list_run_workspace_roots(
    workspace_root: Path,
    *,
    after: tuple[str, str] | None = None,
    limit: int,
) -> tuple[Path, ...]:
    """Return a bounded page of candidate run roots in (shard, run_id) order.

    Symlinked shards are skipped rather than traversed. A run root that is itself
    a symlink is included so the sweep can unlink it without following.
    """
    cap = max(1, int(limit))
    try:
        root_mode = workspace_root.lstat().st_mode
    except FileNotFoundError:
        return ()
    except OSError:
        logger.warning("Failed to inspect Agent Workspace root %s", workspace_root, exc_info=True)
        return ()
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        return ()

    shards: list[str] = []
    try:
        shard_entries = tuple(workspace_root.iterdir())
    except OSError:
        logger.warning("Failed to list Agent Workspace shards in %s", workspace_root, exc_info=True)
        return ()
    for entry in shard_entries:
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to inspect Agent Workspace shard %s", entry, exc_info=True)
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            continue
        if _SHARD_NAME.fullmatch(entry.name) is None:
            continue
        shards.append(entry.name)
    shards.sort()

    found: list[Path] = []
    after_shard, after_run = after if after is not None else ("", "")
    for shard in shards:
        if after is not None and shard < after_shard:
            continue
        shard_path = workspace_root / shard
        run_ids: list[str] = []
        try:
            entries = tuple(shard_path.iterdir())
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Failed to list Agent Workspace runs in %s", shard_path, exc_info=True)
            continue
        for entry in entries:
            try:
                mode = entry.lstat().st_mode
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("Failed to inspect Agent Workspace run %s", entry, exc_info=True)
                continue
            if parse_run_id(entry.name) is None:
                continue
            if not (stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
                continue
            if after is not None and shard == after_shard and entry.name <= after_run:
                continue
            run_ids.append(entry.name)
        run_ids.sort()
        for run_id in run_ids:
            found.append(shard_path / run_id)
            if len(found) >= cap:
                return tuple(found)
    return tuple(found)


def _expected_run_root(workspace_root: Path, owner_id: str, run_id: str) -> Path:
    if parse_run_id(run_id) is None or Path(run_id).name != run_id:
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    shard = owner_shard(owner_id)
    path = run_root(workspace_root, owner_id, run_id)
    if path.name != run_id or path.parent.name != shard:
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    try:
        path.parent.relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspaceIntegrityError("run workspace path is not a run root") from exc
    return path


def _require_discovered_run_root(workspace_root: Path, path: Path) -> None:
    if parse_run_id(path.name) is None:
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    if _SHARD_NAME.fullmatch(path.parent.name) is None:
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    try:
        parent_mode = path.parent.lstat().st_mode
    except OSError as exc:
        raise WorkspaceIntegrityError("run workspace path is not a run root") from exc
    if stat.S_ISLNK(parent_mode):
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    try:
        path.parent.parent.relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspaceIntegrityError("run workspace path is not a run root") from exc
    if path.parent.parent != workspace_root:
        raise WorkspaceIntegrityError("run workspace path is not a run root")


def _remove_run_root(path: Path) -> None:
    try:
        parent_mode = path.parent.lstat().st_mode
    except OSError as exc:
        raise WorkspaceIntegrityError("run workspace path is not a run root") from exc
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        path.unlink()
        return
    if not stat.S_ISDIR(mode):
        raise WorkspaceIntegrityError("run workspace path is not a run root")
    # shutil.rmtree unlinks nested symlinks instead of following them.
    shutil.rmtree(path)


__all__ = [
    "AgentWorkspaceReclaimer",
    "RunWorkspace",
    "WorkspaceIntegrityError",
    "WorkspaceRecoveryFailed",
    "WorkspaceUnavailableError",
    "active_epoch_workspace",
    "agent_workspace_reclaimer",
    "bind_run_workspace",
    "carry_run_notes",
    "copy_epoch_verified",
    "epoch_paths",
    "list_run_workspace_roots",
    "owner_shard",
    "reclaim_discovered_run_root",
    "reclaim_run_workspace",
    "run_root",
    "spill_receipt",
    "write_spill_file",
]
