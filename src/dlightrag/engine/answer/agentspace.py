# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Agent Workspace layout and epoch handoff for configured execution adapters."""

from __future__ import annotations

import asyncio
import datetime
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import stat
import tarfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn

from dlightrag.engine.agent.environment.confinement import ConfinementPolicy
from dlightrag.engine.agent.environment.errors import (
    WORKSPACE_MAX_BYTES,
)
from dlightrag.engine.agent.environment.execution import (
    ExecutionEnvironment,
    ExecutionEnvironmentAdapter,
    ExecutionMode,
    TrustExecutionAdapter,
)
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.output import OutputStage
from dlightrag.engine.answer.continuation_handles import SESSION_NOTE_DIRECTORY, is_session_note
from dlightrag.engine.answer.execution_settings import default_local_agentspace_root
from dlightrag.engine.dependencies import TransientDependencyError
from dlightrag.engine.runtime.agentspace import (
    SESSION_NOTES_MATERIALIZE_FAILED,
    AgentspaceSnapshotFile,
    AgentspaceStore,
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    HandoffLeaseLost,
    SessionNoteRecord,
    note_digest,
)
from dlightrag.engine.runtime.coordinator import LeaseLostError
from dlightrag.engine.runtime.records import DeletedRun, parse_run_id
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.store import RunExistenceReader


class AgentspaceUnavailable(TransientDependencyError):
    """The Agent Workspace volume did not answer, so the Run defers rather than fails.

    An OSError while observing, copying, or creating an epoch, a source that moved
    while it was copied, or no room for one maximum epoch copy. None is a verdict on
    the Run: nothing recorded changes before a handoff commits, so the next claim
    copies the recorded epoch again from the start.
    """

    def __init__(self) -> None:
        super().__init__("agent_workspace", "Agent Workspace is temporarily unavailable")


class AgentspaceIntegrityError(RuntimeError):
    """Unsupported entries or a stable source/destination digest mismatch."""


logger = logging.getLogger(__name__)

_SPILL_RECOVERY_PAGE_SIZE = 128
_ORPHAN_SWEEP_PAGE_SIZE = 128
_EPOCH_COPY_TEMP_NAME = re.compile(r"\.tmp-([1-9][0-9]*)-[0-9a-f]{32}")
_SHARD_NAME = re.compile(r"^[0-9a-f]{2}$")

#: The plane of a Run's workspace its processes are given as ``HOME`` and ``TMPDIR``,
#: where they leave what they generate on the way — a script, a chart preview, a
#: downloaded page. ``prepare_process_directories`` recreates it the next time a
#: process is run.
_SCRATCH_DIRECTORY = "tmp"

#: How long a finished Run keeps its tree. A constant rather than a configuration key,
#: like the grace the Runtime gives a finished Run's working state and the one a stage
#: without a Run gets: this is hygiene, not retention — nothing a reader can ask for is
#: removed, and a deployment that wants a different span changes one number. It is
#: deliberately far below ``run_retention_days``, which the record and its events keep:
#: a week leaves an operator time to look at what a Run did before it goes.
_TREE_RETENTION = datetime.timedelta(days=7)


@dataclass(frozen=True, slots=True)
class RunAgentspace:
    """One claimed run's epoch directories and rooted environment."""

    epoch: int
    workspace: Path
    spill_dir: Path
    environment: ExecutionEnvironment
    #: Why the Session's notes are missing from this working copy, when they are.
    #: Memory degrades rather than failing the Run that could not be given it.
    notes_degraded: str | None = None
    #: The Session head this fresh epoch was seeded from, when it was.
    seeded_head_run_id: str | None = None


#: Where a Session's committed workspace trees live, beside its Run trees: the head
#: belongs to the Session, and a Run's own epoch stays disposable.
_SESSION_HEADS_DIRECTORY = "_session-heads"
_HEAD_TREE_DIRECTORY = "workspace"

#: What a head never carries. Scratch is not state, and notes are the memory plane's
#: (ADR 0022): they are materialized from it, not copied through the workspace.
_HEAD_EXCLUDED_TOP_LEVEL = frozenset({"tmp", "notes"})

#: Content-addressed snapshot bytes beside the heads. Every head tree links the bytes it
#: carries, so an unchanged file is stored once across a Session's retained snapshots and the
#: committed tree outlives the Run tree it was copied from.
_SNAPSHOT_BLOBS_DIRECTORY = ".blobs"
_SNAPSHOT_MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class AgentspaceSeed:
    """One committed head a fresh epoch is allowed to start from."""

    head_run_id: str
    manifest_digest: str
    workspace: Path


@dataclass(frozen=True, slots=True)
class AgentspaceHeadTarget:
    """Where a research attempt stages its head if it settles succeeded, and on what."""

    heads_root: Path
    session_id: str
    expected_head_run_id: str | None
    #: What the committed head already carried, when this attempt was seeded from it.
    seeded_manifest_digest: str | None = None
    #: The branch of the Session this attempt continues; a fork owns its own branch.
    lane_id: str = "main"
    #: The Run whose snapshot this attempt built on, for the snapshot's lineage.
    parent_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class StagedAgentspaceHead:
    """One succeeded tree copied out as a candidate Session head."""

    workspace: Path
    manifest_digest: str
    files: tuple[AgentspaceSnapshotFile, ...] = ()


def owner_tag(owner_id: str) -> str:
    """A stable directory tag for one owner: the shard alone is shared by owners."""
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:16]


def session_heads_root(workspace_root: Path, owner_id: str, session_id: str) -> Path:
    """Where one Session's head trees live, isolated by owner under its shard.

    A Session id is globally unique, but head cleanup is owner-scoped (a reader
    deletes conversations, not the whole volume), so the owner must be a path
    component the cleanup can address without walking everyone else's sessions.
    """
    return (
        workspace_root
        / owner_shard(owner_id)
        / owner_tag(owner_id)
        / _SESSION_HEADS_DIRECTORY
        / session_id
    )


def prune_session_head_trees(heads_root: Path, *, keep_run_ids: Set[str] = frozenset()) -> int:
    """Remove every head tree but the committed one, and stale staging leftovers.

    A Run stages its tree before the terminal write, so a crash or a refused CAS can
    leave a directory no row names. The next claim of the same Session is the one safe
    moment to collect them: it holds the lane and has just read the authoritative head.
    Trees a snapshot row still names are kept — every settled Run's tree is restorable —
    and staging leftovers are only removed once they are older than the tree clock, so a
    concurrent attempt's copy is never pulled from under it.
    """
    try:
        entries = tuple(heads_root.iterdir())
    except FileNotFoundError:
        return 0
    removed = 0
    cutoff = time.time() - _TREE_RETENTION.total_seconds()
    for entry in entries:
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(mode):
            continue
        try:
            stale_staging = entry.name.startswith(".staging-") and entry.lstat().st_mtime < cutoff
        except FileNotFoundError:
            continue
        if entry.name in {_SNAPSHOT_BLOBS_DIRECTORY, *keep_run_ids} or (
            entry.name.startswith(".staging-") and not stale_staging
        ):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    _collect_snapshot_blobs(heads_root)
    return removed


def reclaim_session_heads(root: Path, owner_id: str, session_id: str) -> None:
    """Remove one Session's head trees. Missing is success."""
    shutil.rmtree(session_heads_root(root, owner_id, session_id), ignore_errors=True)


def reclaim_owner_session_heads(root: Path, owner_id: str) -> None:
    """Remove every head tree an owner had. Missing is success."""
    shutil.rmtree(
        root / owner_shard(owner_id) / owner_tag(owner_id) / _SESSION_HEADS_DIRECTORY,
        ignore_errors=True,
    )


def head_run_agentspace(heads_root: Path, head_run_id: str) -> Path:
    return heads_root / head_run_id / _HEAD_TREE_DIRECTORY


def owner_shard(owner_id: str) -> str:
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:2]


def run_root(workspace_root: Path, owner_id: str, run_id: str) -> Path:
    return workspace_root / owner_shard(owner_id) / run_id


def epoch_paths(root: Path, epoch: int) -> tuple[Path, Path]:
    base = root / "epochs" / str(epoch)
    return base / "workspace", base / "internal" / "tool-results"


async def bind_run_agentspace(
    *,
    workspace_root: Path,
    owner_id: str,
    run_id: str,
    fencing_epoch: int,
    recorded_epoch: int | None,
    store: AgentspaceStore | None,
    execution_adapter: ExecutionEnvironmentAdapter | None = None,
    notes: Sequence[SessionNoteRecord] = (),
    seed: AgentspaceSeed | None = None,
) -> RunAgentspace:
    """Create or recover the active epoch and return a rooted environment.

    The Session's notes are materialized into a fresh epoch *before* the handoff
    records the inventory, so a crash cannot leave the files on disk with an empty
    observation. Recovery copies the whole epoch and must not materialize again: this
    Run may have written notes of its own since. Materialization is the caller's to
    attempt: memory degrades rather than failing the Run that could not read it.

    Epoch I/O runs off the event loop, and a volume that does not answer raises
    :class:`AgentspaceUnavailable`: the recorded epoch is never modified before a
    handoff commits, so the deferred Run's next claim copies it again.
    """
    root = run_root(workspace_root, owner_id, run_id)
    # A caller that binds without an adapter still gets a confined environment: the
    # confinement is part of what an enabled execution environment is (ADR 0024).
    adapter = execution_adapter or TrustExecutionAdapter(ConfinementPolicy())
    source_epoch = recorded_epoch
    destination = fencing_epoch
    head_bytes = (
        await _volume(_tree_bytes, seed.workspace)
        if seed is not None and recorded_epoch is None
        else 0
    )
    await _volume(_require_epoch_headroom, workspace_root, head_bytes)
    # This is deliberately claim-local, not a startup sweep: bind's caller already owns
    # this run's current fenced claim before stale epoch-copy trees may be reclaimed.
    await _volume(_cleanup_stale_epoch_copy_temps, root, current_epoch=destination)
    if source_epoch is None:
        # Nothing is recorded for this Run, so no numbered epoch below this attempt
        # is authoritative: an interrupted bind (crash or a lost claim between the
        # copy and the handoff) left one behind, and fencing epochs only grow.
        await _volume(_discard_unrecorded_epochs, root, below=destination)
        workspace, spill = await _volume(_prepare_epoch_dirs, root, destination)
        inventory: tuple[InventoryPathRecord, ...] = ()
        notes_degraded: str | None = None
        if seed is not None:
            inventory = await _volume(seed_epoch_from_head, seed, workspace)
        if notes:
            note_inventory, notes_degraded = await _volume(
                materialize_session_notes, notes, workspace
            )
            inventory = inventory + note_inventory
        if store is not None:
            committed = await store.handoff_epoch(
                expected_epoch=None, destination_epoch=destination, inventory=inventory
            )
            if not isinstance(committed, HandoffCommit):
                # A fenced-out worker must not compose a request that claims notes are
                # in a workspace this Run does not own.
                await _volume(_discard_unrecorded_epochs, root, below=destination + 1)
                _refuse_handoff(committed)
        await _volume(_cleanup_stale_epoch_copy_temps, root, current_epoch=destination)
        return RunAgentspace(
            epoch=destination,
            workspace=workspace,
            spill_dir=spill,
            environment=adapter.create(workspace, owner_id=owner_id),
            notes_degraded=notes_degraded,
            seeded_head_run_id=(seed.head_run_id if seed is not None else None),
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
                # A copy no handoff recorded is no epoch: discard it as the fresh path
                # does, and keep the recorded source for the claim that can record one.
                await _volume(
                    _discard_unrecorded_epochs, root, below=destination + 1, keep=source_epoch
                )
                _refuse_handoff(result)
        await _volume(_retire_epoch, root, source_epoch)
        await _volume(_discard_unrecorded_epochs, root, below=destination, keep=destination)
        # Narrow the race in which the fenced-out worker creates its unique temp tree
        # after the pre-copy cleanup. A later creation remains safe but may survive.
        await _volume(_cleanup_stale_epoch_copy_temps, root, current_epoch=destination)
    workspace, spill = epoch_paths(root, destination)
    await _volume(workspace.mkdir, parents=True, exist_ok=True)
    await _volume(spill.mkdir, parents=True, exist_ok=True)
    return RunAgentspace(
        epoch=destination,
        workspace=workspace,
        spill_dir=spill,
        environment=adapter.create(workspace, owner_id=owner_id),
    )


async def copy_epoch_verified(
    root: Path, source_epoch: int, destination: int, store: AgentspaceStore | None
) -> tuple[InventoryPathRecord, ...]:
    """Copy a stable source; the one fenced claimed run keeps spills immutable while paging.

    Returns the verified observation of the copied workspace, which is the source
    manifest the copy had to match: the caller records it as the new epoch's
    Workspace Inventory, so a reader after a recovery sees what the Run holds
    rather than an empty table.
    """
    spills = await _committed_spills(store) if store is not None else ()
    return await _volume(_copy_epoch, root, source_epoch, destination, spills)


def _copy_epoch(
    root: Path, source_epoch: int, destination: int, spills: Sequence[CommittedSpillRecord]
) -> tuple[InventoryPathRecord, ...]:
    """Copy into a unique temp tree, verify it, and move it into place.

    It runs whole in one thread, which removes its temp tree whatever fails: a
    cancelled bind cannot race its own cleanup, and the source is only ever read.
    """
    source_ws, source_spill = epoch_paths(root, source_epoch)
    dest_parent = root / "epochs" / str(destination)
    temp_parent = root / "epochs" / f".tmp-{destination}-{uuid.uuid4().hex}"
    try:
        temp_ws = temp_parent / "workspace"
        temp_spill = temp_parent / "internal" / "tool-results"
        temp_ws.mkdir(parents=True)
        temp_spill.mkdir(parents=True)
        manifest = _agentspace_manifest(source_ws)
        _copy_tree_regular_files(source_ws, temp_ws)
        if _agentspace_manifest(source_ws) != manifest:
            # A fenced-out writer is still settling into it: copy again later.
            logger.warning("Agent Workspace %s changed while it was copied", source_ws)
            raise AgentspaceUnavailable()
        if _agentspace_manifest(temp_ws) != manifest:
            raise AgentspaceIntegrityError("copied workspace does not match the source manifest")
        for spill in spills:
            _copy_committed_spill(source_spill, temp_spill, spill)
        if dest_parent.exists():
            shutil.rmtree(dest_parent)
        temp_parent.rename(dest_parent)
    except BaseException:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise
    return _inventory_observation(manifest)


async def _volume[**P, T](operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run blocking workspace I/O off the event loop; an OSError there is an outage.

    A copy on a slow shared mount must not stall the loop that renews leases. The
    OSError names files the Agent chose, and dependency classification reads the text
    of the whole cause chain, so it is logged here and the outage is raised outside
    the handler, chained to nothing.
    """
    try:
        return await asyncio.to_thread(operation, *args, **kwargs)
    except OSError:
        logger.warning("Agent Workspace volume did not answer", exc_info=True)
    raise AgentspaceUnavailable()


def _require_epoch_headroom(workspace_root: Path, extra_bytes: int = 0) -> None:
    """Defer a claim the volume cannot hold one maximum epoch copy for right now.

    Free space is the volume's state at this claim, not configuration: other Runs
    write and retention reclaims, so a shortfall is an outage rather than a refusal.
    A seeded claim also keeps a committed head beside its epoch, so its copy headroom
    is raised by the head's own size.
    """
    usage = os.statvfs(workspace_root)
    if usage.f_bavail * usage.f_frsize < WORKSPACE_MAX_BYTES + max(0, extra_bytes):
        logger.warning("Agent Workspace volume %s has no room for an epoch copy", workspace_root)
        raise AgentspaceUnavailable()


def _refuse_handoff(result: HandoffConflict | HandoffLeaseLost) -> NoReturn:
    """A handoff that did not commit: this worker lost its claim, or the record moved."""
    if isinstance(result, HandoffLeaseLost):
        raise LeaseLostError
    raise AgentspaceIntegrityError("the recorded workspace epoch moved under this claim")


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


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, files in _carried_tree(root):
        del directory
        for path in files:
            total += path.stat().st_size
    return total


def active_epoch_agentspace(root: Path) -> Path | None:
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


def materialize_session_notes(
    notes: Sequence[SessionNoteRecord],
    destination_workspace: Path,
) -> tuple[tuple[InventoryPathRecord, ...], str | None]:
    """Write the Session's notes into a fresh epoch, degrading rather than refusing.

    The plane owns the bytes, so there is nothing to verify them against: the epoch's
    Inventory is recorded from what this wrote. Staging then swapping keeps a failed
    materialization from leaving a half-written notes tree behind the handoff, and an
    empty note set materializes nothing at all.

    A failure here is memory, not the Run (ADR 0022): the notes are discarded and the
    typed reason is returned for the Run's trace, because a Run that cannot be given
    its notes still has its transcript, its Evidence, and its Products.
    """
    if not notes:
        return (), None
    staging = destination_workspace / f".materialize-notes-{uuid.uuid4().hex}"
    try:
        written: list[InventoryPathRecord] = []
        for note in notes:
            if not is_session_note(note.relative_path):
                raise AgentspaceIntegrityError("materialized path is not a Session note")
            staged = staging / note.relative_path
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(note.content)
            written.append(
                InventoryPathRecord(
                    relative_path=note.relative_path,
                    entry_type="file",
                    size_bytes=len(note.content),
                    content_digest=note_digest(note.content),
                )
            )
        _replace_notes_directory(destination_workspace, staging / SESSION_NOTE_DIRECTORY)
    except AgentspaceIntegrityError, OSError:
        logger.warning("Could not materialize Session notes", exc_info=True)
        shutil.rmtree(staging, ignore_errors=True)
        return (), SESSION_NOTES_MATERIALIZE_FAILED
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging, ignore_errors=True)
    return tuple(written), None


def _replace_notes_directory(destination_workspace: Path, staged_notes: Path) -> None:
    """Swap the staged notes tree into place. A symlink at the reserved path is refused."""
    if not staged_notes.exists():
        raise AgentspaceIntegrityError("materialized notes directory was not staged")
    final_notes = destination_workspace / SESSION_NOTE_DIRECTORY
    try:
        mode = final_notes.lstat().st_mode
    except FileNotFoundError:
        staged_notes.rename(final_notes)
        return
    if stat.S_ISLNK(mode):
        raise AgentspaceIntegrityError("notes directory is a symbolic link")
    if not stat.S_ISDIR(mode):
        raise AgentspaceIntegrityError("notes path is not a directory")
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
    # The reserved notes directory is created with the other two so the one plane the
    # framework carries is as visible as the publication plane and the scratch plane.
    # A live Run measured the alternative: state a later step needed was written to
    # `tmp/`, which nothing carries, while `notes/` stayed empty until the end of the
    # Run, by which time no summary could name it.
    (workspace / SESSION_NOTE_DIRECTORY).mkdir(exist_ok=True)
    (workspace / _SCRATCH_DIRECTORY).mkdir(exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    _discard_notes_staging(workspace)
    return workspace, spill


def _discard_notes_staging(workspace: Path) -> None:
    """Remove a staging tree an interrupted materialization left behind.

    The swap that installs the notes tree is atomic, so anything still named for a
    staging pass is residue: a recovery copy would otherwise record it as content.
    The retired carry's staging name stays in the list: a tree left by an older
    deployment is the same residue.
    """
    prefixes = (".materialize-notes-", ".carry-notes-")
    for entry in tuple(workspace.iterdir()):
        if not entry.name.startswith(prefixes):
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
            logger.warning("Failed to discard notes staging %s", entry, exc_info=True)


def _agentspace_tree(root: Path) -> Iterator[tuple[Path, tuple[Path, ...]]]:
    """Yield every directory under ``root`` with the regular files directly in it.

    Strict: ``os.walk`` skips a directory it cannot list and ``Path`` predicates read
    an error as "no", so either lets an unreadable subtree verify as a smaller tree
    whose source the handoff then retires. Here a failed listing or lstat raises, and
    an entry is only ever what lstat says it is.
    """
    pending = [root]
    while pending:
        directory = pending.pop()
        files: list[Path] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(mode):
                    files.append(Path(entry.path))
                else:
                    raise AgentspaceIntegrityError("workspace contains a special or linked file")
        yield directory, tuple(files)


def _agentspace_manifest(root: Path) -> dict[str, tuple[str, int, str]]:
    manifest: dict[str, tuple[str, int, str]] = {}
    for _, files in _agentspace_tree(root):
        for path in files:
            rel = str(path.relative_to(root))
            data = path.read_bytes()
            manifest[rel] = ("file", len(data), hashlib.sha256(data).hexdigest())
    return manifest


def _copy_tree_regular_files(source: Path, dest: Path) -> None:
    for directory, files in _agentspace_tree(source):
        target_dir = dest / directory.relative_to(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            shutil.copy2(path, target_dir / path.name)


def _carried_tree(root: Path) -> Iterator[tuple[Path, tuple[Path, ...]]]:
    """Yield the head content of one tree: everything but top-level scratch and notes.

    Strict like ``_agentspace_tree``: an unreadable directory or a linked/special entry
    refuses rather than verifying as a smaller tree, because a head that silently lost
    a subtree would be carried as if it were complete.
    """
    pending = [root]
    while pending:
        directory = pending.pop()
        files: list[Path] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    if directory == root and entry.name in _HEAD_EXCLUDED_TOP_LEVEL:
                        continue
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(mode):
                    files.append(Path(entry.path))
                else:
                    raise AgentspaceIntegrityError("workspace contains a special or linked file")
        yield directory, tuple(files)


def _manifest_digest(manifest: Mapping[str, tuple[str, int, str]]) -> str:
    """The digest of a head's carried content: names, sizes and bytes, sorted."""
    canonical = json.dumps(
        {relative: list(value) for relative, value in sorted(manifest.items())},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_epoch_from_head(
    seed: AgentspaceSeed, destination_workspace: Path
) -> tuple[InventoryPathRecord, ...]:
    """Copy a committed head's carried files into a fresh epoch, verifying its digest."""
    source = seed.workspace
    if not source.is_dir():
        raise AgentspaceIntegrityError("Session workspace head is missing")
    manifest: dict[str, tuple[str, int, str]] = {}
    inventory: list[InventoryPathRecord] = []
    for directory, files in _carried_tree(source):
        target_dir = destination_workspace / directory.relative_to(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            data = path.read_bytes()
            relative = str(path.relative_to(source))
            digest = hashlib.sha256(data).hexdigest()
            shutil.copy2(path, target_dir / path.name)
            manifest[relative] = ("file", len(data), digest)
            inventory.append(
                InventoryPathRecord(
                    relative_path=relative,
                    entry_type="file",
                    size_bytes=len(data),
                    content_digest=digest,
                )
            )
    if _manifest_digest(manifest) != seed.manifest_digest:
        raise AgentspaceIntegrityError("Session workspace head does not match its digest")
    return tuple(inventory)


def archive_carried_agentspace(workspace: Path) -> bytes | None:
    """One completed goal's carried workspace as a reproducible ``.tar.gz``.

    The archive is the same content a Session head carries — everything but top-level
    scratch and memory — with sorted entries and no environment metadata, so the same
    workspace archives to the same bytes. ``None`` means there is nothing to archive.
    """
    entries: list[tuple[str, bytes]] = []
    for _directory, files in _carried_tree(workspace):
        for path in files:
            entries.append((str(path.relative_to(workspace)), path.read_bytes()))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for relative, data in entries:
            info = tarfile.TarInfo(name=relative)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), mtime=0)


def carried_manifest_digest(workspace: Path) -> str:
    """The digest of what a Run's tree would commit, without copying it first."""
    manifest: dict[str, tuple[str, int, str]] = {}
    for _, files in _carried_tree(workspace):
        for path in files:
            data = path.read_bytes()
            manifest[str(path.relative_to(workspace))] = (
                "file",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
    return _manifest_digest(manifest)


def _snapshot_blob_path(heads_root: Path, digest: str) -> Path:
    return heads_root / _SNAPSHOT_BLOBS_DIRECTORY / digest[:2] / digest


def _link_snapshot_blob(heads_root: Path, digest: str, data: bytes, target: Path) -> None:
    """Store the bytes once, then link them into the snapshot tree.

    A missing blob is written to a temp file and moved into place, so no reader sees a
    partial blob; an existing one is already content-addressed by this digest. The link
    keeps the snapshot independent of the Run tree that produced it and makes an
    unchanged file cost no disk per retained phase.
    """
    blob = _snapshot_blob_path(heads_root, digest)
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        temporary = blob.with_name(f".tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, blob)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(blob, target)
    except OSError:
        # A filesystem without hard links still keeps the snapshot correct; only the
        # dedupe is lost, so a deployment detail never fails a settled phase.
        shutil.copyfile(blob, target)


def _collect_snapshot_blobs(heads_root: Path) -> None:
    """Remove blob bytes no retained snapshot tree references.

    Only trees that carry a manifest are enumerated; a tree from before the blob store
    (no manifest) makes the pass a no-op, because deleting bytes it might reference
    would corrupt a snapshot the pointer can still name.
    """
    trees = [
        entry
        for entry in heads_root.iterdir()
        if entry.is_dir() and entry.name != _SNAPSHOT_BLOBS_DIRECTORY
    ]
    referenced: set[str] = set()
    for tree in trees:
        manifest_path = tree / _SNAPSHOT_MANIFEST_FILE
        if not manifest_path.is_file():
            return
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return
        if not isinstance(entries, list):
            return
        referenced.update(
            str(entry["digest"])
            for entry in entries
            if isinstance(entry, dict) and "digest" in entry
        )
    blobs = heads_root / _SNAPSHOT_BLOBS_DIRECTORY
    if not blobs.is_dir():
        return
    for shard in tuple(blobs.iterdir()):
        if not shard.is_dir():
            continue
        for blob in tuple(shard.iterdir()):
            if blob.is_file() and blob.name not in referenced:
                blob.unlink(missing_ok=True)
        try:
            shard.rmdir()
        except OSError:
            continue


def stage_session_head(*, workspace: Path, heads_root: Path, run_id: str) -> StagedAgentspaceHead:
    """Copy a succeeded Run's carried files out as the Session's next head.

    The tree is written to a unique staging directory and moved into place, so no
    reader ever sees a half-written head; the target name is the Run's own id, which
    makes retrying the same commit idempotent.
    """
    heads_root.mkdir(parents=True, exist_ok=True)
    final = heads_root / run_id
    staging = heads_root / f".staging-{run_id}-{uuid.uuid4().hex}"
    manifest: dict[str, tuple[str, int, str]] = {}
    try:
        staged_workspace = staging / _HEAD_TREE_DIRECTORY
        staged_workspace.mkdir(parents=True)
        for directory, files in _carried_tree(workspace):
            target_dir = staged_workspace / directory.relative_to(workspace)
            target_dir.mkdir(parents=True, exist_ok=True)
            for path in files:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                _link_snapshot_blob(heads_root, digest, data, target_dir / path.name)
                manifest[str(path.relative_to(workspace))] = ("file", len(data), digest)
        (staging / _SNAPSHOT_MANIFEST_FILE).write_text(
            json.dumps(
                [
                    {"path": relative, "digest": value[2], "size": value[1]}
                    for relative, value in sorted(manifest.items())
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    files = tuple(
        AgentspaceSnapshotFile(
            relative_path=relative,
            content_digest=value[2],
            size_bytes=value[1],
        )
        for relative, value in sorted(manifest.items())
    )
    return StagedAgentspaceHead(
        workspace=final / _HEAD_TREE_DIRECTORY,
        manifest_digest=_manifest_digest(manifest),
        files=files,
    )


async def _committed_spills(store: AgentspaceStore) -> tuple[CommittedSpillRecord, ...]:
    spills: list[CommittedSpillRecord] = []
    cursor: str | None = None
    while True:
        page = await store.load_spills_page(
            after_resource_id=cursor, limit=_SPILL_RECOVERY_PAGE_SIZE
        )
        if len(page) > _SPILL_RECOVERY_PAGE_SIZE:
            raise AgentspaceIntegrityError("committed spill page exceeded the requested limit")
        for spill in page:
            if cursor is not None and spill.resource_id <= cursor:
                raise AgentspaceIntegrityError(
                    "committed spill pages are not strictly ordered by resource_id"
                )
            spills.append(spill)
            cursor = spill.resource_id
        if len(page) < _SPILL_RECOVERY_PAGE_SIZE:
            return tuple(spills)


def _copy_committed_spill(source_dir: Path, dest_dir: Path, spill: CommittedSpillRecord) -> None:
    name = f"{spill.resource_id}.txt"
    src = source_dir / name
    try:
        mode = src.lstat().st_mode
    except FileNotFoundError:
        mode = 0
    if not stat.S_ISREG(mode):
        raise AgentspaceIntegrityError(f"committed spill {spill.resource_id} is missing")
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spill.content_digest or len(data) != spill.size_bytes:
        raise AgentspaceIntegrityError(f"committed spill {spill.resource_id} failed digest check")
    dest = dest_dir / name
    dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise AgentspaceIntegrityError(f"copied spill {spill.resource_id} does not match")


def _retire_epoch(root: Path, epoch: int) -> None:
    stale = root / "epochs" / str(epoch)
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)


class AgentspaceReclaimer:
    """Keeps the Agent Workspace volume to what a Run still needs.

    One walk decides each run root: a root whose Run row is gone is removed, a live
    Run keeps everything, and a Run that has been over longer than the tree clock
    loses its whole root. Nothing reads a finished Run's tree: its notes are the
    Session's rows, which the next Run materializes; a published Artifact is a Blob its
    row pins; a spill is read only by the Run that wrote it, and a terminal Run never
    runs again. The row itself and its events keep the deployment's Run retention.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        page_size: int = _ORPHAN_SWEEP_PAGE_SIZE,
        now: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        if not workspace_root.is_absolute():
            raise ValueError("workspace root must be an absolute path")
        self._root = workspace_root
        self._page_size = max(1, int(page_size))
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))

    async def reclaim(self, runs: Sequence[DeletedRun]) -> None:
        for run in runs:
            if run.run_kind != "answer":
                continue
            try:
                # A retention batch is bounded but a working tree is not small: the
                # walk must not run on the coordinator's event loop.
                await asyncio.to_thread(
                    reclaim_run_agentspace, self._root, run.owner_id, run.run_id
                )
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
        """Walk every run root and act on the ones that need it, in bounded pages.

        Returns how many roots the sweep removed: an orphan's, or a finished Run's. The
        read of a Run is the sweep's one question — is it still there, and has it been
        over long enough that its tree is nobody's — so the same walk answers both, which
        is what the Corpus Mutation stage sweep does with its own root.
        """
        changed = 0
        after: tuple[str, str] | None = None
        while True:
            page = await asyncio.to_thread(
                list_run_agentspace_roots,
                self._root,
                after=after,
                limit=self._page_size,
            )
            if not page:
                return changed
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
                if record is not None and not self._tree_is_done(record):
                    # A live claim owns its epoch directory, and a live claim has a row.
                    continue
                try:
                    await asyncio.to_thread(reclaim_discovered_run_root, self._root, path)
                    changed += 1
                except FileNotFoundError:
                    continue
                except Exception:
                    logger.warning(
                        "Failed to reclaim Agent Workspace at %s",
                        path,
                        exc_info=True,
                    )
            if len(page) < self._page_size:
                return changed

    def _tree_is_done(self, record: object) -> bool:
        """Whether this Run is over and has been for longer than its tree is kept.

        A Run that never finished keeps its tree: nothing else may decide that an agent
        still working has stopped needing it. The clock starts at ``finished_at``, so a
        Run that ran for days is not swept the moment it ends.
        """
        if getattr(record, "terminal", False) is not True:
            return False
        finished_at = getattr(record, "finished_at", None)
        if not isinstance(finished_at, datetime.datetime):
            return False
        return self._now() - finished_at >= _TREE_RETENTION

    async def reclaim_session(self, *, owner_id: str, session_id: str) -> None:
        """Remove one deleted Session's committed heads, best effort."""
        try:
            await asyncio.to_thread(reclaim_session_heads, self._root, owner_id, session_id)
        except Exception:
            logger.warning(
                "Failed to reclaim Session workspace heads for Session %s",
                session_id,
                extra={"owner_id": owner_id},
                exc_info=True,
            )

    async def reclaim_owner(self, *, owner_id: str) -> None:
        """Remove every deleted owner's committed heads, best effort."""
        try:
            await asyncio.to_thread(reclaim_owner_session_heads, self._root, owner_id)
        except Exception:
            logger.warning(
                "Failed to reclaim Session workspace heads for owner",
                extra={"owner_id": owner_id},
                exc_info=True,
            )


def agentspace_reclaimer(
    *,
    execution_environment: ExecutionMode,
    workspace_root: str | None,
) -> AgentspaceReclaimer | None:
    """Return a reclaimer when a workspace root is configured.

    Deletion creates nothing, so a deployment that turns execution off still
    reclaims the trees earlier enabled runs left behind. No configured root
    means none: disabled does not invent the default path, because that path
    was never this process's workspace.
    """
    root = resolve_agentspace_root(
        execution_environment=execution_environment, workspace_root=workspace_root
    )
    return None if root is None else AgentspaceReclaimer(root)


def resolve_agentspace_root(
    *,
    execution_environment: str,
    workspace_root: str | None,
) -> Path | None:
    """Return the Agent Workspace root this deployment owns, or nothing.

    A named root is the deployment's own path and is used whatever the execution
    mode is: reclamation and auditing must reach trees an earlier enabled
    configuration left there. An unnamed root means the default path, which only an
    enabled configuration owns — disabled does not invent it, because that path was
    never this process's workspace.
    """
    raw = (workspace_root or "").strip()
    if not raw or raw in {"null", "None"}:
        if execution_environment == "disabled":
            return None
        return default_local_agentspace_root()
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise ValueError("agent.workspace_root must be an absolute path")
    return root.resolve()


@dataclass(frozen=True, slots=True)
class RunAgentspaceAudit:
    """What one read-only pass over a workspace root observed."""

    roots: int
    orphans: tuple[str, ...]
    unreadable: int


async def audit_run_agentspaces(
    *,
    workspace_root: Path,
    store: RunExistenceReader,
    page_size: int = _ORPHAN_SWEEP_PAGE_SIZE,
    sample: int = 20,
) -> RunAgentspaceAudit:
    """Report Run roots, and the ones whose Run row is gone. Deletes nothing.

    The sweep is the thing that deletes, and it runs where a deployment owns a root.
    This is the operator's look at the same fact: a root left by an earlier enabled
    configuration whose path this deployment does not own can be counted before
    anything is asked to remove it.
    """
    page_size = max(1, int(page_size))
    sample = max(0, int(sample))
    roots = 0
    orphans: list[str] = []
    unreadable = 0
    after: tuple[str, str] | None = None
    while True:
        page = await asyncio.to_thread(
            list_run_agentspace_roots, workspace_root, after=after, limit=page_size
        )
        if not page:
            return RunAgentspaceAudit(roots=roots, orphans=tuple(orphans), unreadable=unreadable)
        for path in page:
            after = (path.parent.name, path.name)
            roots += 1
            try:
                record = await store.get_run_global(run_id=path.name)
            except Exception:
                logger.warning("Failed to read run %s during Agent Workspace audit", path.name)
                unreadable += 1
                continue
            if record is None and len(orphans) < sample:
                orphans.append(f"{path.parent.name}/{path.name}")
        if len(page) < page_size:
            return RunAgentspaceAudit(roots=roots, orphans=tuple(orphans), unreadable=unreadable)


def reclaim_run_agentspace(workspace_root: Path, owner_id: str, run_id: str) -> None:
    """Remove one Run's workspace root. Missing is success; a bad path is refused."""
    _remove_run_root(_expected_run_root(workspace_root, owner_id, run_id))


def reclaim_discovered_run_root(workspace_root: Path, path: Path) -> None:
    """Remove a listed run root. The path must be the expected shard/run layout."""
    _require_discovered_run_root(workspace_root, path)
    _remove_run_root(path)


def list_run_agentspace_roots(
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
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    shard = owner_shard(owner_id)
    path = run_root(workspace_root, owner_id, run_id)
    if path.name != run_id or path.parent.name != shard:
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    try:
        path.parent.relative_to(workspace_root)
    except ValueError as exc:
        raise AgentspaceIntegrityError("run workspace path is not a run root") from exc
    return path


def _require_discovered_run_root(workspace_root: Path, path: Path) -> None:
    if parse_run_id(path.name) is None:
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    if _SHARD_NAME.fullmatch(path.parent.name) is None:
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    try:
        parent_mode = path.parent.lstat().st_mode
    except OSError as exc:
        raise AgentspaceIntegrityError("run workspace path is not a run root") from exc
    if stat.S_ISLNK(parent_mode):
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    try:
        path.parent.parent.relative_to(workspace_root)
    except ValueError as exc:
        raise AgentspaceIntegrityError("run workspace path is not a run root") from exc
    if path.parent.parent != workspace_root:
        raise AgentspaceIntegrityError("run workspace path is not a run root")


def _remove_run_root(path: Path) -> None:
    try:
        parent_mode = path.parent.lstat().st_mode
    except OSError as exc:
        raise AgentspaceIntegrityError("run workspace path is not a run root") from exc
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        path.unlink()
        return
    if not stat.S_ISDIR(mode):
        raise AgentspaceIntegrityError("run workspace path is not a run root")
    # shutil.rmtree unlinks nested symlinks instead of following them.
    shutil.rmtree(path)


__all__ = [
    "AgentspaceReclaimer",
    "RunAgentspaceAudit",
    "audit_run_agentspaces",
    "resolve_agentspace_root",
    "RunAgentspace",
    "AgentspaceIntegrityError",
    "AgentspaceUnavailable",
    "active_epoch_agentspace",
    "agentspace_reclaimer",
    "bind_run_agentspace",
    "copy_epoch_verified",
    "epoch_paths",
    "list_run_agentspace_roots",
    "owner_shard",
    "reclaim_discovered_run_root",
    "reclaim_run_agentspace",
    "run_root",
    "spill_receipt",
    "write_spill_file",
]
