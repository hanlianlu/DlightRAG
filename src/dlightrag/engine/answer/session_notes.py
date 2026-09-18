# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session's notes plane: materialization, promotion, and its degradation.

Memory belongs to the Agent Session, not to the Run (ADR 0022). A Run holds a
materialized working copy under the reserved notes path, and Tool settlement
promotes the difference back under the Run's own lease. Nothing is copied from a
parent Run, so memory never depends on how a Run arrived.

Nothing here may fail a Run. Every failure is a typed reason the caller records on
the Run's trace and continues, because a Run that cannot read its Session's memory
still has its transcript, its Evidence, and its Products.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dlightrag.engine.answer.continuation_handles import (
    MAX_SESSION_NOTE_PATH_CHARS,
    SESSION_NOTE_DIRECTORY,
    is_session_note,
)
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    SESSION_NOTES_BUDGET_REFUSED,
    SESSION_NOTES_LEASE_LOST,
    SESSION_NOTES_MAX_BYTES,
    SESSION_NOTES_MAX_COUNT,
    SessionNoteRecord,
    WorkspaceStore,
    note_digest,
)

logger = logging.getLogger(__name__)

#: The Run trace key that carries the reason memory degraded. One key, so a Run that
#: degrades twice reports its latest reason rather than a growing list.
SESSION_NOTES_DEGRADED_KEY = "session_notes_degraded"

#: Typed reasons, all of them degradations rather than failures.
SESSION_NOTES_PLANE_UNREADABLE = "plane_unreadable"
SESSION_NOTES_WORKING_COPY_UNREADABLE = "working_copy_unreadable"
SESSION_NOTES_PROMOTION_REFUSED = "promotion_refused"


@dataclass(frozen=True, slots=True)
class WorkingCopyNotes:
    """What one read of a Run's working copy found, and what it could not take."""

    records: tuple[SessionNoteRecord, ...] = ()
    #: Notes larger than the whole plane budget: refused, and named so the Run's trace
    #: can say so rather than leaving the write silently unremembered.
    oversized: tuple[str, ...] = ()
    #: Paths under the reserved directory that exist but were not read as notes: a
    #: symbolic link, an unreadable file, or something that is not a regular file.
    #: They may still hold memory, so they are protected from the delete diff.
    unreadable: tuple[str, ...] = ()
    degraded_reason: str | None = None


def read_working_copy_notes(workspace: Path) -> WorkingCopyNotes:
    """Return the notes this Run's working copy holds, by path, and what it skipped.

    A note that is a symbolic link, a non-file, unreadable, or larger than the plane's
    whole budget never enters memory: the working copy is the agent's own tree, memory
    is not worth failing a Run over, and a refused note is reported rather than dropped.
    """
    root = workspace / SESSION_NOTE_DIRECTORY
    records: list[SessionNoteRecord] = []
    oversized: list[str] = []
    unreadable: list[str] = []
    try:
        # Every other surface in this system refuses a symbolic link at a workspace
        # path, and the promotion read is the one that must not follow one: a linked
        # root reads bytes outside the workspace into memory, and an empty linked
        # target would make a Lane's whole memory look deleted.
        if root.is_symlink():
            logger.warning("Refused a symbolic link at the Session notes root")
            return WorkingCopyNotes(degraded_reason=SESSION_NOTES_WORKING_COPY_UNREADABLE)
        if not root.is_dir():
            return WorkingCopyNotes()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(workspace).as_posix()
            if not is_session_note(relative) or len(relative) > MAX_SESSION_NOTE_PATH_CHARS:
                continue
            try:
                if path.is_symlink():
                    unreadable.append(relative)
                    continue
                if not path.is_file():
                    continue
                if path.stat().st_size > SESSION_NOTES_MAX_BYTES:
                    oversized.append(relative)
                    continue
                records.append(SessionNoteRecord(relative_path=relative, content=path.read_bytes()))
            except OSError:
                logger.warning("Skipped an unreadable Session note %s", relative, exc_info=True)
                unreadable.append(relative)
    except OSError:
        logger.warning("Could not read the Run's notes working copy", exc_info=True)
        return WorkingCopyNotes(degraded_reason=SESSION_NOTES_WORKING_COPY_UNREADABLE)
    return WorkingCopyNotes(
        records=tuple(records),
        oversized=tuple(oversized),
        unreadable=tuple(unreadable),
    )


def read_working_copy(workspace: Path) -> tuple[SessionNoteRecord, ...]:
    """Return just the notes a working copy holds, for callers that need nothing else."""
    return read_working_copy_notes(workspace).records


def read_legacy_notes(
    *,
    source_workspace: Path | None,
    records: Sequence[InventoryPathRecord],
) -> tuple[SessionNoteRecord, ...]:
    """Read one legacy parent Run's registered notes for the one-time migration.

    Best effort by decision: a note this cannot read is left behind rather than
    refusing the Run that happens to bind first. A registered digest that disagrees
    with the bytes on disk is a refusal for that note alone, because memory that never
    verified is worse than memory that was not migrated.
    """
    if source_workspace is None:
        return ()
    selected: list[SessionNoteRecord] = []
    total_bytes = 0
    for record in records:
        if record.entry_type != "file" or not is_session_note(record.relative_path):
            continue
        if len(record.relative_path) > MAX_SESSION_NOTE_PATH_CHARS:
            continue
        if len(selected) >= SESSION_NOTES_MAX_COUNT:
            break
        if total_bytes + record.size_bytes > SESSION_NOTES_MAX_BYTES:
            break
        path = source_workspace / record.relative_path
        try:
            if path.is_symlink() or not path.is_file():
                continue
            content = path.read_bytes()
        except OSError:
            logger.warning("Could not read the legacy Run note %s", record.relative_path)
            continue
        if len(content) != record.size_bytes:
            continue
        if record.content_digest is not None and note_digest(content) != record.content_digest:
            continue
        selected.append(SessionNoteRecord(relative_path=record.relative_path, content=content))
        total_bytes += len(content)
    return tuple(selected)


def read_working_copy_safely(workspace: Path) -> WorkingCopyNotes:
    """Return a working copy read that never raises: a failure is a typed reason."""
    try:
        return read_working_copy_notes(workspace)
    except OSError:
        logger.warning("Could not read the Run's notes working copy", exc_info=True)
        return WorkingCopyNotes(degraded_reason=SESSION_NOTES_WORKING_COPY_UNREADABLE)


def read_working_copy_or_reason(
    workspace: Path,
) -> tuple[tuple[SessionNoteRecord, ...], str | None]:
    """Return a working copy's notes, or the reason the Run holds no memory from it."""
    read = read_working_copy_safely(workspace)
    reason = read.degraded_reason
    if reason is None and read.oversized:
        reason = SESSION_NOTES_BUDGET_REFUSED
    return read.records, reason


@dataclass(frozen=True, slots=True)
class SessionNotesBinding:
    """The notes a Run materialized into its working copy, and any degradation."""

    records: tuple[SessionNoteRecord, ...] = ()
    degraded_reason: str | None = None


class SessionNotesPlane:
    """One Run's view of its Session's memory: a baseline and the promotion of changes.

    The baseline is what this Run laid down, so deletion questions only ever address
    the notes this Run was given: a path another Lane wrote after this Run bound is not
    this Run's to remove, and a candidate whose plane bytes are no longer the ones this
    Run materialized is left alone. Writes are last-settled-wins, which ADR 0022 records
    as a residual.
    """

    def __init__(
        self,
        *,
        store: WorkspaceStore,
        session_id: str,
        workspace: Path | None = None,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._workspace = workspace
        self._baseline: dict[str, str] = {}

    def rebind(self, *, workspace: Path, records: Sequence[SessionNoteRecord]) -> None:
        """Attach the epoch this Run bound and the notes it materialized into it."""
        self._workspace = workspace
        self.materialized(records)

    def materialized(self, records: Sequence[SessionNoteRecord]) -> None:
        """Record the notes currently in this Run's working copy as its baseline."""
        self._baseline = {record.relative_path: note_digest(record.content) for record in records}

    async def load(self) -> SessionNotesBinding:
        """Read the Session's notes; unreadable memory is a reason, never an error."""
        try:
            records = await self._store.load_session_notes(session_id=self._session_id)
        except Exception:
            logger.warning("Could not read the Session notes plane", exc_info=True)
            return SessionNotesBinding(degraded_reason=SESSION_NOTES_PLANE_UNREADABLE)
        return SessionNotesBinding(records=tuple(records))

    async def promote(
        self,
        *,
        upserts: Sequence[SessionNoteRecord],
        deletes: Sequence[str] = (),
    ) -> str | None:
        """Promote notes into the Session's plane. Returns a reason, or ``None``."""
        if not upserts and not deletes:
            return None
        try:
            result = await self._store.promote_session_notes(
                session_id=self._session_id,
                upserts=upserts,
                deletes=deletes,
            )
        except Exception:
            logger.warning("Could not promote Session notes", exc_info=True)
            return SESSION_NOTES_PROMOTION_REFUSED
        if result.degraded_reason == SESSION_NOTES_LEASE_LOST:
            return result.degraded_reason
        refused = set(result.refused_paths)
        for note in upserts:
            if note.relative_path in refused:
                continue
            self._baseline[note.relative_path] = note_digest(note.content)
        for path in deletes:
            self._baseline.pop(path, None)
        return result.degraded_reason

    async def _confirm_deletions(self, candidates: Sequence[str]) -> tuple[str, ...]:
        """Return the candidates whose plane bytes are still what this Run materialized.

        A path another Lane rewrote after this Run bound is not this Run's to remove:
        that Lane's note is newer than the copy this Run deleted from, and memory is
        last-settled-wins rather than first-deleter-wins.
        """
        if not candidates:
            return ()
        try:
            live = {
                note.relative_path: note_digest(note.content)
                for note in await self._store.load_session_notes(session_id=self._session_id)
            }
        except Exception:
            logger.warning("Could not confirm Session note deletions", exc_info=True)
            return ()
        return tuple(path for path in candidates if live.get(path) == self._baseline.get(path))

    async def reconcile(self) -> str | None:
        """Promote this Run's working-copy differences from its materialized baseline."""
        workspace = self._workspace
        if workspace is None:
            return None
        read = read_working_copy_safely(workspace)
        if read.degraded_reason is not None:
            return read.degraded_reason
        current = read.records
        present: dict[str, str] = {
            record.relative_path: note_digest(record.content) for record in current
        }
        upserts = tuple(
            record
            for record in current
            if self._baseline.get(record.relative_path) != present[record.relative_path]
        )
        protected = {*read.oversized, *read.unreadable}
        candidates = tuple(
            path for path in self._baseline if path not in present and path not in protected
        )
        deletes = await self._confirm_deletions(candidates)
        reason = await self.promote(upserts=upserts, deletes=deletes)
        if reason is not None:
            return reason
        # A note too large for the whole plane is refused by name rather than dropped:
        # the Run's trace states it, and the bytes stay in the working copy.
        return SESSION_NOTES_BUDGET_REFUSED if read.oversized else None


__all__ = [
    "SESSION_NOTES_DEGRADED_KEY",
    "SESSION_NOTES_PLANE_UNREADABLE",
    "SESSION_NOTES_PROMOTION_REFUSED",
    "SESSION_NOTES_WORKING_COPY_UNREADABLE",
    "SessionNotesBinding",
    "SessionNotesPlane",
    "WorkingCopyNotes",
    "read_legacy_notes",
    "read_working_copy",
    "read_working_copy_notes",
    "read_working_copy_or_reason",
    "read_working_copy_safely",
]
