# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Compose the re-readable identities one compaction summary carries forward.

A projection's ``durable_handles`` are continuation memory: the only way the next
turn can reach content the covered prefix took out of the request. Two classes of
identity belong there, and they have no common monotone order to merge by. An
Evidence handle is ordered by admission; a committed spill carries only the
UUIDv7 identity of the effect intent that produced it. Selection is therefore by
class, not by a global recency merge: spills claim a reserved share of the handle
budget, and Evidence fills the remainder in admission order.

A Session Note is a third kind of continuation identity and is deliberately not a
handle: memory belongs to the Agent Session (ADR 0022), the Run holds a materialized
working copy, and the next turn names the *path* and reads current bytes. Which paths
are notes is declared by the reserved directory rather than by a second registry, so
the note set and the working copy can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence

from dlightrag.engine.answer.publication import artifact_read_call
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    CommittedSpillRecord,
    RunArtifactRecord,
    SessionNoteRecord,
)

#: How many committed spills one summary may name. The share is reserved, not
#: first-come: the omitted handle is the expensive failure, and Evidence, which
#: a retrieval-heavy Run admits in tens, must not be able to crowd spills out.
#: The existing summary cap bounds the total, so this share also bounds what a
#: Run with many spills spends of it.
MAX_SPILL_HANDLES = 20

#: The one Workspace directory a Session's memory lives at, inside every Run's own
#: working copy. The directory is not a resource plane and grants nothing: it is where
#: a file has to sit for the framework to treat it as memory. `artifacts/` is elsewhere
#: on purpose, because an attached file means publication to the user.
SESSION_NOTE_DIRECTORY = "notes"

#: How many published Artifacts one summary may name. Like the spill share, it is
#: reserved rather than first-come: a conversation that iterates on one deliverable
#: must keep the handle that reads its last published version even after the receipt
#: that taught it has been compacted away.
MAX_ARTIFACT_HANDLES = 8

#: How many Session Notes one compaction summary may name. A note is written to be read
#: again, so the cap is generous for a plan file and a findings file and still bounds
#: the summary's prose. Ordering is the working copy's own: by path, so a note the
#: agent keeps updating keeps its place instead of moving to the front on every write.
MAX_NAMED_SESSION_NOTES = 8

#: The longest note path one summary may name. The count cap alone does not bound
#: the prose: a path may be thousands of characters, and the summary's own
#: strictly-reducing check counts it, so an absurd name would spend the summary's
#: budget instead of the note's.
MAX_SESSION_NOTE_PATH_CHARS = 512


def published_artifact_handle(record: RunArtifactRecord) -> str:
    """Render one published Artifact as the call that reads its version again.

    The address is the one publication binds (the Artifact path, hashed), so the handle
    a Run names and the bytes a later Run adopts are the same statement: a conversation
    that keeps working on one deliverable reads the version it published earlier, and
    the next publication is a new version rather than a rewrite of that one.
    """
    return (
        f"[artifact] {record.relative_path} ({record.size_bytes} bytes) — "
        f"revisit it with {artifact_read_call(record.relative_path)} before editing it again"
    )


def session_notes_message(records: Sequence[SessionNoteRecord]) -> str:
    """Return the one static prefix that names the notes this Run holds.

    The text is a function of the note set alone: no clock, no Run id, and no per-turn
    remainder. Empty input is empty output, so a Run that bound no notes says nothing
    about memory. Naming a call to read implies a tool that can make it, so an inert
    Fast workspace composes nothing.
    """
    if not records:
        return ""
    lines = [
        "These Session notes are already in this workspace. Read one with "
        "read(path=...) and continue it; do not re-derive what one states.",
        "",
    ]
    lines.extend(f"- {record.relative_path} ({len(record.content)} bytes)" for record in records)
    return "\n".join(lines)


def spill_handle(spill: CommittedSpillRecord) -> str:
    """Render one committed spill as a handle the next turn can act on.

    The receipt that taught the model this call is inside the covered prefix by
    the time this handle is the only thing left, so the call is spelled out. The
    cursor is omitted rather than shown as ``0``: the Tool takes a string cursor,
    and an omitted one already starts at the first line.
    """
    return (
        f"[spill] {spill.resource_id} ({spill.size_bytes} bytes) — "
        f're-read with read(resource_id="{spill.resource_id}")'
    )


def is_session_note(relative_path: str) -> bool:
    """Return whether one Workspace-relative path sits in the reserved notes directory."""
    path = relative_path.strip()
    prefix = f"{SESSION_NOTE_DIRECTORY}/"
    if not path.startswith(prefix) or path == prefix:
        return False
    return ".." not in path.split("/")


def session_note_handle(record: InventoryPathRecord) -> str:
    """Render one Run Note as the path call that reads it again.

    The recorded digest is deliberately absent: the Inventory keeps a digest only
    for paths this framework wrote itself (a `bash` call re-observes the whole
    workspace without one), and a note is live memory — the Session owns the bytes
    and this Run holds a working copy (ADR 0022) — so the read that follows serves
    the current copy rather than bytes frozen at a compaction.
    The path is rendered with its repr for the same reason the spill handle is: a
    name may hold a quote, and a call the model cannot reproduce is worse than no
    call.

    The moment is named as well as the call. A handle that only states the capability
    leaves the reader to decide when a re-read is worth it, and a live Research Run
    measured that decision: two compactions named the note and the Run re-derived the
    values with fresh searches instead of reading them. Naming the moment the read is
    for costs one clause and points at exactly the step where a value the summary does
    not state is needed.
    """
    return (
        f"[note] {record.relative_path} ({record.size_bytes} bytes) — "
        f"re-read with read(path={record.relative_path!r}) before a step that needs a "
        f"value this summary does not state"
    )


def compose_session_notes(records: Sequence[InventoryPathRecord]) -> list[str]:
    """Return the summary's Session Note lines, in working-copy order, bounded by the cap."""
    notes = [
        session_note_handle(record)
        for record in records
        if record.entry_type == "file"
        and is_session_note(record.relative_path)
        and len(record.relative_path) <= MAX_SESSION_NOTE_PATH_CHARS
    ]
    return notes[:MAX_NAMED_SESSION_NOTES]


def published_artifact_handles(
    records: Sequence[RunArtifactRecord],
) -> list[str]:
    """Return this Run's published Artifact handles, by path, bounded by their share."""
    return [published_artifact_handle(record) for record in records[:MAX_ARTIFACT_HANDLES]]


def compose_durable_handles(
    *,
    spills: Sequence[CommittedSpillRecord],
    evidence_handles: Sequence[str],
    artifacts: Sequence[RunArtifactRecord] = (),
) -> list[str]:
    """Return the summary's handle list: the newest spills, published products, Evidence.

    ``spills`` arrives newest-first from the Run's durable spill rows. Each non-Evidence
    class has its own reserved share, because the omitted handle is the expensive
    failure and a retrieval-heavy Run admits Evidence in tens; Evidence keeps its own
    admission order behind them and is bounded by the summary's cap, never here.
    """
    handles = [spill_handle(spill) for spill in spills[:MAX_SPILL_HANDLES]]
    handles.extend(published_artifact_handles(artifacts))
    handles.extend(evidence_handles)
    return handles


__all__ = [
    "MAX_ARTIFACT_HANDLES",
    "MAX_NAMED_SESSION_NOTES",
    "MAX_SESSION_NOTE_PATH_CHARS",
    "MAX_SPILL_HANDLES",
    "SESSION_NOTE_DIRECTORY",
    "compose_durable_handles",
    "compose_session_notes",
    "published_artifact_handle",
    "published_artifact_handles",
    "is_session_note",
    "session_note_handle",
    "session_notes_message",
    "spill_handle",
]
