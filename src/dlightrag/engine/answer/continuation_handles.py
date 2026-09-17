# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Compose the re-readable identities one compaction summary carries forward.

A projection's ``durable_handles`` are continuation memory: the only way the next
turn can reach content the covered prefix took out of the request. Two classes of
identity belong there, and they have no common monotone order to merge by. An
Evidence handle is ordered by admission; a committed spill carries only the
UUIDv7 identity of the effect intent that produced it. Selection is therefore by
class, not by a global recency merge: spills claim a reserved share of the handle
budget, and Evidence fills the remainder in admission order.

A Run Note is a third kind of continuation identity and is deliberately not a
handle: it is a file the Run keeps, so the next turn names its *path* and reads
current bytes. The Workspace Inventory is the one observation authority for what
the workspace holds, and the note set is a filter over that observation rather than
a second registry that could disagree with it.
"""

from __future__ import annotations

from collections.abc import Sequence

from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import CommittedSpillRecord

#: How many committed spills one summary may name. The share is reserved, not
#: first-come: the omitted handle is the expensive failure, and Evidence, which
#: a retrieval-heavy Run admits in tens, must not be able to crowd spills out.
#: The existing summary cap bounds the total, so this share also bounds what a
#: Run with many spills spends of it.
MAX_SPILL_HANDLES = 20

#: The one Workspace directory a Run declares its continuation files in. The
#: directory is not a resource plane and grants nothing: it is where a file has to
#: sit for the framework to name it in the next summary. `artifacts/` is elsewhere
#: on purpose, because an attached file means publication to the user.
RUN_NOTE_DIRECTORY = "notes"

#: How many Run Notes one summary may name. A note is written to be read again, so
#: the cap is generous for a plan file and a findings file and still bounds the
#: summary's prose. Ordering is the Inventory's own: by path, so a note the agent
#: keeps updating keeps its place instead of moving to the front on every write.
MAX_RUN_NOTES = 8

#: The longest note path one summary may name. The count cap alone does not bound
#: the prose: a path may be thousands of characters, and the summary's own
#: strictly-reducing check counts it, so an absurd name would spend the summary's
#: budget instead of the note's.
MAX_RUN_NOTE_PATH_CHARS = 512

#: Total bytes one continuation may copy. A note is conclusions, not a dump, and
#: the copy multiplies storage by every follow-up; 1 MiB is generous for the count
#: cap and still bounds a carry that would otherwise clone an unbounded tree.
MAX_CARRIED_RUN_NOTE_BYTES = 1_048_576


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


def is_run_note(relative_path: str) -> bool:
    """Return whether one Workspace-relative path sits in the notes directory."""
    path = relative_path.strip()
    prefix = f"{RUN_NOTE_DIRECTORY}/"
    if not path.startswith(prefix) or path == prefix:
        return False
    return ".." not in path.split("/")


def run_note_handle(record: InventoryPathRecord) -> str:
    """Render one Run Note as the path call that reads it again.

    The recorded digest is deliberately absent: the Inventory keeps a digest only
    for paths this framework wrote itself (a `bash` call re-observes the whole
    workspace without one), and a note is a live file the Run owns, so the read
    that follows serves its current bytes rather than bytes frozen at a compaction.
    The path is rendered with its repr for the same reason the spill handle is: a
    name may hold a quote, and a call the model cannot reproduce is worse than no
    call.
    """
    return (
        f"[note] {record.relative_path} ({record.size_bytes} bytes) — "
        f"re-read with read(path={record.relative_path!r})"
    )


def compose_run_notes(records: Sequence[InventoryPathRecord]) -> list[str]:
    """Return the summary's Run Note lines, in Inventory order, bounded by the cap."""
    notes = [
        run_note_handle(record)
        for record in records
        if record.entry_type == "file"
        and is_run_note(record.relative_path)
        and len(record.relative_path) <= MAX_RUN_NOTE_PATH_CHARS
    ]
    return notes[:MAX_RUN_NOTES]


def select_carried_run_notes(
    records: Sequence[InventoryPathRecord],
) -> tuple[InventoryPathRecord, ...]:
    """Return the notes a continuation copies, in Inventory order, within the carry bound.

    A path over the read limit is skipped rather than truncated: the name is the
    identity, and a clipped one would point at a file that does not exist. The
    count cap is the summary's own, so a carry cannot invent a note the next
    compaction would refuse to name. The first note that would exceed the byte
    ceiling stops the selection; later files are not reordered to squeeze in.
    """
    selected: list[InventoryPathRecord] = []
    total_bytes = 0
    for record in records:
        if record.entry_type != "file" or not is_run_note(record.relative_path):
            continue
        if len(record.relative_path) > MAX_RUN_NOTE_PATH_CHARS:
            continue
        if len(selected) >= MAX_RUN_NOTES:
            break
        if total_bytes + record.size_bytes > MAX_CARRIED_RUN_NOTE_BYTES:
            break
        selected.append(record)
        total_bytes += record.size_bytes
    return tuple(selected)


def carried_run_notes_message(records: Sequence[InventoryPathRecord]) -> str:
    """Return the one static prefix that names the notes this Run inherited.

    The text is a function of the carried set alone: no clock, no Run id, and no
    per-turn remainder. Empty input is empty output so a Run that inherited
    nothing says nothing about carrying.
    """
    if not records:
        return ""
    lines = [
        "The parent Run's notes are already in this workspace. "
        "Read them with read(path=...); do not re-derive their contents.",
        "",
    ]
    lines.extend(f"- {record.relative_path} ({record.size_bytes} bytes)" for record in records)
    return "\n".join(lines)


def compose_durable_handles(
    *,
    spills: Sequence[CommittedSpillRecord],
    evidence_handles: Sequence[str],
) -> list[str]:
    """Return the summary's handle list: the newest spills, then Evidence.

    ``spills`` arrives newest-first from the Run's durable spill rows. Evidence
    handles keep their own admission order behind them and are bounded by the
    summary's cap, never by this function.
    """
    handles = [spill_handle(spill) for spill in spills[:MAX_SPILL_HANDLES]]
    handles.extend(evidence_handles)
    return handles


__all__ = [
    "MAX_CARRIED_RUN_NOTE_BYTES",
    "MAX_RUN_NOTES",
    "MAX_RUN_NOTE_PATH_CHARS",
    "MAX_SPILL_HANDLES",
    "RUN_NOTE_DIRECTORY",
    "carried_run_notes_message",
    "compose_durable_handles",
    "compose_run_notes",
    "is_run_note",
    "run_note_handle",
    "select_carried_run_notes",
    "spill_handle",
]
