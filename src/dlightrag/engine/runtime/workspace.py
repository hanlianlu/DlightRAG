# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Claim-bound workspace epoch, inventory, committed spills, and Session notes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from dlightrag.engine.runtime.settlements import InventoryPathRecord

#: How many notes one Agent Session keeps, and how many bytes it may hold. Memory
#: belongs to the Session (ADR 0022), so the bound is the Session's rather than a
#: Run's: a note outlives the Run that wrote it, and the plane must still stay small
#: enough to be memory rather than a second transcript. Refusal, never eviction: an
#: over-budget note is the writing Run's to see refused, not an older note's to lose.
SESSION_NOTES_MAX_COUNT = 64
SESSION_NOTES_MAX_BYTES = 256 * 1024

#: Why memory degraded instead of landing. The Run records the reason and continues:
#: a Run that cannot be given its notes still has its transcript, its Evidence, and
#: its Products.
SESSION_NOTES_BUDGET_REFUSED = "budget_refused"
SESSION_NOTES_LEASE_LOST = "lease_lost"
SESSION_NOTES_MATERIALIZE_FAILED = "materialize_failed"


def note_digest(content: bytes) -> str:
    """Return the digest a note's bytes are compared by."""
    return hashlib.sha256(content).hexdigest()


def validate_note_path(relative_path: str) -> bool:
    """Return whether a promotion may address this Session-relative path at all.

    Note semantics belong to the answer layer (`is_session_note`); this is the plane's own
    path safety, so a plane never stores an absolute path or escapes its Session.
    """
    path = relative_path.strip()
    if not path or path != relative_path or path.startswith("/") or len(path) > 1024:
        return False
    return ".." not in path.split("/") and not path.endswith("/")


@dataclass(frozen=True, slots=True)
class SessionNotesLimits:
    """The plane's bounds, as one value the deployment can set (``answer.agent``).

    The defaults are the ADR's: 64 notes and 256 KiB per Agent Session.
    """

    max_count: int = SESSION_NOTES_MAX_COUNT
    max_bytes: int = SESSION_NOTES_MAX_BYTES

    def __post_init__(self) -> None:
        if self.max_count < 1:
            raise ValueError("session notes count bound must be positive")
        if self.max_bytes < 1:
            raise ValueError("session notes byte bound must be positive")


#: The bounds every caller uses unless the deployment configured its own.
DEFAULT_SESSION_NOTES_LIMITS = SessionNotesLimits()


def select_promotable_session_notes(
    *,
    existing: Mapping[str, int],
    upserts: Sequence[SessionNoteRecord],
    deletes: Sequence[str],
    limits: SessionNotesLimits = DEFAULT_SESSION_NOTES_LIMITS,
) -> tuple[tuple[SessionNoteRecord, ...], tuple[str, ...]]:
    """Return the upserts the plane's budget admits, and the paths it refuses.

    ``existing`` is the plane's current state as path to byte size. Deletion is
    priced first, so a Run that removes a note frees that budget in the same
    promotion. Admission is by path, so the same payload always admits the same
    subset rather than depending on the order a Run happened to write in.
    """
    removed = set(deletes)
    kept = {path: size for path, size in existing.items() if path not in removed}
    total_bytes = sum(kept.values())
    accepted: list[SessionNoteRecord] = []
    refused: list[str] = []
    for note in sorted(upserts, key=lambda item: item.relative_path):
        path = note.relative_path
        replacing = path in kept
        size_bytes = len(note.content)
        next_count = len(kept) + (0 if replacing else 1)
        next_bytes = total_bytes - (kept[path] if replacing else 0) + size_bytes
        if next_count > limits.max_count or next_bytes > limits.max_bytes:
            refused.append(path)
            continue
        kept[path] = size_bytes
        total_bytes = next_bytes
        accepted.append(note)
    return tuple(accepted), tuple(refused)


@dataclass(frozen=True, slots=True)
class SessionNoteRecord:
    """The bytes of one Session note, as materialized into or promoted from a copy."""

    relative_path: str
    content: bytes

    def __post_init__(self) -> None:
        if not validate_note_path(self.relative_path):
            raise ValueError("session note path must be a Session-relative file path")


@dataclass(frozen=True, slots=True)
class SessionNotesPromotion:
    """What one promotion landed, and the reason memory degraded when it did not."""

    promoted: int = 0
    deleted: int = 0
    refused_paths: tuple[str, ...] = ()
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CommittedSpillRecord:
    """One committed spill digest for volume recovery."""

    resource_id: str
    content_digest: str
    size_bytes: int
    session_id: str
    intent_id: str


@dataclass(frozen=True, slots=True)
class HandoffCommit:
    """CAS moved workspace_epoch to the destination fencing generation."""

    workspace_epoch: int


@dataclass(frozen=True, slots=True)
class HandoffConflict:
    """The expected workspace_epoch no longer matches the stored value."""

    expected_epoch: int | None
    current_epoch: int | None


@dataclass(frozen=True, slots=True)
class HandoffLeaseLost:
    """The caller no longer holds the live lease."""


type HandoffResult = HandoffCommit | HandoffConflict | HandoffLeaseLost


@dataclass(frozen=True, slots=True)
class RunArtifactRecord:
    """One Artifact root this Run attached, as its own settlement recorded it.

    The address is derived from the relative path, so a summary can name the handle a
    later turn will read the published version with before that publication exists.
    """

    relative_path: str
    label: str
    size_bytes: int
    content_digest: str
    presentation: str

    def __post_init__(self) -> None:
        if not self.relative_path.strip():
            raise ValueError("artifact record path cannot be empty")
        if self.size_bytes < 0:
            raise ValueError("artifact record size cannot be negative")
        if len(self.content_digest) != 64:
            raise ValueError("artifact record digest must be a SHA-256 hex digest")


class WorkspaceStore(Protocol):
    """Fenced workspace metadata. Handoff never advances durable progress."""

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult: ...

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]: ...

    async def load_run_artifacts(self) -> tuple[RunArtifactRecord, ...]: ...

    async def load_session_notes(self, *, session_id: str) -> tuple[SessionNoteRecord, ...]: ...

    async def promote_session_notes(
        self,
        *,
        session_id: str,
        upserts: Sequence[SessionNoteRecord],
        deletes: Sequence[str] = (),
        limits: SessionNotesLimits = DEFAULT_SESSION_NOTES_LIMITS,
    ) -> SessionNotesPromotion: ...

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]: ...

    async def load_recent_spills(self, *, limit: int) -> tuple[CommittedSpillRecord, ...]: ...


_MAX_SPILL_PAGE_LIMIT = 1_000


def _validate_spill_page_limit(limit: int) -> None:
    if limit < 1 or limit > _MAX_SPILL_PAGE_LIMIT:
        raise ValueError(f"spill page limit must be between 1 and {_MAX_SPILL_PAGE_LIMIT}")


__all__ = [
    "CommittedSpillRecord",
    "HandoffCommit",
    "HandoffConflict",
    "HandoffLeaseLost",
    "HandoffResult",
    "RunArtifactRecord",
    "SESSION_NOTES_BUDGET_REFUSED",
    "SESSION_NOTES_LEASE_LOST",
    "SESSION_NOTES_MATERIALIZE_FAILED",
    "SESSION_NOTES_MAX_BYTES",
    "SESSION_NOTES_MAX_COUNT",
    "DEFAULT_SESSION_NOTES_LIMITS",
    "SessionNoteRecord",
    "SessionNotesLimits",
    "SessionNotesPromotion",
    "WorkspaceStore",
    "note_digest",
    "select_promotable_session_notes",
    "validate_note_path",
]
