# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A process-local WorkspaceStore for unit tests of what a Run does with its workspace.

PostgreSQL's PGWorkspaceStore is the product's store; its handoff, inventory,
spill and Session-note rules run against a real database in
tests/integration/test_answer_runs_pg.py and test_agent_session_pg.py.
"""

from collections.abc import Sequence
from heapq import nsmallest

from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    DEFAULT_SESSION_NOTES_LIMITS,
    SESSION_NOTES_BUDGET_REFUSED,
    SESSION_NOTES_LEASE_LOST,
    CommittedSpillRecord,
    HandoffCommit,
    HandoffConflict,
    HandoffLeaseLost,
    HandoffResult,
    RunArtifactRecord,
    SessionNoteRecord,
    SessionNotesLimits,
    SessionNotesPromotion,
    _validate_spill_page_limit,
    select_promotable_session_notes,
)


class InMemoryWorkspaceStore:
    """Process-local workspace store with the product store's read orders."""

    def __init__(
        self,
        *,
        workspace_epoch: int | None = None,
        live: bool = True,
    ) -> None:
        self.workspace_epoch = workspace_epoch
        self.live = live
        self.inventory: list[InventoryPathRecord] = []
        self.spills: list[CommittedSpillRecord] = []
        self.artifacts: list[RunArtifactRecord] = []
        self.session_notes: dict[str, dict[str, bytes]] = {}

    async def handoff_epoch(
        self,
        *,
        expected_epoch: int | None,
        destination_epoch: int,
        inventory: Sequence[InventoryPathRecord],
    ) -> HandoffResult:
        if not self.live:
            return HandoffLeaseLost()
        if self.workspace_epoch != expected_epoch:
            return HandoffConflict(
                expected_epoch=expected_epoch, current_epoch=self.workspace_epoch
            )
        if destination_epoch < 1:
            raise ValueError("destination epoch must be positive")
        self.workspace_epoch = destination_epoch
        self.inventory = list(inventory)
        return HandoffCommit(workspace_epoch=destination_epoch)

    async def load_inventory(self) -> tuple[InventoryPathRecord, ...]:
        # Path order, as the durable adapter's own ORDER BY gives: the note set is
        # a filter over this observation, and two orders would be two answers.
        return tuple(sorted(self.inventory, key=lambda record: record.relative_path))

    async def load_run_artifacts(self) -> tuple[RunArtifactRecord, ...]:
        return tuple(sorted(self.artifacts, key=lambda record: record.relative_path))

    async def load_session_notes(self, *, session_id: str) -> tuple[SessionNoteRecord, ...]:
        return tuple(
            SessionNoteRecord(relative_path=path, content=content)
            for path, content in sorted(self.session_notes.get(session_id, {}).items())
        )

    async def promote_session_notes(
        self,
        *,
        session_id: str,
        upserts: Sequence[SessionNoteRecord],
        deletes: Sequence[str] = (),
        limits: SessionNotesLimits = DEFAULT_SESSION_NOTES_LIMITS,
    ) -> SessionNotesPromotion:
        if not self.live:
            return SessionNotesPromotion(degraded_reason=SESSION_NOTES_LEASE_LOST)
        plane = dict(self.session_notes.get(session_id, {}))
        accepted, refused = select_promotable_session_notes(
            existing={path: len(content) for path, content in plane.items()},
            upserts=upserts,
            deletes=deletes,
            limits=limits,
        )
        removed = [path for path in deletes if path in plane]
        for path in removed:
            del plane[path]
        for note in accepted:
            plane[note.relative_path] = note.content
        self.session_notes[session_id] = plane
        return SessionNotesPromotion(
            promoted=len(accepted),
            deleted=len(removed),
            refused_paths=refused,
            degraded_reason=(SESSION_NOTES_BUDGET_REFUSED if refused else None),
        )

    async def load_spills_page(
        self, *, after_resource_id: str | None, limit: int
    ) -> tuple[CommittedSpillRecord, ...]:
        _validate_spill_page_limit(limit)
        matching = (
            spill
            for spill in self.spills
            if after_resource_id is None or spill.resource_id > after_resource_id
        )
        return tuple(nsmallest(limit, matching, key=lambda spill: spill.resource_id))

    async def load_recent_spills(self, *, limit: int) -> tuple[CommittedSpillRecord, ...]:
        """Return this Run's newest committed spills first.

        The producing effect intent is the only monotone marker a committed spill
        carries: its resource id is a random handle and the row records no
        settlement time. The cursor-paged recovery read stays a separate method
        because it walks every spill, while this one only needs the newest few.
        """
        _validate_spill_page_limit(limit)
        return tuple(
            sorted(
                self.spills,
                key=lambda spill: (spill.intent_id, spill.resource_id),
                reverse=True,
            )[:limit]
        )
