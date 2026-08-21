# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deep model/owner-facing Memory façade.

One interface for every cross-conversation Profile operation. The host owns
identity, eligibility, and context placement; this module owns the checklist,
atomic write commit, standing recall, and lifecycle policy.
"""

from __future__ import annotations

from datetime import datetime

from dlightrag_memory.models import MemoryProvenance, MemoryRecord, MemoryWrite
from dlightrag_memory.recall import render_auto_recall, select_auto_recall
from dlightrag_memory.store import MemoryStore, commit_memory_write

_MANAGEMENT_PROVENANCE = MemoryProvenance(run_id="management", session_id="management")


class Memory:
    """Cross-conversation Owner Profile Memory bound to one storage adapter."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def remember(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        kind: str,
        body: str,
        confidence: float,
        provenance: MemoryProvenance,
        supersedes_id: str | None = None,
    ) -> MemoryRecord | None:
        """Store one preference or fact, optionally superseding an older record."""
        return await commit_memory_write(
            self._store,
            MemoryWrite(
                owner_id=owner_id,
                auth_mode=auth_mode,
                kind=kind,  # type: ignore[arg-type]
                body=body,
                confidence=confidence,
                provenance=provenance,
                action="remember",
                supersedes_id=supersedes_id,
            ),
        )

    async def forget(
        self,
        *,
        owner_id: str,
        auth_mode: str,
        memory_id: str | None = None,
        body: str | None = None,
        provenance: MemoryProvenance | None = None,
    ) -> None:
        """Hard-delete one record by id, or every record with an exact body."""
        await commit_memory_write(
            self._store,
            MemoryWrite(
                owner_id=owner_id,
                auth_mode=auth_mode,
                kind="preference",
                body=body or "",
                confidence=1.0,
                provenance=provenance or _MANAGEMENT_PROVENANCE,
                action="forget",
                supersedes_id=memory_id,
            ),
        )

    async def list_active(self, *, owner_id: str) -> tuple[MemoryRecord, ...]:
        """Return one owner's active records, newest first."""
        return await self._store.list_active(owner_id=owner_id)

    async def standing_text(self, *, owner_id: str) -> str:
        """Render the bounded non-citable standing block for one owner."""
        return render_auto_recall(select_auto_recall(await self.list_active(owner_id=owner_id)))

    async def purge_superseded(self, *, older_than: datetime) -> int:
        """Delete superseded history past the retention floor."""
        return await self._store.purge_superseded(older_than=older_than)


__all__ = ["Memory"]
