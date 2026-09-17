# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Compose the re-readable identities one compaction summary carries forward.

A projection's ``durable_handles`` are continuation memory: the only way the next
turn can reach content the covered prefix took out of the request. Two classes of
identity belong there, and they have no common monotone order to merge by. An
Evidence handle is ordered by admission; a committed spill carries only the
UUIDv7 identity of the effect intent that produced it. Selection is therefore by
class, not by a global recency merge: spills claim a reserved share of the handle
budget, and Evidence fills the remainder in admission order.
"""

from __future__ import annotations

from collections.abc import Sequence

from dlightrag.engine.runtime.workspace import CommittedSpillRecord

#: How many committed spills one summary may name. The share is reserved, not
#: first-come: the omitted handle is the expensive failure, and Evidence, which
#: a retrieval-heavy Run admits in tens, must not be able to crowd spills out.
#: The existing summary cap bounds the total, so this share also bounds what a
#: Run with many spills spends of it.
MAX_SPILL_HANDLES = 20


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


__all__ = ["MAX_SPILL_HANDLES", "compose_durable_handles", "spill_handle"]
