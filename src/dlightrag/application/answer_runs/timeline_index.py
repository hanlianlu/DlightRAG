# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run timeline as a derived index, kept for as long as its version holds.

A Run's timeline is derived at Run grain and read a page at a time: the merged step order and
the whole Run's marks are the same facts for every page of the same Run and for its overview.
Reading the canonical entries once per page recomputed that whole derivation every time, which is
what made page k as expensive as page one.

This module holds the derivation where a reader surface can reuse it. An index is built from one
version token and answers only for that token: :meth:`TimelineIndexCache.get` compares the token
the store states now with the one the index was built under, so a changed Run is a miss rather
than stale steps. The cache is process-local, bounded, and disposable — it creates no durable
second truth ([ADR 0069](../../../../docs/adr/0069-the-web-reads-run-activity-from-the-runs-it-keeps.md)),
and losing it costs one rebuild, never correctness.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .timeline import (
    AgentIdentity,
    RunTimelineStep,
    decision_marks,
)

#: How many Runs keep a derived index: the one on screen and the few a reader jumps between.
TIMELINE_INDEX_RUNS = 8

_TOKEN_VERSION_CHARS = 20
"""How much of the token hash a client and a cache validator see."""


@dataclass(frozen=True, slots=True)
class RunTimelineToken:
    """What a derived timeline index is valid for: the Run's version and the roster it merged.

    ``run_version`` is the Run's durable progress version, which its writers bump once per
    durable entry transaction and never for a token or reasoning batch. ``last_sequences`` is
    each Session's highest stored sequence: a recovery write can append an entry without moving
    the progress version, and a page must not miss it. ``roster`` names the Run's Children, in
    the order the roster states them, because a Child appearing changes the agent identities and
    order the merge uses without touching the Run's own version.
    """

    run_version: int
    status: str
    roster: tuple[tuple[str, str], ...]
    last_sequences: tuple[tuple[str, int], ...]

    @classmethod
    def from_store(cls, row: Mapping[str, Any]) -> RunTimelineToken:
        """One token from the store's row: counts as integers, pairs as tuples."""
        roster = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in row.get("roster") or ()
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        )
        sequences = tuple(
            (str(pair[0]), int(pair[1]))
            for pair in row.get("sequences") or ()
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        )
        return cls(
            run_version=int(row.get("run_version") or 0),
            status=str(row.get("status") or ""),
            roster=roster,
            last_sequences=sequences,
        )

    @property
    def version(self) -> str:
        """The opaque version string this token states, stable across processes and restarts."""
        material = json.dumps(
            [self.run_version, self.status, self.roster, self.last_sequences],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return digest[:_TOKEN_VERSION_CHARS]


@dataclass(frozen=True, slots=True)
class TimelineIndex:
    """One Run's timeline, derived once for one token and reused by every read of it."""

    token: RunTimelineToken
    agents: tuple[AgentIdentity, ...]
    steps: tuple[RunTimelineStep, ...]
    marks: Mapping[tuple[str, int], tuple[str, ...]]

    @property
    def version(self) -> str:
        """The token's version: what a search page and a cache validator state."""
        return self.token.version

    @classmethod
    def build(
        cls,
        *,
        token: RunTimelineToken,
        agents: tuple[AgentIdentity, ...],
        steps: tuple[RunTimelineStep, ...],
    ) -> TimelineIndex:
        """Derive one index from the merge and its marks."""
        marks = decision_marks(steps)
        return cls(
            token=token,
            agents=agents,
            steps=steps,
            marks=marks,
        )


class TimelineIndexCache:
    """The last few Runs' derived indexes, keyed by the version token they were built under.

    One instance lives on the process's Answer service. Reads are single-threaded on the event
    loop, so no lock is needed; the byte cost of an index is the merge's Python objects, and
    :data:`TIMELINE_INDEX_RUNS` bounds how many of them are kept.
    """

    def __init__(self, limit: int = TIMELINE_INDEX_RUNS) -> None:
        self._limit = max(1, int(limit))
        self._held: OrderedDict[tuple[str, str], TimelineIndex] = OrderedDict()

    def get(self, owner_id: str, run_id: str, token: RunTimelineToken) -> TimelineIndex | None:
        """The index for this owner, Run and token, or None when the Run moved since it was built."""
        key = (owner_id, run_id)
        held = self._held.get(key)
        if held is None or held.token != token:
            return None
        self._held.move_to_end(key)
        return held

    def put(self, owner_id: str, run_id: str, index: TimelineIndex) -> None:
        """Keep one index, evicting the Run read longest ago beyond the bound."""
        key = (owner_id, run_id)
        self._held[key] = index
        self._held.move_to_end(key)
        while len(self._held) > self._limit:
            self._held.popitem(last=False)

    def clear(self) -> None:
        """Drop every derived index."""
        self._held.clear()


__all__ = [
    "RunTimelineToken",
    "TIMELINE_INDEX_RUNS",
    "TimelineIndex",
    "TimelineIndexCache",
]
