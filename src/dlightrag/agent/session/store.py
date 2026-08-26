# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The agent session store contract and its closed commit outcomes.

Every mutating operation returns an explicit commit outcome value; expected
conflicts are values, never database exceptions. Canonical transactions use
exact register-sequence CAS so unrelated Lanes never conflict, while one
Session-wide commit sequence orders successful transactions. The legacy
main-Lane append methods remain only until M3 migrates JournalRunBoundaries.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from dlightrag.agent.session.effects import EffectSettlement, HostUpdateT
from dlightrag.agent.session.entries import SessionEntry
from dlightrag.agent.session.graph import AgentSessionGraph
from dlightrag.agent.session.ids import EntryId, IntentId, LaneId, SessionId
from dlightrag.agent.session.projection import ContextProjection
from dlightrag.agent.session.registers import LaneHead, LaneState, RegisterRecord
from dlightrag.agent.session.transactions import (
    SessionTransaction,
    TransactionOutcome,
)
from dlightrag.agent.session.tree import AgentSessionTree, LaneSnapshot

type SessionProgressClass = Literal["live", "prelude"]


@dataclass(frozen=True, slots=True)
class SessionCommit:
    """One committed append: new version and the contiguous sequences written."""

    version: int
    appended_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EffectCommit:
    """One committed settlement: new version, sequences, and the settled intent."""

    version: int
    appended_sequences: tuple[int, ...]
    intent_id: IntentId
    outcome: str


@dataclass(frozen=True, slots=True)
class VersionConflict:
    """The expected session version no longer matches the stored version."""

    expected_version: int
    current_version: int


@dataclass(frozen=True, slots=True)
class LeaseLost:
    """The caller's lease no longer owns this session."""


@dataclass(frozen=True, slots=True)
class EffectMissing:
    """No unsettled intent with this id exists in the session."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EffectAlreadySettled:
    """The intent was already settled; load and fold the committed settlement."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EffectContractChanged:
    """The stored intent no longer matches the settlement's tool contract."""

    intent_id: IntentId


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    """A host update collided with existing evidence or resource identity."""


type AppendCommit = SessionCommit | VersionConflict | LeaseLost
type SettleCommit = (
    EffectCommit
    | VersionConflict
    | LeaseLost
    | EffectMissing
    | EffectAlreadySettled
    | EffectContractChanged
    | EvidenceConflict
)


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    """One immutable Session Tree and its exact current-register snapshot."""

    session_id: SessionId
    commit_sequence: int
    entries: tuple[SessionEntry, ...]
    active_projection: ContextProjection | None
    registers: tuple[RegisterRecord, ...] = ()

    @property
    def version(self) -> int:
        """Transitional alias removed with JournalRunBoundaries in M3."""
        return self.commit_sequence

    @property
    def graph(self) -> AgentSessionGraph:
        """Return the physical Entry Tree selected at the main Lane Head."""
        graph = AgentSessionGraph.from_entries(self.session_id, self.entries)
        for record in self.registers:
            if (
                isinstance(record.value, LaneHead)
                and record.value.lane_id == LaneId.main()
                and record.value.entry_id is not None
            ):
                return graph.select_head(record.value.entry_id)
        return graph

    @property
    def tree(self) -> AgentSessionTree:
        """Return the Lane-addressed immutable read interface."""
        heads = {
            record.value.lane_id: record
            for record in self.registers
            if isinstance(record.value, LaneHead)
        }
        states = {
            record.value.lane_id: record
            for record in self.registers
            if isinstance(record.value, LaneState)
        }
        if set(heads) != set(states):
            raise ValueError("Agent Session Lane registers are incomplete")
        lanes = tuple(
            LaneSnapshot(lane_id=lane_id, head=head, state=states[lane_id])
            for lane_id, head in sorted(heads.items(), key=lambda item: item[0].value)
        )
        return AgentSessionTree(
            session_id=self.session_id,
            commit_sequence=self.commit_sequence,
            entries=self.entries,
            lanes=lanes,
        )


class AgentSessionStore[HostUpdateT](Protocol):
    """Durable journal storage for one agent session.

    The PostgreSQL adapter commits host updates, ordered result entries,
    projection, settlement, session version, and — for live settlements —
    durable run progress in one transaction. Recovery prelude settlements use
    ``progress="prelude"`` and must not advance durable progress.
    """

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        """Return the current snapshot for one session."""
        ...

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[HostUpdateT],
    ) -> TransactionOutcome:
        """Atomically mutate Entries, exact-CAS Registers, projection, and HostDelta."""
        ...

    async def append_to_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        expected_head: RegisterRecord,
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
    ) -> TransactionOutcome:
        """Place one Entry chain and exact-CAS a single Lane Head."""
        ...

    async def fork_lane(
        self,
        *,
        session_id: SessionId,
        source_lane_id: LaneId,
        lane_id: LaneId,
        at_entry_id: EntryId | None = None,
    ) -> TransactionOutcome:
        """Create one stable Lane at a stable checkpoint."""
        ...

    async def archive_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
    ) -> TransactionOutcome:
        """Archive one idle non-main Lane without deleting shared Entries."""
        ...

    async def append(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
    ) -> AppendCommit:
        """Append ordered entries atomically; never settles an effect."""
        ...

    async def settle_effect(
        self,
        *,
        session_id: SessionId,
        expected_version: int,
        intent_id: IntentId,
        settlement: EffectSettlement[HostUpdateT],
        entries: Sequence[SessionEntry],
        projection: ContextProjection | None = None,
        progress: SessionProgressClass = "live",
        lane_id: LaneId | None = None,
    ) -> SettleCommit:
        """Settle one existing unsettled intent atomically with its results."""
        ...


__all__ = [
    "AgentSessionSnapshot",
    "AgentSessionStore",
    "AppendCommit",
    "EffectAlreadySettled",
    "EffectCommit",
    "EffectContractChanged",
    "EffectMissing",
    "EvidenceConflict",
    "HostUpdateT",
    "LeaseLost",
    "SessionCommit",
    "SessionProgressClass",
    "SettleCommit",
    "VersionConflict",
]
