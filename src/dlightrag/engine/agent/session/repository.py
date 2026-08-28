# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Agent Session repository/store read and transaction contract."""

from dataclasses import dataclass, replace
from typing import Any, Protocol

from dlightrag.engine.agent.session.entries import SessionEntry
from dlightrag.engine.agent.session.graph import AgentSessionGraph
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.projection import ContextProjection
from dlightrag.engine.agent.session.registers import (
    ContextProjectionRegister,
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
    SetRegister,
)
from dlightrag.engine.agent.session.transactions import (
    SessionTransaction,
    TransactionCommit,
    TransactionOutcome,
)
from dlightrag.engine.agent.session.tree import AgentSessionTree, LaneSnapshot


@dataclass(frozen=True, slots=True)
class AgentSessionCursor:
    """Authoritative mutable and immutable high-water marks for one Session."""

    commit_sequence: int
    last_entry_sequence: int

    def __post_init__(self) -> None:
        if self.commit_sequence < 0 or self.last_entry_sequence < 0:
            raise ValueError("Agent Session cursor sequences cannot be negative")


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    """One immutable Session Tree and its exact current-register snapshot."""

    session_id: SessionId
    commit_sequence: int
    last_entry_sequence: int
    entries: tuple[SessionEntry, ...]
    registers: tuple[RegisterRecord, ...] = ()
    selected_lane_id: LaneId = LaneId.main()

    def __post_init__(self) -> None:
        AgentSessionCursor(self.commit_sequence, self.last_entry_sequence)
        if len(self.entries) != self.last_entry_sequence:
            raise ValueError("Agent Session Entry count does not match its cursor")
        # Keep incremental construction O(delta): full repository loads scrub every
        # decoded row, while refresh/projection validate only the newly appended suffix.
        if self.entries and (
            self.entries[0].sequence != 1 or self.entries[-1].sequence != self.last_entry_sequence
        ):
            raise ValueError("Agent Session snapshot Entry sequence is not gap-free")

    @property
    def cursor(self) -> AgentSessionCursor:
        return AgentSessionCursor(self.commit_sequence, self.last_entry_sequence)

    @property
    def active_projection(self) -> ContextProjection | None:
        """Return the selected Lane's typed branch-local projection register."""
        for record in self.registers:
            if (
                isinstance(record.value, ContextProjectionRegister)
                and record.value.lane_id == self.selected_lane_id
            ):
                return record.value.projection
        return None

    @property
    def graph(self) -> AgentSessionGraph:
        """Return the physical Entry Tree selected at the chosen Lane Head."""
        graph = AgentSessionGraph.from_entries(self.session_id, self.entries)
        for record in self.registers:
            if (
                isinstance(record.value, LaneHead)
                and record.value.lane_id == self.selected_lane_id
                and record.value.entry_id is not None
            ):
                return graph.select_head(record.value.entry_id)
        return graph

    @property
    def tree(self) -> AgentSessionTree:
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


def project_transaction_commit(
    snapshot: AgentSessionSnapshot,
    transaction: SessionTransaction[Any],
    commit: TransactionCommit,
) -> AgentSessionSnapshot | None:
    """Project one contiguous authoritative commit, or require an authoritative reload."""
    if commit.commit_sequence != snapshot.commit_sequence + 1:
        return None
    expected_sequences = tuple(
        range(
            snapshot.last_entry_sequence + 1,
            snapshot.last_entry_sequence + 1 + len(transaction.entries),
        )
    )
    if commit.appended_sequences != expected_sequences:
        return None
    expected_register_sequences = tuple(
        (write.ref, commit.commit_sequence) for write in transaction.register_writes
    )
    if commit.register_sequences != expected_register_sequences:
        return None
    stamped_entries = tuple(
        replace(entry, sequence=sequence)
        for entry, sequence in zip(
            transaction.entries,
            commit.appended_sequences,
            strict=True,
        )
    )
    registers = {record.ref: record for record in snapshot.registers}
    for write in transaction.register_writes:
        if isinstance(write, SetRegister):
            registers[write.ref] = RegisterRecord(
                value=write.value,
                sequence=commit.commit_sequence,
            )
        elif isinstance(write, DeleteRegister):
            registers.pop(write.ref, None)
    try:
        # Tuple concatenation preserves the identity of every decoded historical Entry.
        return AgentSessionSnapshot(
            session_id=snapshot.session_id,
            commit_sequence=commit.commit_sequence,
            last_entry_sequence=snapshot.last_entry_sequence + len(stamped_entries),
            entries=snapshot.entries + stamped_entries,
            registers=tuple(
                record
                for _, record in sorted(
                    registers.items(),
                    key=lambda item: (item[0].kind, item[0].key),
                )
            ),
            selected_lane_id=snapshot.selected_lane_id,
        )
    except TypeError, ValueError:
        return None


class AgentSessionRepository[HostDeltaT](Protocol):
    """Coherent snapshot reads plus the atomic transaction adapter seam."""

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot: ...

    async def refresh(
        self,
        session_id: SessionId,
        *,
        previous: AgentSessionSnapshot,
    ) -> AgentSessionSnapshot: ...

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[HostDeltaT],
    ) -> TransactionOutcome: ...


__all__ = [
    "AgentSessionCursor",
    "AgentSessionRepository",
    "AgentSessionSnapshot",
    "project_transaction_commit",
]
