# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Agent Session repository/store read and transaction contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dlightrag.agent.session.entries import SessionEntry
from dlightrag.agent.session.graph import AgentSessionGraph
from dlightrag.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.agent.session.projection import ContextProjection
from dlightrag.agent.session.registers import (
    ContextProjectionRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
)
from dlightrag.agent.session.transactions import SessionTransaction, TransactionOutcome
from dlightrag.agent.session.tree import AgentSessionTree, LaneSnapshot


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    """One immutable Session Tree and its exact current-register snapshot."""

    session_id: SessionId
    commit_sequence: int
    entries: tuple[SessionEntry, ...]
    registers: tuple[RegisterRecord, ...] = ()
    selected_lane_id: LaneId = LaneId.main()

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


class AgentSessionStore[HostDeltaT](Protocol):
    """Deep storage seam: immutable reads plus atomic exact-CAS transactions."""

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot: ...

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[HostDeltaT],
    ) -> TransactionOutcome: ...

    async def append_to_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        expected_head: RegisterRecord,
        entries: Sequence[SessionEntry],
    ) -> TransactionOutcome: ...

    async def fork_lane(
        self,
        *,
        session_id: SessionId,
        source_lane_id: LaneId,
        lane_id: LaneId,
        at_entry_id: EntryId | None = None,
    ) -> TransactionOutcome: ...

    async def archive_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
    ) -> TransactionOutcome: ...


__all__ = ["AgentSessionSnapshot", "AgentSessionStore"]
