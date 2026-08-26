# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable Session Tree snapshots and stable checkpoint queries."""

from dataclasses import dataclass

from dlightrag.agent.session.entries import (
    AssistantMessageEntry,
    SessionEntry,
    ToolResultMessageEntry,
)
from dlightrag.agent.session.graph import AgentSessionGraph
from dlightrag.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.agent.session.registers import LaneHead, LaneState, RegisterRecord


@dataclass(frozen=True, slots=True)
class LaneSnapshot:
    lane_id: LaneId
    head: RegisterRecord
    state: RegisterRecord

    def __post_init__(self) -> None:
        if not isinstance(self.head.value, LaneHead) or self.head.value.lane_id != self.lane_id:
            raise ValueError("Lane snapshot head does not match its Lane")
        if not isinstance(self.state.value, LaneState) or self.state.value.lane_id != self.lane_id:
            raise ValueError("Lane snapshot state does not match its Lane")

    @property
    def head_entry_id(self) -> EntryId | None:
        value = self.head.value
        if not isinstance(value, LaneHead):
            raise TypeError("Lane Head register has the wrong value type")
        return value.entry_id

    @property
    def archived(self) -> bool:
        value = self.state.value
        if not isinstance(value, LaneState):
            raise TypeError("Lane State register has the wrong value type")
        return value.archived


@dataclass(frozen=True, slots=True)
class AgentSessionTree:
    """One commit-sequence-bound immutable view exposed to Hosts and tests."""

    session_id: SessionId
    commit_sequence: int
    entries: tuple[SessionEntry, ...]
    lanes: tuple[LaneSnapshot, ...]

    def __post_init__(self) -> None:
        if self.commit_sequence < 0:
            raise ValueError("Session commit sequence cannot be negative")
        graph = AgentSessionGraph.from_entries(self.session_id, self.entries)
        known = {entry.entry_id for entry in graph.entries}
        lane_ids: set[LaneId] = set()
        for lane in self.lanes:
            if lane.lane_id in lane_ids:
                raise ValueError("Session Tree Lane identities must be unique")
            lane_ids.add(lane.lane_id)
            if lane.head_entry_id is not None and lane.head_entry_id not in known:
                raise ValueError("Lane head must name an Entry in the same Session")
        if self.entries and LaneId.main() not in lane_ids:
            raise ValueError("a non-empty Session Tree requires the main Lane")

    @property
    def graph(self) -> AgentSessionGraph:
        return AgentSessionGraph.from_entries(self.session_id, self.entries)

    def lane(self, lane_id: LaneId = LaneId.main()) -> LaneSnapshot:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        raise KeyError(f"unknown Agent Session Lane: {lane_id}")

    def ancestry(self, lane_id: LaneId = LaneId.main()) -> tuple[SessionEntry, ...]:
        head = self.lane(lane_id).head_entry_id
        if head is None:
            return ()
        return self.graph.ancestry(head)

    def is_stable_checkpoint(self, entry_id: EntryId | None) -> bool:
        """Return whether one Head has no unmatched provider Tool Call."""
        if entry_id is None:
            return True
        ancestry = self.graph.ancestry(entry_id)
        pending: set[str] = set()
        for entry in ancestry:
            if isinstance(entry, AssistantMessageEntry):
                pending.update(call.id for call in entry.tool_calls)
            elif isinstance(entry, ToolResultMessageEntry):
                pending.discard(entry.result.call_id)
        return not pending


__all__ = ["AgentSessionTree", "LaneSnapshot"]
