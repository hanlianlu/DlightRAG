# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical parent-linked view over immutable Agent Session entries."""

from dataclasses import dataclass

from dlightrag.agent.session.entries import RunSegmentEntry, SessionEntry
from dlightrag.agent.session.ids import EntryId, SessionId


@dataclass(frozen=True, slots=True)
class SessionNode:
    """One immutable entry and the preceding entry on its branch."""

    entry: SessionEntry
    parent_entry_id: EntryId | None


@dataclass(frozen=True, slots=True)
class AgentSessionGraph:
    """A selected Agent Session head and its parent-linked immutable entries.

    Durable journals are linear in 3.0, so ``from_linear_entries`` derives
    parent links from committed order. ``select_head`` validates and projects
    an in-memory ancestry view; stores do not persist alternate heads.
    """

    session_id: SessionId
    nodes: tuple[SessionNode, ...]
    head_entry_id: EntryId | None

    @classmethod
    def from_linear_entries(
        cls,
        session_id: SessionId,
        entries: tuple[SessionEntry, ...],
    ) -> AgentSessionGraph:
        parent: EntryId | None = None
        nodes: list[SessionNode] = []
        expected_sequence = 1
        for entry in entries:
            if entry.session_id != session_id:
                raise ValueError("session graph entry belongs to another session")
            if entry.sequence != expected_sequence:
                raise ValueError("session graph requires a contiguous committed sequence")
            if (
                isinstance(entry, RunSegmentEntry)
                and entry.parent_head_id is not None
                and (parent is None or entry.parent_head_id != parent.value)
            ):
                raise ValueError("run segment parent head does not match the selected head")
            nodes.append(SessionNode(entry=entry, parent_entry_id=parent))
            parent = entry.entry_id
            expected_sequence += 1
        return cls(session_id=session_id, nodes=tuple(nodes), head_entry_id=parent)

    def select_head(self, head_entry_id: EntryId) -> AgentSessionGraph:
        """Return a branch view rooted at an existing immutable entry."""
        if all(node.entry.entry_id != head_entry_id for node in self.nodes):
            raise KeyError(f"unknown Agent Session head: {head_entry_id}")
        return AgentSessionGraph(
            session_id=self.session_id,
            nodes=self.nodes,
            head_entry_id=head_entry_id,
        )

    @property
    def entries(self) -> tuple[SessionEntry, ...]:
        return tuple(node.entry for node in self.nodes)

    def ancestry(self, head_entry_id: EntryId | None = None) -> tuple[SessionEntry, ...]:
        head = self.head_entry_id if head_entry_id is None else head_entry_id
        if head is None:
            return ()
        by_id = {node.entry.entry_id: node for node in self.nodes}
        reverse: list[SessionEntry] = []
        current: EntryId | None = head
        while current is not None:
            node = by_id.get(current)
            if node is None:
                raise KeyError(f"unknown Agent Session head: {current}")
            reverse.append(node.entry)
            current = node.parent_entry_id
        return tuple(reversed(reverse))


__all__ = ["AgentSessionGraph", "SessionNode"]
