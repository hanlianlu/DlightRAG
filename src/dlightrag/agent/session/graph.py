# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable parent-linked views over one canonical Agent Session Tree."""

from dataclasses import dataclass, replace

from dlightrag.agent.session.entries import RunSegmentEntry, SessionEntry
from dlightrag.agent.session.ids import EntryId, SessionId


@dataclass(frozen=True, slots=True)
class SessionNode:
    """One immutable Entry at its physical parent placement."""

    entry: SessionEntry

    @property
    def parent_entry_id(self) -> EntryId | None:
        return self.entry.parent_entry_id


@dataclass(frozen=True, slots=True)
class AgentSessionGraph:
    """One immutable Entry Tree, optionally viewed from a selected Head."""

    session_id: SessionId
    nodes: tuple[SessionNode, ...]
    head_entry_id: EntryId | None = None

    @classmethod
    def from_entries(
        cls,
        session_id: SessionId,
        entries: tuple[SessionEntry, ...],
        *,
        head_entry_id: EntryId | None = None,
    ) -> AgentSessionGraph:
        """Validate physical placement and build one immutable Tree view."""
        by_id: dict[EntryId, SessionEntry] = {}
        previous_sequence = 0
        roots = 0
        for entry in entries:
            if entry.session_id != session_id:
                raise ValueError("session graph entry belongs to another session")
            if entry.sequence <= previous_sequence:
                raise ValueError("session graph entry sequences must be strictly increasing")
            if entry.entry_id in by_id:
                raise ValueError("session graph entry identities must be unique")
            if entry.parent_entry_id is None:
                roots += 1
                if by_id:
                    raise ValueError("only the first Session Entry can be a root")
            elif entry.parent_entry_id not in by_id:
                raise ValueError("session graph entry parent must precede its child")
            by_id[entry.entry_id] = entry
            previous_sequence = entry.sequence
        if entries and roots != 1:
            raise ValueError("a non-empty Session Tree requires exactly one root")
        selected = head_entry_id
        if selected is not None and selected not in by_id:
            raise KeyError(f"unknown Agent Session head: {selected}")
        return cls(
            session_id=session_id,
            nodes=tuple(SessionNode(entry=entry) for entry in entries),
            head_entry_id=selected,
        )

    @classmethod
    def from_linear_entries(
        cls,
        session_id: SessionId,
        entries: tuple[SessionEntry, ...],
    ) -> AgentSessionGraph:
        """Build physical placement for pre-M2 in-process callers.

        Durable adapters always load ``parent_entry_id`` from storage. This
        helper remains only while the existing Answer interpreter migrates in
        M3 and deliberately does not model alternate Heads.
        """
        parent: EntryId | None = None
        placed: list[SessionEntry] = []
        for entry in entries:
            if (
                isinstance(entry, RunSegmentEntry)
                and entry.parent_head_id is not None
                and (parent is None or entry.parent_head_id != parent.value)
            ):
                raise ValueError("run segment parent head does not match the selected head")
            placed_entry = replace(entry, parent_entry_id=parent)
            placed.append(placed_entry)
            parent = placed_entry.entry_id
        return cls.from_entries(
            session_id,
            tuple(placed),
            head_entry_id=parent,
        )

    def select_head(self, head_entry_id: EntryId) -> AgentSessionGraph:
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
