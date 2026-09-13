# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Exact selected Entry attachments requested from retained Run references."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
from dlightrag.engine.agent.session.fold import retained_session_entries
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.agent.tool_content import (
    ToolResourceAttachmentPart,
    decode_tool_content,
    encode_tool_content,
)


@dataclass(frozen=True, slots=True)
class AttachmentOccurrence:
    entry_id: str
    part_index: int
    attachment: ToolResourceAttachmentPart

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "part_index": self.part_index,
            "attachment": encode_tool_content((self.attachment,))[0],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AttachmentOccurrence:
        parts = decode_tool_content([payload.get("attachment")])
        part = parts[0]
        if (
            not isinstance(part, ToolResourceAttachmentPart)
            or not isinstance(payload.get("entry_id"), str)
            or not payload["entry_id"]
            or type(payload.get("part_index")) is not int
            or payload["part_index"] < 0
        ):
            raise ValueError("invalid attachment occurrence")
        occurrence = cls(payload["entry_id"], payload["part_index"], part)
        if occurrence.canonical_payload() != payload:
            raise ValueError("attachment occurrence must use the closed durable encoding")
        return occurrence

    @property
    def reference_id(self) -> str:
        """Retention identity is an Entry position, never a source handle or digest."""
        return f"attachment-occurrence:{self.entry_id}:{self.part_index}"


@dataclass(frozen=True, slots=True)
class AttachmentReplaySelection:
    session_id: str
    lane_id: str
    head_entry_id: str
    occurrences: tuple[AttachmentOccurrence, ...]

    @classmethod
    def from_snapshot(cls, snapshot: AgentSessionSnapshot) -> AttachmentReplaySelection:
        lane = snapshot.tree.lane(snapshot.selected_lane_id)
        entries = retained_session_entries(
            snapshot.tree.ancestry(snapshot.selected_lane_id), snapshot.active_projection
        )
        return cls(
            session_id=snapshot.session_id.value,
            lane_id=snapshot.selected_lane_id.value,
            head_entry_id=lane.head_entry_id.value if lane.head_entry_id is not None else "",
            occurrences=tuple(
                AttachmentOccurrence(entry.entry_id.value, index, replace(part, data=b""))
                for entry in entries
                if isinstance(entry, ToolResultMessageEntry)
                for index, part in enumerate(entry.result.parts)
                if isinstance(part, ToolResourceAttachmentPart)
            ),
        )
