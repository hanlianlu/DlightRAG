# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed semantic Entry Tree union with canonical serialization."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from dlightrag.agent.session.effects import JsonValue, ToolResultEntry
from dlightrag.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    ProjectionId,
    SessionId,
)
from dlightrag.agent.tool_content import decode_tool_content, encode_tool_content
from dlightrag.ai.messages import ToolCall

SESSION_ENTRY_SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionEntry:
    """Base record physically placed under exactly one parent Entry."""

    entry_id: EntryId
    session_id: SessionId
    timestamp: datetime
    parent_entry_id: EntryId | None = None
    sequence: int = 0
    schema_version: int = SESSION_ENTRY_SCHEMA_VERSION

    entry_type: ClassVar[str] = "session_entry"

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("Entry sequence cannot be negative")
        if self.schema_version != SESSION_ENTRY_SCHEMA_VERSION:
            raise ValueError("Entry schema version is not current")
        if self.timestamp.tzinfo is None:
            raise ValueError("Entry timestamp must be timezone-aware")

    def canonical_payload(self) -> JsonValue:
        raise NotImplementedError

    def to_canonical_json(self) -> JsonValue:
        return {
            "entry_id": self.entry_id.value,
            "session_id": self.session_id.value,
            "parent_entry_id": (
                self.parent_entry_id.value if self.parent_entry_id is not None else None
            ),
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "schema_version": self.schema_version,
            "payload": self.canonical_payload(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessageEntry(SessionEntry):
    content: JsonValue

    entry_type: ClassVar[str] = "user_message"

    def canonical_payload(self) -> JsonValue:
        return {"content": self.content}


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessageEntry(SessionEntry):
    """One complete validated provider response; partial streams are never durable."""

    content: str
    stop_reason: Literal["stop", "length", "tool_use"]
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: JsonValue | None = None
    cost: JsonValue | None = None
    provider_state: JsonValue | None = None

    entry_type: ClassVar[str] = "assistant_message"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stop_reason == "tool_use" and not self.tool_calls:
            raise ValueError("tool_use AssistantMessage requires Tool calls")

    def canonical_payload(self) -> JsonValue:
        payload: dict[str, Any] = {
            "content": self.content,
            "reasoning": self.reasoning,
            "stop_reason": self.stop_reason,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "argument_error": call.argument_error,
                    "thought_signature": call.thought_signature,
                }
                for call in self.tool_calls
            ],
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.provider_state is not None:
            payload["provider_state"] = self.provider_state
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultMessageEntry(SessionEntry):
    """One source-position ToolResult with permanent recovery provenance."""

    result: ToolResultEntry
    intent_id: IntentId | None
    source_index: int
    contract_version: int
    input_schema_digest: str
    replay_policy: Literal["replayable", "never"]
    attempt_id: AttemptId | None
    effective_input_digest: str

    entry_type: ClassVar[str] = "tool_result"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source_index < 0 or self.contract_version < 0:
            raise ValueError("ToolResult provenance position/version is invalid")
        if self.intent_id is not None and (
            self.contract_version < 1
            or len(self.input_schema_digest) != 64
            or len(self.effective_input_digest) != 64
        ):
            raise ValueError("executable ToolResult provenance must be complete")

    def canonical_payload(self) -> JsonValue:
        return {
            "intent_id": self.intent_id.value if self.intent_id is not None else None,
            "source_index": self.source_index,
            "tool_name": self.result.tool_name,
            "call_id": self.result.call_id,
            "outcome": self.result.outcome,
            "content": encode_tool_content(self.result.parts),
            "cached": self.result.cached,
            "contract_version": self.contract_version,
            "input_schema_digest": self.input_schema_digest,
            "replay_policy": self.replay_policy,
            "attempt_id": self.attempt_id.value if self.attempt_id is not None else None,
            "effective_input_digest": self.effective_input_digest,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlMessageEntry(SessionEntry):
    """One accepted Steer consumed at a stable checkpoint."""

    control_id: str
    content: JsonValue

    entry_type: ClassVar[str] = "control_message"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.control_id:
            raise ValueError("ControlMessage identity cannot be empty")

    def canonical_payload(self) -> JsonValue:
        return {"control_id": self.control_id, "content": self.content}


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionEntry(SessionEntry):
    """One branch-local immutable context projection checkpoint."""

    projection_id: ProjectionId
    summary: str | None
    covered_through_sequence: int
    first_retained_sequence: int
    covered_through_entry_id: EntryId | None = None
    first_retained_entry_id: EntryId | None = None
    source_digest: str = ""

    entry_type: ClassVar[str] = "compaction"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.covered_through_sequence < 0:
            raise ValueError("compaction covered-through sequence cannot be negative")
        if self.first_retained_sequence < 1:
            raise ValueError("compaction retained start must be positive")
        if self.first_retained_sequence <= self.covered_through_sequence:
            raise ValueError("compaction retained start must follow the covered prefix")
        if self.summary is not None and not self.summary.strip():
            raise ValueError("compaction summary cannot be empty when present")
        if self.covered_through_sequence > 0 and (
            self.covered_through_entry_id is None or len(self.source_digest) != 64
        ):
            raise ValueError("compaction requires branch Entry identity and source digest")

    def canonical_payload(self) -> JsonValue:
        return {
            "projection_id": self.projection_id.value,
            "summary": self.summary,
            "covered_through_sequence": self.covered_through_sequence,
            "first_retained_sequence": self.first_retained_sequence,
            "covered_through_entry_id": (
                self.covered_through_entry_id.value
                if self.covered_through_entry_id is not None
                else None
            ),
            "first_retained_entry_id": (
                self.first_retained_entry_id.value
                if self.first_retained_entry_id is not None
                else None
            ),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AdoptionEntry(SessionEntry):
    """One explicit bounded cross-Lane adoption with immutable provenance."""

    source_session_id: SessionId
    source_entry_id: EntryId
    content: JsonValue

    entry_type: ClassVar[str] = "adoption"

    def canonical_payload(self) -> JsonValue:
        return {
            "source_session_id": self.source_session_id.value,
            "source_entry_id": self.source_entry_id.value,
            "content": self.content,
        }


type SessionEntryKind = (
    UserMessageEntry
    | AssistantMessageEntry
    | ToolResultMessageEntry
    | ControlMessageEntry
    | CompactionEntry
    | AdoptionEntry
)

ENTRY_TYPE_TO_CLASS: dict[str, type[SessionEntry]] = {
    entry.entry_type: entry  # type: ignore[type-abstract]
    for entry in (
        UserMessageEntry,
        AssistantMessageEntry,
        ToolResultMessageEntry,
        ControlMessageEntry,
        CompactionEntry,
        AdoptionEntry,
    )
}


def entry_type_of(entry: SessionEntry) -> str:
    return entry.entry_type


def new_session_entry(
    *,
    entry_type: str,
    session_id: SessionId,
    sequence: int = 0,
    timestamp: datetime | None = None,
    **payload: Any,
) -> SessionEntry:
    entry_class = ENTRY_TYPE_TO_CLASS.get(entry_type)
    if entry_class is None:
        raise ValueError(f"unknown Entry type: {entry_type}")
    return entry_class(
        entry_id=EntryId.new(),
        session_id=session_id,
        sequence=sequence,
        timestamp=timestamp or _utc_now(),
        **payload,
    )


def decode_entry_payload(
    *,
    entry_type: str,
    entry_id: EntryId,
    session_id: SessionId,
    sequence: int,
    timestamp: datetime,
    payload: Mapping[str, Any],
    parent_entry_id: EntryId | None = None,
) -> SessionEntry:
    common = {
        "entry_id": entry_id,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp": timestamp,
        "parent_entry_id": parent_entry_id,
    }
    if entry_type == "user_message":
        return UserMessageEntry(**common, content=payload["content"])
    if entry_type == "assistant_message":
        calls = tuple(
            ToolCall(
                id=str(call["id"]),
                name=str(call["name"]),
                arguments=dict(call.get("arguments") or {}),
                argument_error=call.get("argument_error"),
                thought_signature=call.get("thought_signature"),
            )
            for call in payload.get("tool_calls") or ()
        )
        return AssistantMessageEntry(
            **common,
            content=str(payload["content"]),
            stop_reason=payload["stop_reason"],
            reasoning=str(payload.get("reasoning") or ""),
            tool_calls=calls,
            usage=payload.get("usage"),
            cost=payload.get("cost"),
            provider_state=payload.get("provider_state"),
        )
    if entry_type == "tool_result":
        return ToolResultMessageEntry(
            **common,
            intent_id=(IntentId(str(payload["intent_id"])) if payload.get("intent_id") else None),
            source_index=int(payload["source_index"]),
            result=ToolResultEntry(
                tool_name=str(payload["tool_name"]),
                call_id=str(payload["call_id"]),
                outcome=payload["outcome"],
                parts=decode_tool_content(payload["content"]),
                details=None,
                cached=bool(payload.get("cached") or False),
            ),
            contract_version=int(payload["contract_version"]),
            input_schema_digest=str(payload["input_schema_digest"]),
            replay_policy=payload["replay_policy"],
            attempt_id=(
                AttemptId(str(payload["attempt_id"])) if payload.get("attempt_id") else None
            ),
            effective_input_digest=str(payload["effective_input_digest"]),
        )
    if entry_type == "control_message":
        return ControlMessageEntry(
            **common,
            control_id=str(payload["control_id"]),
            content=payload["content"],
        )
    if entry_type == "compaction":
        return CompactionEntry(
            **common,
            projection_id=ProjectionId(str(payload["projection_id"])),
            summary=payload.get("summary"),
            covered_through_sequence=int(payload["covered_through_sequence"]),
            first_retained_sequence=int(payload["first_retained_sequence"]),
            covered_through_entry_id=(
                EntryId(str(payload["covered_through_entry_id"]))
                if payload.get("covered_through_entry_id")
                else None
            ),
            first_retained_entry_id=(
                EntryId(str(payload["first_retained_entry_id"]))
                if payload.get("first_retained_entry_id")
                else None
            ),
            source_digest=str(payload.get("source_digest") or ""),
        )
    if entry_type == "adoption":
        return AdoptionEntry(
            **common,
            source_session_id=SessionId(str(payload["source_session_id"])),
            source_entry_id=EntryId(str(payload["source_entry_id"])),
            content=payload["content"],
        )
    raise ValueError(f"unknown Entry type: {entry_type}")


__all__ = [
    "ENTRY_TYPE_TO_CLASS",
    "SESSION_ENTRY_SCHEMA_VERSION",
    "AdoptionEntry",
    "AssistantMessageEntry",
    "CompactionEntry",
    "ControlMessageEntry",
    "SessionEntry",
    "SessionEntryKind",
    "ToolResultMessageEntry",
    "UserMessageEntry",
    "decode_entry_payload",
    "entry_type_of",
    "new_session_entry",
]
