# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The finite Agent Session entry union with canonical serialization.

Each concrete entry is immutable and carries common identity fields plus a
typed payload. The closed union grows only together with a real writer.

``to_canonical_json`` returns the exact ``payload_json`` a durable store keeps;
``canonical_entry_json`` wraps it with the common columns for tests and
adapters. The fold consumes the typed records, never a raw payload mapping.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from dlightrag.agent.session.effects import (
    EffectIntent,
    JsonValue,
    ToolResultEntry,
)
from dlightrag.agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag.ai.messages import ToolCall

SESSION_ENTRY_SCHEMA_VERSION = 1

SessionTerminalReason = Literal["completed", "cancelled", "abandoned"]
RunSegmentKind = Literal["start", "resume"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionEntry:
    """Base record for every immutable journal entry.

    ``sequence`` is allocated by the store as one contiguous range per
    transaction (M3-D16); entries under construction may carry ``0`` and are
    stamped with their durable sequence when committed.
    """

    entry_id: EntryId
    session_id: SessionId
    sequence: int = 0
    timestamp: datetime
    schema_version: int = SESSION_ENTRY_SCHEMA_VERSION

    entry_type: ClassVar[str] = "session_entry"

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("journal entry sequence cannot be negative")
        if self.schema_version != SESSION_ENTRY_SCHEMA_VERSION:
            raise ValueError("journal entry schema version is not current")
        if self.timestamp.tzinfo is None:
            raise ValueError("journal entry timestamp must be timezone-aware")

    def canonical_payload(self) -> JsonValue:
        """Return this variant's typed payload as canonical JSON data."""
        raise NotImplementedError

    def to_canonical_json(self) -> JsonValue:
        """Return the durable payload_json for this entry."""
        return {
            "entry_id": str(self.entry_id),
            "session_id": str(self.session_id),
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "schema_version": self.schema_version,
            "payload": self.canonical_payload(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSegmentEntry(SessionEntry):
    """One start or resume interval over a selected Agent Session head."""

    segment_id: str
    kind: RunSegmentKind
    parent_head_id: str | None = None

    entry_type: ClassVar[str] = "run_segment"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.segment_id.strip():
            raise ValueError("run segment id cannot be empty")

    def canonical_payload(self) -> JsonValue:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "parent_head_id": self.parent_head_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessageEntry(SessionEntry):
    """One user message: a question, a pinned history turn, or a prompt injection."""

    content: JsonValue

    entry_type: ClassVar[str] = "user_message"

    def canonical_payload(self) -> JsonValue:
        return {"content": self.content}


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessageEntry(SessionEntry):
    """One complete assistant response: content, stop reason, and usage/cost anchor.

    Only a fully exhausted provider response is journaled; cancellation or crash
    mid-stream leaves no partial assistant entry (M3-D11).
    """

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
            raise ValueError("tool_use assistant entry requires tool calls")

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
class EffectIntentEntry(SessionEntry):
    """One ordered effect intent, persisted before its effect executes."""

    intent: EffectIntent

    entry_type: ClassVar[str] = "effect_intent"

    @property
    def intent_id(self) -> IntentId:
        return self.intent.intent_id

    def canonical_payload(self) -> JsonValue:
        intent = self.intent
        return {
            "intent_id": str(intent.intent_id),
            "tool_name": intent.tool_name,
            "replay_policy": intent.replay_policy,
            "contract_version": intent.contract_version,
            "input_schema_digest": intent.input_schema_digest,
            "canonical_input": intent.canonical_input,
            "source_call_id": intent.source_call_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectResultEntry(SessionEntry):
    """One ordered effect result or one deterministic validation result."""

    result: ToolResultEntry
    intent_id: IntentId | None = None

    entry_type: ClassVar[str] = "effect_result"

    def canonical_payload(self) -> JsonValue:
        return {
            "intent_id": str(self.intent_id) if self.intent_id is not None else None,
            "tool_name": self.result.tool_name,
            "call_id": self.result.call_id,
            "outcome": self.result.outcome,
            "content": self.result.content,
            "details": self.result.details,
            "cached": self.result.cached,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextInjectionEntry(SessionEntry):
    """One injected context message the framework, not the user, authored."""

    content: JsonValue
    label: str | None = None

    entry_type: ClassVar[str] = "context_injection"

    def canonical_payload(self) -> JsonValue:
        payload: dict[str, Any] = {"content": self.content}
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionEntry(SessionEntry):
    """One committed compaction: summary, covered prefix, and retained start."""

    projection_id: ProjectionId
    summary: str | None
    covered_through_sequence: int
    first_retained_sequence: int

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

    def canonical_payload(self) -> JsonValue:
        return {
            "projection_id": str(self.projection_id),
            "summary": self.summary,
            "covered_through_sequence": self.covered_through_sequence,
            "first_retained_sequence": self.first_retained_sequence,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileFactEntry(SessionEntry):
    """One pinned run/session fact recorded for audit and recovery inspection."""

    key: str
    value: JsonValue

    entry_type: ClassVar[str] = "profile_fact"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.key.strip():
            raise ValueError("profile fact key cannot be empty")

    def canonical_payload(self) -> JsonValue:
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionTerminalEntry(SessionEntry):
    """One terminal session fact: how and why the session ended."""

    reason: SessionTerminalReason
    detail: str | None = None

    entry_type: ClassVar[str] = "session_terminal"

    def canonical_payload(self) -> JsonValue:
        payload: dict[str, Any] = {"reason": self.reason}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


type SessionEntryKind = (
    RunSegmentEntry
    | UserMessageEntry
    | AssistantMessageEntry
    | EffectIntentEntry
    | EffectResultEntry
    | ContextInjectionEntry
    | CompactionEntry
    | ProfileFactEntry
    | SessionTerminalEntry
)

#: The closed M3 entry union by durable type tag.
ENTRY_TYPE_TO_CLASS: dict[str, type[SessionEntry]] = {
    entry.entry_type: entry  # type: ignore[type-abstract]
    for entry in (
        RunSegmentEntry,
        UserMessageEntry,
        AssistantMessageEntry,
        EffectIntentEntry,
        EffectResultEntry,
        ContextInjectionEntry,
        CompactionEntry,
        ProfileFactEntry,
        SessionTerminalEntry,
    )
}


def entry_type_of(entry: SessionEntry) -> str:
    """Return the durable type tag of one journal entry."""
    return entry.entry_type


def new_session_entry(
    *,
    entry_type: str,
    session_id: SessionId,
    sequence: int = 0,
    timestamp: datetime | None = None,
    **payload: Any,
) -> SessionEntry:
    """Construct one journal entry of the closed union.

    Unknown type tags raise immediately, before any durable store is touched.
    """
    entry_class = ENTRY_TYPE_TO_CLASS.get(entry_type)
    if entry_class is None:
        raise ValueError(f"unknown journal entry type: {entry_type}")
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
) -> SessionEntry:
    """Rebuild one typed entry from its canonical durable payload."""
    common = {
        "entry_id": entry_id,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp": timestamp,
    }
    if entry_type == "run_segment":
        return RunSegmentEntry(
            **common,
            segment_id=str(payload["segment_id"]),
            kind=payload["kind"],
            parent_head_id=(
                str(payload["parent_head_id"]) if payload.get("parent_head_id") else None
            ),
        )
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
    if entry_type == "effect_intent":
        return EffectIntentEntry(
            **common,
            intent=EffectIntent(
                intent_id=IntentId(str(payload["intent_id"])),
                tool_name=str(payload["tool_name"]),
                replay_policy=payload["replay_policy"],
                contract_version=int(payload["contract_version"]),
                input_schema_digest=str(payload["input_schema_digest"]),
                canonical_input=str(payload["canonical_input"]),
                source_call_id=payload.get("source_call_id"),
            ),
        )
    if entry_type == "effect_result":
        return EffectResultEntry(
            **common,
            intent_id=(IntentId(str(payload["intent_id"])) if payload.get("intent_id") else None),
            result=ToolResultEntry(
                tool_name=str(payload["tool_name"]),
                call_id=str(payload["call_id"]),
                outcome=payload["outcome"],
                content=str(payload["content"]),
                details=payload.get("details"),
                cached=bool(payload.get("cached") or False),
            ),
        )
    if entry_type == "context_injection":
        return ContextInjectionEntry(
            **common,
            content=payload["content"],
            label=payload.get("label"),
        )
    if entry_type == "compaction":
        return CompactionEntry(
            **common,
            projection_id=ProjectionId(str(payload["projection_id"])),
            summary=payload.get("summary"),
            covered_through_sequence=int(payload["covered_through_sequence"]),
            first_retained_sequence=int(payload["first_retained_sequence"]),
        )
    if entry_type == "profile_fact":
        return ProfileFactEntry(**common, key=str(payload["key"]), value=payload["value"])
    if entry_type == "session_terminal":
        return SessionTerminalEntry(
            **common, reason=payload["reason"], detail=payload.get("detail")
        )
    raise ValueError(f"unknown journal entry type: {entry_type}")


__all__ = [
    "ENTRY_TYPE_TO_CLASS",
    "SESSION_ENTRY_SCHEMA_VERSION",
    "AssistantMessageEntry",
    "CompactionEntry",
    "ContextInjectionEntry",
    "EffectIntentEntry",
    "EffectResultEntry",
    "ProfileFactEntry",
    "RunSegmentEntry",
    "RunSegmentKind",
    "SessionEntry",
    "SessionEntryKind",
    "SessionTerminalEntry",
    "SessionTerminalReason",
    "UserMessageEntry",
    "decode_entry_payload",
    "entry_type_of",
    "new_session_entry",
]
