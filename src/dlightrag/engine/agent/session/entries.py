# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed semantic Entry Tree union with canonical serialization."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal

from dlightrag.engine.agent.session.effects import JsonValue, ToolResultEntry
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    ProjectionId,
    SessionId,
)
from dlightrag.engine.agent.tool_content import decode_tool_content, encode_tool_content
from dlightrag.engine.ai.messages import ToolCall

SESSION_ENTRY_SCHEMA_VERSION = 2


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


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessageEntry(SessionEntry):
    """One message that replays to the model as any user message does.

    ``origin`` says where it came from when that was not the owner: ``child_notification`` marks
    a Child's settled result or guidance question that the Run handed its own agent as the next
    input, so a reader that wants only the owner's words (the Session Memory Pass) can skip it.
    """

    content: JsonValue
    acceptance_id: str | None = None
    origin: Literal["child_notification"] | None = None

    entry_type: ClassVar[str] = "user_message"

    def canonical_payload(self) -> JsonValue:
        payload: dict[str, Any] = {"content": self.content}
        if self.acceptance_id is not None:
            payload["acceptance_id"] = self.acceptance_id
        if self.origin is not None:
            payload["origin"] = self.origin
        return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessageEntry(SessionEntry):
    """One complete validated provider response; partial streams are never durable.

    ``call`` is the model, parameters, prompt digests and timings the response was made with
    (ADR 0043). It is written in the entry's own transaction, and is absent on an entry written
    before it existed, on one whose model call left no record, and on one a recovery wrote from
    a result whose call was gone with its process.
    """

    content: str
    stop_reason: Literal["stop", "length", "tool_use"]
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: JsonValue | None = None
    cost: JsonValue | None = None
    provider_state: JsonValue | None = None
    call: JsonValue | None = None
    acceptance_id: str | None = None

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
        if self.call is not None:
            payload["call"] = self.call
        if self.acceptance_id is not None:
            payload["acceptance_id"] = self.acceptance_id
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
        payload: dict[str, Any] = {
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
        # Absent rather than null on a call no Tool reported facts for, so the payload a
        # reader parses is the payload an older writer wrote.
        if self.result.facts is not None:
            payload["facts"] = self.result.facts
        return payload


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
class DeliveryWarningEntry(SessionEntry):
    """One pre-delivery warning: what the reader's record would say if the Run stopped now.

    It is internal evidence, not a human instruction: the runtime appends it after a final
    response is proposed, so the model can correct the record before the operation completes.
    """

    operation_id: str
    attempt: int
    findings: tuple[str, ...]

    entry_type: ClassVar[str] = "delivery_warning"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.operation_id:
            raise ValueError("Delivery warning identity cannot be empty")
        if self.attempt < 1:
            raise ValueError("Delivery warning attempt must be positive")
        if not self.findings:
            raise ValueError("Delivery warning must carry findings")

    def canonical_payload(self) -> JsonValue:
        return {
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "findings": list(self.findings),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionEntry(SessionEntry):
    """One branch-local immutable context projection checkpoint.

    ``call`` is the model, parameters, prompt digests and timings of the call that wrote its
    summary (ADR 0043), absent where the entry predates it or that call left no record.
    """

    projection_id: ProjectionId
    summary: str | None
    covered_through_sequence: int
    first_retained_sequence: int
    covered_through_entry_id: EntryId | None = None
    first_retained_entry_id: EntryId | None = None
    source_digest: str = ""
    call: JsonValue | None = None

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
        payload: dict[str, Any] = {
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
        if self.call is not None:
            payload["call"] = self.call
        return payload


type SessionEntryKind = (
    UserMessageEntry
    | AssistantMessageEntry
    | ToolResultMessageEntry
    | ControlMessageEntry
    | DeliveryWarningEntry
    | CompactionEntry
)

ENTRY_TYPE_TO_CLASS: dict[str, type[SessionEntry]] = {
    entry.entry_type: entry  # type: ignore[type-abstract]
    for entry in (
        UserMessageEntry,
        AssistantMessageEntry,
        ToolResultMessageEntry,
        ControlMessageEntry,
        DeliveryWarningEntry,
        CompactionEntry,
    )
}


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
        origin = payload.get("origin")
        if origin not in (None, "child_notification"):
            raise ValueError(f"unknown UserMessage origin: {origin}")
        return UserMessageEntry(
            **common,
            content=payload["content"],
            acceptance_id=(str(payload["acceptance_id"]) if payload.get("acceptance_id") else None),
            origin=origin,
        )
    if entry_type == "assistant_message":
        calls = tuple(
            ToolCall(
                id=str(call["id"]),
                name=str(call["name"]),
                arguments=dict(call.get("arguments") or {}),
                argument_error=call.get("argument_error"),
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
            call=payload.get("call"),
            acceptance_id=(str(payload["acceptance_id"]) if payload.get("acceptance_id") else None),
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
                facts=payload.get("facts"),
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
    if entry_type == "delivery_warning":
        return DeliveryWarningEntry(
            **common,
            operation_id=str(payload["operation_id"]),
            attempt=int(payload["attempt"]),
            findings=tuple(str(item) for item in payload.get("findings") or ()),
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
            call=payload.get("call"),
        )
    raise ValueError(f"unknown Entry type: {entry_type}")


__all__ = [
    "ENTRY_TYPE_TO_CLASS",
    "SESSION_ENTRY_SCHEMA_VERSION",
    "AssistantMessageEntry",
    "CompactionEntry",
    "ControlMessageEntry",
    "DeliveryWarningEntry",
    "SessionEntry",
    "SessionEntryKind",
    "ToolResultMessageEntry",
    "UserMessageEntry",
    "decode_entry_payload",
]
