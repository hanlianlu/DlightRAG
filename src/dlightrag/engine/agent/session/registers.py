# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed typed current-state registers for one Agent Session.

Entries carry immutable conversation meaning. Registers are exact-CAS cells
for Lane cursors and complete current Runtime state. There is no generic JSON
register namespace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.ids import (
    EntryId,
    IntentId,
    LaneId,
    OperationId,
    ProjectionId,
)
from dlightrag.engine.agent.session.operation import (
    OperationMeta,
    RunOperationState,
    decode_operation_state,
    operation_state_payload,
)
from dlightrag.engine.agent.session.projection import ContextProjection

REGISTER_SCHEMA_VERSION = 2

RegisterKind = Literal[
    "lane_head",
    "lane_state",
    "operation_meta",
    "operation_state",
    "request_snapshot",
    "tool_arguments",
    "pending_input",
    "host_turn_reservation",
    "context_projection",
    "session_fault",
]
_REGISTER_KINDS = frozenset(
    {
        "lane_head",
        "lane_state",
        "operation_meta",
        "operation_state",
        "request_snapshot",
        "tool_arguments",
        "pending_input",
        "host_turn_reservation",
        "context_projection",
        "session_fault",
    }
)


@dataclass(frozen=True, slots=True)
class RegisterRef:
    """Stable identity of one closed register cell."""

    kind: RegisterKind
    key: str

    def __post_init__(self) -> None:
        if self.kind not in _REGISTER_KINDS:
            raise ValueError("unknown Agent Session register kind")
        if not self.key:
            raise ValueError("register key cannot be empty")


@dataclass(frozen=True, slots=True)
class LaneHead:
    """The immutable-entry leaf where one Lane appends next."""

    lane_id: LaneId
    entry_id: EntryId | None

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("lane_head", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "entry_id": self.entry_id.value if self.entry_id is not None else None,
        }


@dataclass(frozen=True, slots=True)
class LaneState:
    """The bounded mutable lifecycle of one Lane cursor."""

    lane_id: LaneId
    archived: bool = False
    active_operation_id: str | None = None
    last_operation_id: str | None = None

    def __post_init__(self) -> None:
        if self.archived and self.active_operation_id is not None:
            raise ValueError("an archived Lane cannot own an active Operation")

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("lane_state", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "archived": self.archived,
            "active_operation_id": self.active_operation_id,
            "last_operation_id": self.last_operation_id,
        }


@dataclass(frozen=True, slots=True)
class SessionFault:
    """One exhaustively durable invariant fault that closes this Session."""

    detail: str

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError("Session fault detail cannot be empty")

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("session_fault", "session")

    def canonical_payload(self) -> dict[str, Any]:
        return {"schema_version": REGISTER_SCHEMA_VERSION, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class OperationMetaRegister:
    meta: OperationMeta

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("operation_meta", self.meta.operation_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {"schema_version": REGISTER_SCHEMA_VERSION, **self.meta.canonical_payload()}


@dataclass(frozen=True, slots=True)
class OperationStateRegister:
    state: RunOperationState

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("operation_state", self.state.operation_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {"schema_version": REGISTER_SCHEMA_VERSION, **operation_state_payload(self.state)}


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    """Exact transient provider request persisted before an attempt begins."""

    operation_id: OperationId
    turn_number: int
    plan_digest: str
    model_role: str
    messages_json: str
    tools_json: str
    tool_choice: str
    max_tokens: int | None

    def __post_init__(self) -> None:
        if self.turn_number < 1:
            raise ValueError("provider Request Snapshot turn must be positive")
        if len(self.plan_digest) != 64:
            raise ValueError("provider Request Snapshot Plan digest must be SHA-256")
        if not self.model_role or not self.tool_choice:
            raise ValueError("provider Request Snapshot identity cannot be empty")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("provider Request Snapshot output limit must be positive")
        messages = json.loads(self.messages_json)
        tools = json.loads(self.tools_json)
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError("provider Request Snapshot messages/tools must be arrays")

    @classmethod
    def from_values(
        cls,
        *,
        operation_id: OperationId,
        turn_number: int,
        plan_digest: str,
        model_role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        max_tokens: int | None,
    ) -> RequestSnapshot:
        return cls(
            operation_id=operation_id,
            turn_number=turn_number,
            plan_digest=plan_digest,
            model_role=model_role,
            messages_json=canonical_json(messages),
            tools_json=canonical_json(tools),
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        return json.loads(self.messages_json)

    @property
    def tools(self) -> list[dict[str, Any]]:
        return json.loads(self.tools_json)

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("request_snapshot", self.operation_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "operation_id": self.operation_id.value,
            "turn_number": self.turn_number,
            "plan_digest": self.plan_digest,
            "model_role": self.model_role,
            "messages": self.messages,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True, slots=True)
class ToolArguments:
    """Transient validated arguments retained only while one Tool can recover."""

    intent_id: IntentId
    canonical_input: str

    def __post_init__(self) -> None:
        value = json.loads(self.canonical_input)
        if not isinstance(value, dict):
            raise ValueError("Tool Arguments must be a canonical object")
        if canonical_json(value) != self.canonical_input:
            raise ValueError("Tool Arguments must use canonical JSON")

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.canonical_input)

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("tool_arguments", self.intent_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "intent_id": self.intent_id.value,
            "arguments": self.arguments,
        }


@dataclass(frozen=True, slots=True)
class HostTurnReservation:
    """One Fast Host turn accepted on a Lane but not yet settled."""

    lane_id: LaneId
    reservation_id: str
    idempotency_key: str
    user_entry_id: EntryId

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.idempotency_key:
            raise ValueError("Host turn reservation identity cannot be empty")

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("host_turn_reservation", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "reservation_id": self.reservation_id,
            "idempotency_key": self.idempotency_key,
            "user_entry_id": self.user_entry_id.value,
        }


@dataclass(frozen=True, slots=True)
class ContextProjectionRegister:
    """Current branch-local compaction projection for one Lane."""

    lane_id: LaneId
    projection: ContextProjection

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("context_projection", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        value = self.projection
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "projection_id": value.projection_id.value,
            "first_retained_sequence": value.first_retained_sequence,
            "covered_through_sequence": value.covered_through_sequence,
            "covered_through_entry_id": (
                value.covered_through_entry_id.value
                if value.covered_through_entry_id is not None
                else None
            ),
            "first_retained_entry_id": (
                value.first_retained_entry_id.value
                if value.first_retained_entry_id is not None
                else None
            ),
            "source_digest": value.source_digest,
            "summary": value.summary,
        }


@dataclass(frozen=True, slots=True)
class FollowUpInput:
    input_id: str
    idempotency_key: str
    content_json: str

    def __post_init__(self) -> None:
        if not self.input_id or not self.idempotency_key or not self.content_json:
            raise ValueError("Pending follow-up identity and content cannot be empty")

    @classmethod
    def from_content(cls, *, input_id: str, idempotency_key: str, content: Any) -> FollowUpInput:
        return cls(
            input_id=input_id,
            idempotency_key=idempotency_key,
            content_json=canonical_json(content),
        )

    @property
    def content(self) -> Any:
        return json.loads(self.content_json)


@dataclass(frozen=True, slots=True)
class PendingInput:
    """Bounded FIFO of unaccepted follow-ups for one busy Lane."""

    lane_id: LaneId
    items: tuple[FollowUpInput, ...]

    @property
    def ref(self) -> RegisterRef:
        return RegisterRef("pending_input", self.lane_id.value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTER_SCHEMA_VERSION,
            "lane_id": self.lane_id.value,
            "items": [
                {
                    "input_id": item.input_id,
                    "idempotency_key": item.idempotency_key,
                    "content": item.content,
                }
                for item in self.items
            ],
        }


type SessionRegister = (
    LaneHead
    | LaneState
    | OperationMetaRegister
    | OperationStateRegister
    | RequestSnapshot
    | ToolArguments
    | PendingInput
    | HostTurnReservation
    | ContextProjectionRegister
    | SessionFault
)


@dataclass(frozen=True, slots=True)
class RegisterRecord:
    """One current register value and its exact CAS sequence token."""

    value: SessionRegister
    sequence: int

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("register sequence must be positive")

    @property
    def ref(self) -> RegisterRef:
        return self.value.ref


@dataclass(frozen=True, slots=True)
class SetRegister:
    value: SessionRegister

    @property
    def ref(self) -> RegisterRef:
        return self.value.ref


@dataclass(frozen=True, slots=True)
class DeleteRegister:
    ref: RegisterRef


type RegisterWrite = SetRegister | DeleteRegister


def decode_register(*, kind: str, payload: dict[str, Any]) -> SessionRegister:
    """Decode one value from the closed durable register union."""
    if int(payload.get("schema_version") or 0) != REGISTER_SCHEMA_VERSION:
        raise ValueError("Agent Session register schema version is not current")
    if kind == "lane_head":
        lane_id = LaneId(str(payload["lane_id"]))
        raw_entry_id = payload.get("entry_id")
        return LaneHead(
            lane_id=lane_id,
            entry_id=EntryId(str(raw_entry_id)) if raw_entry_id is not None else None,
        )
    if kind == "lane_state":
        return LaneState(
            lane_id=LaneId(str(payload["lane_id"])),
            archived=bool(payload.get("archived") or False),
            active_operation_id=(
                str(payload["active_operation_id"])
                if payload.get("active_operation_id") is not None
                else None
            ),
            last_operation_id=(
                str(payload["last_operation_id"])
                if payload.get("last_operation_id") is not None
                else None
            ),
        )
    if kind == "session_fault":
        return SessionFault(detail=str(payload["detail"]))
    if kind == "operation_meta":
        return OperationMetaRegister(
            OperationMeta(
                operation_id=OperationId(str(payload["operation_id"])),
                lane_id=LaneId(str(payload["lane_id"])),
                idempotency_key=str(payload["idempotency_key"]),
                acceptance_digest=str(payload["acceptance_digest"]),
                plan_json=str(payload["plan_json"]),
                plan_digest=str(payload["plan_digest"]),
            )
        )
    if kind == "operation_state":
        return OperationStateRegister(decode_operation_state(payload))
    if kind == "request_snapshot":
        return RequestSnapshot.from_values(
            operation_id=OperationId(str(payload["operation_id"])),
            turn_number=int(payload["turn_number"]),
            plan_digest=str(payload["plan_digest"]),
            model_role=str(payload["model_role"]),
            messages=list(payload.get("messages") or []),
            tools=list(payload.get("tools") or []),
            tool_choice=str(payload["tool_choice"]),
            max_tokens=(int(payload["max_tokens"]) if payload.get("max_tokens") else None),
        )
    if kind == "tool_arguments":
        return ToolArguments(
            intent_id=IntentId(str(payload["intent_id"])),
            canonical_input=canonical_json(dict(payload.get("arguments") or {})),
        )
    if kind == "host_turn_reservation":
        return HostTurnReservation(
            lane_id=LaneId(str(payload["lane_id"])),
            reservation_id=str(payload["reservation_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            user_entry_id=EntryId(str(payload["user_entry_id"])),
        )
    if kind == "context_projection":
        return ContextProjectionRegister(
            lane_id=LaneId(str(payload["lane_id"])),
            projection=ContextProjection(
                projection_id=ProjectionId(str(payload["projection_id"])),
                first_retained_sequence=int(payload["first_retained_sequence"]),
                covered_through_sequence=int(payload["covered_through_sequence"]),
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
                summary=payload.get("summary"),
            ),
        )
    if kind == "pending_input":
        return PendingInput(
            lane_id=LaneId(str(payload["lane_id"])),
            items=tuple(
                FollowUpInput.from_content(
                    input_id=str(item["input_id"]),
                    idempotency_key=str(item["idempotency_key"]),
                    content=item["content"],
                )
                for item in payload.get("items") or ()
            ),
        )
    raise ValueError(f"unknown Agent Session register kind: {kind}")


__all__ = [
    "REGISTER_SCHEMA_VERSION",
    "ContextProjectionRegister",
    "DeleteRegister",
    "FollowUpInput",
    "HostTurnReservation",
    "LaneHead",
    "LaneState",
    "OperationMetaRegister",
    "OperationStateRegister",
    "PendingInput",
    "RegisterRecord",
    "RegisterRef",
    "RegisterWrite",
    "RequestSnapshot",
    "SessionFault",
    "SessionRegister",
    "SetRegister",
    "ToolArguments",
    "decode_register",
]
