# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Total durable state for one Agent operation.

The operation register is the Runtime program counter.  Every variant contains
all data needed by the pure interpreter for its next decision; no absent Entry
or nullable phase field is interpreted as state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dlightrag.agent.session.effects import ReplayPolicy, canonical_json
from dlightrag.agent.session.ids import AttemptId, EntryId, IntentId, LaneId, OperationId

ToolCallDisposition = Literal[
    "executable",
    "unknown_tool",
    "invalid_arguments",
    "plan_denied",
    "truncated_call",
    "contract_changed",
]
TerminalFailureKind = Literal[
    "provider_unavailable",
    "provider_invalid",
    "plan_unavailable",
    "context_overflow",
    "compaction_failed",
    "runtime_fault",
]


@dataclass(frozen=True, slots=True)
class OperationMeta:
    """Immutable accepted identity and Plan pin for one operation."""

    operation_id: OperationId
    lane_id: LaneId
    idempotency_key: str
    acceptance_digest: str
    plan_json: str
    plan_digest: str

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            raise ValueError("Operation idempotency key cannot be empty")
        if len(self.acceptance_digest) != 64 or len(self.plan_digest) != 64:
            raise ValueError("Operation Meta digests must be SHA-256")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id.value,
            "lane_id": self.lane_id.value,
            "idempotency_key": self.idempotency_key,
            "acceptance_digest": self.acceptance_digest,
            "plan_json": self.plan_json,
            "plan_digest": self.plan_digest,
        }


@dataclass(frozen=True, slots=True)
class AcceptedSteer:
    """One accepted correction waiting for the next stable checkpoint."""

    control_id: str
    content_json: str

    def __post_init__(self) -> None:
        if not self.control_id or not self.content_json:
            raise ValueError("accepted Steer requires identity and content")

    @classmethod
    def from_content(cls, control_id: str, content: Any) -> AcceptedSteer:
        return cls(control_id=control_id, content_json=canonical_json(content))

    @property
    def content(self) -> Any:
        import json

        return json.loads(self.content_json)

    def canonical_payload(self) -> dict[str, Any]:
        return {"control_id": self.control_id, "content": self.content}


@dataclass(frozen=True, slots=True)
class ToolBatchItem:
    """One provider source position with reserved durable identities."""

    source_index: int
    call_id: str
    tool_name: str
    disposition: ToolCallDisposition
    result_entry_id: EntryId
    intent_id: IntentId | None = None
    replay_policy: ReplayPolicy = "never"
    contract_version: int = 1
    input_schema_digest: str = ""
    effective_input_digest: str = ""
    synthetic_message: str = ""

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("Tool Batch source index cannot be negative")
        if not self.call_id or not self.tool_name:
            raise ValueError("Tool Batch item requires call and Tool identity")
        executable = self.disposition == "executable"
        if executable != (self.intent_id is not None):
            raise ValueError("only executable Tool Batch items carry an IntentId")
        if executable:
            if self.contract_version < 1:
                raise ValueError("Tool contract version must be positive")
            if len(self.input_schema_digest) != 64 or len(self.effective_input_digest) != 64:
                raise ValueError("executable Tool Batch item digests must be SHA-256")
        elif not self.synthetic_message:
            raise ValueError("non-executable Tool Batch item requires a synthetic result")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "disposition": self.disposition,
            "result_entry_id": self.result_entry_id.value,
            "intent_id": self.intent_id.value if self.intent_id is not None else None,
            "replay_policy": self.replay_policy,
            "contract_version": self.contract_version,
            "input_schema_digest": self.input_schema_digest,
            "effective_input_digest": self.effective_input_digest,
            "synthetic_message": self.synthetic_message,
        }


@dataclass(frozen=True, slots=True)
class ToolBatchPlan:
    """Complete ordered plan for every call in one Assistant response."""

    assistant_entry_id: EntryId
    items: tuple[ToolBatchItem, ...]

    def __post_init__(self) -> None:
        if tuple(item.source_index for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("Tool Batch Plan must cover contiguous provider source positions")
        call_ids = [item.call_id for item in self.items]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Tool Batch Plan call identities must be unique")
        result_ids = [item.result_entry_id for item in self.items]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Tool Batch Plan result identities must be unique")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "assistant_entry_id": self.assistant_entry_id.value,
            "items": [item.canonical_payload() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class ReadyForProvider:
    operation_id: OperationId
    turn_count: int = 0
    provider_attempts: int = 0
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["ready_for_provider"] = "ready_for_provider"


@dataclass(frozen=True, slots=True)
class ProviderRequestPending:
    operation_id: OperationId
    turn_number: int
    provider_attempts: int
    attempt_ids: tuple[AttemptId, ...] = ()
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["provider_request_pending"] = "provider_request_pending"

    def __post_init__(self) -> None:
        if self.provider_attempts != len(self.attempt_ids):
            raise ValueError("provider attempt count and identities must match")


@dataclass(frozen=True, slots=True)
class ToolBatchReady:
    operation_id: OperationId
    turn_number: int
    batch: ToolBatchPlan
    next_source_index: int = 0
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["tool_batch_ready"] = "tool_batch_ready"

    def __post_init__(self) -> None:
        if not 0 <= self.next_source_index <= len(self.batch.items):
            raise ValueError("Tool Batch cursor is outside its complete Plan")


@dataclass(frozen=True, slots=True)
class ToolEffectPending:
    operation_id: OperationId
    turn_number: int
    batch: ToolBatchPlan
    source_index: int
    attempt_id: AttemptId
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["tool_effect_pending"] = "tool_effect_pending"

    def __post_init__(self) -> None:
        if not 0 <= self.source_index < len(self.batch.items):
            raise ValueError("pending Tool source position is outside its Plan")
        if self.batch.items[self.source_index].disposition != "executable":
            raise ValueError("only an executable Tool item may be pending")


@dataclass(frozen=True, slots=True)
class CompactionPending:
    operation_id: OperationId
    turn_count: int
    attempt: int
    max_attempts: int
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["compaction_pending"] = "compaction_pending"

    def __post_init__(self) -> None:
        if self.attempt < 0 or self.max_attempts < 1 or self.attempt > self.max_attempts:
            raise ValueError("Compaction attempt is outside its bounded policy")


@dataclass(frozen=True, slots=True)
class CompletionReady:
    """A terminal-looking provider response awaiting checkpoint controls."""

    operation_id: OperationId
    turn_count: int
    assistant_entry_id: EntryId
    steers: tuple[AcceptedSteer, ...] = ()
    state_type: Literal["completion_ready"] = "completion_ready"


@dataclass(frozen=True, slots=True)
class Cancelling:
    operation_id: OperationId
    turn_count: int
    batch: ToolBatchPlan | None = None
    next_source_index: int = 0
    uncertain_source_index: int | None = None
    uncertain_attempt_id: AttemptId | None = None
    state_type: Literal["cancelling"] = "cancelling"


@dataclass(frozen=True, slots=True)
class OperationCompleted:
    operation_id: OperationId
    turn_count: int
    assistant_entry_id: EntryId
    state_type: Literal["completed"] = "completed"


@dataclass(frozen=True, slots=True)
class OperationCancelled:
    operation_id: OperationId
    turn_count: int
    state_type: Literal["cancelled"] = "cancelled"


@dataclass(frozen=True, slots=True)
class OperationFailed:
    operation_id: OperationId
    turn_count: int
    kind: TerminalFailureKind
    detail: str
    provider_attempt_ids: tuple[AttemptId, ...] = ()
    state_type: Literal["failed"] = "failed"


type RunOperationState = (
    ReadyForProvider
    | ProviderRequestPending
    | ToolBatchReady
    | ToolEffectPending
    | CompactionPending
    | CompletionReady
    | Cancelling
    | OperationCompleted
    | OperationCancelled
    | OperationFailed
)


def operation_state_payload(state: RunOperationState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_type": state.state_type,
        "operation_id": state.operation_id.value,
        "turn_count": getattr(state, "turn_count", 0),
    }
    if isinstance(state, ReadyForProvider):
        payload.update(
            provider_attempts=state.provider_attempts,
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, ProviderRequestPending):
        payload.update(
            turn_number=state.turn_number,
            provider_attempts=state.provider_attempts,
            attempt_ids=[attempt.value for attempt in state.attempt_ids],
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, ToolBatchReady):
        payload.update(
            turn_number=state.turn_number,
            batch=state.batch.canonical_payload(),
            next_source_index=state.next_source_index,
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, ToolEffectPending):
        payload.update(
            turn_number=state.turn_number,
            batch=state.batch.canonical_payload(),
            source_index=state.source_index,
            attempt_id=state.attempt_id.value,
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, CompactionPending):
        payload.update(
            attempt=state.attempt,
            max_attempts=state.max_attempts,
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, CompletionReady):
        payload.update(
            assistant_entry_id=state.assistant_entry_id.value,
            steers=[item.canonical_payload() for item in state.steers],
        )
    elif isinstance(state, Cancelling):
        payload.update(
            batch=state.batch.canonical_payload() if state.batch is not None else None,
            next_source_index=state.next_source_index,
            uncertain_source_index=state.uncertain_source_index,
            uncertain_attempt_id=(
                state.uncertain_attempt_id.value if state.uncertain_attempt_id is not None else None
            ),
        )
    elif isinstance(state, OperationCompleted):
        payload["assistant_entry_id"] = state.assistant_entry_id.value
    elif isinstance(state, OperationFailed):
        payload.update(
            kind=state.kind,
            detail=state.detail,
            provider_attempt_ids=[attempt.value for attempt in state.provider_attempt_ids],
        )
    return payload


def _steers(payload: dict[str, Any]) -> tuple[AcceptedSteer, ...]:
    return tuple(
        AcceptedSteer.from_content(str(item["control_id"]), item["content"])
        for item in payload.get("steers") or ()
    )


def _batch(payload: dict[str, Any]) -> ToolBatchPlan:
    return ToolBatchPlan(
        assistant_entry_id=EntryId(str(payload["assistant_entry_id"])),
        items=tuple(
            ToolBatchItem(
                source_index=int(item["source_index"]),
                call_id=str(item["call_id"]),
                tool_name=str(item["tool_name"]),
                disposition=item["disposition"],
                result_entry_id=EntryId(str(item["result_entry_id"])),
                intent_id=(IntentId(str(item["intent_id"])) if item.get("intent_id") else None),
                replay_policy=item.get("replay_policy", "never"),
                contract_version=int(item.get("contract_version") or 1),
                input_schema_digest=str(item.get("input_schema_digest") or ""),
                effective_input_digest=str(item.get("effective_input_digest") or ""),
                synthetic_message=str(item.get("synthetic_message") or ""),
            )
            for item in payload.get("items") or ()
        ),
    )


def decode_operation_state(payload: dict[str, Any]) -> RunOperationState:
    kind = str(payload["state_type"])
    operation_id = OperationId(str(payload["operation_id"]))
    turn_count = int(payload.get("turn_count") or 0)
    if kind == "ready_for_provider":
        return ReadyForProvider(
            operation_id,
            turn_count=turn_count,
            provider_attempts=int(payload.get("provider_attempts") or 0),
            steers=_steers(payload),
        )
    if kind == "provider_request_pending":
        return ProviderRequestPending(
            operation_id,
            turn_number=int(payload["turn_number"]),
            provider_attempts=int(payload.get("provider_attempts") or 0),
            attempt_ids=tuple(
                AttemptId(str(attempt_id)) for attempt_id in payload.get("attempt_ids") or ()
            ),
            steers=_steers(payload),
        )
    if kind == "tool_batch_ready":
        return ToolBatchReady(
            operation_id,
            turn_number=int(payload["turn_number"]),
            batch=_batch(dict(payload["batch"])),
            next_source_index=int(payload.get("next_source_index") or 0),
            steers=_steers(payload),
        )
    if kind == "tool_effect_pending":
        return ToolEffectPending(
            operation_id,
            turn_number=int(payload["turn_number"]),
            batch=_batch(dict(payload["batch"])),
            source_index=int(payload["source_index"]),
            attempt_id=AttemptId(str(payload["attempt_id"])),
            steers=_steers(payload),
        )
    if kind == "compaction_pending":
        return CompactionPending(
            operation_id,
            turn_count=turn_count,
            attempt=int(payload["attempt"]),
            max_attempts=int(payload["max_attempts"]),
            steers=_steers(payload),
        )
    if kind == "completion_ready":
        return CompletionReady(
            operation_id,
            turn_count=turn_count,
            assistant_entry_id=EntryId(str(payload["assistant_entry_id"])),
            steers=_steers(payload),
        )
    if kind == "cancelling":
        raw_batch = payload.get("batch")
        return Cancelling(
            operation_id,
            turn_count=turn_count,
            batch=_batch(dict(raw_batch)) if isinstance(raw_batch, dict) else None,
            next_source_index=int(payload.get("next_source_index") or 0),
            uncertain_source_index=(
                int(payload["uncertain_source_index"])
                if payload.get("uncertain_source_index") is not None
                else None
            ),
            uncertain_attempt_id=(
                AttemptId(str(payload["uncertain_attempt_id"]))
                if payload.get("uncertain_attempt_id")
                else None
            ),
        )
    if kind == "completed":
        return OperationCompleted(
            operation_id,
            turn_count=turn_count,
            assistant_entry_id=EntryId(str(payload["assistant_entry_id"])),
        )
    if kind == "cancelled":
        return OperationCancelled(operation_id, turn_count=turn_count)
    if kind == "failed":
        return OperationFailed(
            operation_id,
            turn_count=turn_count,
            kind=payload["kind"],
            detail=str(payload["detail"]),
            provider_attempt_ids=tuple(
                AttemptId(str(attempt_id))
                for attempt_id in payload.get("provider_attempt_ids") or ()
            ),
        )
    raise ValueError(f"unknown Agent Operation state type: {kind}")


__all__ = [
    "AcceptedSteer",
    "Cancelling",
    "CompactionPending",
    "CompletionReady",
    "OperationCancelled",
    "OperationCompleted",
    "OperationFailed",
    "OperationMeta",
    "ProviderRequestPending",
    "ReadyForProvider",
    "RunOperationState",
    "TerminalFailureKind",
    "ToolBatchItem",
    "ToolBatchPlan",
    "ToolBatchReady",
    "ToolCallDisposition",
    "ToolEffectPending",
    "decode_operation_state",
    "operation_state_payload",
]
