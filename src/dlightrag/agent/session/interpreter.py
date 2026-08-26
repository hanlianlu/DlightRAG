# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pure total interpreter from OperationState to one closed NextAction."""

from dataclasses import dataclass
from typing import Literal

from dlightrag.agent.session.operation import (
    Cancelling,
    CompactionPending,
    CompletionReady,
    OperationCancelled,
    OperationCompleted,
    OperationFailed,
    ProviderRequestPending,
    ReadyForProvider,
    RunOperationState,
    ToolBatchItem,
    ToolBatchReady,
    ToolEffectPending,
)


@dataclass(frozen=True, slots=True)
class AssembleProviderRequest:
    turn_number: int


@dataclass(frozen=True, slots=True)
class CallProvider:
    turn_number: int


@dataclass(frozen=True, slots=True)
class CommitSyntheticToolResult:
    item: ToolBatchItem
    outcome: Literal[
        "unknown_tool",
        "invalid_arguments",
        "plan_denied",
        "truncated_arguments",
        "tool_contract_changed",
    ]


@dataclass(frozen=True, slots=True)
class BeginToolEffect:
    item: ToolBatchItem


@dataclass(frozen=True, slots=True)
class RecoverToolEffect:
    item: ToolBatchItem
    replay: bool


@dataclass(frozen=True, slots=True)
class ContinueAfterToolBatch:
    turn_number: int


@dataclass(frozen=True, slots=True)
class ConsumeSteer:
    control_id: str


@dataclass(frozen=True, slots=True)
class CompleteOperation:
    pass


@dataclass(frozen=True, slots=True)
class RunCompaction:
    attempt: int


@dataclass(frozen=True, slots=True)
class CloseCancellationPosition:
    item: ToolBatchItem
    outcome_unknown: bool


@dataclass(frozen=True, slots=True)
class FinishCancellation:
    pass


@dataclass(frozen=True, slots=True)
class NoAction:
    terminal: bool = True


type NextAction = (
    AssembleProviderRequest
    | CallProvider
    | CommitSyntheticToolResult
    | BeginToolEffect
    | RecoverToolEffect
    | ContinueAfterToolBatch
    | ConsumeSteer
    | CompleteOperation
    | RunCompaction
    | CloseCancellationPosition
    | FinishCancellation
    | NoAction
)


def next_action(state: RunOperationState) -> NextAction:
    """Return the only legal next action for one complete current state."""
    if isinstance(state, ReadyForProvider):
        if state.steers:
            return ConsumeSteer(state.steers[0].control_id)
        return AssembleProviderRequest(state.turn_count + 1)
    if isinstance(state, ProviderRequestPending):
        return CallProvider(state.turn_number)
    if isinstance(state, ToolBatchReady):
        if state.next_source_index == len(state.batch.items):
            return ContinueAfterToolBatch(state.turn_number)
        item = state.batch.items[state.next_source_index]
        if item.disposition == "executable":
            return BeginToolEffect(item)
        outcomes: dict[
            str,
            Literal[
                "unknown_tool",
                "invalid_arguments",
                "plan_denied",
                "truncated_arguments",
                "tool_contract_changed",
            ],
        ] = {
            "unknown_tool": "unknown_tool",
            "invalid_arguments": "invalid_arguments",
            "plan_denied": "plan_denied",
            "truncated_call": "truncated_arguments",
            "contract_changed": "tool_contract_changed",
        }
        return CommitSyntheticToolResult(item, outcomes[item.disposition])
    if isinstance(state, ToolEffectPending):
        item = state.batch.items[state.source_index]
        return RecoverToolEffect(item, replay=item.replay_policy == "replayable")
    if isinstance(state, CompletionReady):
        if state.steers:
            return ConsumeSteer(state.steers[0].control_id)
        return CompleteOperation()
    if isinstance(state, CompactionPending):
        return RunCompaction(state.attempt)
    if isinstance(state, Cancelling):
        if state.batch is None or state.next_source_index >= len(state.batch.items):
            return FinishCancellation()
        item = state.batch.items[state.next_source_index]
        return CloseCancellationPosition(
            item,
            outcome_unknown=state.uncertain_source_index == state.next_source_index,
        )
    if isinstance(state, (OperationCompleted, OperationCancelled, OperationFailed)):
        return NoAction()
    raise AssertionError(f"unhandled Operation state: {type(state).__name__}")


__all__ = [
    "AssembleProviderRequest",
    "BeginToolEffect",
    "CallProvider",
    "CloseCancellationPosition",
    "CommitSyntheticToolResult",
    "CompleteOperation",
    "ConsumeSteer",
    "ContinueAfterToolBatch",
    "FinishCancellation",
    "NextAction",
    "NoAction",
    "RecoverToolEffect",
    "RunCompaction",
    "next_action",
]
