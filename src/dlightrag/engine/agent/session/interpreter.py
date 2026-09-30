# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pure total interpreter from OperationState to one closed NextAction."""

from dataclasses import dataclass
from itertools import takewhile
from typing import Literal

from dlightrag.engine.agent.session.operation import (
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
    ToolBatchPlan,
    ToolBatchReady,
    ToolEffectPending,
)

type SyntheticToolDisposition = Literal[
    "unknown_tool",
    "invalid_arguments",
    "plan_denied",
    "truncated_call",
    "contract_changed",
]
type SyntheticToolResultOutcome = Literal[
    "unknown_tool",
    "invalid_arguments",
    "plan_denied",
    "truncated_arguments",
    "tool_contract_changed",
]

TOOL_DISPOSITION_OUTCOME: dict[SyntheticToolDisposition, SyntheticToolResultOutcome] = {
    "unknown_tool": "unknown_tool",
    "invalid_arguments": "invalid_arguments",
    "plan_denied": "plan_denied",
    "truncated_call": "truncated_arguments",
    "contract_changed": "tool_contract_changed",
}

#: Most read-only calls one Tool Batch runs at once. A web-heavy turn issues four
#: to eight searches, so eight runs such a turn in one round while still bounding
#: what one batch asks of a search provider or the corpus at a time.
MAX_CONCURRENT_TOOL_CALLS = 8


@dataclass(frozen=True, slots=True)
class AssembleProviderRequest:
    turn_number: int
    #: Set when this turn already declined a compaction: the effect returns the
    #: request rather than asking to compact a prefix that cannot advance.
    compaction_declined: bool = False


@dataclass(frozen=True, slots=True)
class CallProvider:
    turn_number: int


@dataclass(frozen=True, slots=True)
class CommitSyntheticToolResult:
    item: ToolBatchItem
    outcome: SyntheticToolResultOutcome


@dataclass(frozen=True, slots=True)
class BeginToolEffect:
    """Start ``item``, and the read-only calls right after it that run beside it."""

    item: ToolBatchItem
    concurrent: tuple[ToolBatchItem, ...] = ()


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
        return AssembleProviderRequest(
            state.turn_count + 1, compaction_declined=state.compaction_declined
        )
    if isinstance(state, ProviderRequestPending):
        return CallProvider(state.turn_number)
    if isinstance(state, ToolBatchReady):
        if state.next_source_index == len(state.batch.items):
            return ContinueAfterToolBatch(state.turn_number)
        item = state.batch.items[state.next_source_index]
        if item.disposition == "executable":
            return BeginToolEffect(item, _concurrent_with(item, state.batch))
        return CommitSyntheticToolResult(item, TOOL_DISPOSITION_OUTCOME[item.disposition])
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


def _concurrent_with(item: ToolBatchItem, batch: ToolBatchPlan) -> tuple[ToolBatchItem, ...]:
    """The read-only calls right after a read-only ``item``, within the bound.

    Any other call, and any position that never executes, runs alone and ends the
    run of neighbours, so a call with side effects is a barrier in source order.
    """
    if not _read_only(item):
        return ()
    following = batch.items[item.source_index + 1 : item.source_index + MAX_CONCURRENT_TOOL_CALLS]
    return tuple(takewhile(_read_only, following))


def _read_only(item: ToolBatchItem) -> bool:
    return item.disposition == "executable" and item.read_only


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
    "MAX_CONCURRENT_TOOL_CALLS",
    "NextAction",
    "NoAction",
    "RecoverToolEffect",
    "RunCompaction",
    "TOOL_DISPOSITION_OUTCOME",
    "next_action",
]
