# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Atomic Session transactions over immutable entries and typed registers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dlightrag.engine.agent.session.entries import SessionEntry
from dlightrag.engine.agent.session.ids import IntentId, LaneId, SessionId
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    OperationMetaRegister,
    RegisterRef,
    RegisterWrite,
    SessionFault,
    SetRegister,
)


@dataclass(frozen=True, slots=True)
class RegisterExpectation:
    """Exact current sequence required for one register mutation.

    ``None`` means the register must not exist. There is deliberately no
    Session-wide expected version: unrelated Lane commits cannot conflict.
    """

    ref: RegisterRef
    sequence: int | None

    def __post_init__(self) -> None:
        if self.sequence is not None and self.sequence < 1:
            raise ValueError("expected register sequence must be positive")


@dataclass(frozen=True, slots=True)
class HostDeltaSettlement[HostDeltaT]:
    """One typed Host mutation bound to the Tool intent that produced it."""

    intent_id: IntentId
    value: HostDeltaT


@dataclass(frozen=True, slots=True)
class SessionTransaction[HostDeltaT]:
    """One all-or-none mutation over Runtime state and one typed HostDelta.

    ``advances_durable_progress`` defaults from whether the transaction appends
    semantic Entries or a HostDelta. Recovery closure that makes no new external
    progress overrides it to false.
    """

    entries: tuple[SessionEntry, ...] = ()
    register_writes: tuple[RegisterWrite, ...] = ()
    expectations: tuple[RegisterExpectation, ...] = ()
    host_delta: HostDeltaSettlement[HostDeltaT] | None = None
    advances_durable_progress: bool = False

    def __post_init__(self) -> None:
        if not self.entries and not self.register_writes:
            raise ValueError("a Session transaction requires at least one mutation")
        expectation_refs = [expectation.ref for expectation in self.expectations]
        if len(expectation_refs) != len(set(expectation_refs)):
            raise ValueError("a Session transaction cannot repeat a register expectation")
        write_refs = [write.ref for write in self.register_writes]
        if len(write_refs) != len(set(write_refs)):
            raise ValueError("a Session transaction cannot write one register twice")
        expected = set(expectation_refs)
        expectation_by_ref = {
            expectation.ref: expectation.sequence for expectation in self.expectations
        }
        missing = [ref for ref in write_refs if ref not in expected]
        if missing:
            raise ValueError("every register write requires an exact expectation")
        for write in self.register_writes:
            if (
                isinstance(write, SetRegister)
                and isinstance(write.value, OperationMetaRegister | SessionFault)
                and expectation_by_ref[write.ref] is not None
            ):
                raise ValueError("Operation Meta and Session Fault are immutable")
            if isinstance(write, DeleteRegister) and write.ref.kind in {
                "operation_meta",
                "operation_state",
                "session_fault",
            }:
                raise ValueError("Operation Meta/State registers cannot be deleted")
            if (
                isinstance(write, DeleteRegister)
                and write.ref.kind in {"lane_head", "lane_state"}
                and write.ref.key == LaneId.main().value
            ):
                raise ValueError("main Lane registers cannot be deleted")
            if (
                isinstance(write, SetRegister)
                and isinstance(write.value, LaneState)
                and write.value.lane_id == LaneId.main()
                and write.value.archived
            ):
                raise ValueError("the main Lane cannot be archived")
        if self.entries:
            final_entry_id = self.entries[-1].entry_id
            advances = [
                write
                for write in self.register_writes
                if isinstance(write, SetRegister)
                and isinstance(write.value, LaneHead)
                and write.value.entry_id == final_entry_id
            ]
            if not advances:
                raise ValueError("an Entry transaction must advance a Lane Head")

    @classmethod
    def from_parts(
        cls,
        *,
        entries: Sequence[SessionEntry] = (),
        register_writes: Sequence[RegisterWrite] = (),
        expectations: Sequence[RegisterExpectation] = (),
        host_delta: HostDeltaSettlement[HostDeltaT] | None = None,
        advances_durable_progress: bool | None = None,
    ) -> SessionTransaction[HostDeltaT]:
        frozen_entries = tuple(entries)
        return cls(
            entries=frozen_entries,
            register_writes=tuple(register_writes),
            expectations=tuple(expectations),
            host_delta=host_delta,
            advances_durable_progress=(
                bool(frozen_entries or host_delta)
                if advances_durable_progress is None
                else advances_durable_progress
            ),
        )


@dataclass(frozen=True, slots=True)
class TransactionCommit:
    """One committed transaction and its assigned durable sequences."""

    commit_sequence: int
    appended_sequences: tuple[int, ...]
    register_sequences: tuple[tuple[RegisterRef, int], ...]


@dataclass(frozen=True, slots=True)
class RegisterConflict:
    """One exact register expectation no longer matches current state."""

    ref: RegisterRef
    expected_sequence: int | None
    current_sequence: int | None


@dataclass(frozen=True, slots=True)
class TransactionLeaseLost:
    """The supplied fencing epoch no longer owns this Session."""


type TransactionOutcome = TransactionCommit | RegisterConflict | TransactionLeaseLost


class SessionTransactionPort[HostDeltaT](Protocol):
    """Atomic write seam used by the durable Session Runtime."""

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[HostDeltaT],
    ) -> TransactionOutcome: ...


__all__ = [
    "HostDeltaSettlement",
    "RegisterConflict",
    "RegisterExpectation",
    "SessionTransaction",
    "SessionTransactionPort",
    "TransactionCommit",
    "TransactionLeaseLost",
    "TransactionOutcome",
]
