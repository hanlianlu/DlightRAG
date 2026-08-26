# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Atomic Session transactions over immutable entries and typed registers."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dlightrag.agent.session.entries import SessionEntry
from dlightrag.agent.session.ids import LaneId, SessionId
from dlightrag.agent.session.projection import ContextProjection
from dlightrag.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRef,
    RegisterWrite,
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
class SessionTransaction[HostDeltaT]:
    """One all-or-none mutation over the two canonical durable forms."""

    entries: tuple[SessionEntry, ...] = ()
    register_writes: tuple[RegisterWrite, ...] = ()
    expectations: tuple[RegisterExpectation, ...] = ()
    projection: ContextProjection | None = None
    host_delta: HostDeltaT | None = None

    def __post_init__(self) -> None:
        if not self.entries and not self.register_writes and self.projection is None:
            raise ValueError("a Session transaction requires at least one mutation")
        expectation_refs = [expectation.ref for expectation in self.expectations]
        if len(expectation_refs) != len(set(expectation_refs)):
            raise ValueError("a Session transaction cannot repeat a register expectation")
        write_refs = [write.ref for write in self.register_writes]
        if len(write_refs) != len(set(write_refs)):
            raise ValueError("a Session transaction cannot write one register twice")
        expected = set(expectation_refs)
        missing = [ref for ref in write_refs if ref not in expected]
        if missing:
            raise ValueError("every register write requires an exact expectation")
        for write in self.register_writes:
            if isinstance(write, DeleteRegister) and write.ref.key == LaneId.main().value:
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
        projection: ContextProjection | None = None,
        host_delta: HostDeltaT | None = None,
    ) -> SessionTransaction[HostDeltaT]:
        return cls(
            entries=tuple(entries),
            register_writes=tuple(register_writes),
            expectations=tuple(expectations),
            projection=projection,
            host_delta=host_delta,
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
    "RegisterConflict",
    "RegisterExpectation",
    "SessionTransaction",
    "SessionTransactionPort",
    "TransactionCommit",
    "TransactionLeaseLost",
    "TransactionOutcome",
]
