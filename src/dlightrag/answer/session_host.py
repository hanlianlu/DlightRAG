# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned Fast turns over the canonical Agent Session tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dlightrag.agent.session.entries import AssistantMessageEntry, UserMessageEntry
from dlightrag.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.agent.session.registers import (
    DeleteRegister,
    HostTurnReservation,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SessionFault,
    SetRegister,
)
from dlightrag.agent.session.runtime import OperationConflictError, SessionLeaseLostError
from dlightrag.agent.session.store import AgentSessionSnapshot
from dlightrag.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    SessionTransactionPort,
    TransactionCommit,
    TransactionLeaseLost,
)


@dataclass(frozen=True, slots=True)
class AcceptedFastTurn:
    session_id: SessionId
    lane_id: LaneId
    reservation_id: str
    user_entry_id: EntryId
    created: bool
    settled: bool = False


class FastSessionHost:
    """Atomically reserve and settle Fast Host turns on one Session Lane."""

    def __init__(
        self,
        *,
        transactions: SessionTransactionPort[Any],
        load: Any,
        fencing_epoch: int,
    ) -> None:
        self._transactions = transactions
        self._load = load
        self._fencing_epoch = fencing_epoch

    async def accept(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        idempotency_key: str,
        content: Any,
    ) -> AcceptedFastTurn:
        snapshot = await self._load(session_id)
        _reject_fault(snapshot)
        reservation_ref = RegisterRef("host_turn_reservation", lane_id.value)
        existing = _register(snapshot, reservation_ref)
        if existing is not None:
            value = existing.value
            if not isinstance(value, HostTurnReservation):
                raise TypeError("Host turn reservation register has the wrong value type")
            if value.reservation_id == reservation_id and value.idempotency_key == idempotency_key:
                return AcceptedFastTurn(
                    session_id,
                    lane_id,
                    reservation_id,
                    value.user_entry_id,
                    created=False,
                )
            raise OperationConflictError("Lane already owns another Fast Host turn")

        try:
            lane = snapshot.tree.lane(lane_id)
        except KeyError:
            if snapshot.entries or lane_id != LaneId.main():
                raise OperationConflictError(f"unknown Lane: {lane_id.value}") from None
            parent = None
            head_sequence = None
            state_sequence = None
            lane_state = LaneState(lane_id)
            state_write = True
        else:
            lane_state = lane.state.value
            if not isinstance(lane_state, LaneState):
                raise TypeError("Lane State register has the wrong value type")
            if lane.archived:
                raise OperationConflictError("an archived Lane cannot accept a Fast turn")
            if lane_state.active_operation_id is not None:
                raise OperationConflictError("Lane already owns an Agent operation")
            parent = lane.head_entry_id
            head_sequence = lane.head.sequence
            state_sequence = lane.state.sequence
            state_write = False
            ancestry = snapshot.tree.ancestry(lane_id)
            if ancestry:
                latest = ancestry[-1]
                if (
                    isinstance(latest, AssistantMessageEntry)
                    and isinstance(latest.provider_state, dict)
                    and latest.provider_state.get("fast_reservation_id") == reservation_id
                ):
                    user = next(
                        (
                            entry
                            for entry in reversed(ancestry[:-1])
                            if isinstance(entry, UserMessageEntry)
                            and entry.acceptance_id == reservation_id
                        ),
                        None,
                    )
                    if user is None:
                        raise OperationConflictError(
                            "settled Fast turn lost its accepted User Entry"
                        )
                    if user.content != content:
                        raise OperationConflictError(
                            "Fast turn replay changed its accepted content"
                        )
                    return AcceptedFastTurn(
                        session_id,
                        lane_id,
                        reservation_id,
                        user.entry_id,
                        created=False,
                        settled=True,
                    )
                if isinstance(latest, UserMessageEntry) and latest.acceptance_id == reservation_id:
                    if latest.content != content:
                        raise OperationConflictError(
                            "Fast turn replay changed its accepted content"
                        )
                    reservation = HostTurnReservation(
                        lane_id=lane_id,
                        reservation_id=reservation_id,
                        idempotency_key=idempotency_key,
                        user_entry_id=latest.entry_id,
                    )
                    await self._transact(
                        session_id,
                        SessionTransaction.from_parts(
                            register_writes=[SetRegister(reservation)],
                            expectations=[RegisterExpectation(reservation.ref, None)],
                        ),
                    )
                    return AcceptedFastTurn(
                        session_id,
                        lane_id,
                        reservation_id,
                        latest.entry_id,
                        created=False,
                    )

        message = UserMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=parent,
            content=content,
            acceptance_id=reservation_id,
        )
        reservation = HostTurnReservation(
            lane_id=lane_id,
            reservation_id=reservation_id,
            idempotency_key=idempotency_key,
            user_entry_id=message.entry_id,
        )
        writes = [
            SetRegister(LaneHead(lane_id, message.entry_id)),
            *((SetRegister(lane_state),) if state_write else ()),
            SetRegister(reservation),
        ]
        expectations = [
            RegisterExpectation(writes[0].ref, head_sequence),
            *((RegisterExpectation(lane_state.ref, state_sequence),) if state_write else ()),
            RegisterExpectation(reservation.ref, None),
        ]
        await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[message],
                register_writes=writes,
                expectations=expectations,
            ),
        )
        return AcceptedFastTurn(
            session_id,
            lane_id,
            reservation_id,
            message.entry_id,
            created=True,
        )

    async def complete(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
        content: str,
        usage: Any | None = None,
        cost: Any | None = None,
    ) -> TransactionCommit | None:
        snapshot = await self._load(session_id)
        _reject_fault(snapshot)
        reservation_record = _reservation(snapshot, lane_id)
        if reservation_record is None:
            ancestry = snapshot.tree.ancestry(lane_id)
            if ancestry and isinstance(ancestry[-1], AssistantMessageEntry):
                state = ancestry[-1].provider_state
                if isinstance(state, dict) and state.get("fast_reservation_id") == reservation_id:
                    return None
            raise OperationConflictError("Fast Host turn reservation is not active")
        if not isinstance(reservation_record.value, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        if reservation_record.value.reservation_id != reservation_id:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        head = snapshot.tree.lane(lane_id).head
        if not isinstance(head.value, LaneHead):
            raise TypeError("Lane Head register has the wrong value type")
        entry = AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=datetime.now(UTC),
            parent_entry_id=head.value.entry_id,
            content=content,
            stop_reason="stop",
            usage=usage,
            cost=cost,
            provider_state={"fast_reservation_id": reservation_id},
        )
        return await self._transact(
            session_id,
            SessionTransaction.from_parts(
                entries=[entry],
                register_writes=[
                    SetRegister(LaneHead(lane_id, entry.entry_id)),
                    DeleteRegister(reservation_record.ref),
                ],
                expectations=[
                    RegisterExpectation(head.ref, head.sequence),
                    RegisterExpectation(reservation_record.ref, reservation_record.sequence),
                ],
            ),
        )

    async def fail(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        reservation_id: str,
    ) -> TransactionCommit | None:
        snapshot = await self._load(session_id)
        reservation = _reservation(snapshot, lane_id)
        if reservation is None:
            ancestry = snapshot.tree.ancestry(lane_id)
            if ancestry and isinstance(ancestry[-1], UserMessageEntry):
                if ancestry[-1].acceptance_id == reservation_id:
                    return None
            raise OperationConflictError("Fast Host turn reservation is not active")
        if not isinstance(reservation.value, HostTurnReservation):
            raise TypeError("Host turn reservation register has the wrong value type")
        if reservation.value.reservation_id != reservation_id:
            raise OperationConflictError("Fast Host turn reservation identity changed")
        return await self._transact(
            session_id,
            SessionTransaction.from_parts(
                register_writes=[DeleteRegister(reservation.ref)],
                expectations=[RegisterExpectation(reservation.ref, reservation.sequence)],
            ),
        )

    async def _transact(
        self,
        session_id: SessionId,
        transaction: SessionTransaction[Any],
    ) -> TransactionCommit:
        outcome = await self._transactions.transact(
            session_id=session_id,
            fencing_epoch=self._fencing_epoch,
            transaction=transaction,
        )
        if isinstance(outcome, RegisterConflict):
            raise OperationConflictError(f"register {outcome.ref.kind}:{outcome.ref.key} changed")
        if isinstance(outcome, TransactionLeaseLost):
            raise SessionLeaseLostError(session_id.value)
        return outcome


def _register(snapshot: AgentSessionSnapshot, ref: RegisterRef) -> RegisterRecord | None:
    return next((record for record in snapshot.registers if record.ref == ref), None)


def _reject_fault(snapshot: AgentSessionSnapshot) -> None:
    if any(isinstance(record.value, SessionFault) for record in snapshot.registers):
        raise OperationConflictError("a faulted Session cannot accept a Fast turn")


def _reservation(
    snapshot: AgentSessionSnapshot,
    lane_id: LaneId,
) -> RegisterRecord | None:
    record = _register(snapshot, RegisterRef("host_turn_reservation", lane_id.value))
    if record is None:
        return None
    if not isinstance(record.value, HostTurnReservation):
        raise TypeError("Host turn reservation register has the wrong value type")
    return record


__all__ = ["AcceptedFastTurn", "FastSessionHost"]
