# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""In-memory adapter for the canonical Agent Session transaction seam."""

from collections.abc import Sequence
from dataclasses import replace

from dlightrag.agent.session.entries import SessionEntry
from dlightrag.agent.session.ids import EntryId, IntentId, LaneId, SessionId
from dlightrag.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SetRegister,
)
from dlightrag.agent.session.store import (
    AgentSessionSnapshot,
)
from dlightrag.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
    TransactionOutcome,
)


class MemoryAgentSessionStore[HostDeltaT]:
    """Process-local adapter with the PostgreSQL adapter's exact CAS semantics."""

    def __init__(self, *, fencing_epoch: int = 1) -> None:
        if fencing_epoch < 1:
            raise ValueError("fencing epoch must be positive")
        self._sessions: dict[SessionId, _Session[HostDeltaT]] = {}
        self._fencing_epoch = fencing_epoch

    def applied_host_deltas(self, session_id: SessionId) -> tuple[tuple[IntentId, HostDeltaT], ...]:
        """Return HostDelta settlements committed atomically in this adapter."""
        session = self._sessions.get(session_id)
        return () if session is None else tuple(session.host_deltas)

    def transfer_lease(self, fencing_epoch: int) -> None:
        """Fence prior writers in tests just as a durable lease transfer would."""
        if fencing_epoch <= self._fencing_epoch:
            raise ValueError("a transferred fencing epoch must increase")
        self._fencing_epoch = fencing_epoch

    async def load(self, session_id: SessionId) -> AgentSessionSnapshot:
        session = self._sessions.get(session_id)
        if session is None:
            return AgentSessionSnapshot(
                session_id=session_id,
                commit_sequence=0,
                entries=(),
                registers=(),
            )
        return self._snapshot(session_id, session)

    async def transact(
        self,
        *,
        session_id: SessionId,
        fencing_epoch: int,
        transaction: SessionTransaction[HostDeltaT],
    ) -> TransactionOutcome:
        if fencing_epoch != self._fencing_epoch:
            return TransactionLeaseLost()
        session = self._sessions.setdefault(session_id, _Session())
        for expectation in transaction.expectations:
            current = session.registers.get(expectation.ref)
            current_sequence = current.sequence if current is not None else None
            if current_sequence != expectation.sequence:
                return RegisterConflict(
                    ref=expectation.ref,
                    expected_sequence=expectation.sequence,
                    current_sequence=current_sequence,
                )
        self._validate_entries(session_id, session, transaction.entries)
        self._validate_register_writes(session, transaction)
        commit_sequence = session.commit_sequence + 1
        sequences = tuple(
            range(
                session.last_entry_sequence + 1,
                session.last_entry_sequence + 1 + len(transaction.entries),
            )
        )
        stamped = tuple(
            replace(entry, sequence=sequence)
            for entry, sequence in zip(transaction.entries, sequences, strict=True)
        )
        session.entries.extend(stamped)
        session.last_entry_sequence += len(stamped)
        register_sequences: list[tuple[RegisterRef, int]] = []
        for write in transaction.register_writes:
            if isinstance(write, SetRegister):
                session.registers[write.ref] = RegisterRecord(
                    value=write.value,
                    sequence=commit_sequence,
                )
                register_sequences.append((write.ref, commit_sequence))
            elif isinstance(write, DeleteRegister):
                session.registers.pop(write.ref, None)
                register_sequences.append((write.ref, commit_sequence))
        if transaction.host_delta is not None:
            session.host_deltas.append(
                (transaction.host_delta.intent_id, transaction.host_delta.value)
            )
        session.commit_sequence = commit_sequence
        return TransactionCommit(
            commit_sequence=commit_sequence,
            appended_sequences=sequences,
            register_sequences=tuple(register_sequences),
        )

    async def append_to_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
        expected_head: RegisterRecord,
        entries: Sequence[SessionEntry],
    ) -> TransactionOutcome:
        """Place a short Entry chain and advance exactly one Lane Head."""
        if not entries:
            raise ValueError("a Lane append requires at least one Entry")
        if not isinstance(expected_head.value, LaneHead):
            raise TypeError("expected_head must be a LaneHead record")
        if expected_head.value.lane_id != lane_id:
            raise ValueError("expected Lane Head belongs to another Lane")
        snapshot = await self.load(session_id)
        lane = snapshot.tree.lane(lane_id)
        if lane.archived:
            raise ValueError("an archived Lane is not writable")
        parent = expected_head.value.entry_id
        placed: list[SessionEntry] = []
        for entry in entries:
            placed_entry = replace(entry, parent_entry_id=parent)
            placed.append(placed_entry)
            parent = placed_entry.entry_id
        return await self.transact(
            session_id=session_id,
            fencing_epoch=self._fencing_epoch,
            transaction=SessionTransaction.from_parts(
                entries=placed,
                register_writes=[SetRegister(LaneHead(lane_id=lane_id, entry_id=parent))],
                expectations=[
                    RegisterExpectation(expected_head.ref, expected_head.sequence),
                    RegisterExpectation(lane.state.ref, lane.state.sequence),
                ],
            ),
        )

    async def fork_lane(
        self,
        *,
        session_id: SessionId,
        source_lane_id: LaneId,
        lane_id: LaneId,
        at_entry_id: EntryId | None = None,
    ) -> TransactionOutcome:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown Agent Session: {session_id}")
        snapshot = self._snapshot(session_id, session)
        source = snapshot.tree.lane(source_lane_id)
        target_head = source.head_entry_id if at_entry_id is None else at_entry_id
        if target_head is None:
            raise ValueError("a Lane cannot fork from an empty Head")
        ancestry_ids = {entry.entry_id for entry in snapshot.tree.ancestry(source_lane_id)}
        if target_head is not None and target_head not in ancestry_ids:
            raise ValueError("a fork target must belong to the source Lane ancestry")
        if not snapshot.tree.is_stable_checkpoint(target_head):
            raise ValueError("a Lane can fork only from a stable checkpoint")
        head = LaneHead(lane_id=lane_id, entry_id=target_head)
        state = LaneState(lane_id=lane_id)
        return await self.transact(
            session_id=session_id,
            fencing_epoch=self._fencing_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
            ),
        )

    async def archive_lane(
        self,
        *,
        session_id: SessionId,
        lane_id: LaneId,
    ) -> TransactionOutcome:
        if lane_id == LaneId.main():
            raise ValueError("the main Lane cannot be archived")
        snapshot = await self.load(session_id)
        lane = snapshot.tree.lane(lane_id)
        state = lane.state.value
        if not isinstance(state, LaneState):
            raise TypeError("Lane State register has the wrong value type")
        if state.active_operation_id is not None:
            raise ValueError("an active Lane cannot be archived")
        return await self.transact(
            session_id=session_id,
            fencing_epoch=self._fencing_epoch,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(replace(state, archived=True))],
                expectations=[RegisterExpectation(lane.state.ref, lane.state.sequence)],
            ),
        )

    @staticmethod
    def _validate_register_writes(
        session: _Session,
        transaction: SessionTransaction[HostDeltaT],
    ) -> None:
        values = {ref: record.value for ref, record in session.registers.items()}
        for write in transaction.register_writes:
            if isinstance(write, SetRegister):
                values[write.ref] = write.value
            elif isinstance(write, DeleteRegister):
                values.pop(write.ref, None)
        refs = set(values)
        heads = {ref.key for ref in refs if ref.kind == "lane_head"}
        states = {ref.key for ref in refs if ref.kind == "lane_state"}
        if heads != states or LaneId.main().value not in heads:
            raise ValueError("Session registers require complete main and Lane pairs")
        if transaction.entries:
            advanced_lanes = {
                write.value.lane_id
                for write in transaction.register_writes
                if isinstance(write, SetRegister)
                and isinstance(write.value, LaneHead)
                and write.value.entry_id == transaction.entries[-1].entry_id
            }
            for lane_id in advanced_lanes:
                state = values.get(LaneState(lane_id).ref)
                if isinstance(state, LaneState) and state.archived:
                    raise ValueError("an archived Lane is not writable")

    @staticmethod
    def _validate_entries(
        session_id: SessionId,
        session: _Session,
        entries: Sequence[SessionEntry],
    ) -> None:
        known = {entry.entry_id for entry in session.entries}
        roots = sum(entry.parent_entry_id is None for entry in session.entries)
        for entry in entries:
            if entry.session_id != session_id:
                raise ValueError("transaction Entry belongs to another Session")
            if entry.entry_id in known:
                raise ValueError("transaction Entry identity already exists")
            if entry.parent_entry_id is None:
                if known or roots:
                    raise ValueError("only the first Session Entry can be a root")
                roots += 1
            elif entry.parent_entry_id not in known:
                raise ValueError("transaction Entry parent is missing")
            known.add(entry.entry_id)

    @staticmethod
    def _main_lane(session: _Session[HostDeltaT] | None) -> tuple[RegisterRecord, bool]:
        ref = LaneHead(LaneId.main(), None).ref
        if session is not None and ref in session.registers:
            return session.registers[ref], False
        placeholder = RegisterRecord(value=LaneHead(LaneId.main(), None), sequence=1)
        return placeholder, True

    @staticmethod
    def _snapshot(session_id: SessionId, session: _Session[HostDeltaT]) -> AgentSessionSnapshot:
        return AgentSessionSnapshot(
            session_id=session_id,
            commit_sequence=session.commit_sequence,
            entries=tuple(session.entries),
            registers=tuple(
                record
                for _, record in sorted(
                    session.registers.items(),
                    key=lambda item: (item[0].kind, item[0].key),
                )
            ),
        )


class _Session[HostDeltaT]:
    def __init__(self) -> None:
        self.entries: list[SessionEntry] = []
        self.registers: dict[RegisterRef, RegisterRecord] = {}
        self.last_entry_sequence = 0
        self.commit_sequence = 0
        self.host_deltas: list[tuple[IntentId, HostDeltaT]] = []


__all__ = ["MemoryAgentSessionStore"]
