# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Delta-bounded recording-connection tests for PostgreSQL Session transactions."""

import hashlib
import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest

import dlightrag.adapters.postgres.answer.session_repository as session_repository_module
from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
from dlightrag.engine.agent.session.entries import UserMessageEntry
from dlightrag.engine.agent.session.ids import EntryId, IntentId, LaneId, SessionId
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    RegisterRecord,
    RegisterRef,
    SetRegister,
)
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.agent.session.transactions import (
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
)
from dlightrag.engine.runtime.settlements import EffectHostUpdate, OpaqueEvidenceWrite


class _TransactionContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _AcquireContext:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def __aenter__(self) -> Any:
        return self._connection

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _SnapshotPool:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._connection)


class _SnapshotConnection:
    def __init__(
        self,
        *,
        metadata: dict[str, int] | None,
        entry_rows: list[dict[str, Any]] | None = None,
        register_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.metadata = metadata
        self.entry_rows = entry_rows or []
        self.register_rows = register_rows or []
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.transaction_kwargs: list[dict[str, Any]] = []

    def transaction(self, **kwargs: Any) -> _TransactionContext:
        self.transaction_kwargs.append(kwargs)
        return _TransactionContext()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, int] | None:
        self.calls.append(("fetchrow", query, args))
        return self.metadata

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query, args))
        if "sequence > $3 AND sequence <= $4" in query:
            return self.entry_rows
        if "FROM dlightrag_agent_session_registers" in query:
            return self.register_rows
        raise AssertionError(f"unexpected snapshot fetch: {query}")


class _RecordingConnection:
    def __init__(
        self,
        *,
        register_sequences: dict[tuple[str, str], int | None] | None = None,
        entry_ids: set[str] | None = None,
        lane_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.register_sequences = register_sequences or {}
        self.entry_ids = entry_ids or set()
        self.lane_rows = lane_rows or []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchvals: list[tuple[str, tuple[Any, ...]]] = []
        self.executes: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        if "WITH expected(register_kind" in query:
            kinds, keys = args[2], args[3]
            return [
                {
                    "ordinal": ordinal,
                    "register_kind": kind,
                    "register_key": key,
                    "sequence": self.register_sequences.get((kind, key)),
                }
                for ordinal, (kind, key) in enumerate(zip(kinds, keys, strict=True), start=1)
            ]
        if "FROM dlightrag_agent_session_entries" in query:
            requested = {str(value) for value in args[2]}
            return [{"entry_id": value} for value in self.entry_ids & requested]
        if "register_kind IN ('lane_head', 'lane_state')" in query:
            return self.lane_rows
        raise AssertionError(f"unexpected fetch: {query}")

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchvals.append((query, args))
        if "LEFT JOIN dlightrag_answer_evidence" in query:
            return None
        raise AssertionError(f"unexpected fetchval: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        self.executes.append((query, args))
        return "OK"


def _repository() -> PGAgentSessionRepository:
    return PGAgentSessionRepository(
        pool=None,
        owner_id="owner",
        run_id=uuid.uuid4(),
        worker_id="worker",
        lease_owner="worker",
        fencing_epoch=1,
    )


def _snapshot_repository(connection: _SnapshotConnection) -> PGAgentSessionRepository:
    return PGAgentSessionRepository(
        pool=cast(Any, _SnapshotPool(connection)),
        owner_id="owner",
        run_id=uuid.uuid4(),
        worker_id="worker",
        lease_owner="worker",
        fencing_epoch=1,
    )


def _entry_row(entry: UserMessageEntry) -> dict[str, Any]:
    return {
        "sequence": entry.sequence,
        "entry_id": entry.entry_id.value,
        "parent_entry_id": (
            entry.parent_entry_id.value if entry.parent_entry_id is not None else None
        ),
        "entry_type": entry.entry_type,
        "schema_version": entry.schema_version,
        "timestamp": entry.timestamp,
        "payload_json": entry.canonical_payload(),
    }


def _user(session_id: SessionId, content: str, *, entry_id: EntryId | None = None):
    return UserMessageEntry(
        entry_id=entry_id or EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content=content,
    )


async def test_refresh_unchanged_is_metadata_only_and_returns_previous_identity() -> None:
    session_id = SessionId.new()
    root = replace(_user(session_id, "root"), sequence=1)
    previous = AgentSessionSnapshot(
        session_id=session_id,
        commit_sequence=1,
        last_entry_sequence=1,
        entries=(root,),
    )
    connection = _SnapshotConnection(metadata={"commit_sequence": 1, "last_sequence": 1})

    refreshed = await _snapshot_repository(connection).refresh(session_id, previous=previous)

    assert refreshed is previous
    assert [
        (method, "SELECT commit_sequence" in query) for method, query, _ in connection.calls
    ] == [("fetchrow", True)]
    assert connection.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]


async def test_refresh_fetches_and_decodes_only_bounded_suffix_and_replaces_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = SessionId.new()
    root = replace(_user(session_id, "root"), sequence=1)
    second = replace(
        _user(session_id, "second"),
        sequence=2,
        parent_entry_id=root.entry_id,
    )
    third = replace(
        _user(session_id, "third"),
        sequence=3,
        parent_entry_id=second.entry_id,
    )
    previous = AgentSessionSnapshot(
        session_id=session_id,
        commit_sequence=1,
        last_entry_sequence=1,
        entries=(root,),
        registers=(RegisterRecord(LaneHead(LaneId.main(), root.entry_id), 1),),
    )
    connection = _SnapshotConnection(
        metadata={"commit_sequence": 3, "last_sequence": 3},
        entry_rows=[_entry_row(second), _entry_row(third)],
        register_rows=[],
    )
    decoded_sequences: list[int] = []
    original_decode = session_repository_module._decode_entry  # pyright: ignore[reportPrivateUsage]

    def count_decode(row: Any, *, session_id: SessionId):
        decoded_sequences.append(int(row["sequence"]))
        return original_decode(row, session_id=session_id)

    monkeypatch.setattr(session_repository_module, "_decode_entry", count_decode)

    refreshed = await _snapshot_repository(connection).refresh(session_id, previous=previous)

    assert [entry.sequence for entry in refreshed.entries] == [1, 2, 3]
    assert refreshed.entries[0] is root
    assert refreshed.registers == ()
    assert decoded_sequences == [2, 3]
    delta_call = next(call for call in connection.calls if call[0] == "fetch")
    assert "sequence > $3 AND sequence <= $4" in delta_call[1]
    assert delta_call[2][2:] == (1, 3)
    assert connection.transaction_kwargs == [{"isolation": "repeatable_read", "readonly": True}]


@pytest.mark.parametrize(
    ("metadata", "rows", "match"),
    [
        ({"commit_sequence": 3, "last_sequence": 3}, [], "not gap-free"),
        ({"commit_sequence": 3, "last_sequence": 3}, [{"sequence": 3}], "not gap-free"),
        ({"commit_sequence": 0, "last_sequence": 0}, [], "regressed"),
    ],
)
async def test_refresh_rejects_missing_gapped_or_regressed_rows(
    metadata: dict[str, int],
    rows: list[dict[str, Any]],
    match: str,
) -> None:
    session_id = SessionId.new()
    root = replace(_user(session_id, "root"), sequence=1)
    previous = AgentSessionSnapshot(
        session_id=session_id,
        commit_sequence=1,
        last_entry_sequence=1,
        entries=(root,),
    )
    connection = _SnapshotConnection(metadata=metadata, entry_rows=rows)

    with pytest.raises(ValueError, match=match):
        await _snapshot_repository(connection).refresh(session_id, previous=previous)

    assert not any(
        method == "fetch" and "dlightrag_agent_session_registers" in query
        for method, query, _ in connection.calls
    )


def _expectation_transaction(count: int) -> SessionTransaction[Any]:
    mutation = RegisterRef("pending_input", "mutation")
    expectations = [RegisterExpectation(mutation, None)]
    expectations.extend(
        RegisterExpectation(RegisterRef("request_snapshot", str(uuid.uuid4())), None)
        for _ in range(count - 1)
    )
    return SessionTransaction.from_parts(
        register_writes=[DeleteRegister(mutation)],
        expectations=expectations,
    )


@pytest.mark.parametrize("count", [1, 50])
async def test_expectations_use_one_set_query_regardless_of_count(count: int) -> None:
    connection = _RecordingConnection()
    transaction = _expectation_transaction(count)

    conflict = await _repository()._check_register_expectations(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), SessionId.new(), transaction
    )

    assert conflict is None
    assert len(connection.fetches) == 1
    query, args = connection.fetches[0]
    assert "unnest($3::text[], $4::text[]) WITH ORDINALITY" in query
    assert "FOR UPDATE" not in query
    assert len(args[2]) == len(args[3]) == count


async def test_expectation_conflict_is_first_in_transaction_order() -> None:
    first = RegisterRef("request_snapshot", "first")
    second = RegisterRef("tool_arguments", "second")
    transaction = SessionTransaction.from_parts(
        register_writes=[DeleteRegister(RegisterRef("pending_input", "mutation"))],
        expectations=[
            RegisterExpectation(first, 3),
            RegisterExpectation(second, 5),
            RegisterExpectation(RegisterRef("pending_input", "mutation"), None),
        ],
    )
    connection = _RecordingConnection(
        register_sequences={(first.kind, first.key): 4, (second.kind, second.key): 6}
    )

    conflict = await _repository()._check_register_expectations(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), SessionId.new(), transaction
    )

    assert isinstance(conflict, RegisterConflict)
    assert conflict.ref == first
    assert conflict.current_sequence == 4
    assert len(connection.fetches) == 1


async def test_entry_validation_probes_only_incoming_and_external_parent_ids() -> None:
    session_id = SessionId.new()
    external_parent = EntryId.new()
    first = replace(_user(session_id, "first"), parent_entry_id=external_parent)
    second = replace(_user(session_id, "second"), parent_entry_id=first.entry_id)
    connection = _RecordingConnection(entry_ids={external_parent.value})

    await _repository()._validate_transaction_entries(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), session_id, [first, second], last_sequence=1000
    )

    assert len(connection.fetches) == 1
    query, args = connection.fetches[0]
    assert "entry_id = ANY($3::uuid[])" in query
    assert "parent_entry_id" not in query
    assert {str(value) for value in args[2]} == {
        external_parent.value,
        first.entry_id.value,
        second.entry_id.value,
    }


async def test_entry_validation_rejects_local_duplicate_before_sql() -> None:
    session_id = SessionId.new()
    identity = EntryId.new()
    first = _user(session_id, "first", entry_id=identity)
    duplicate = replace(_user(session_id, "duplicate", entry_id=identity), parent_entry_id=identity)
    connection = _RecordingConnection()

    with pytest.raises(ValueError, match="identity already exists"):
        await _repository()._validate_transaction_entries(  # pyright: ignore[reportPrivateUsage]
            cast(Any, connection), session_id, [first, duplicate], last_sequence=0
        )

    assert connection.fetches == []


async def test_register_validation_reads_only_affected_lane_pairs() -> None:
    session_id = SessionId.new()
    branch = LaneId.new()
    main = LaneId.main().value
    rows = [
        {"register_kind": "lane_head", "register_key": main, "payload_json": None},
        {"register_kind": "lane_state", "register_key": main, "payload_json": None},
    ]
    connection = _RecordingConnection(lane_rows=rows)
    branch_head = LaneHead(branch, None)
    branch_state = LaneState(branch)
    transaction = SessionTransaction.from_parts(
        register_writes=[SetRegister(branch_head), SetRegister(branch_state)],
        expectations=[
            RegisterExpectation(branch_head.ref, None),
            RegisterExpectation(branch_state.ref, None),
        ],
    )

    await _repository()._validate_transaction_registers(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), session_id, transaction
    )

    assert len(connection.fetches) == 1
    query, args = connection.fetches[0]
    assert "register_kind IN ('lane_head', 'lane_state')" in query
    assert "register_key = ANY($3::text[])" in query
    assert set(args[2]) == {main, branch.value}
    assert args[3] == []


@pytest.mark.parametrize(
    ("lane_rows", "writes", "expectations"),
    [
        (
            [
                {
                    "register_kind": "lane_head",
                    "register_key": LaneId.main().value,
                    "payload_json": None,
                }
            ],
            [DeleteRegister(RegisterRef("pending_input", "unrelated"))],
            [RegisterExpectation(RegisterRef("pending_input", "unrelated"), None)],
        ),
        (
            [
                {
                    "register_kind": "lane_head",
                    "register_key": LaneId.main().value,
                    "payload_json": None,
                },
                {
                    "register_kind": "lane_state",
                    "register_key": LaneId.main().value,
                    "payload_json": None,
                },
            ],
            [SetRegister(LaneHead(LaneId("11111111-1111-7111-8111-111111111111"), None))],
            [
                RegisterExpectation(
                    LaneHead(LaneId("11111111-1111-7111-8111-111111111111"), None).ref,
                    None,
                )
            ],
        ),
    ],
    ids=["one-sided-main", "one-sided-touched-branch"],
)
async def test_register_validation_rejects_one_sided_lane_pairs(
    lane_rows: list[dict[str, Any]],
    writes: list[Any],
    expectations: list[RegisterExpectation],
) -> None:
    connection = _RecordingConnection(lane_rows=lane_rows)
    transaction = SessionTransaction.from_parts(
        register_writes=writes,
        expectations=expectations,
    )

    with pytest.raises(ValueError, match="complete main and Lane pairs"):
        await _repository()._validate_transaction_registers(  # pyright: ignore[reportPrivateUsage]
            cast(Any, connection), SessionId.new(), transaction
        )


@pytest.mark.parametrize("count", [1, 1000])
async def test_agent_evidence_write_uses_two_statements_regardless_of_count(count: int) -> None:
    repository = _repository()
    connection = _RecordingConnection()
    content = b"evidence"
    locator = b"locator"
    session_id = SessionId.new()
    intent_id = IntentId.new()
    evidence = tuple(
        OpaqueEvidenceWrite(
            session_id=session_id.value,
            intent_id=intent_id.value,
            result_ordinal=ordinal,
            content_digest=hashlib.sha256(content).hexdigest(),
            locator_digest=hashlib.sha256(locator).hexdigest(),
            content=content,
            locator=locator,
        )
        for ordinal in range(count)
    )

    await repository._write_host_update(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection),
        session_id,
        intent_id,
        EffectHostUpdate(evidence=evidence),
    )

    assert len(connection.executes) == 1
    assert len(connection.fetchvals) == 1
    insert, insert_args = connection.executes[0]
    probe, probe_args = connection.fetchvals[0]
    assert "INSERT INTO dlightrag_answer_evidence" in insert
    assert "LEFT JOIN dlightrag_answer_evidence" in probe
    assert all(len(values) == count for values in insert_args[2:])
    assert all(len(values) == count for values in probe_args[2:])


async def test_entry_and_register_mutations_are_one_statement_per_nonempty_category() -> None:
    repository = _repository()
    session_id = SessionId.new()
    root = _user(session_id, "root")
    child = replace(_user(session_id, "child"), parent_entry_id=root.entry_id)
    branch = LaneId.new()
    sets = [
        SetRegister(LaneHead(LaneId.main(), child.entry_id)),
        SetRegister(LaneState(LaneId.main())),
    ]
    deletes = [
        DeleteRegister(RegisterRef("lane_head", branch.value)),
        DeleteRegister(RegisterRef("lane_state", branch.value)),
    ]
    connection = _RecordingConnection()

    await repository._insert_entries(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), session_id, [root, child], [41, 42]
    )
    await repository._write_registers(  # pyright: ignore[reportPrivateUsage]
        cast(Any, connection), session_id, [*sets, *deletes], sequence=9
    )

    assert len(connection.executes) == 3
    assert (
        sum(
            "INSERT INTO dlightrag_agent_session_entries" in query
            for query, _ in connection.executes
        )
        == 1
    )
    assert (
        sum(
            "INSERT INTO dlightrag_agent_session_registers" in query
            for query, _ in connection.executes
        )
        == 1
    )
    assert (
        sum(
            "DELETE FROM dlightrag_agent_session_registers" in query
            for query, _ in connection.executes
        )
        == 1
    )
    assert all("executemany" not in query.lower() for query, _ in connection.executes)
    entry_payload = json.loads(connection.executes[0][1][2])
    set_payload = json.loads(connection.executes[1][1][3])
    delete_args = connection.executes[2][1]
    assert [item["sequence"] for item in entry_payload] == [41, 42]
    assert [item["register_kind"] for item in set_payload] == ["lane_head", "lane_state"]
    assert delete_args[2] == ["lane_head", "lane_state"]
    assert delete_args[3] == [branch.value, branch.value]

    empty = _RecordingConnection()
    await repository._insert_entries(  # pyright: ignore[reportPrivateUsage]
        cast(Any, empty), session_id, [], []
    )
    await repository._write_registers(  # pyright: ignore[reportPrivateUsage]
        cast(Any, empty), session_id, [], sequence=10
    )
    assert empty.executes == []
