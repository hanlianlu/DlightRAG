# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL Memory Undo restoration inherits the restored row's vector."""

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from dlightrag_memory import Memory, MemoryProvenance
from dlightrag_memory._storage import pg
from dlightrag_memory.ports import NullEmbedder, TextEmbedder

_OWNER = "alpha"
_ORIGINAL_ID = str(uuid.uuid4())
_CURRENT_ID = str(uuid.uuid4())
_PROVENANCE = {"origin_kind": "answer_run", "origin_id": "run-1", "run_id": "run-1"}


class _RecordingEmbedder:
    dim = 3

    def __init__(self) -> None:
        self.calls = 0

    @property
    def embedding_fingerprint(self) -> str:
        return "test:recording@local"

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _record_row(memory_id: str, body: str, status: str) -> dict[str, Any]:
    return {
        "owner_id": _OWNER,
        "memory_id": memory_id,
        "kind": "preference",
        "body": body,
        "origin_kind": "answer_run",
        "origin_id": "run-1",
        "run_id": "run-1",
        "session_id": None,
        "status": status,
        "supersedes_id": None,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


class _SupersedeUndoConnection:
    """Scripted settlement of an undo whose target superseded the original row."""

    def __init__(self) -> None:
        self.inserts: list[tuple[str, tuple[Any, ...]]] = []
        original = {
            "owner_id": _OWNER,
            "memory_id": _ORIGINAL_ID,
            "kind": "preference",
            "body": "Drinks tea.",
            "provenance": _PROVENANCE,
            "status": "active",
        }
        self._target = {
            "request_fingerprint": "unused",
            "undone_by": None,
            "receipt": {
                "action": "remember",
                "outcome": "changed",
                "change_id": str(uuid.uuid4()),
                "memory_ids": [_CURRENT_ID],
                "supersedes_id": _ORIGINAL_ID,
                "provenance": _PROVENANCE,
            },
            "before_records": [original],
        }

    async def fetchval(self, query: str, *args: Any) -> Any:
        return None

    async def fetchrow(self, query: str, *args: Any) -> Any:
        if query is pg._SELECT_OPERATION_FOR_UPDATE:
            return self._target
        if query is pg._SELECT_ONE_FOR_UPDATE:
            return _record_row(_CURRENT_ID, "Drinks coffee.", "active")
        return None  # no replay

    async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
        self.inserts.append((query, args))
        return [{"memory_id": item["memory_id"]} for item in json.loads(args[4])]

    async def execute(self, query: str, *args: Any) -> str:
        return "UPDATE 1"

    def transaction(self) -> _Transaction:
        return _Transaction()


class _Acquire:
    def __init__(self, conn: _SupersedeUndoConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _SupersedeUndoConnection:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: _SupersedeUndoConnection) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "sparse-only"])
async def test_supersede_undo_restores_from_the_original_row_without_embedding(
    dense: bool,
) -> None:
    conn = _SupersedeUndoConnection()
    embedder = _RecordingEmbedder()
    bound: TextEmbedder = embedder if dense else NullEmbedder()
    memory = Memory(pg.PostgresMemoryStore(pool=_Pool(conn), embedder=bound))

    receipt = await memory.undo(
        owner_id=_OWNER,
        change_id=str(uuid.uuid4()),
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )

    assert receipt.outcome == "changed"
    [(statement, args)] = conn.inserts
    assert statement is (
        pg._INSERT_RESTORED_BATCH_WITH_EMBEDDING if dense else pg._INSERT_RESTORED_BATCH
    )
    [restored] = json.loads(args[4])
    assert restored["memory_id"] == receipt.memory_id
    # The vector comes from the row being restored; the lineage points at the
    # row the restoration supersedes.
    assert restored["source_id"] == _ORIGINAL_ID
    assert restored["supersedes_id"] == _CURRENT_ID
    assert restored["body"] == "Drinks tea."
    assert embedder.calls == 0
