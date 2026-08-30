# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the workspace-partitioning foundation seam."""

import re
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.partition_foundation import (
    PartitionedTableSpec,
    child_partition_name,
    default_child_name,
    ensure_partitioned_tables,
    verify_partitioned_tables,
)
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError


class _Tx:
    async def __aenter__(self) -> _Tx:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def _spec(name: str = "lightrag_doc_chunks") -> PartitionedTableSpec:
    return PartitionedTableSpec(
        name=name,
        required_columns=("id", "workspace"),
        primary_key=("workspace", "id"),
        required_indexes=("idx_lightrag_doc_chunks_id",),
    )


class TestPartitionNaming:
    def test_default_child_name_is_deterministic_and_never_raw(self) -> None:
        name = default_child_name("lightrag_doc_chunks")
        assert name == default_child_name("lightrag_doc_chunks")
        assert re.fullmatch(r"p_[0-9a-f]{10}_w_default", name)
        assert len(name) <= 63

    def test_child_name_hides_the_workspace_identifier(self) -> None:
        name = child_partition_name("lightrag_doc_chunks", "malicious; DROP TABLE x")
        assert name == child_partition_name("lightrag_doc_chunks", "malicious; DROP TABLE x")
        assert "malicious" not in name
        assert re.fullmatch(r"p_[0-9a-f]{10}_w_[0-9a-f]{16}", name)

    def test_names_differ_per_parent_and_workspace(self) -> None:
        a = child_partition_name("lightrag_doc_chunks", "ws-a")
        b = child_partition_name("lightrag_doc_chunks", "ws-b")
        c = child_partition_name("lightrag_vdb_chunks_x", "ws-a")
        assert len({a, b, c}) == 3

    def test_unsafe_parent_names_are_rejected_before_sql(self) -> None:
        with pytest.raises(ValueError):
            child_partition_name("bad name; drop", "ws")


class _FakeConn:
    """Catalog fake: relkind answers and statement recording."""

    def __init__(self, *, relkind: str | None, empty: bool = True) -> None:
        self._relkind = relkind
        self._empty = empty
        self.executed: list[str] = []

    def transaction(self) -> _Tx:
        return _Tx()

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "c.relkind::text FROM pg_catalog.pg_class" in query:
            if args and str(args[0]).startswith("t_"):
                return None
            return self._relkind
        if "NOT EXISTS" in query:
            return self._empty
        return None

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append(query)


async def test_writer_fails_loudly_on_a_populated_unpartitioned_table() -> None:
    conn = _FakeConn(relkind="r", empty=False)

    with pytest.raises(CorpusSchemaError) as excinfo:
        await ensure_partitioned_tables(conn, specs=(_spec(),))

    message = str(excinfo.value)
    assert "lightrag_doc_chunks" in message
    assert "reset_development.py" in message
    assert "never rebuilt destructively" in message
    assert "LOCK TABLE lightrag_doc_chunks IN ACCESS EXCLUSIVE MODE" in conn.executed
    # The table lock serialized the emptiness decision, but nothing was
    # renamed, dropped, or rebuilt.
    assert not any(
        statement.startswith(("CREATE", "ALTER", "DROP", "RENAME")) for statement in conn.executed
    )


async def test_writer_skips_a_missing_ok_table_so_migrations_can_create_it() -> None:
    spec = PartitionedTableSpec(name="dlightrag_doc_metadata", missing_ok=True)
    conn = _FakeConn(relkind=None)

    await ensure_partitioned_tables(conn, specs=(spec,))

    # Only the advisory lock ran: the table itself is left to its migration scope.
    assert not any(
        statement.startswith(("CREATE", "ALTER", "DROP", "RENAME")) for statement in conn.executed
    )


async def test_writer_rejects_an_empty_plain_dlightrag_owned_legacy_table() -> None:
    spec = PartitionedTableSpec(
        name="dlightrag_doc_metadata",
        missing_ok=True,
        convert_empty_plain=False,
    )
    conn = _FakeConn(relkind="r", empty=True)

    with pytest.raises(CorpusSchemaError, match="fresh-schema release"):
        await ensure_partitioned_tables(conn, specs=(spec,))

    assert not any(statement.startswith(("LOCK", "ALTER", "DROP")) for statement in conn.executed)


async def test_writer_rejects_a_missing_lightrag_owned_table() -> None:
    conn = _FakeConn(relkind=None)

    with pytest.raises(CorpusSchemaError, match="missing after storage"):
        await ensure_partitioned_tables(conn, specs=(_spec(),))


async def test_reader_rejects_a_plain_table() -> None:
    conn = _FakeConn(relkind="r")

    with pytest.raises(CorpusSchemaError) as excinfo:
        await verify_partitioned_tables(conn, specs=(_spec(),))

    message = str(excinfo.value)
    assert "not partitioned by workspace" in message
    assert "reset_development.py" in message


async def test_reader_rejects_a_missing_table() -> None:
    conn = _FakeConn(relkind=None)

    with pytest.raises(CorpusSchemaError, match="is missing"):
        await verify_partitioned_tables(conn, specs=(_spec(),))


async def test_spec_names_are_validated_before_any_sql() -> None:
    conn = _FakeConn(relkind="p")

    with pytest.raises(ValueError, match="Unsafe PostgreSQL identifier"):
        await verify_partitioned_tables(conn, specs=(_spec(name="bad;name"),))

    assert conn.executed == []
