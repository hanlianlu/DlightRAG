# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The PostgreSQL inventory write: a bounded statement count whatever the file count."""

import pytest

from dlightrag.engine.runtime.settlements import InventoryPathRecord


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> str:
        self.statements.append((query, args))
        return "OK"


def _file(path: str, size: int = 1) -> InventoryPathRecord:
    return InventoryPathRecord(relative_path=path, entry_type="file", size_bytes=size)


@pytest.mark.asyncio
async def test_pg_inventory_rescan_is_two_statements_whatever_the_file_count() -> None:
    """A rescan runs under the Run lock, so its write must not grow per file."""
    import uuid

    from dlightrag.adapters.postgres.answer.workspace import write_inventory

    conn = _RecordingConnection()
    records = tuple(_file(f"src/file_{index}.py", index) for index in range(2_000))

    await write_inventory(conn, "owner", uuid.uuid4(), upserts=records, replace_all=True)

    assert len(conn.statements) == 2
    (removal, removal_args), (upsert, upsert_args) = conn.statements
    assert "NOT EXISTS" in removal and "unnest($3::text[])" in removal
    assert removal_args[2] == [record.relative_path for record in records]
    assert "FROM unnest(" in upsert and "IS DISTINCT FROM" in upsert
    assert upsert_args[2] == [record.relative_path for record in records]


@pytest.mark.asyncio
async def test_pg_inventory_delta_deletes_then_keeps_the_last_observation() -> None:
    import uuid

    from dlightrag.adapters.postgres.answer.workspace import write_inventory

    conn = _RecordingConnection()

    await write_inventory(
        conn,
        "owner",
        uuid.uuid4(),
        upserts=(_file("a.md", 1), _file("a.md", 7), _file("b.md", 2)),
        deletes=("gone.md", "a.md"),
    )

    (removal, removal_args), (_upsert, upsert_args) = conn.statements
    assert "relative_path = ANY($3::text[])" in removal
    assert removal_args[2] == ["gone.md", "a.md"]
    assert upsert_args[2] == ["a.md", "b.md"]
    assert upsert_args[5] == [7, 2]
