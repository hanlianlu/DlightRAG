# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Partition naming and spec validation, which hold before any statement runs.

Partitioned parents, legacy shapes and missing tables run against PostgreSQL in
tests/integration/test_partition_foundation_pg.py.
"""

import re
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.partition_foundation import (
    PartitionedTableSpec,
    child_partition_name,
    default_child_name,
    verify_partitioned_tables,
)


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


class _NoStatements:
    """A connection that refuses every statement, so a refusal shows it ran none."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"no statement may run before the spec is validated: {name}")


async def test_spec_names_are_validated_before_any_sql() -> None:
    with pytest.raises(ValueError, match="Unsafe PostgreSQL identifier"):
        await verify_partitioned_tables(_NoStatements(), specs=(_spec(name="bad;name"),))
