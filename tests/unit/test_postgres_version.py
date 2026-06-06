# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL and pgvector version validation tests."""

import pytest

from dlightrag.storage.postgres_version import (
    ensure_postgres_extensions,
    parse_pgvector_version,
    parse_server_version_num,
    validate_pgvector_halfvec,
    validate_postgres_major,
)


@pytest.mark.parametrize(
    ("server_version_num", "expected"),
    [
        ("180004", 18),
        ("180000", 18),
        ("170010", 17),
        (180004, 18),
    ],
)
def test_parse_server_version_num(server_version_num, expected) -> None:
    assert parse_server_version_num(server_version_num) == expected


def test_validate_postgres_major_accepts_pg18() -> None:
    validate_postgres_major("180004", required_major=18)


def test_validate_postgres_major_rejects_pg17() -> None:
    with pytest.raises(RuntimeError, match="PostgreSQL 18"):
        validate_postgres_major("170010", required_major=18)


def test_parse_pgvector_version() -> None:
    assert parse_pgvector_version("0.8.2") == (0, 8, 2)
    assert parse_pgvector_version("0.7.0") == (0, 7, 0)


def test_validate_pgvector_halfvec_accepts_modern_pgvector() -> None:
    validate_pgvector_halfvec("0.8.2")


def test_validate_pgvector_halfvec_rejects_old_pgvector() -> None:
    with pytest.raises(RuntimeError, match="halfvec"):
        validate_pgvector_halfvec("0.6.2")


class _FakeConn:
    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, statement: str) -> None:
        self.executed.append(statement)


async def test_ensure_postgres_extensions_runs_idempotent_create_extension() -> None:
    conn = _FakeConn()

    await ensure_postgres_extensions(conn, ("pg_textsearch", "pg_jieba"))

    assert conn.executed == [
        "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
        "CREATE EXTENSION IF NOT EXISTS pg_jieba",
    ]


async def test_ensure_postgres_extensions_rejects_unknown_names() -> None:
    conn = _FakeConn()

    with pytest.raises(ValueError, match="unsupported PostgreSQL extension"):
        await ensure_postgres_extensions(conn, ("pg_textsearch; DROP SCHEMA public",))
