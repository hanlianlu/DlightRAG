# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL and pgvector version validation tests."""

import logging

import pytest

from dlightrag.adapters.postgres.core._version import (
    ensure_postgres_extensions,
    parse_pgvector_version,
    parse_server_version_num,
    update_extension_if_behind,
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
    """A connection whose server has pg_textsearch installed at ``installed``, carrying ``available``."""

    def __init__(
        self,
        installed: str | None = "1.5.1",
        available: str | None = "1.5.1",
        refusal: Exception | None = None,
    ) -> None:
        self.executed: list[str] = []
        self._versions = {"installed_version": installed, "default_version": available}
        self._refusal = refusal

    async def execute(self, statement: str) -> None:
        if self._refusal is not None and statement.startswith("ALTER EXTENSION"):
            raise self._refusal
        self.executed.append(statement)

    async def fetchrow(self, query: str, name: str) -> dict[str, str | None] | None:
        assert "pg_available_extensions" in query and name == "pg_textsearch"
        return self._versions


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


async def test_ensure_postgres_extensions_names_the_way_out() -> None:
    """A managed endpoint reports the server's own error, which never mentions BM25."""

    class _RefusingConn:
        async def execute(self, statement: str) -> None:
            raise RuntimeError("pg_textsearch library not loaded.")

    with pytest.raises(RuntimeError, match="bm25_enabled: false") as excinfo:
        await ensure_postgres_extensions(_RefusingConn(), ("pg_textsearch",))

    assert "pg_textsearch library not loaded." in str(excinfo.value)


@pytest.mark.parametrize(
    ("installed", "available"),
    [("1.4.0", "1.5.1"), ("1.5.0", "1.5.1"), ("0.6.1", "1.0.0"), ("1.9.0", "1.10.0")],
)
async def test_an_installed_pg_textsearch_behind_the_server_is_updated(
    installed: str, available: str
) -> None:
    conn = _FakeConn(installed=installed, available=available)

    await ensure_postgres_extensions(conn, ("pg_textsearch",))

    assert conn.executed == [
        "CREATE EXTENSION IF NOT EXISTS pg_textsearch",
        "ALTER EXTENSION pg_textsearch UPDATE",
    ]


@pytest.mark.parametrize(
    ("installed", "available"),
    [
        ("1.5.1", "1.5.1"),  # current
        ("1.6.0", "1.5.1"),  # an older image started over a newer database: never downgraded
        (None, "1.5.1"),  # not installed
        ("1.5.1-dev", "1.5.1"),  # not a release number: left to the operator
        ("1.4.0", None),  # the server carries no control file for it
    ],
)
async def test_pg_textsearch_is_left_alone_unless_the_server_carries_a_newer_release(
    installed: str | None, available: str | None
) -> None:
    conn = _FakeConn(installed=installed, available=available)

    await ensure_postgres_extensions(conn, ("pg_textsearch",))

    assert conn.executed == ["CREATE EXTENSION IF NOT EXISTS pg_textsearch"]


async def test_an_extension_without_a_startup_update_is_only_created() -> None:
    conn = _FakeConn(installed="1.0", available="2.0")

    await ensure_postgres_extensions(conn, ("pg_jieba", "pg_trgm"))

    assert conn.executed == [
        "CREATE EXTENSION IF NOT EXISTS pg_jieba",
        "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    ]


async def test_a_refused_update_is_logged_and_the_start_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn = _FakeConn(installed="1.4.0", available="1.5.1", refusal=RuntimeError("must be owner"))

    with caplog.at_level(logging.WARNING):
        await ensure_postgres_extensions(conn, ("pg_textsearch",))

    assert conn.executed == ["CREATE EXTENSION IF NOT EXISTS pg_textsearch"]
    (record,) = caplog.records
    assert "stays at 1.4.0 while the server carries 1.5.1" in record.getMessage()
    assert "ALTER EXTENSION pg_textsearch UPDATE" in record.getMessage()


async def test_updating_names_only_a_known_extension() -> None:
    with pytest.raises(ValueError, match="unsupported PostgreSQL extension update"):
        await update_extension_if_behind(_FakeConn(), "pg_textsearch; DROP SCHEMA public")
