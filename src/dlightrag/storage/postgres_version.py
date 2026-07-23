# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL server and pgvector capability validation."""

from typing import Any

_BOOTSTRAPPABLE_EXTENSIONS = frozenset({"pg_textsearch", "pg_jieba"})

# DlightRAG's PostgreSQL major is a hard runtime requirement, not a tunable:
# Apache AGE, pgvector halfvec, pg_textsearch, and pg_jieba are all pinned to
# the PG18 ecosystem. It is a named constant (not config) so it cannot be
# lowered into a cryptic runtime failure.
REQUIRED_POSTGRES_MAJOR = 18


def parse_server_version_num(value: str | int) -> int:
    """Return PostgreSQL major version from SHOW server_version_num."""
    number = int(value)
    return number // 10000


def validate_postgres_major(
    value: str | int, *, required_major: int = REQUIRED_POSTGRES_MAJOR
) -> None:
    """Raise if the connected server is below the required PostgreSQL major."""
    actual = parse_server_version_num(value)
    if actual < required_major:
        raise RuntimeError(
            f"DlightRAG requires PostgreSQL {required_major}.x or newer in the "
            f"{required_major}.x line; connected server reports major {actual}."
        )


def parse_pgvector_version(value: str) -> tuple[int, int, int]:
    """Return pgvector extension version as a comparable tuple."""
    major, minor, patch = (value.split(".") + ["0", "0"])[:3]
    return int(major), int(minor), int(patch)


def validate_pgvector_halfvec(value: str) -> None:
    """Raise if the installed pgvector version cannot support halfvec."""
    if parse_pgvector_version(value) < (0, 7, 0):
        raise RuntimeError(
            "POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC requires pgvector halfvec "
            "support from pgvector >= 0.7.0."
        )


async def ensure_postgres_major(
    conn: Any, *, required_major: int = REQUIRED_POSTGRES_MAJOR
) -> None:
    """Validate an asyncpg connection against the required PostgreSQL major."""
    value = await conn.fetchval("SHOW server_version_num")
    validate_postgres_major(value, required_major=required_major)


async def ensure_pgvector_halfvec(conn: Any) -> None:
    """Validate halfvec support before LightRAG creates vector stores."""
    value = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    if value is None:
        raise RuntimeError("pgvector extension is required for DlightRAG vector storage.")
    validate_pgvector_halfvec(str(value))


async def ensure_postgres_extensions(conn: Any, extensions: tuple[str, ...]) -> None:
    """Create DlightRAG-owned PostgreSQL extensions on writable startup paths."""
    for extension in extensions:
        if extension not in _BOOTSTRAPPABLE_EXTENSIONS:
            raise ValueError(f"unsupported PostgreSQL extension bootstrap: {extension!r}")
        await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
