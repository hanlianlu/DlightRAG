# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL server and pgvector capability validation."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BOOTSTRAPPABLE_EXTENSIONS = frozenset({"pg_textsearch", "pg_jieba", "pg_trgm"})
# Extensions whose shared library moves with the image while the SQL objects stay at the version
# CREATE EXTENSION IF NOT EXISTS installed: it never updates an extension that already exists.
_UPDATED_ON_STARTUP = frozenset({"pg_textsearch"})
_EXTENSION_VERSIONS_SQL = (
    "SELECT installed_version, default_version FROM pg_available_extensions WHERE name = $1"
)

# DlightRAG's PostgreSQL major is a hard runtime requirement, not a tunable:
# pgvector halfvec, pg_textsearch, and pg_jieba are all pinned to the PG18
# ecosystem. It is a named constant (not config) so it cannot be
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
        try:
            await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
        except Exception as exc:
            if extension in ("pg_textsearch", "pg_jieba"):
                # Both bootstrappable text-search extensions exist only to serve BM25.
                raise RuntimeError(
                    f"PostgreSQL extension {extension!r} is unavailable ({exc}). "
                    "It backs BM25 retrieval; set bm25_enabled: false to run vector-only."
                ) from exc
            raise RuntimeError(
                f"PostgreSQL extension {extension!r} is unavailable ({exc})."
            ) from exc
        if extension in _UPDATED_ON_STARTUP:
            await update_extension_if_behind(conn, extension)


def _release_numbers(version: str) -> tuple[int, ...] | None:
    """The dotted numbers of an extension version, or None for one that is not numeric."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


async def update_extension_if_behind(conn: Any, extension: str) -> None:
    """Run ``ALTER EXTENSION ... UPDATE`` when the server carries a newer release than the one installed.

    The server's own control file names the release it carries, so an image that moved
    pg_textsearch forward brings every database up to it at the writer's next start, under the
    startup lock. A database already at or above that release is left alone, and so is one
    whose extension is not installed. A refusal (a managed endpoint whose role does not own the
    extension) is logged, not raised: the installed SQL objects keep working under the newer
    library, and the operator runs the statement as the extension's owner
    (docs/operations.md, "Updating an installed extension").
    """
    if extension not in _BOOTSTRAPPABLE_EXTENSIONS:
        raise ValueError(f"unsupported PostgreSQL extension update: {extension!r}")
    row = await conn.fetchrow(_EXTENSION_VERSIONS_SQL, extension)
    if row is None or row["installed_version"] is None or row["default_version"] is None:
        return
    installed = _release_numbers(row["installed_version"])
    available = _release_numbers(row["default_version"])
    if installed is None or available is None or installed >= available:
        return
    try:
        await conn.execute(f"ALTER EXTENSION {extension} UPDATE")
    except Exception as exc:
        logger.warning(
            "PostgreSQL extension %s stays at %s while the server carries %s (%s); "
            "run ALTER EXTENSION %s UPDATE as its owner.",
            extension,
            row["installed_version"],
            row["default_version"],
            exc,
            extension,
        )
        return
    logger.info(
        "PostgreSQL extension %s updated from %s to %s",
        extension,
        row["installed_version"],
        row["default_version"],
    )
