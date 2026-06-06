# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LightRAG patches still required by LightRAG 1.5.0.

PostgreSQL AGE bug: LightRAG's PostgreSQLDB.configure_age() calls create_graph() unconditionally,
which causes Apache AGE (and PG) to log an ERROR on every startup when the graph
already exists — even though the error is caught. This pollutes PG logs.
Additionally, Apache AGE raises DuplicateSchemaError (a SyntaxOrAccessError
subclass, *not* related to InvalidSchemaNameError), which the original code does
not catch, causing init to crash after the first graph-creation call succeeds.

Fix: Replace configure_age() with a pre-check version that queries
ag_catalog.ag_graph before calling create_graph(), skipping creation entirely
when the graph already exists.  Wrap execute() to also catch DuplicateSchemaError
as defence-in-depth.

As of LightRAG 1.5.0 both patched surfaces are still missing upstream.

MinerU parser hygiene: current LightRAG MinerU IR builder serializes unknown
content-list item types as body text. MinerU emits page furniture such as
headers, footers, and printed page numbers in content_list outputs; indexing
those as body text pollutes chunks, KG extraction, BM25, and citations.
DlightRAG patches the builder at the parser boundary when a runtime behavior
probe shows upstream still indexes those auxiliary blocks. More ambiguous page
notes are controlled by ``DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY``.

Keep this module small and delete patches as upstream covers them.

Patches are:
- Idempotent (safe to call multiple times)
- Guarded (source-inspected for AGE, behavior-probed for parser hygiene)
- Preserves PostgreSQLDB.execute keyword arguments via **kwargs
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, Literal

import asyncpg.exceptions

logger = logging.getLogger(__name__)

_PATCHED = False
PatchName = Literal["configure_age", "execute"]


def apply() -> None:
    """Apply all LightRAG patches. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    applied = []
    if _patch_configure_age():
        applied.append("configure_age")
    if _patch_execute():
        applied.append("execute")
    from dlightrag.core.ingestion.parser_hygiene import apply_mineru_auxiliary_block_filter

    if apply_mineru_auxiliary_block_filter():
        applied.append("mineru_auxiliary_blocks")
    _PATCHED = True
    if applied:
        logger.info("Applied LightRAG patches: %s", ", ".join(applied))
    else:
        logger.debug("LightRAG patches already covered upstream")


def required_patch_names(postgresql_db_cls: Any) -> tuple[PatchName, ...]:
    """Return the LightRAG AGE patches still required for a PostgreSQLDB class."""
    required: list[PatchName] = []
    if _configure_age_needs_patch(postgresql_db_cls.configure_age):
        required.append("configure_age")
    if _execute_needs_patch(postgresql_db_cls.execute):
        required.append("execute")
    return tuple(required)


def _source_contains(method: Callable[..., Any], needle: str) -> bool | None:
    """Return whether *needle* appears in method source, or None if unavailable."""
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return None
    return needle in source


def _configure_age_needs_patch(method: Callable[..., Any]) -> bool:
    """Return True if configure_age still lacks DlightRAG's required guards."""
    has_precheck = _source_contains(method, "ag_catalog.ag_graph")
    has_duplicate_schema = _source_contains(method, "DuplicateSchemaError")
    if has_precheck is None or has_duplicate_schema is None:
        return True
    return not (has_precheck and has_duplicate_schema)


def _execute_needs_patch(method: Callable[..., Any]) -> bool:
    """Return True if execute still fails to tolerate DuplicateSchemaError."""
    has_duplicate_schema = _source_contains(method, "DuplicateSchemaError")
    if has_duplicate_schema is None:
        return True
    return not has_duplicate_schema


def _patch_configure_age() -> bool:
    """Replace PostgreSQLDB.configure_age with a pre-check version.

    The original calls ``create_graph()`` unconditionally, causing PG to log
    ERROR for every "already exists" case.  Even though these are caught, they
    pollute PG logs on every startup.

    This patch checks ``ag_catalog.ag_graph`` first and skips creation if
    the graph already exists — no ERROR logged at all.
    """
    from lightrag.kg.postgres_impl import PostgreSQLDB

    original = PostgreSQLDB.configure_age

    if not _configure_age_needs_patch(original):
        logger.debug("configure_age already has pre-check and DuplicateSchemaError guard")
        return False

    @staticmethod  # type: ignore[misc]
    async def patched_configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        await connection.execute('SET search_path = ag_catalog, "$user", public')
        exists = await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1)",
            graph_name,
        )
        if not exists:
            try:
                await connection.execute("SELECT create_graph($1)", graph_name)
            except (
                asyncpg.exceptions.InvalidSchemaNameError,
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateSchemaError,
            ):
                pass  # race condition

    PostgreSQLDB.configure_age = patched_configure_age  # type: ignore[assignment]
    return True


def _patch_execute() -> bool:
    """Wrap PostgreSQLDB.execute to also catch DuplicateSchemaError.

    Defence-in-depth: the primary fix is configure_age(), but execute()'s
    inner _operation also misses DuplicateSchemaError in its catch tuple.
    Wrapping (vs replacing) preserves the original's full logic and forwards
    unknown upstream kwargs.
    """
    from lightrag.kg.postgres_impl import PostgreSQLDB

    original = PostgreSQLDB.execute

    if not _execute_needs_patch(original):
        logger.debug("execute already handles DuplicateSchemaError, skipping patch")
        return False

    async def wrapped_execute(  # type: ignore[no-untyped-def]
        self,
        sql,
        data=None,
        upsert=False,
        ignore_if_exists=False,
        **kwargs,
    ):
        try:
            return await original(
                self,
                sql,
                data,
                upsert=upsert,
                ignore_if_exists=ignore_if_exists,
                **kwargs,
            )
        except asyncpg.exceptions.DuplicateSchemaError as e:
            if ignore_if_exists:
                logger.debug("Ignoring DuplicateSchemaError (patched): %r", e)
                return None
            if upsert:
                logger.info("DuplicateSchemaError treated as upsert success (patched): %r", e)
                return None
            raise

    PostgreSQLDB.execute = wrapped_execute  # type: ignore[assignment]
    return True
