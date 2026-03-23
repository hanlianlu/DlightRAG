# src/dlightrag/core/_lightrag_patches.py
# Copyright 2025-2026 Hanlian Lu. All rights reserved.
"""Monkey-patches for LightRAG upstream bugs.

Bug: LightRAG's PostgreSQLDB.configure_age() only catches
InvalidSchemaNameError and UniqueViolationError when create_graph()
fails on an already-existing graph. But Apache AGE raises
DuplicateSchemaError (a SyntaxOrAccessError subclass, *not* related
to InvalidSchemaNameError). This causes init to crash after the first
graph-creation call succeeds — subsequent configure_age() calls in
_run_with_retry() hit the uncaught exception, aborting label/index
creation (create_vlabel, create_elabel) and leaving the graph in a
half-initialized state. Ingestion then fails with:
    UndefinedTableError: relation "<workspace>.base" does not exist

Fix: Wrap (not replace) configure_age() and execute() to also catch
asyncpg.exceptions.DuplicateSchemaError. The original methods are
called first; we only intercept the missing exception.

Patches are:
- Idempotent (safe to call multiple times)
- Wrap-based (delegate to original, only add missing exception handling)
- Source-inspected (skip automatically if upstream fixes the bug)
- Forward-compatible (unknown kwargs passed through via **kwargs)
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

import asyncpg.exceptions

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> None:
    """Apply all LightRAG patches. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    _patch_configure_age()
    _patch_execute()
    _PATCHED = True
    logger.info("Applied LightRAG AGE patches (DuplicateSchemaError)")


def _needs_patch(method: Callable[..., Any]) -> bool:
    """Return True if the method's source still lacks DuplicateSchemaError handling."""
    try:
        source = inspect.getsource(method)
        return "DuplicateSchemaError" not in source
    except (OSError, TypeError):
        return True


def _patch_configure_age() -> None:
    """Wrap PostgreSQLDB.configure_age to also catch DuplicateSchemaError."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    original = PostgreSQLDB.configure_age

    if not _needs_patch(original):
        logger.debug("configure_age already handles DuplicateSchemaError, skipping patch")
        return

    @staticmethod  # type: ignore[misc]
    async def wrapped_configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        try:
            await original(connection, graph_name)
        except asyncpg.exceptions.DuplicateSchemaError:
            pass

    PostgreSQLDB.configure_age = wrapped_configure_age  # type: ignore[assignment]


def _patch_execute() -> None:
    """Wrap PostgreSQLDB.execute to also catch DuplicateSchemaError.

    Defence-in-depth: the primary fix is configure_age(), but execute()'s
    inner _operation also misses DuplicateSchemaError in its catch tuple.
    Wrapping (vs replacing) preserves the original's full logic and forwards
    unknown kwargs for future-compatibility.
    """
    from lightrag.kg.postgres_impl import PostgreSQLDB

    original = PostgreSQLDB.execute

    if not _needs_patch(original):
        logger.debug("execute already handles DuplicateSchemaError, skipping patch")
        return

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
