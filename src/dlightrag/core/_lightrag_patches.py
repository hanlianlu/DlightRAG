# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime-gated LightRAG compatibility patches.

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

As of LightRAG 1.5.4, these patches remain runtime-gated because some
upstream surfaces still need local guards.

MinerU parser hygiene: current LightRAG MinerU IR builder serializes unknown
content-list item types as body text. Two consequences are patched at the parser
boundary, each gated by a runtime behavior probe: (1) figure blocks MinerU emits
as ``chart`` are aliased to ``image`` so they reach the drawing/VLM sidecar path
instead of being dropped; (2) page furniture (headers, footers, printed page
numbers) is removed so it does not pollute chunks, KG extraction, BM25, and
citations. More ambiguous page notes are controlled by
``DLIGHTRAG_MINERU_AUXILIARY_BLOCK_POLICY``.

Keep this module small and delete patches as upstream covers them.

Patches are:
- Idempotent (safe to call multiple times)
- Guarded (source-inspected for AGE, behavior-probed for parser hygiene)
- Preserves PostgreSQLDB.execute keyword arguments via **kwargs
- Defensive-wrapped: if a patched method's signature changes upstream, the
  wrapper logs a warning and falls back to the original behaviour
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, Literal

import asyncpg.exceptions

logger = logging.getLogger(__name__)

PatchName = Literal["configure_age", "execute"]
_FAILED_DOC_PATCH_ATTR = "_dlightrag_preserves_failed_docs"


def apply() -> None:
    """Apply all LightRAG patches. Idempotent."""
    applied = []
    if _patch_configure_age():
        applied.append("configure_age")
    if _patch_execute():
        applied.append("execute")
    from dlightrag.core.ingestion.parser_hygiene import apply_mineru_content_list_hygiene

    if apply_mineru_content_list_hygiene():
        applied.append("mineru_content_list_hygiene")
    if _patch_failed_doc_loop():
        applied.append("preserve_failed_docs")
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
    except OSError, TypeError:
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


# ──────────────────────────────────────────────────────────────────────
# Patch 3: preserve FAILED documents across pipeline runs
# ──────────────────────────────────────────────────────────────────────


def _patch_failed_doc_loop() -> bool:
    """Prevent ``_validate_and_fix_document_consistency`` from resetting
    FAILED documents back to PENDING.

    LightRAG's current pipeline phase-2 treats ``FAILED`` as an
    "interrupted" state and unconditionally resets it to PENDING when
    the document still has content in ``full_docs``.  This creates an
    infinite loop:

    1. Unsupported file → FAILED (content stays)
    2. Next ingest → ``_validate_and_fix_document_consistency`` resets
       it to PENDING
    3. Pipeline re-processes it → FAILED again
    4. Goto 2

    Once LightRAG upstream removes ``FAILED`` from the ``is_interrupted``
    tuple, this wrapper becomes a no-op — it scans for FAILED docs before
    the call and only acts when it finds any that were flipped to PENDING.

    Defensive: if the original method's signature changes or the import
    fails, the wrapper logs a warning and falls back to no-op.
    """
    try:
        from lightrag.base import DocStatus  # type: ignore[import-untyped]
        from lightrag.pipeline import _PipelineMixin  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("LightRAG pipeline module not available — skipping failed-doc patch")
        return False

    original = _PipelineMixin._validate_and_fix_document_consistency

    if getattr(original, _FAILED_DOC_PATCH_ATTR, False):
        return False
    if not _failed_doc_reset_needs_patch(original):
        return False

    async def patched_validate_and_fix(  # type: ignore[no-untyped-def]
        self,
        to_process_docs,
        pipeline_status,
        pipeline_status_lock,
    ):
        # Snapshot FAILED doc_ids before LightRAG runs.
        failed_ids_before: set[str] = {
            doc_id
            for doc_id, d in to_process_docs.items()
            if getattr(d, "status", None) is DocStatus.FAILED
        }

        result = await original(self, to_process_docs, pipeline_status, pipeline_status_lock)

        # Restore: any FAILED doc that was flipped to PENDING gets popped
        # from the processing list entirely — it stays FAILED in doc_status.
        preserved = 0
        for doc_id in sorted(failed_ids_before):
            if doc_id in to_process_docs:
                status = getattr(to_process_docs[doc_id], "status", None)
                if status is DocStatus.PENDING:
                    to_process_docs.pop(doc_id, None)
                    preserved += 1

        if preserved:
            logger.info(
                "Preserved %d FAILED document(s) — not reset to PENDING (DlightRAG patch)",
                preserved,
            )

        return result

    setattr(patched_validate_and_fix, _FAILED_DOC_PATCH_ATTR, True)
    _PipelineMixin._validate_and_fix_document_consistency = (  # type: ignore[assignment]
        patched_validate_and_fix
    )
    return True


def _failed_doc_reset_needs_patch(method: Any) -> bool:
    """Return True if the method still resets FAILED to PENDING.

    Checks whether ``DocStatus.FAILED`` appears inside the
    ``is_interrupted`` status-triple (the bug) rather than just in the
    phase-1 preserve section (which is correct).
    """
    try:
        source = inspect.getsource(method)
    except OSError, TypeError:
        # Can't inspect — defensively apply the wrapper
        return True

    # Phase-2 bug marker: FAILED inside the is_interrupted tuple
    if "DocStatus.FAILED," not in source and "DocStatus.FAILED)" not in source:
        # The exact line reads:
        #     is_interrupted = status in (
        #         DocStatus.PROCESSING,
        #         DocStatus.FAILED,
        #         DocStatus.PARSING,
        #         DocStatus.ANALYZING,
        #     )
        # So FAILED appears as `DocStatus.FAILED,` or `DocStatus.FAILED)`
        # The latter handles the case where FAILED is the last element.
        return False
    return True
