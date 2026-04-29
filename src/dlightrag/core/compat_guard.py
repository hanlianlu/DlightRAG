# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized startup validation for LightRAG coupling assumptions.

DlightRAG extends LightRAG through wrapping (FilteredVectorStorage), monkey
patching (_lightrag_patches), and direct DB access (PGJsonbKVStorage,
PGMetadataIndex, PGHashIndex). This guard validates all coupling assumptions
once at startup and fails fast with a complete error report if anything
has drifted between LightRAG releases.

Called once in RAGService init after LightRAG.initialize_storages() and
before chunks_vdb is wrapped by FilteredVectorStorage.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LightRAGCompatGuard:
    """Validates LightRAG internal API assumptions at startup.

    Collects all errors before raising — produces one report instead of
    failing on the first issue. PG-specific checks are skipped automatically
    when the active backend is not PostgreSQL.
    """

    _CHUNKS_VDB_COLUMNS = {"id", "content", "content_vector", "workspace", "file_path"}
    _BM25_TABLE = "lightrag_doc_chunks"
    _BM25_COLUMNS = {"id", "content", "file_path"}
    _CONFIGURE_AGE_PARAMS = ("connection", "graph_name")
    _EXECUTE_PARAMS = ("self", "sql", "data", "upsert", "ignore_if_exists")

    def __init__(self, lightrag: Any) -> None:
        self._lightrag = lightrag

    async def verify_all(self) -> None:
        """Run all checks, collect errors, raise if any."""
        errors: list[str] = []
        is_pg = self._is_pg_backend()
        if is_pg:
            await self._check_chunks_table_schema(errors)
            await self._check_bm25_table(errors)
        self._check_embedding_func_attr(errors)
        self._check_pool_access(errors, require_pg=is_pg)
        self._check_patch_signatures(errors)
        if errors:
            raise RuntimeError(
                f"LightRAG compatibility check failed "
                f"({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors)
            )
        logger.info(
            "LightRAG compatibility check passed (backend=%s)",
            "postgresql" if is_pg else "non-pg",
        )

    def _is_pg_backend(self) -> bool:
        """Detect whether chunks_vdb is backed by PostgreSQL with a live pool."""
        vdb = getattr(self._lightrag, "chunks_vdb", None)
        if vdb is None:
            return False
        db = getattr(vdb, "db", None)
        if db is None:
            return False
        return hasattr(db, "pool") and getattr(db, "pool", None) is not None

    async def _check_chunks_table_schema(self, errors: list[str]) -> None:
        """Check A: chunks_vdb table has all columns we depend on."""
        vdb = self._lightrag.chunks_vdb
        table_name = getattr(vdb, "table_name", None)
        if not table_name:
            errors.append("chunks_vdb missing 'table_name' attribute (PG path)")
            return
        pool = vdb.db.pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                table_name.lower(),
            )
        actual = {r["column_name"] for r in rows}
        missing = self._CHUNKS_VDB_COLUMNS - actual
        if missing:
            errors.append(f"chunks_vdb table '{table_name}' missing columns: {missing}")

    async def _check_bm25_table(self, errors: list[str]) -> None:
        """Check B: BM25 table exists with required columns."""
        pool = self._lightrag.chunks_vdb.db.pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                self._BM25_TABLE.lower(),
            )
        if not rows:
            errors.append(f"BM25 table '{self._BM25_TABLE}' does not exist")
            return
        actual = {r["column_name"] for r in rows}
        missing = self._BM25_COLUMNS - actual
        if missing:
            errors.append(f"BM25 table '{self._BM25_TABLE}' missing columns: {missing}")

    def _check_embedding_func_attr(self, errors: list[str]) -> None:
        """Check C: chunks_vdb.embedding_func exists and is writable.

        DlightRAG's FilteredVectorStorage wrapper substitutes embedding_func at
        runtime. If upstream makes it a read-only property, our wrap breaks.
        """
        vdb = self._lightrag.chunks_vdb
        if not hasattr(vdb, "embedding_func"):
            errors.append("chunks_vdb missing 'embedding_func' attribute")
            return
        for cls in type(vdb).__mro__:
            desc = cls.__dict__.get("embedding_func")
            if isinstance(desc, property) and desc.fset is None:
                errors.append("chunks_vdb.embedding_func is a read-only property")
                return

    def _check_pool_access(self, errors: list[str], require_pg: bool) -> None:
        """Check D: chunks_vdb.db.pool attribute chain reachable on PG backends."""
        if not require_pg:
            return
        vdb = self._lightrag.chunks_vdb
        if not hasattr(vdb, "db"):
            errors.append("chunks_vdb missing 'db' attribute (PG backend expected)")
            return
        if not hasattr(vdb.db, "pool"):
            errors.append("chunks_vdb.db missing 'pool' attribute (PG backend expected)")

    def _check_patch_signatures(self, errors: list[str]) -> None:
        """Check E: PostgreSQLDB method signatures match _lightrag_patches assumptions.

        Only runs when PG storage is in use; non-PG backends never load the patch
        target module so the import-then-inspect dance would be misleading.
        """
        if not self._is_pg_backend():
            return

        import inspect

        try:
            from lightrag.kg.postgres_impl import PostgreSQLDB
        except ImportError as e:
            errors.append(f"Cannot import PostgreSQLDB for signature check: {e}")
            return

        try:
            sig = inspect.signature(PostgreSQLDB.configure_age)
            params = tuple(sig.parameters.keys())
            if params != self._CONFIGURE_AGE_PARAMS:
                errors.append(
                    f"PostgreSQLDB.configure_age signature changed: "
                    f"expected {self._CONFIGURE_AGE_PARAMS}, got {params}"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Cannot inspect configure_age: {e}")

        try:
            sig = inspect.signature(PostgreSQLDB.execute)
            params = tuple(sig.parameters.keys())
            expected = self._EXECUTE_PARAMS
            if params[: len(expected)] != expected:
                errors.append(
                    f"PostgreSQLDB.execute signature changed: "
                    f"expected prefix {expected}, got {params}"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Cannot inspect execute: {e}")
