# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized startup validation for LightRAG coupling assumptions.

DlightRAG extends LightRAG through wrapping (FilteredVectorStorage), monkey
patching (_lightrag_patches), and direct DB access (PGMetadataIndex,
LightRAG text_chunks/chunks_vdb/doc_status). This guard validates all
coupling assumptions once at startup and fails fast with a complete error
report if anything has drifted between LightRAG releases.

Called once in RAGService init after LightRAG.initialize_storages() and
before chunks_vdb is wrapped by FilteredVectorStorage.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LightRAGContractGuard:
    """Validates LightRAG internal API assumptions at startup.

    Collects all errors before raising, producing one report instead of
    failing on the first issue. PostgreSQL is the only supported backend.
    """

    _CHUNKS_VDB_COLUMNS = {"id", "content", "content_vector", "workspace", "file_path"}
    _BM25_TABLE = "lightrag_doc_chunks"
    _BM25_COLUMNS = {"id", "content", "file_path"}
    _CONFIGURE_AGE_PARAMS = ("connection", "graph_name")
    _EXECUTE_PARAMS = ("self", "sql", "data", "upsert", "ignore_if_exists")
    _CLIENT_MANAGER_CONFIG_PARAMS = ("vector_storage",)
    _CLIENT_MANAGER_BUILD_SIGNATURE_PARAMS = ("config", "vector_storage")
    _CLIENT_MANAGER_ASSERT_SIGNATURE_PARAMS = ("requested_signature",)
    _NAMESPACE_TO_TABLE_NAME_PARAMS = ("namespace",)
    _READ_ONLY_STORAGE_ATTRS = (
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
        "llm_response_cache",
        "doc_status",
    )

    def __init__(self, lightrag: Any) -> None:
        self._lightrag = lightrag

    async def verify_all(self, *, include_read_only_attach_contract: bool = False) -> None:
        """Run all checks, collect errors, raise if any."""
        errors: list[str] = []
        self._require_pg_backend(errors)
        if not errors:
            await self._check_chunks_table_schema(errors)
            await self._check_bm25_table(errors)
            self._check_embedding_func_attr(errors)
            self._check_pool_access(errors)
            if include_read_only_attach_contract:
                self._check_read_only_attach_contract(errors)
        self._check_patch_signatures(errors)
        if errors:
            raise RuntimeError(
                f"LightRAG contract check failed "
                f"({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors)
            )
        logger.info("LightRAG contract check passed (backend=postgresql)")

    def verify_read_only_attach_contract(self) -> None:
        """Validate the private surfaces the read-only attach adapter relies on."""
        errors: list[str] = []
        self._check_read_only_attach_contract(errors)
        if errors:
            raise RuntimeError(
                f"LightRAG contract check failed "
                f"({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def _require_pg_backend(self, errors: list[str]) -> None:
        """Require chunks_vdb to expose PostgreSQL pool access."""
        vdb = getattr(self._lightrag, "chunks_vdb", None)
        if vdb is None:
            errors.append("chunks_vdb missing (PostgreSQL backend required)")
            return
        db = getattr(vdb, "db", None)
        if db is None:
            errors.append("chunks_vdb.db missing (PostgreSQL backend required)")
            return
        if not hasattr(db, "pool") or getattr(db, "pool", None) is None:
            errors.append("chunks_vdb.db.pool missing (PostgreSQL backend required)")

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
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = $1",
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
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = $1",
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

    def _check_pool_access(self, errors: list[str]) -> None:
        """Check D: chunks_vdb.db.pool attribute chain remains reachable."""
        vdb = self._lightrag.chunks_vdb
        if not hasattr(vdb, "db"):
            errors.append("chunks_vdb missing 'db' attribute (PG backend expected)")
            return
        if not hasattr(vdb.db, "pool"):
            errors.append("chunks_vdb.db missing 'pool' attribute (PG backend expected)")

    def _check_read_only_attach_contract(self, errors: list[str]) -> None:
        """Check E: reader attach adapter surfaces remain available."""
        import inspect

        def _matches_expected_prefix_with_optional_suffix(
            signature: inspect.Signature, expected: tuple[str, ...]
        ) -> bool:
            parameters = tuple(signature.parameters.values())
            param_names = tuple(parameter.name for parameter in parameters)
            if param_names[: len(expected)] != expected:
                return False
            for parameter in parameters[len(expected) :]:
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if parameter.default is inspect.Parameter.empty:
                    return False
            return True

        try:
            from lightrag.kg.postgres_impl import ClientManager, namespace_to_table_name
        except ImportError as e:
            errors.append(f"Cannot import reader attach surfaces: {e}")
            return

        for attr in self._READ_ONLY_STORAGE_ATTRS:
            if not hasattr(self._lightrag, attr):
                errors.append(f"LightRAG missing '{attr}' storage attribute for reader attach")

        graph_storage = getattr(self._lightrag, "chunk_entity_relation_graph", None)
        if graph_storage is not None and not callable(
            getattr(graph_storage, "_get_workspace_graph_name", None)
        ):
            errors.append(
                "chunk_entity_relation_graph missing callable '_get_workspace_graph_name'"
            )

        required_client_attrs = (
            "get_config",
            "_build_vector_signature",
            "_assert_compatible_vector_signature",
            "_lock",
            "_instances",
        )
        for attr in required_client_attrs:
            if not hasattr(ClientManager, attr):
                errors.append(f"ClientManager.{attr} missing for reader attach")

        instances = getattr(ClientManager, "_instances", None)
        if instances is not None:
            for key in ("db", "ref_count", "vector_signature"):
                if key not in instances:
                    errors.append(f"ClientManager._instances missing key '{key}'")

        signature_checks = (
            (
                "ClientManager.get_config",
                getattr(ClientManager, "get_config", None),
                self._CLIENT_MANAGER_CONFIG_PARAMS,
            ),
            (
                "ClientManager._build_vector_signature",
                getattr(ClientManager, "_build_vector_signature", None),
                self._CLIENT_MANAGER_BUILD_SIGNATURE_PARAMS,
            ),
            (
                "ClientManager._assert_compatible_vector_signature",
                getattr(ClientManager, "_assert_compatible_vector_signature", None),
                self._CLIENT_MANAGER_ASSERT_SIGNATURE_PARAMS,
            ),
            (
                "namespace_to_table_name",
                namespace_to_table_name,
                self._NAMESPACE_TO_TABLE_NAME_PARAMS,
            ),
        )
        for name, value, expected in signature_checks:
            if value is None or not callable(value):
                continue
            try:
                signature = inspect.signature(value)
                params = tuple(signature.parameters.keys())
            except (ValueError, TypeError) as e:
                errors.append(f"Cannot inspect {name}: {e}")
                continue
            if not _matches_expected_prefix_with_optional_suffix(signature, expected):
                errors.append(f"{name} signature changed: expected prefix {expected}, got {params}")

    def _check_patch_signatures(self, errors: list[str]) -> None:
        """Check F: PostgreSQLDB method signatures match _lightrag_patches assumptions.

        PostgreSQLDB is part of the supported storage contract.
        """
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
