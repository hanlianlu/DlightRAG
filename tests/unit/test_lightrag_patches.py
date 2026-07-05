# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG AGE monkey-patches."""

from unittest.mock import AsyncMock, MagicMock

import asyncpg.exceptions
import pytest
from lightrag.kg.postgres_impl import PostgreSQLDB

from dlightrag.core._lightrag_patches import required_patch_names


def test_current_lightrag_main_requires_both_age_patches() -> None:
    """Pinned LightRAG main still lacks the AGE guards DlightRAG needs."""
    assert required_patch_names(PostgreSQLDB) == ("configure_age", "execute")


class TestConfigureAgePatch:
    """Test patched configure_age with pre-check to avoid PG ERROR log spam."""

    async def test_skips_create_when_graph_exists(self):
        """Pre-check finds graph → create_graph is never called."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)  # graph exists
        conn.execute = AsyncMock()

        await PostgreSQLDB.configure_age(conn, "test_graph")
        # SET search_path called, but create_graph NOT called
        conn.execute.assert_called_once()  # only SET search_path
        conn.fetchval.assert_called_once()

    async def test_creates_graph_when_not_exists(self):
        """Pre-check finds no graph → create_graph is called."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=False)  # graph does not exist
        conn.execute = AsyncMock()

        await PostgreSQLDB.configure_age(conn, "new_graph")
        # SET search_path + create_graph
        assert conn.execute.call_count == 2
        create_call = conn.execute.call_args_list[1]
        assert "create_graph" in str(create_call)

    async def test_race_condition_caught(self):
        """Race: pre-check says not exists, but create_graph raises DuplicateSchemaError."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=False)  # race: check says no
        conn.execute = AsyncMock(
            side_effect=[
                None,  # SET search_path
                asyncpg.exceptions.DuplicateSchemaError("race condition"),
            ]
        )

        # Should NOT raise
        await PostgreSQLDB.configure_age(conn, "test_graph")

    async def test_other_exceptions_propagate(self):
        """Unrelated exceptions propagate."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=False)
        conn.execute = AsyncMock(
            side_effect=[
                None,  # SET search_path
                asyncpg.exceptions.UndefinedTableError("no such table"),
            ]
        )

        with pytest.raises(asyncpg.exceptions.UndefinedTableError):
            await PostgreSQLDB.configure_age(conn, "test_graph")


class TestExecutePatch:
    """Test that wrapped execute catches DuplicateSchemaError."""

    async def test_duplicate_schema_ignored_when_ignore_if_exists(self):
        """Wrapper catches DuplicateSchemaError when ignore_if_exists=True."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        db = MagicMock(spec=PostgreSQLDB)
        db._run_with_retry = AsyncMock(
            side_effect=asyncpg.exceptions.DuplicateSchemaError("exists")
        )

        # Should NOT raise
        await PostgreSQLDB.execute(
            db,
            "CREATE SOMETHING",
            ignore_if_exists=True,
        )

    async def test_duplicate_schema_ignored_when_upsert(self):
        """Wrapper catches DuplicateSchemaError when upsert=True."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        db = MagicMock(spec=PostgreSQLDB)
        db._run_with_retry = AsyncMock(
            side_effect=asyncpg.exceptions.DuplicateSchemaError("exists")
        )

        await PostgreSQLDB.execute(db, "CREATE SOMETHING", upsert=True)

    async def test_duplicate_schema_raises_when_not_ignored(self):
        """Wrapper re-raises DuplicateSchemaError when neither flag is set."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        db = MagicMock(spec=PostgreSQLDB)
        db._run_with_retry = AsyncMock(
            side_effect=asyncpg.exceptions.DuplicateSchemaError("exists")
        )

        with pytest.raises(asyncpg.exceptions.DuplicateSchemaError):
            await PostgreSQLDB.execute(db, "CREATE SOMETHING")

    async def test_forwards_unknown_kwargs(self):
        """Future kwargs should pass through to the original via **kwargs."""
        from dlightrag.core._lightrag_patches import apply

        apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        db = MagicMock(spec=PostgreSQLDB)
        db._run_with_retry = AsyncMock()  # succeeds

        await PostgreSQLDB.execute(
            db,
            "SELECT 1",
            with_age=True,
            graph_name="test_graph",
        )
        # Verify _run_with_retry was called (original was invoked, not short-circuited)
        db._run_with_retry.assert_called_once()


class TestNeedsPatch:
    """Test the source-inspection guard."""

    def test_configure_age_detects_missing_precheck(self, monkeypatch: pytest.MonkeyPatch):
        import dlightrag.core._lightrag_patches as mod

        def missing_precheck():
            pass

        monkeypatch.setattr(
            mod,
            "_source_contains",
            lambda _method, needle: needle == "DuplicateSchemaError",
        )
        assert mod._configure_age_needs_patch(missing_precheck) is True

    def test_configure_age_detects_missing_duplicate_schema_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import dlightrag.core._lightrag_patches as mod

        def missing_duplicate_schema():
            pass

        monkeypatch.setattr(
            mod,
            "_source_contains",
            lambda _method, needle: needle == "ag_catalog.ag_graph",
        )
        assert mod._configure_age_needs_patch(missing_duplicate_schema) is True

    def test_configure_age_detects_already_fixed_method(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import dlightrag.core._lightrag_patches as mod

        def already_fixed():
            pass

        monkeypatch.setattr(mod, "_source_contains", lambda _method, _needle: True)
        assert mod._configure_age_needs_patch(already_fixed) is False

    def test_execute_detects_missing_duplicate_schema_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import dlightrag.core._lightrag_patches as mod

        def missing_duplicate_schema():
            pass

        monkeypatch.setattr(mod, "_source_contains", lambda _method, _needle: False)
        assert mod._execute_needs_patch(missing_duplicate_schema) is True

    def test_execute_detects_already_fixed_method(self, monkeypatch: pytest.MonkeyPatch):
        import dlightrag.core._lightrag_patches as mod

        def already_fixed():
            pass

        monkeypatch.setattr(mod, "_source_contains", lambda _method, _needle: True)
        assert mod._execute_needs_patch(already_fixed) is False

    def test_uninspectable_returns_true(self):
        """If getsource fails, assume patch is needed (safe default)."""
        from dlightrag.core._lightrag_patches import (
            _configure_age_needs_patch,
            _execute_needs_patch,
        )

        # Built-in functions can't be inspected
        assert _configure_age_needs_patch(len) is True
        assert _execute_needs_patch(len) is True


class TestPatchIdempotency:
    """Test that patches can be applied multiple times safely."""

    async def test_apply_also_enables_mineru_parser_hygiene(self):
        """Patch entrypoint covers both AGE and MinerU parser hygiene."""
        from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

        import dlightrag.core._lightrag_patches as mod
        from dlightrag.core.ingestion.parser_hygiene import (
            mineru_ir_builder_needs_auxiliary_filter,
        )

        mod.apply()

        assert mineru_ir_builder_needs_auxiliary_filter(MinerUIRBuilder) is False

    async def test_apply_twice_no_double_patch(self):
        """Applying patches twice should not double-patch."""
        import dlightrag.core._lightrag_patches as mod

        mod.apply()
        mod.apply()

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)  # graph exists
        conn.execute = AsyncMock()
        await PostgreSQLDB.configure_age(conn, "g")
        # Should still work normally
        conn.execute.assert_called_once()
