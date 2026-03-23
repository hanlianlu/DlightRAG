# tests/unit/test_lightrag_patches.py
# Copyright 2025-2026 Hanlian Lu. All rights reserved.
"""Tests for LightRAG AGE monkey-patches."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncpg.exceptions
import pytest


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

    def test_detects_unpatched_method(self):
        from dlightrag.core._lightrag_patches import _needs_patch

        def missing_precheck():
            pass

        assert _needs_patch(missing_precheck) is True

    def test_detects_already_patched_method(self):
        from dlightrag.core._lightrag_patches import _needs_patch

        def already_fixed():
            # SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1)
            pass

        assert _needs_patch(already_fixed) is False

    def test_uninspectable_returns_true(self):
        """If getsource fails, assume patch is needed (safe default)."""
        from dlightrag.core._lightrag_patches import _needs_patch

        # Built-in functions can't be inspected
        assert _needs_patch(len) is True


class TestPatchIdempotency:
    """Test that patches can be applied multiple times safely."""

    async def test_apply_twice_no_double_patch(self):
        """Applying patches twice should not double-patch."""
        import dlightrag.core._lightrag_patches as mod

        mod._PATCHED = False
        mod.apply()
        mod.apply()  # second call returns early (_PATCHED=True)

        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)  # graph exists
        conn.execute = AsyncMock()
        await PostgreSQLDB.configure_age(conn, "g")
        # Should still work normally
        conn.execute.assert_called_once()

    async def test_idempotent_apply(self):
        """_PATCHED flag prevents re-application."""
        import dlightrag.core._lightrag_patches as mod

        mod._PATCHED = False
        mod.apply()

        mod._PATCHED = False
        mod.apply()

        # Patch replaces (not wraps) — just verify it still works
        from lightrag.kg.postgres_impl import PostgreSQLDB

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)
        conn.execute = AsyncMock()
        await PostgreSQLDB.configure_age(conn, "g")
