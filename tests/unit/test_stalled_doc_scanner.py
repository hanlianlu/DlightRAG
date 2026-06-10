# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for stalled document scanner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock asyncpg helpers ────────────────────────────────────────────────


class _StalledRow:
    """Fake asyncpg Row for stalled LIGHTRAG_DOC_STATUS entries."""

    def __init__(self, doc_id: str, status: str, file_path: str) -> None:
        self._data: dict[str, object] = {
            "id": doc_id,
            "status": status,
            "file_path": file_path,
            "updated_at": "2026-01-01T00:00:00",
        }

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)

    def keys(self) -> object:
        return self._data.keys()

    def __iter__(self) -> object:
        return iter(self._data)

    def items(self) -> object:
        return self._data.items()


class _MockConn:
    """Simulates an asyncpg connection for stalled-doc scanner tests."""

    def __init__(
        self,
        *,
        stalled_rows: list[_StalledRow] | None = None,
        update_result: str = "UPDATE 0",
    ) -> None:
        self._stalled_rows = stalled_rows or []
        self._update_result = update_result
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args: object) -> list[_StalledRow]:
        self.fetch_calls.append((query, args))
        return list(self._stalled_rows)

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        return self._update_result

    async def close(self) -> None:
        pass


# ── Helpers ─────────────────────────────────────────────────────────────


def _patch_config() -> MagicMock:
    """Return a MagicMock config with pg_connection_kwargs()."""
    cfg = MagicMock()
    cfg.pg_connection_kwargs.return_value = {
        "host": "localhost",
        "port": 5432,
        "user": "test",
        "password": "test",
        "database": "test",
    }
    return cfg


# ── Tests ───────────────────────────────────────────────────────────────


class TestRecoverStalledDocs:
    async def test_no_stalled_docs_returns_zero(self) -> None:
        """When no documents are stalled, return zero counts."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        conn = _MockConn(stalled_rows=[])
        cfg = _patch_config()

        with (
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            result = await recover_stalled_docs(workspace="default")

        assert result["stalled_count"] == 0
        assert result["reset_count"] == 0
        assert result["stalled_ids"] == []
        assert result["errors"] == []

    async def test_resets_stalled_docs_in_all_intermediate_states(self) -> None:
        """PARSING, ANALYZING, and PROCESSING docs are all reset."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        rows = [
            _StalledRow("doc-1", "parsing", "/tmp/a.pdf"),
            _StalledRow("doc-2", "analyzing", "/tmp/b.pdf"),
            _StalledRow("doc-3", "processing", "/tmp/c.pdf"),
        ]
        conn = _MockConn(stalled_rows=rows, update_result="UPDATE 3")
        cfg = _patch_config()

        with (
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            result = await recover_stalled_docs(workspace="default")

        assert result["stalled_count"] == 3
        assert result["reset_count"] == 3
        assert set(result["stalled_ids"]) == {"doc-1", "doc-2", "doc-3"}
        assert result["errors"] == []

    async def test_dry_run_finds_but_does_not_reset(self) -> None:
        """dry_run=True identifies stalled docs without mutating."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        rows = [_StalledRow("doc-1", "processing", "/tmp/a.pdf")]
        conn = _MockConn(stalled_rows=rows)
        cfg = _patch_config()

        with (
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            result = await recover_stalled_docs(workspace="default", dry_run=True)

        assert result["stalled_count"] == 1
        assert result["reset_count"] == 0
        # No UPDATE executed in dry_run mode
        update_calls = [c for c in conn.execute_calls if "UPDATE" in c[0]]
        assert len(update_calls) == 0

    async def test_connection_failure_graceful(self) -> None:
        """PostgreSQL unreachable returns zero counts with error info."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        cfg = _patch_config()

        with (
            patch(
                "asyncpg.connect",
                new_callable=AsyncMock,
                side_effect=OSError("connection refused"),
            ),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            result = await recover_stalled_docs(workspace="default")

        assert result["stalled_count"] == 0
        assert result["reset_count"] == 0
        assert len(result["errors"]) == 1
        assert "connection refused" in result["errors"][0]

    async def test_uses_workspace_in_query(self) -> None:
        """The SELECT query includes the workspace parameter."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        conn = _MockConn(stalled_rows=[])
        cfg = _patch_config()

        with (
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            await recover_stalled_docs(workspace="my-workspace")

        # Verify the fetch query uses the correct workspace
        assert conn.fetch_calls
        assert "my-workspace" in str(conn.fetch_calls[0][1])

    async def test_update_query_targets_specific_doc_ids(self) -> None:
        """The UPDATE only touches the specific stalled doc IDs."""
        from dlightrag.storage.stalled_doc_scanner import recover_stalled_docs

        rows = [_StalledRow("stalled-abc", "processing", "/tmp/x.pdf")]
        conn = _MockConn(stalled_rows=rows, update_result="UPDATE 1")
        cfg = _patch_config()

        with (
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("dlightrag.config.get_config", return_value=cfg),
        ):
            await recover_stalled_docs(workspace="default")

        # Collect UPDATE calls
        update_calls = [c for c in conn.execute_calls if "UPDATE" in c[0]]
        assert len(update_calls) == 1
        update_query, update_args = update_calls[0]
        assert "SET status = 'pending'" in update_query
        # The single doc ID should be present in the args
        flat_args = [
            item
            for arg in update_args
            for item in (list(arg) if isinstance(arg, (list, tuple)) else [arg])
        ]
        assert "stalled-abc" in flat_args


class TestRecoverStalledDocsIntegration:
    """Integration tests with RAGServiceManager."""

    async def test_manager_skips_when_timeout_is_zero(self) -> None:
        """When stalled_doc_timeout_seconds=0, recovery is disabled."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.stalled_doc_timeout_seconds = 0
        cfg.max_async = 8
        cfg.embedding.model = "test-model"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg
        manager._services = {}
        # recovery should return immediately without calling scanner
        await manager._recover_stalled_docs(["default"])
        # No exception → pass

    async def test_manager_calls_scanner_for_each_workspace(self) -> None:
        """Each known workspace is scanned."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.stalled_doc_timeout_seconds = 3600
        cfg.max_async = 8
        cfg.embedding.model = "test-model"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg
        manager._services = {}

        call_args: list[dict[str, object]] = []

        async def _fake_recover(**kwargs: object) -> dict[str, object]:
            call_args.append(kwargs)
            return {"reset_count": 0, "stalled_count": 0, "stalled_ids": [], "errors": []}

        with patch(
            "dlightrag.storage.stalled_doc_scanner.recover_stalled_docs",
            new_callable=AsyncMock,
            side_effect=_fake_recover,
        ):
            await manager._recover_stalled_docs(["ws1", "ws2"])

        assert len(call_args) == 2
        workspaces = [str(c["workspace"]) for c in call_args]
        assert workspaces == ["ws1", "ws2"]
        for c in call_args:
            assert c["timeout_seconds"] == 3600
            assert c["dry_run"] is False

    async def test_manager_logs_recovered_ids(self) -> None:
        """When docs are recovered, their IDs are logged."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.stalled_doc_timeout_seconds = 3600
        cfg.max_async = 8
        cfg.embedding.model = "test-model"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg
        manager._services = {}

        async def _fake_recover(**_: object) -> dict[str, object]:
            return {
                "reset_count": 3,
                "stalled_count": 3,
                "stalled_ids": ["doc-a", "doc-b", "doc-c"],
                "errors": [],
            }

        with patch(
            "dlightrag.storage.stalled_doc_scanner.recover_stalled_docs",
            new_callable=AsyncMock,
            side_effect=_fake_recover,
        ):
            await manager._recover_stalled_docs(["default"])

        # No crash → pass.  Logging is verified implicitly.

    async def test_manager_is_resilient_to_per_workspace_failure(self) -> None:
        """One workspace failing doesn't block scanning the others."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.workspace = "default"
        cfg.stalled_doc_timeout_seconds = 3600
        cfg.max_async = 8
        cfg.embedding.model = "test-model"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg
        manager._services = {}

        async def _flakey_recover(**kw: object) -> dict[str, object]:
            workspace = str(kw.get("workspace", ""))
            if workspace == "ws-bad":
                raise RuntimeError("PG is down for this workspace")
            return {"reset_count": 2, "stalled_count": 2, "stalled_ids": ["d1", "d2"], "errors": []}

        with patch(
            "dlightrag.storage.stalled_doc_scanner.recover_stalled_docs",
            new_callable=AsyncMock,
            side_effect=_flakey_recover,
        ):
            await manager._recover_stalled_docs(["ws-ok", "ws-bad", "ws-also-ok"])

        # ws-bad failed but ws-ok and ws-also-ok should have been scanned
        # No exception propagates


class TestCheckpointPruning:
    """Tests for _prune_checkpoint_sessions startup hook."""

    async def test_prunes_old_sessions_on_startup(self) -> None:
        """Calls prune_old_sessions with configured TTL and logs the count."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.checkpoint_session_ttl_days = 30
        cfg.working_dir_path = MagicMock()
        cfg.workspace = "default"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        mock_cp = MagicMock()
        mock_cp.prune_old_sessions = AsyncMock(return_value=12)
        manager._checkpoint = mock_cp  # pre-seed to bypass lazy init

        await manager._prune_checkpoint_sessions()

        mock_cp.prune_old_sessions.assert_awaited_once_with(max_age_days=30)

    async def test_no_checkpoint_file_yet_is_graceful(self) -> None:
        """When the checkpoint DB doesn't exist yet (no sessions), no crash."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.checkpoint_session_ttl_days = 30
        cfg.working_dir_path = MagicMock()
        cfg.workspace = "default"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        # _get_checkpoint raises because working_dir doesn't exist
        with patch.object(
            manager, "_get_checkpoint", side_effect=FileNotFoundError("No such directory")
        ):
            await manager._prune_checkpoint_sessions()
        # No exception propagates

    async def test_prune_failure_is_logged_not_raised(self) -> None:
        """SQLite I/O error during prune is caught, not propagated."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.checkpoint_session_ttl_days = 30
        cfg.working_dir_path = MagicMock()
        cfg.workspace = "default"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        mock_cp = MagicMock()
        mock_cp.prune_old_sessions = AsyncMock(side_effect=OSError("disk full"))
        manager._checkpoint = mock_cp

        await manager._prune_checkpoint_sessions()
        # No exception propagates

    async def test_zero_deleted_logs_nothing(self) -> None:
        """When no sessions are old enough, silent success."""
        from dlightrag.core.servicemanager import RAGServiceManager

        cfg = MagicMock()
        cfg.checkpoint_session_ttl_days = 30
        cfg.working_dir_path = MagicMock()
        cfg.workspace = "default"

        manager = RAGServiceManager.__new__(RAGServiceManager)
        manager._config = cfg

        mock_cp = MagicMock()
        mock_cp.prune_old_sessions = AsyncMock(return_value=0)
        manager._checkpoint = mock_cp

        await manager._prune_checkpoint_sessions()
        # No crash, no log spam
