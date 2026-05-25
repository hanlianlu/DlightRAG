# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.config import (
    AnswerConfig,
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelConfig,
    set_config,
)
from dlightrag.core.query_planner import QueryPlanner
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError


@pytest.fixture()
def test_cfg(tmp_path) -> DlightragConfig:
    cfg = DlightragConfig(
        working_dir=str(tmp_path / "dlightrag_storage"),
        llm=LLMConfig(default=ModelConfig(model="gpt-4.1-mini", api_key="test")),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="test",
            startup_probe=False,
        ),
    )
    set_config(cfg)
    return cfg


class TestGetService:
    """Test workspace-keyed RAGService creation and caching."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_creates_service_for_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc = await manager._get_service("project-a")
        assert svc is mock_create.return_value
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == "project_a"  # normalized

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_caches_per_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws_1")
        svc2 = await manager._get_service("ws_1")
        assert svc1 is svc2
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_different_workspaces_different_services(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = [AsyncMock(), AsyncMock()]
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws_a")
        svc2 = await manager._get_service("ws_b")
        assert svc1 is not svc2
        assert mock_create.await_count == 2

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_concurrent_creates_once(self, mock_create, test_cfg) -> None:
        mock_service = AsyncMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return mock_service

        mock_create.side_effect = slow_create
        manager = RAGServiceManager(config=test_cfg)
        results = await asyncio.gather(
            manager._get_service("ws-x"),
            manager._get_service("ws-x"),
            manager._get_service("ws-x"),
        )
        assert mock_create.await_count == 1
        assert all(r is mock_service for r in results)


class TestWorkspaceCreation:
    """Test workspace creation registers discoverable workspace metadata."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_create_workspace_registers_workspace_meta(self, mock_create, test_cfg) -> None:
        svc = AsyncMock()
        mock_create.return_value = svc
        manager = RAGServiceManager(config=test_cfg)
        manager._wait_after_write = AsyncMock(return_value=None)

        await manager.acreate_workspace("new workspace")

        svc.aregister_workspace.assert_awaited_once()
        manager._wait_after_write.assert_awaited_once()


class TestBackoff:
    """Test exponential backoff on service creation failure."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_failure_sets_error_state(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        assert not manager.is_ready()
        error_info = manager.get_error_info()
        assert "ws_a" in error_info["backoff_workspaces"]

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_backoff_blocks_retry(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_retry_succeeds_after_backoff(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        # Expire the backoff by backdating the timestamp
        ts, interval = manager._backoff["ws_a"]
        manager._backoff["ws_a"] = (ts - interval - 1, interval)
        mock_create.side_effect = None
        mock_create.return_value = AsyncMock()
        svc = await manager._get_service("ws_a")
        assert svc is mock_create.return_value

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_success_resets_error_state(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        # Expire the backoff by backdating the timestamp
        ts, interval = manager._backoff["ws_a"]
        manager._backoff["ws_a"] = (ts - interval - 1, interval)
        mock_create.side_effect = None
        mock_create.return_value = AsyncMock()
        await manager._get_service("ws_a")
        assert "ws_a" not in manager._backoff

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_per_workspace_backoff_isolation(self, mock_create, test_cfg) -> None:
        """Workspace A in backoff does not block workspace B."""

        async def fail_only_a(**kwargs):
            if kwargs["config"].workspace == "ws_a":
                raise RuntimeError("ws_a is down")
            return AsyncMock()

        mock_create.side_effect = fail_only_a
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        # ws_a is now in backoff; ws_b should still succeed
        svc_b = await manager._get_service("ws_b")
        assert svc_b is not None
        assert "ws_a" in manager._backoff
        assert "ws_b" not in manager._backoff

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_backoff_clears_on_success(self, mock_create, test_cfg) -> None:
        """Backoff entry for a workspace is removed after a successful creation."""
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        assert "ws_a" in manager._backoff
        # Expire backoff and let next attempt succeed
        ts, interval = manager._backoff["ws_a"]
        manager._backoff["ws_a"] = (ts - interval - 1, interval)
        mock_create.side_effect = None
        mock_create.return_value = AsyncMock()
        await manager._get_service("ws_a")
        assert "ws_a" not in manager._backoff


class TestRouting:
    """Test single-workspace vs federated routing."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_single_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_multi_workspace_federates(
        self, mock_create, mock_fed, test_cfg
    ) -> None:
        mock_fed.return_value = MagicMock()
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspaces=["ws_a", "ws_b"])
        mock_fed.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_default_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == test_cfg.workspace

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_calls_aretrieve_then_engine(self, mock_create, test_cfg) -> None:
        """aanswer() routes through aretrieve() then AnswerEngine.generate()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts)
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        mock_engine.generate.return_value = MagicMock()

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        await manager.aanswer("query", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "query", mock_contexts, query_images=None, context_top_k=30
        )


class TestAnswerViaEngine:
    """aanswer and aanswer_stream route through AnswerEngine."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_calls_retrieve_then_engine(self, mock_create, test_cfg) -> None:
        """aanswer() calls aretrieve() then AnswerEngine.generate()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_retrieval = MagicMock(contexts=mock_contexts)
        mock_svc.aretrieve.return_value = mock_retrieval
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        expected_result = MagicMock()
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("what is X?", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "what is X?", mock_contexts, query_images=None, context_top_k=30
        )
        assert result is expected_result

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_uses_answer_candidate_and_context_limits(
        self, mock_create, test_cfg
    ) -> None:
        """Answer over-fetches retrieval candidates and caps final prompt contexts."""
        cfg = test_cfg.model_copy(
            update={"answer": AnswerConfig(candidate_top_k=7, context_top_k=3)}
        )
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        expected_result = MagicMock(trace={})
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", workspace="ws_a")

        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == 7
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            query_images=None,
            context_top_k=3,
        )
        assert result is expected_result

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_stream_calls_retrieve_then_engine(self, mock_create, test_cfg) -> None:
        """aanswer_stream() calls aretrieve() then AnswerEngine.generate_stream()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_retrieval = MagicMock(contexts=mock_contexts)
        mock_svc.aretrieve.return_value = mock_retrieval
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        mock_stream = AsyncMock()
        mock_engine.generate_stream.return_value = (mock_contexts, mock_stream)

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        contexts, stream = await manager.aanswer_stream("what is X?", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "what is X?", mock_contexts, query_images=None, context_top_k=30
        )
        assert contexts is mock_contexts
        assert stream is mock_stream

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_multi_workspace_uses_federated_retrieve(
        self, mock_create, mock_fed_retrieve, test_cfg
    ) -> None:
        """aanswer() with multiple workspaces federates retrieval, then uses engine."""
        mock_contexts = {"chunks": [{"text": "ctx"}], "entities": [], "relationships": []}
        mock_fed_retrieve.return_value = MagicMock(contexts=mock_contexts)

        mock_engine = AsyncMock()
        expected_result = MagicMock()
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", workspaces=["ws_a", "ws_b"])
        mock_fed_retrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "query", mock_contexts, query_images=None, context_top_k=30
        )
        assert result is expected_result

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aanswer_stream_multi_workspace(
        self, mock_create, mock_fed_retrieve, test_cfg
    ) -> None:
        """aanswer_stream() with multiple workspaces federates retrieval."""
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_fed_retrieve.return_value = MagicMock(contexts=mock_contexts)

        mock_engine = AsyncMock()
        mock_stream = AsyncMock()
        mock_engine.generate_stream.return_value = (mock_contexts, mock_stream)

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        contexts, stream = await manager.aanswer_stream("query", workspaces=["ws_a", "ws_b"])
        mock_fed_retrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "query", mock_contexts, query_images=None, context_top_k=30
        )
        assert contexts is mock_contexts

    def test_get_answer_engine_lazy_creates(self, test_cfg) -> None:
        """_get_answer_engine() lazily creates an AnswerEngine instance."""
        manager = RAGServiceManager(config=test_cfg)
        assert manager._answer_engine is None
        with patch("dlightrag.models.llm.get_query_model_func") as mock_llm:
            mock_llm.return_value = MagicMock()
            engine = manager._get_answer_engine()
            assert engine is not None
            # Second call returns same instance
            engine2 = manager._get_answer_engine()
            assert engine2 is engine


class TestDelegation:
    """Test write-operation delegation."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aingest_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"status": "ok"}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.aingest("ws_a", source_type="local", path="/tmp/f.pdf")
        mock_svc.aregister_workspace.assert_awaited_once()
        mock_svc.aingest.assert_awaited_once()
        assert result == {"status": "ok"}

    @patch("dlightrag.storage.replication.wait_for_current_wal_replay", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aingest_waits_for_replica_when_configured(
        self, mock_create, mock_wait, test_cfg
    ) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"status": "ok"}
        mock_create.return_value = mock_svc
        mock_wait.return_value = "0/42"
        cfg = test_cfg.model_copy(
            update={"read_after_write_mode": "wait_for_replay", "read_after_write_timeout": 2.5}
        )

        manager = RAGServiceManager(config=cfg)
        result = await manager.aingest("ws_a", source_type="local", path="/tmp/f.pdf")

        mock_wait.assert_awaited_once_with(cfg, timeout=2.5)
        assert result == {"status": "ok", "replica_replay_lsn": "0/42"}

    @patch("dlightrag.storage.replication.wait_for_current_wal_replay", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aingest_surfaces_replica_wait_timeout(
        self, mock_create, mock_wait, test_cfg
    ) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"status": "ok"}
        mock_create.return_value = mock_svc
        mock_wait.side_effect = TimeoutError("replica lag")
        cfg = test_cfg.model_copy(update={"read_after_write_mode": "wait_for_replay"})

        manager = RAGServiceManager(config=cfg)

        with pytest.raises(RAGServiceUnavailableError, match="replica lag"):
            await manager.aingest("ws_a", source_type="local", path="/tmp/f.pdf")

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_list_ingested_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.alist_ingested_files.return_value = [{"doc": "d1"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.list_ingested_files("ws_a")
        assert result == [{"doc": "d1"}]

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_delete_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.adelete_files.return_value = [{"status": "deleted"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.delete_files("ws_a", filenames=["a.pdf"])
        assert result == [{"status": "deleted"}]


class TestDegradedMode:
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_create_sets_ready_on_success(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = await RAGServiceManager.create(config=test_cfg)
        assert manager.is_ready()
        assert not manager.is_degraded()
        # Warnings may include "Workspace registry unavailable" in tests
        # without a running PostgreSQL — that's expected and non-fatal.

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_create_sets_degraded_on_failure(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = await RAGServiceManager.create(config=test_cfg)
        assert not manager.is_ready()
        assert manager.is_degraded()
        assert any("DB down" in w for w in manager.get_warnings())


class TestActionableErrors:
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_connection_refused_gets_hint(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = ConnectionRefusedError("Connection refused")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError, match="Check.*DLIGHTRAG_POSTGRES"):
            await manager._get_service("ws_a")

    def test_actionable_error_default(self) -> None:
        exc = ValueError("something broke")
        result = RAGServiceManager._actionable_error(exc)
        assert result == "ValueError: something broke"

    def test_actionable_error_timeout(self) -> None:
        exc = TimeoutError("request timed out")
        result = RAGServiceManager._actionable_error(exc)
        assert "overloaded" in result


class TestRequestTimeout:
    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_retrieve_timeout(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()

        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(10)

        mock_svc.aretrieve = slow_retrieve
        mock_create.return_value = mock_svc
        test_cfg_short = test_cfg.model_copy(update={"request_timeout": 1})
        manager = RAGServiceManager(config=test_cfg_short)
        with pytest.raises(RAGServiceUnavailableError, match="timed out"):
            await manager.aretrieve("test query", workspace="default")


class TestClose:
    """Test cleanup."""

    async def test_close_all_services(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        svc_a = AsyncMock()
        svc_b = AsyncMock()
        manager._services = {"a": svc_a, "b": svc_b}
        manager._ready = True
        await manager.close()
        svc_a.close.assert_awaited_once()
        svc_b.close.assert_awaited_once()
        assert manager._services == {}
        assert not manager._ready


class TestWorkspaceDiscovery:
    """Test list_workspaces with PostgreSQL-backed metadata."""

    async def test_pg_discovery(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._workspace_registry = AsyncMock()
        manager._workspace_registry.list = AsyncMock(
            return_value=[
                {
                    "workspace": "project-a",
                    "display_name": "Project A",
                    "created_at": datetime(2026, 5, 25, tzinfo=UTC),
                },
                {"workspace": "project-b", "display_name": "Project B"},
            ]
        )

        result = await manager.list_workspaces()

        assert "project-a" in result
        assert "project-b" in result

    async def test_workspace_records_are_json_safe(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._workspace_registry = AsyncMock()
        manager._workspace_registry.list = AsyncMock(
            return_value=[
                {
                    "workspace": "project-a",
                    "display_name": "Project A",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": datetime(2026, 5, 25, 12, 0, tzinfo=UTC),
                    "updated_at": datetime(2026, 5, 25, 12, 1, tzinfo=UTC),
                }
            ]
        )

        records = await manager.list_workspace_records()

        assert records == [
            {
                "workspace": "project-a",
                "display_name": "Project A",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": "2026-05-25T12:00:00+00:00",
                "updated_at": "2026-05-25T12:01:00+00:00",
            }
        ]

    async def test_fallback_returns_default(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.list_workspaces()
        assert test_cfg.workspace in result
