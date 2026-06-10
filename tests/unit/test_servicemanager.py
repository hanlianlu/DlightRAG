# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
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
from dlightrag.core.scope import RequestScope
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError


@pytest.fixture()
def test_cfg(tmp_path) -> DlightragConfig:
    cfg = DlightragConfig(
        working_dir=str(tmp_path / "dlightrag_storage"),
        llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="test",
            startup_probe=False,
        ),
    )
    set_config(cfg)
    return cfg


class _InMemoryIngestJobStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.recoverable_rows: list[dict[str, Any]] = []
        self.deleted_workspaces: list[str] = []
        self.pruned = False

    async def create(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: str,
        request: dict[str, Any],
    ) -> None:
        self.rows[job_id] = {
            "job_id": job_id,
            "workspace": workspace,
            "source_type": source_type,
            "status": "queued",
            "request": request,
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "current_window": 0,
            "result": {},
            "errors": [],
        }

    async def mark_running(self, job_id: str) -> None:
        self.rows[job_id]["status"] = "running"

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        current_window: int,
        errors: list[str],
    ) -> None:
        row = self.rows[job_id]
        row["total_items"] += total_delta
        row["processed_items"] += processed_delta
        row["failed_items"] += failed_delta
        row["current_window"] = current_window
        row["errors"].extend(errors)

    async def finish(self, job_id: str, *, result: dict[str, Any]) -> None:
        self.rows[job_id]["status"] = "succeeded"
        self.rows[job_id]["result"] = result

    async def fail(self, job_id: str, *, error: str) -> None:
        self.rows[job_id]["status"] = "failed"
        self.rows[job_id]["errors"].append(error)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return self.rows.get(job_id)

    async def list_recoverable(self) -> list[dict[str, Any]]:
        return list(self.recoverable_rows)

    async def prune(self) -> dict[str, int]:
        self.pruned = True
        return {"failed_abandoned": 0, "deleted_completed": 0}

    async def delete_for_workspace(self, workspace: str) -> int:
        self.deleted_workspaces.append(workspace)
        before = len(self.rows)
        self.rows = {
            job_id: row for job_id, row in self.rows.items() if row.get("workspace") != workspace
        }
        return before - len(self.rows)


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

    @patch("dlightrag.storage.workspaces.PGWorkspaceRegistry")
    async def test_initialize_registry_uses_canonical_workspace_id(
        self,
        mock_registry_cls: MagicMock,
        test_cfg: DlightragConfig,
    ) -> None:
        registry = MagicMock()
        registry.initialize = AsyncMock()
        registry.upsert = AsyncMock()
        mock_registry_cls.return_value = registry
        cfg = test_cfg.model_copy(update={"workspace": "test-fallback-ws"})
        manager = RAGServiceManager(config=cfg)

        await manager._initialize_workspace_registry()

        registry.upsert.assert_awaited_once_with(
            workspace="test_fallback_ws",
            display_name="test-fallback-ws",
            embedding_model=cfg.embedding.model,
        )

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_create_workspace_registers_workspace_meta(self, mock_create, test_cfg) -> None:
        svc = AsyncMock()
        mock_create.return_value = svc
        manager = RAGServiceManager(config=test_cfg)

        await manager.acreate_workspace("new workspace")

        svc.aregister_workspace.assert_awaited_once()


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
        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == test_cfg.chunk_top_k

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aretrieve_keeps_explicit_retrieval_limits(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", top_k=9, chunk_top_k=4)
        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == 9
        assert retrieve_kwargs["chunk_top_k"] == 4

    async def test_query_image_memory_is_request_scoped(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._query_image_enhancer = AsyncMock()
        manager._query_image_enhancer.enhance = AsyncMock(
            side_effect=lambda query, images: MagicMock(query=query, descriptions=[])
        )
        alice_scope = RequestScope(user_id="alice", auth_mode="jwt").for_workspaces(["reports"])
        bob_scope = RequestScope(user_id="bob", auth_mode="jwt").for_workspaces(["reports"])

        first = await manager._prepare_query_images(
            "query",
            query_images=["data:image/png;base64,abc"],
            session_id="same-session",
            referenced_image_ids=None,
            store_current=True,
            scope=alice_scope,
        )
        second = await manager._prepare_query_images(
            "query",
            query_images=[],
            session_id="same-session",
            referenced_image_ids=first.current_image_ids,
            store_current=False,
            scope=bob_scope,
        )

        assert first.current_image_ids == ["img_0"]
        assert second.answer_images == []

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
            "query",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=30,
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
            "what is X?",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=30,
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
            conversation_history=None,
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
            "what is X?",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=30,
        )
        assert contexts is mock_contexts
        assert stream is not None

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
            "query",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=30,
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
            "query",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=30,
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

    async def test_stream_concurrency_is_held_until_iterator_finishes(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"max_async": 1})
        manager = RAGServiceManager(config=cfg)
        contexts = {"chunks": [], "entities": [], "relationships": []}

        async def one_token_stream():
            yield "token"

        mock_engine = AsyncMock()
        mock_engine.generate_stream = AsyncMock(return_value=(contexts, one_token_stream()))
        manager._answer_engine = mock_engine

        _, first_stream = await manager.agenerate_stream_from_contexts("q1", contexts)
        second = asyncio.create_task(manager.agenerate_stream_from_contexts("q2", contexts))
        await asyncio.sleep(0)

        assert not second.done()
        assert first_stream is not None
        async for _ in first_stream:
            pass

        _, second_stream = await asyncio.wait_for(second, timeout=1.0)
        assert second_stream is not None


class TestDelegation:
    """Test write-operation delegation."""

    @patch("dlightrag.core.servicemanager.RAGService.create", new_callable=AsyncMock)
    async def test_aingest_uses_job_runner_and_returns_result(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"doc_id": "d1", "status": "ok"}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_job_store = store

        result = await manager.aingest("ws_a", source_type="local", path="/tmp/f.pdf")

        mock_svc.aregister_workspace.assert_awaited_once()
        mock_svc.aingest.assert_awaited_once()
        assert result == {"doc_id": "d1", "status": "ok"}
        row = next(iter(store.rows.values()))
        assert row["workspace"] == "ws_a"
        assert row["status"] == "succeeded"
        assert row["total_items"] == 1
        assert row["processed_items"] == 1
        assert row["result"] == {"doc_id": "d1", "status": "ok"}

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


class TestIngestJobs:
    """Test durable background ingest job orchestration."""

    async def test_recover_ingest_jobs_reschedules_running_job_from_window(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        row = {
            "job_id": "job-1",
            "workspace": "project_a",
            "source_type": "s3",
            "status": "running",
            "request": {
                "workspace": "project_a",
                "source_type": "s3",
                "kwargs": {"bucket": "bucket", "prefix": "docs/"},
            },
            "total_items": 128,
            "processed_items": 128,
            "failed_items": 0,
            "current_window": 2,
            "errors": [],
            "result": {},
        }
        store.recoverable_rows = [row]
        store.rows["job-1"] = dict(row)
        manager._ingest_job_store = store
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        await manager._recover_ingest_jobs(store)
        task = manager._ingest_job_tasks["job-1"]

        await asyncio.wait_for(task, timeout=1.0)

        svc.aregister_workspace.assert_awaited_once()
        svc.aingest.assert_awaited_once()
        ingest_kwargs = svc.aingest.await_args.kwargs
        assert ingest_kwargs["bucket"] == "bucket"
        assert ingest_kwargs["prefix"] == "docs/"
        assert ingest_kwargs["_resume_from_window"] == 2
        row = await manager.get_ingest_job("job-1")
        assert row is not None
        assert row["processed_items"] == 129
        assert row["result"]["processed"] == 129

    async def test_astart_ingest_job_records_progress_and_result(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_job_store = store

        async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
            progress_callback = kwargs["_progress_callback"]
            await progress_callback(
                SimpleNamespace(
                    total_delta=2,
                    processed_delta=1,
                    failed_delta=1,
                    batch_index=0,
                    errors=("s3://bucket/docs/bad.pdf: failed",),
                )
            )
            return {"processed": 1, "failed": 1}

        svc = AsyncMock()
        svc.aingest = AsyncMock(side_effect=fake_ingest)
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        job = await manager.astart_ingest_job(
            "Project A",
            source_type="s3",
            bucket="bucket",
            prefix="docs/",
        )
        task = manager._ingest_job_tasks[job["job_id"]]

        await asyncio.wait_for(task, timeout=1.0)
        row = await manager.get_ingest_job(job["job_id"])

        assert row is not None
        assert row["workspace"] == "project_a"
        assert row["status"] == "succeeded"
        assert row["total_items"] == 2
        assert row["processed_items"] == 1
        assert row["failed_items"] == 1
        assert row["current_window"] == 1
        assert row["result"] == {"processed": 1, "failed": 1}
        assert row["errors"] == ["s3://bucket/docs/bad.pdf: failed"]
        svc.aregister_workspace.assert_awaited_once()
        svc.aingest.assert_awaited_once()

    async def test_aingest_timeout_returns_running_job_without_cancelling_task(
        self, test_cfg
    ) -> None:
        cfg = test_cfg.model_copy(update={"ingest_timeout": 0.01})
        manager = RAGServiceManager(config=cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_job_store = store
        release = asyncio.Event()

        async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
            await release.wait()
            return {"doc_id": "d1"}

        svc = AsyncMock()
        svc.aingest = AsyncMock(side_effect=fake_ingest)
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        result = await manager.aingest("default", source_type="local", path="/tmp/slow.pdf")

        assert result["status"] in {"queued", "running"}
        assert result["job_id"] in manager._ingest_job_tasks
        task = manager._ingest_job_tasks[result["job_id"]]
        assert not task.done()

        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        row = await manager.get_ingest_job(result["job_id"])
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["result"] == {"doc_id": "d1"}


class TestDegradedMode:
    @pytest.fixture(autouse=True)
    def _isolate_workspace_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_initialize_workspace_registry(manager: RAGServiceManager) -> None:
            registry = AsyncMock()
            registry.list.return_value = []
            manager._workspace_registry = registry

        monkeypatch.setattr(
            RAGServiceManager,
            "_initialize_workspace_registry",
            fake_initialize_workspace_registry,
        )

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

    async def test_create_warmup_concurrency_follows_config(
        self, monkeypatch: pytest.MonkeyPatch, test_cfg
    ) -> None:
        cfg = test_cfg.model_copy(update={"max_async": 3})
        calls: dict[str, object] = {}

        async def fake_initialize_workspace_registry(self):  # noqa: ANN001, ANN202
            return None

        async def fake_list_all_workspaces(self):  # noqa: ANN001, ANN202
            return ["default", "alpha", "beta"]

        async def fake_get_service(self, workspace: str):  # noqa: ANN001, ANN202
            self._services[workspace] = workspace
            return workspace

        async def fake_bounded_gather(coros, *, max_concurrent: int, task_name: str):  # noqa: ANN001, ANN202
            calls["max_concurrent"] = max_concurrent
            calls["task_name"] = task_name
            return [await coro for coro in coros]

        monkeypatch.setattr(
            RAGServiceManager,
            "_initialize_workspace_registry",
            fake_initialize_workspace_registry,
        )
        monkeypatch.setattr(RAGServiceManager, "_list_all_workspaces", fake_list_all_workspaces)
        monkeypatch.setattr(RAGServiceManager, "_get_service", fake_get_service)
        monkeypatch.setattr("dlightrag.observability.init_tracing", lambda config: None)
        monkeypatch.setattr("dlightrag.utils.concurrency.bounded_gather", fake_bounded_gather)

        manager = await RAGServiceManager.create(config=cfg)

        assert manager.is_ready()
        assert calls == {"max_concurrent": 3, "task_name": "warmup"}


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
                    "workspace": "project_a",
                    "display_name": "Project A",
                    "created_at": datetime(2026, 5, 25, tzinfo=UTC),
                },
                {"workspace": "project_b", "display_name": "Project B"},
            ]
        )

        result = await manager.list_workspaces()

        assert "project_a" in result
        assert "project_b" in result

    async def test_workspace_records_are_json_safe(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._workspace_registry = AsyncMock()
        manager._workspace_registry.list = AsyncMock(
            return_value=[
                {
                    "workspace": "project_a",
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
                "workspace": "project_a",
                "display_name": "Project A",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": "2026-05-25T12:00:00+00:00",
                "updated_at": "2026-05-25T12:01:00+00:00",
            }
        ]

    async def test_fallback_returns_default(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._workspace_registry = AsyncMock()
        manager._workspace_registry.list = AsyncMock(side_effect=RuntimeError("registry down"))

        result = await manager.list_workspaces()

        assert test_cfg.workspace in result


class TestPlannerSchemaScope:
    async def test_aplan_query_uses_schema_for_requested_workspace(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        reports = AsyncMock()
        reports._metadata_index.get_field_schema = AsyncMock(
            return_value={
                "columns": [{"name": "filename", "type": "character varying"}],
                "custom_keys": ["department"],
            }
        )
        legal = AsyncMock()
        legal._metadata_index.get_field_schema = AsyncMock(
            return_value={
                "columns": [{"name": "filename", "type": "character varying"}],
                "custom_keys": ["jurisdiction"],
            }
        )
        manager._services = {"reports": reports, "legal": legal}

        llm = AsyncMock(return_value='{"standalone_query": "q", "filters": {}}')
        manager._query_planner = QueryPlanner(llm_func=llm)

        await manager.aplan_query("q", workspaces=["reports"])
        await manager.aplan_query("q", workspaces=["legal"])

        first_prompt = llm.await_args_list[0].kwargs["messages"][0]["content"]
        second_prompt = llm.await_args_list[1].kwargs["messages"][0]["content"]
        assert "department" in first_prompt
        assert "jurisdiction" not in first_prompt
        assert "jurisdiction" in second_prompt
        assert "department" not in second_prompt
