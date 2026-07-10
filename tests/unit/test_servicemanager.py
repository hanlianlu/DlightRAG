# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

import asyncio
import inspect
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlightrag.citations.schemas import ChunkSnippet, SourceReference
from dlightrag.config import (
    AnswerConfig,
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelConfig,
    RerankConfig,
    set_config,
)
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.query_images import prepare_query_images
from dlightrag.core.query_planner import QueryPlanner
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.scope import RequestScope
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError
from dlightrag.core.session_images import SessionImageStore
from dlightrag.sourcing.base import SourceDocument


def _image_block(url: str = "data:image/png;base64,abc") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _record_trace_calls(calls: list[dict[str, Any]]):
    @asynccontextmanager
    async def _trace(name: str, **kwargs: Any):
        call = {"name": name, **kwargs, "updates": []}
        calls.append(call)

        class _Trace:
            def update(self, **update_kwargs: Any) -> None:
                call["updates"].append(update_kwargs)

        yield _Trace()

    return _trace


def test_public_sdk_signatures_expose_primary_contracts() -> None:
    for method in (
        RAGServiceManager.aingest,
        RAGServiceManager.astart_ingest_job,
        RAGServiceManager.aretrieve,
        RAGServiceManager.aanswer,
        RAGServiceManager.aanswer_stream,
        RAGServiceManager.adelete_files,
    ):
        params = inspect.signature(method).parameters.values()
        assert all(param.kind is not inspect.Parameter.VAR_KEYWORD for param in params)

    ingest_params = inspect.signature(RAGServiceManager.aingest).parameters
    start_job_params = inspect.signature(RAGServiceManager.astart_ingest_job).parameters
    assert tuple(ingest_params) == ("self", "workspace", "request")
    assert tuple(start_job_params) == ("self", "workspace", "request")

    for name in (
        "path",
        "bucket",
        "s3_region",
        "s3_key",
        "prefix",
        "url",
        "urls",
        "source_uri",
        "source_uris",
        "retain_source_file",
        "metadata_policy",
        "replace",
    ):
        assert name in IngestSpec.model_fields

    retrieve_params = inspect.signature(RAGServiceManager.aretrieve).parameters
    for name in (
        "all_workspaces",
        "top_k",
        "chunk_top_k",
        "filters",
        "multimodal_content",
    ):
        assert name in retrieve_params

    answer_params = inspect.signature(RAGServiceManager.aanswer).parameters
    for name in (
        "top_k",
        "chunk_top_k",
        "answer_context_top_k",
        "filters",
        "multimodal_content",
        "session_id",
        "referenced_image_ids",
        "all_workspaces",
    ):
        assert name in answer_params

    stream_params = inspect.signature(RAGServiceManager.aanswer_stream).parameters
    assert "all_workspaces" in stream_params

    delete_params = inspect.signature(RAGServiceManager.adelete_files).parameters
    assert "dry_run" in delete_params


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
        self.claim_results: dict[str, bool] = {}

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

    async def claim_running(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool:
        if self.claim_results.get(job_id, True) is False:
            return False
        self.rows[job_id]["status"] = "running"
        self.rows[job_id]["lease_owner"] = lease_owner
        self.rows[job_id]["lease_seconds"] = lease_seconds
        return True

    async def heartbeat(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool:
        row = self.rows.get(job_id)
        return bool(row and row.get("lease_owner") == lease_owner and lease_seconds > 0)

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        current_window: int,
        errors: list[str],
        lease_owner: str | None = None,
        lease_seconds: int | None = None,
    ) -> bool:
        row = self.rows[job_id]
        row["total_items"] += total_delta
        row["processed_items"] += processed_delta
        row["failed_items"] += failed_delta
        row["current_window"] = current_window
        row["errors"].extend(errors)
        return True

    async def finish(
        self, job_id: str, *, result: dict[str, Any], lease_owner: str | None = None
    ) -> bool:
        self.rows[job_id]["status"] = "succeeded"
        self.rows[job_id]["result"] = result
        return True

    async def fail(self, job_id: str, *, error: str, lease_owner: str | None = None) -> bool:
        self.rows[job_id]["status"] = "failed"
        self.rows[job_id]["errors"].append(error)
        return True

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


class TestDirectLLMSemaphore:
    """The _sem_bound cap replaces the removed DlightRAG completion queue."""

    async def test_serializes_owned_llm_calls(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._direct_llm_sem = asyncio.Semaphore(1)
        peak = 0
        running = 0

        async def fake_llm(*args: Any, **kwargs: Any) -> str:
            nonlocal peak, running
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return "ok"

        bound = manager._sem_bound(fake_llm)
        await asyncio.gather(bound(), bound(), bound())
        assert peak == 1


class TestGetService:
    """Test workspace-keyed RAGService creation and caching."""

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_creates_service_for_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc = await manager._get_service("project-a")
        assert svc is mock_create.return_value
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].workspace == "project_a"  # normalized

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_caches_per_workspace(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws_1")
        svc2 = await manager._get_service("ws_1")
        assert svc1 is svc2
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_different_workspaces_different_services(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = [AsyncMock(), AsyncMock()]
        manager = RAGServiceManager(config=test_cfg)
        svc1 = await manager._get_service("ws_a")
        svc2 = await manager._get_service("ws_b")
        assert svc1 is not svc2
        assert mock_create.await_count == 2

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_create_workspace_registers_workspace_meta(self, mock_create, test_cfg) -> None:
        svc = AsyncMock()
        mock_create.return_value = svc
        manager = RAGServiceManager(config=test_cfg)

        await manager.acreate_workspace("new workspace")

        svc.aregister_workspace.assert_awaited_once()


class TestBackoff:
    """Test exponential backoff on service creation failure."""

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_failure_sets_error_state(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        assert not manager.is_ready()
        error_info = manager.get_error_info()
        assert "ws_a" in error_info["backoff_workspaces"]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_backoff_blocks_retry(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("fail")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        with pytest.raises(RAGServiceUnavailableError):
            await manager._get_service("ws_a")
        assert mock_create.await_count == 1

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aretrieve_single_workspace(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aretrieve_multi_workspace_federates(
        self, mock_create, mock_fed, test_cfg
    ) -> None:
        mock_fed.return_value = MagicMock()
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspaces=["ws_a", "ws_b"])
        mock_fed.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    async def test_aretrieve_all_workspaces_expands_registry(
        self,
        mock_federated,
        test_cfg,
    ) -> None:
        mock_federated.return_value = RetrievalResult(contexts={"chunks": []})
        manager = RAGServiceManager(config=test_cfg)
        manager.alist_workspaces = AsyncMock(return_value=["default", "Research Notes"])

        await manager.aretrieve("query", all_workspaces=True)

        manager.alist_workspaces.assert_awaited_once()
        assert mock_federated.await_args.args[1] == ["default", "research_notes"]

    @pytest.mark.parametrize("method_name", ["aretrieve", "aanswer", "aanswer_stream"])
    @pytest.mark.parametrize(
        "explicit_selection",
        [
            {"workspace": "finance"},
            {"workspaces": ["finance"]},
        ],
    )
    async def test_all_workspaces_conflicts_with_explicit_selection(
        self,
        method_name: str,
        explicit_selection: dict[str, object],
        test_cfg,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)

        with pytest.raises(ValueError, match="all_workspaces"):
            await getattr(manager, method_name)(
                "query",
                all_workspaces=True,
                **explicit_selection,
            )

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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
        session_images = SessionImageStore(
            max_images_per_session=3,
            max_sessions=10,
            ttl_seconds=test_cfg.checkpoint_session_ttl_seconds,
        )
        enhancer = AsyncMock()
        enhancer.enhance = AsyncMock(
            side_effect=lambda query, images: MagicMock(query=query, descriptions=[])
        )
        alice_scope = RequestScope(user_id="alice", auth_mode="jwt").for_workspaces(["reports"])
        bob_scope = RequestScope(user_id="bob", auth_mode="jwt").for_workspaces(["reports"])

        first = await prepare_query_images(
            "query",
            query_images=[_image_block()],
            session_id="same-session",
            referenced_image_ids=None,
            store_current=True,
            session_images=session_images,
            enhancer=enhancer,
            scope=alice_scope,
        )
        second = await prepare_query_images(
            "query",
            query_images=[],
            session_id="same-session",
            referenced_image_ids=first.current_image_ids,
            store_current=False,
            session_images=session_images,
            enhancer=enhancer,
            scope=bob_scope,
        )

        assert first.current_image_ids == ["img_0"]
        assert second.answer_images == []

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    async def test_aplan_query_emits_query_planning_observation(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def llm_func(*, messages, **kwargs) -> str:
            return (
                '{"standalone_query":"rewritten query","bm25_query":"rewritten query",'
                '"referenced_image_ids":[],"filters":{},'
                '"filter_confidence":"low","filter_evidence":[]}'
            )

        trace_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "dlightrag.observability.trace_observation",
            _record_trace_calls(trace_calls),
        )

        manager = RAGServiceManager(config=test_cfg)
        manager._query_planner = QueryPlanner(llm_func=llm_func)
        manager._get_schema = AsyncMock(return_value={})  # type: ignore[method-assign]

        plan = await manager.aplan_query("raw query", workspaces=["ws_a"])

        assert plan.standalone_query == "rewritten query"
        assert trace_calls == [
            {
                "name": "query_planning",
                "as_type": "chain",
                "input": {"query": "raw query"},
                "metadata": {"workspaces": ["ws_a"], "history_turns": 0},
                "updates": [
                    {
                        "output": {
                            "standalone_query": "rewritten query",
                            "has_metadata_filter": False,
                            "referenced_image_count": 0,
                        }
                    }
                ],
            }
        ]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_calls_retrieve_then_engine(
        self, mock_create, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aanswer() calls aretrieve() then AnswerEngine.generate()."""
        trace_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "dlightrag.observability.trace_observation",
            _record_trace_calls(trace_calls),
        )
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_retrieval = MagicMock(contexts=mock_contexts, trace={})
        mock_svc.aretrieve.return_value = mock_retrieval
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        answer_text = "Answer [1-1]."
        expected_result = MagicMock(
            answer=answer_text, contexts=mock_contexts, sources=[], trace={}
        )
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
        retrieve = next(call for call in trace_calls if call["name"] == "retrieve")
        answer_pipeline = next(call for call in trace_calls if call["name"] == "answer_pipeline")
        answer_generation = next(
            call for call in trace_calls if call["name"] == "answer_generation"
        )
        assert retrieve["input"] == {"query": "what is X?"}
        assert "query" not in retrieve["metadata"]
        assert retrieve["metadata"] == {
            "workspaces": ["ws_a"],
            "top_k": test_cfg.top_k,
            "chunk_top_k": test_cfg.chunk_top_k,
            "has_filters": False,
            "multimodal_count": 0,
        }
        assert retrieve["updates"] == [
            {
                "output": {
                    "context_chunk_count": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "query_image_description_count": 0,
                }
            }
        ]
        assert answer_pipeline["input"] == {"query": "what is X?"}
        assert "query" not in answer_pipeline["metadata"]
        assert answer_pipeline["updates"] == [
            {
                "output": {
                    "answer_len": len(answer_text),
                    "source_count": 0,
                    "context_chunk_count": 0,
                }
            }
        ]
        assert answer_generation["input"] == {"query": "what is X?"}
        assert answer_generation["metadata"] == {"context_chunks": 0, "context_top_k": 30}
        assert answer_generation["updates"] == [
            {
                "output": {
                    "answer_len": len(answer_text),
                    "source_count": 0,
                    "context_chunk_count": 0,
                }
            }
        ]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_derives_candidate_and_context_limits(
        self, mock_create, test_cfg
    ) -> None:
        """Answer over-fetches retrieval candidates and caps final prompt contexts."""
        cfg = test_cfg.model_copy(update={"answer": AnswerConfig(context_top_k=3)})
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
        assert retrieve_kwargs["chunk_top_k"] == test_cfg.chunk_top_k
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            query_images=None,
            conversation_history=None,
            context_top_k=3,
        )
        assert result is expected_result

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_uses_chunk_top_k_as_candidate_override(
        self, mock_create, test_cfg
    ) -> None:
        """Answer chunk_top_k remains the explicit retrieval candidate override."""
        cfg = test_cfg.model_copy(update={"answer": AnswerConfig(context_top_k=3)})
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

        result = await manager.aanswer("query", workspace="ws_a", chunk_top_k=7)

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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_semantic_highlights_are_opt_in(
        self, mock_create, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK answer highlights are disabled by default and enabled per call."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        async def llm_func(*, messages, **kwargs) -> str:
            return '{"phrases": ["market growth"], "confidence": 1.0}'

        monkeypatch.setattr("dlightrag.models.llm.get_keyword_model_func", lambda _cfg: llm_func)
        trace_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "dlightrag.observability.trace_observation",
            _record_trace_calls(trace_calls),
        )

        def _answer_result() -> MagicMock:
            return MagicMock(
                answer="Market growth improved [1-1].",
                contexts=mock_contexts,
                sources=[
                    SourceReference(
                        id="1",
                        path="/docs/report.pdf",
                        chunks=[
                            ChunkSnippet(
                                chunk_id="c1",
                                chunk_idx=1,
                                content="The report says market growth improved in 2025.",
                            )
                        ],
                    )
                ],
                trace={},
            )

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_engine = AsyncMock()
        manager._answer_engine.generate.side_effect = [_answer_result(), _answer_result()]
        manager._query_planner = QueryPlanner(llm_func=None)

        plain = await manager.aanswer("query", workspace="ws_a")
        highlighted = await manager.aanswer(
            "query",
            workspace="ws_a",
            semantic_highlights=True,
        )

        plain_chunks = plain.sources[0].chunks
        highlighted_chunks = highlighted.sources[0].chunks
        assert plain_chunks is not None
        assert highlighted_chunks is not None
        assert plain_chunks[0].highlight_phrases is None
        assert highlighted_chunks[0].highlight_phrases == ["market growth"]
        semantic_highlights = next(
            call for call in trace_calls if call["name"] == "semantic_highlights"
        )
        assert semantic_highlights["metadata"] == {"source_count": 1, "text_chunk_count": 1}
        assert semantic_highlights["updates"] == [
            {"output": {"highlighted_source_count": 1, "highlighted_chunk_count": 1}}
        ]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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
    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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
    async def test_aanswer_all_workspaces_uses_federated_retrieve(
        self,
        mock_federated,
        test_cfg,
    ) -> None:
        contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_federated.return_value = RetrievalResult(contexts=contexts)
        engine = AsyncMock()
        expected = RetrievalResult(answer="answer", contexts=contexts)
        engine.generate.return_value = expected
        manager = RAGServiceManager(config=test_cfg)
        manager.alist_workspaces = AsyncMock(return_value=["ws_a", "ws_b"])
        manager._answer_engine = engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", all_workspaces=True)

        manager.alist_workspaces.assert_awaited_once()
        assert mock_federated.await_args.args[1] == ["ws_a", "ws_b"]
        assert result is expected

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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

    @patch("dlightrag.core.servicemanager.federated_retrieve", new_callable=AsyncMock)
    async def test_aanswer_stream_all_workspaces_uses_federated_retrieve(
        self,
        mock_federated,
        test_cfg,
    ) -> None:
        contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_federated.return_value = RetrievalResult(contexts=contexts)
        engine = AsyncMock()
        answer_stream = AsyncMock()
        engine.generate_stream.return_value = (contexts, answer_stream)
        manager = RAGServiceManager(config=test_cfg)
        manager.alist_workspaces = AsyncMock(return_value=["ws_a", "ws_b"])
        manager._answer_engine = engine
        manager._query_planner = QueryPlanner(llm_func=None)

        resolved_contexts, stream = await manager.aanswer_stream(
            "query",
            all_workspaces=True,
        )

        manager.alist_workspaces.assert_awaited_once()
        assert mock_federated.await_args.args[1] == ["ws_a", "ws_b"]
        assert resolved_contexts is contexts
        assert stream is not None

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

    def test_get_query_planner_uses_planner_model_func(self, test_cfg) -> None:
        """QueryPlanner uses the text planning factory, not the answer/query role."""
        manager = RAGServiceManager(config=test_cfg)
        planner_func = MagicMock()

        with (
            patch(
                "dlightrag.models.llm.get_planner_model_func",
                return_value=planner_func,
                create=True,
            ) as mock_planner,
            patch("dlightrag.models.llm.get_query_model_func") as mock_query,
        ):
            planner = manager._get_query_planner()
            planner2 = manager._get_query_planner()

        mock_planner.assert_called_once_with(test_cfg)
        mock_query.assert_not_called()
        assert planner2 is planner
        # llm_func is the planner factory wrapped by the direct-LLM semaphore
        assert callable(planner._llm_func)
        assert planner._llm_func is not planner_func

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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aingest_uses_job_runner_and_returns_result(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"doc_id": "d1", "status": "ok"}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store

        result = await manager.aingest(
            "ws_a",
            IngestSpec(source_type="local", path="/tmp/f.pdf"),
        )

        mock_svc.aregister_workspace.assert_awaited_once()
        mock_svc.aingest.assert_awaited_once()
        assert result == {"doc_id": "d1", "status": "ok"}
        row = next(iter(store.rows.values()))
        assert row["workspace"] == "ws_a"
        assert row["status"] == "succeeded"
        assert row["total_items"] == 1
        assert row["processed_items"] == 1
        assert row["result"] == {"doc_id": "d1", "status": "ok"}

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_s3_region_reaches_service_ingest(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aingest.return_value = {"processed": 1, "errors": []}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        manager._ingest_jobs._store = _InMemoryIngestJobStore()

        await manager.aingest(
            "ws_a",
            IngestSpec(
                source_type="s3",
                bucket="bucket",
                s3_key="docs/report.pdf",
                s3_region="eu-north-1",
            ),
        )

        assert mock_svc.aingest.await_args.kwargs["s3_region"] == "eu-north-1"

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aingest_source_delegates_directly_to_service(
        self, mock_create, test_cfg
    ) -> None:
        source = AsyncMock()
        mock_svc = AsyncMock()
        mock_svc.aingest_source.return_value = {"processed": 1}
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)

        result = await manager.aingest_source(
            "ws_a",
            source,
            source_type="bynder",
            documents=[SourceDocument(key="asset.pdf")],
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            retain_source_file=True,
        )

        assert result == {"processed": 1}
        mock_svc.aingest_source.assert_awaited_once()
        assert mock_svc.aingest_source.await_args.kwargs["documents"] == [
            SourceDocument(key="asset.pdf")
        ]
        assert mock_svc.aingest_source.await_args.kwargs["retain_source_file"] is True
        assert manager._ingest_jobs._tasks == {}

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_list_ingested_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.alist_ingested_files.return_value = [{"doc": "d1"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.alist_ingested_files("ws_a")
        assert result == [{"doc": "d1"}]

    async def test_file_panel_snapshot_does_not_initialize_cold_workspace(self, test_cfg) -> None:
        store = AsyncMock()
        store.list_processed_files.return_value = [
            {"doc_id": "d1", "file_path": "/tmp/report.pdf", "status": "processed"}
        ]
        manager = RAGServiceManager(config=test_cfg)
        manager._get_file_panel_store = MagicMock(return_value=store)  # type: ignore[method-assign]
        manager._get_service = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("files panel snapshot must not initialize services")
        )

        result = await manager.aget_file_panel_snapshot("Ws-A")

        assert result == {
            "files": [{"doc_id": "d1", "file_path": "/tmp/report.pdf", "status": "processed"}],
            "pipeline_status": {"busy": False, "pending_enqueues": 0, "latest_message": ""},
        }
        store.list_processed_files.assert_awaited_once_with("ws_a")
        manager._get_service.assert_not_awaited()

    async def test_file_panel_snapshot_reads_pipeline_status_for_warm_workspace(
        self, test_cfg
    ) -> None:
        store = AsyncMock()
        store.list_processed_files.return_value = []
        svc = AsyncMock()
        svc.aget_pipeline_status.return_value = {
            "busy": True,
            "pending_enqueues": 1,
            "latest_message": "Indexing",
        }
        manager = RAGServiceManager(config=test_cfg)
        manager._get_file_panel_store = MagicMock(return_value=store)  # type: ignore[method-assign]
        manager._services["ws_a"] = svc

        result = await manager.aget_file_panel_snapshot("Ws-A")

        assert result["pipeline_status"] == {
            "busy": True,
            "pending_enqueues": 1,
            "latest_message": "Indexing",
        }
        svc.aget_pipeline_status.assert_awaited_once()

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_delete_files_delegates(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.adelete_files.return_value = [{"status": "deleted"}]
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        result = await manager.adelete_files("ws_a", filenames=["a.pdf"], dry_run=True)
        assert result == [{"status": "deleted"}]
        mock_svc.adelete_files.assert_awaited_once_with(
            file_paths=None,
            filenames=["a.pdf"],
            dry_run=True,
        )


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
        manager._ingest_jobs._store = store
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        assert manager._ingest_jobs.schedule_recovered_job(row, store) is True
        task = manager._ingest_jobs._tasks["job-1"]

        await asyncio.wait_for(task, timeout=1.0)

        svc.aregister_workspace.assert_awaited_once()
        svc.aingest.assert_awaited_once()
        ingest_kwargs = svc.aingest.await_args.kwargs
        assert ingest_kwargs["bucket"] == "bucket"
        assert ingest_kwargs["prefix"] == "docs/"
        assert ingest_kwargs["_resume_from_window"] == 2
        row = await manager.aget_ingest_job("job-1")
        assert row is not None
        assert row["processed_items"] == 129
        assert row["result"]["processed"] == 129

    async def test_recover_url_ingest_job(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        row = {
            "job_id": "job-1",
            "workspace": "project_a",
            "source_type": "url",
            "status": "running",
            "request": {
                "workspace": "project_a",
                "source_type": "url",
                "kwargs": {
                    "url": "https://api.bynder.com/docs/getting-started",
                    "filename": "getting-started.html",
                },
            },
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "current_window": 0,
            "errors": [],
            "result": {},
        }
        store.rows["job-1"] = dict(row)
        manager._ingest_jobs._store = store
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        assert manager._ingest_jobs.schedule_recovered_job(row, store) is True
        await asyncio.wait_for(manager._ingest_jobs._tasks["job-1"], timeout=1.0)

        svc.aingest.assert_awaited_once()
        assert svc.aingest.await_args.kwargs["source_type"] == "url"
        assert svc.aingest.await_args.kwargs["url"] == (
            "https://api.bynder.com/docs/getting-started"
        )
        assert svc.aingest.await_args.kwargs["filename"] == "getting-started.html"

    async def test_recovered_job_does_not_run_when_database_claim_is_lost(self, test_cfg) -> None:
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
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "current_window": 0,
            "errors": [],
            "result": {},
        }
        store.rows["job-1"] = dict(row)
        store.claim_results["job-1"] = False
        manager._ingest_jobs._store = store
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        assert manager._ingest_jobs.schedule_recovered_job(row, store) is True
        await asyncio.wait_for(manager._ingest_jobs._tasks["job-1"], timeout=1.0)

        manager._get_service.assert_not_awaited()
        svc.aingest.assert_not_awaited()

    async def test_astart_ingest_job_records_progress_and_result(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store

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
            IngestSpec(source_type="s3", bucket="bucket", prefix="docs/"),
        )
        task = manager._ingest_jobs._tasks[job["job_id"]]

        await asyncio.wait_for(task, timeout=1.0)
        row = await manager.aget_ingest_job(job["job_id"])

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

    async def test_upload_batch_local_ingest_cleanup_is_durable_job_metadata(
        self, test_cfg
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store
        staged_dir = test_cfg.input_dir_path / "project_a" / "__uploads__" / "batch-1"
        staged_dir.mkdir(parents=True)
        (staged_dir / "report.pdf").write_text("pdf", encoding="utf-8")
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        job = await manager.astart_ingest_job(
            "Project A",
            IngestSpec(source_type="local", path=str(staged_dir)),
        )
        await asyncio.wait_for(manager._ingest_jobs._tasks[job["job_id"]], timeout=1.0)
        row = await manager.aget_ingest_job(job["job_id"])

        assert row is not None
        assert row["request"]["kwargs"] == {"path": str(staged_dir)}
        assert row["request"]["cleanup_paths"] == [str(staged_dir)]
        assert svc.aingest.await_args.kwargs["path"] == str(staged_dir)
        assert "_cleanup_paths" not in svc.aingest.await_args.kwargs
        assert "cleanup_paths" not in svc.aingest.await_args.kwargs
        assert not staged_dir.exists()

    async def test_regular_local_ingest_source_is_not_cleanup_path(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store
        source_file = test_cfg.input_dir_path / "project_a" / "report.pdf"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("pdf", encoding="utf-8")
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        job = await manager.astart_ingest_job(
            "Project A",
            IngestSpec(source_type="local", path=str(source_file)),
        )
        await asyncio.wait_for(manager._ingest_jobs._tasks[job["job_id"]], timeout=1.0)
        row = await manager.aget_ingest_job(job["job_id"])

        assert row is not None
        assert "cleanup_paths" not in row["request"]
        assert source_file.exists()

    async def test_recovered_upload_batch_ingest_cleans_durable_cleanup_paths(
        self, test_cfg
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        staged_dir = test_cfg.input_dir_path / "project_a" / "__uploads__" / "batch-1"
        staged_dir.mkdir(parents=True)
        (staged_dir / "report.pdf").write_text("pdf", encoding="utf-8")
        row = {
            "job_id": "job-1",
            "workspace": "project_a",
            "source_type": "local",
            "status": "running",
            "request": {
                "workspace": "project_a",
                "source_type": "local",
                "kwargs": {"path": str(staged_dir)},
                "cleanup_paths": [str(staged_dir)],
            },
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "current_window": 0,
            "errors": [],
            "result": {},
        }
        store.rows["job-1"] = dict(row)
        manager._ingest_jobs._store = store
        svc = AsyncMock()
        svc.aingest = AsyncMock(return_value={"processed": 1, "errors": []})
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        assert manager._ingest_jobs.schedule_recovered_job(row, store) is True
        await asyncio.wait_for(manager._ingest_jobs._tasks["job-1"], timeout=1.0)

        assert not staged_dir.exists()
        assert "cleanup_paths" not in svc.aingest.await_args.kwargs

    async def test_aingest_timeout_returns_running_job_without_cancelling_task(
        self, test_cfg
    ) -> None:
        cfg = test_cfg.model_copy(update={"ingest_timeout": 0.01})
        manager = RAGServiceManager(config=cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store
        release = asyncio.Event()

        async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
            await release.wait()
            return {"doc_id": "d1"}

        svc = AsyncMock()
        svc.aingest = AsyncMock(side_effect=fake_ingest)
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        result = await manager.aingest(
            "default",
            IngestSpec(source_type="local", path="/tmp/slow.pdf"),
        )

        assert result["status"] in {"queued", "running"}
        assert result["job_id"] in manager._ingest_jobs._tasks
        task = manager._ingest_jobs._tasks[result["job_id"]]
        assert not task.done()

        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        row = await manager.aget_ingest_job(result["job_id"])
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["result"] == {"doc_id": "d1"}

    async def test_manager_close_leaves_running_ingest_job_recoverable(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store
        started = asyncio.Event()

        async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
            started.set()
            await asyncio.Event().wait()
            return {"doc_id": "d1"}

        svc = AsyncMock()
        svc.aingest = AsyncMock(side_effect=fake_ingest)
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        job = await manager.astart_ingest_job(
            "default",
            IngestSpec(source_type="local", path="/tmp/slow.pdf"),
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await manager.aclose()

        row = await store.get(job["job_id"])
        assert row is not None
        assert row["status"] == "running"
        assert row["errors"] == []

    async def test_manager_close_keeps_upload_batch_files_for_recovery(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        store = _InMemoryIngestJobStore()
        manager._ingest_jobs._store = store
        staged_dir = test_cfg.input_dir_path / "default" / "__uploads__" / "batch-1"
        staged_dir.mkdir(parents=True)
        (staged_dir / "report.pdf").write_text("pdf", encoding="utf-8")
        started = asyncio.Event()

        async def fake_ingest(**kwargs: Any) -> dict[str, Any]:
            started.set()
            await asyncio.Event().wait()
            return {"doc_id": "d1"}

        svc = AsyncMock()
        svc.aingest = AsyncMock(side_effect=fake_ingest)
        manager._get_service = AsyncMock(return_value=svc)  # type: ignore[method-assign]

        await manager.astart_ingest_job(
            "default",
            IngestSpec(source_type="local", path=str(staged_dir)),
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await manager.aclose()

        assert staged_dir.exists()
        assert (staged_dir / "report.pdf").exists()


async def test_vision_probe_result_is_manager_scoped(
    monkeypatch: pytest.MonkeyPatch,
    test_cfg: DlightragConfig,
) -> None:
    test_cfg = test_cfg.model_copy(update={"rerank": RerankConfig(strategy="chat_llm_reranker")})
    first = RAGServiceManager(config=test_cfg)
    first._supports_vision = False
    second = RAGServiceManager(config=test_cfg)
    provider = SimpleNamespace(aclose=AsyncMock())
    probe = AsyncMock(return_value=True)

    monkeypatch.setattr("dlightrag.models.providers.get_provider", MagicMock(return_value=provider))
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_vision_support", probe)

    await second._probe_vision_support()

    assert first._supports_vision is False
    assert second._supports_vision is True
    assert second._rerank_supports_vision is True
    probe.assert_awaited_once()


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
        for name in (
            "_start_ingest_job_recovery",
            "_recover_stalled_docs",
            "_prune_checkpoint_sessions",
            "_probe_vision_support",
        ):
            monkeypatch.setattr(RAGServiceManager, name, AsyncMock())

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_create_sets_ready_on_success(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = await RAGServiceManager.acreate(config=test_cfg)
        assert manager.is_ready()
        assert not manager.is_degraded()
        # Warnings may include "Workspace registry unavailable" in tests
        # without a running PostgreSQL — that's expected and non-fatal.

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_create_sets_degraded_on_failure(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = await RAGServiceManager.acreate(config=test_cfg)
        assert not manager.is_ready()
        assert manager.is_degraded()
        assert any("DB down" in w for w in manager.get_warnings())

    async def test_create_warms_default_workspace_only(
        self, monkeypatch: pytest.MonkeyPatch, test_cfg
    ) -> None:
        cfg = test_cfg.model_copy(update={"max_async": 3})
        created: list[str] = []
        recovered: list[str] = []

        async def fake_initialize_workspace_registry(self):  # noqa: ANN001, ANN202
            return None

        async def fake_list_all_workspaces(self):  # noqa: ANN001, ANN202
            return ["default", "alpha", "beta"]

        async def fake_get_service(self, workspace: str):  # noqa: ANN001, ANN202
            created.append(workspace)
            self._services[workspace] = workspace
            return workspace

        async def fake_recover_stalled_docs(self, workspaces: list[str]):  # noqa: ANN001, ANN202
            recovered.extend(workspaces)

        monkeypatch.setattr(
            RAGServiceManager,
            "_initialize_workspace_registry",
            fake_initialize_workspace_registry,
        )
        monkeypatch.setattr(RAGServiceManager, "_list_all_workspaces", fake_list_all_workspaces)
        monkeypatch.setattr(RAGServiceManager, "_get_service", fake_get_service)
        monkeypatch.setattr(RAGServiceManager, "_recover_stalled_docs", fake_recover_stalled_docs)
        monkeypatch.setattr("dlightrag.observability.init_tracing", lambda config: None)

        manager = await RAGServiceManager.acreate(config=cfg)

        assert manager.is_ready()
        assert created == ["default"]
        assert recovered == ["default", "alpha", "beta"]


class TestActionableErrors:
    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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
    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
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
        await manager.aclose()
        svc_a.aclose.assert_awaited_once()
        svc_b.aclose.assert_awaited_once()
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

        result = await manager.alist_workspaces()

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

        records = await manager.alist_workspace_records()

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

        result = await manager.alist_workspaces()

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
