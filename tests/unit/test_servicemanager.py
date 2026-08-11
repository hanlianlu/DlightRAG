# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

import asyncio
import importlib
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from dlightrag.citations.schemas import ChunkSnippet, SourceReference
from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelConfig,
    RerankConfig,
    WebSearchConfig,
    set_config,
)
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.request.images import prepare_query_images
from dlightrag.core.request.planner import QueryPlan, QueryPlanner
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError
from dlightrag.sourcing.base import SourceDocument


def _image_block(url: str = "data:image/png;base64,abc") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (20, 80, 160)).save(output, "PNG")
    return output.getvalue()


class _AttrStream:
    """Async token stream that accepts dynamic ``trace``/``answer`` attributes."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.trace: dict[str, Any] = {}
        self.image_descriptions: list[str] = []
        self.answer = "".join(tokens)

    def __aiter__(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for token in self._tokens:
                yield token

        return _gen()


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


class _CapturingOrchestrator:
    """Stand-in for AnswerOrchestrator that records how the manager drives it."""

    last: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        _CapturingOrchestrator.last = {"init": kwargs}

    @property
    def uses_research_path(self) -> bool:
        return (
            bool(self.kwargs.get("resource_manifest")) or self.kwargs.get("search_web") is not None
        )

    async def answer(self, query: str, **kwargs: Any) -> Any:
        _CapturingOrchestrator.last["answer"] = {"query": query, **kwargs}
        return RetrievalResult(answer="ok", contexts={"chunks": []})

    async def answer_stream(self, query: str, **kwargs: Any) -> Any:
        _CapturingOrchestrator.last["answer_stream"] = {"query": query, **kwargs}
        return {"chunks": []}, None


async def test_prepared_stream_keeps_server_history_internal(
    test_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_turn = importlib.import_module("dlightrag.core.answer.turn")
    turn = answer_turn.PreparedAnswerTurn(
        current_query="Follow up",
        retrieval_query="Standalone follow up",
        text_history=({"role": "user", "content": "Earlier"},),
    )
    manager = RAGServiceManager(config=test_cfg)
    manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            QueryPlan(original_query="Follow up", standalone_query="Standalone follow up"),
            SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
        )
    )
    monkeypatch.setattr("dlightrag.core.servicemanager.AnswerOrchestrator", _CapturingOrchestrator)

    await manager._aanswer_stream_prepared(turn, workspaces=["default"])

    assert _CapturingOrchestrator.last["answer_stream"]["conversation_history"].messages == [
        {"role": "user", "content": "Earlier"}
    ]


async def test_private_planner_helper_hands_prepared_history_to_planner(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    planner = AsyncMock()
    planner.plan.return_value = QueryPlan(
        original_query="follow up",
        standalone_query="standalone",
    )
    manager._query_planner = planner
    manager._get_schema = AsyncMock(return_value={})  # type: ignore[method-assign]
    history = PriorTurns([{"role": "user", "content": "Earlier"}])

    await manager._aplan_query_prepared(
        "follow up",
        text_history=history,
        workspaces=["default"],
    )

    assert planner.plan.await_args.kwargs["conversation_history"] is history


async def test_request_scope_starts_workspace_warmup_before_planning(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    warm_started = asyncio.Event()
    release_warm = asyncio.Event()

    async def get_service(_workspace: str) -> AsyncMock:
        warm_started.set()
        await release_warm.wait()
        return AsyncMock()

    async def plan_query(*_args: object, **_kwargs: object) -> QueryPlan:
        await warm_started.wait()
        return QueryPlan(original_query="query", standalone_query="planned")

    manager._get_service = AsyncMock(side_effect=get_service)  # type: ignore[method-assign]
    describer = AsyncMock()
    describer.describe.return_value = {}
    manager._aget_query_image_describer = AsyncMock(  # type: ignore[method-assign]
        return_value=describer
    )
    manager._aplan_query_prepared = AsyncMock(  # type: ignore[method-assign]
        side_effect=plan_query
    )

    manager._start_query_service_warmup(["reports"])
    try:
        plan, _prepared = await manager._describe_and_plan(
            "query",
            text_history=None,
            query_images=None,
            ws_list=["reports"],
        )
    finally:
        release_warm.set()

    # Planning completed while the workspace was still initializing.
    assert plan.standalone_query == "planned"
    manager._get_service.assert_awaited_once_with("reports")
    await asyncio.gather(*manager._warmups, return_exceptions=True)


async def test_warmup_skips_workspaces_that_already_have_a_service(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    manager._services["reports"] = cast(Any, AsyncMock())
    manager._get_service = AsyncMock()  # type: ignore[method-assign]

    manager._start_query_service_warmup(["reports"])

    assert manager._warmups == set()
    manager._get_service.assert_not_awaited()


async def test_warmup_failure_is_observed_instead_of_escaping_the_task(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    manager._get_service = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("workspace failed")
    )

    manager._start_query_service_warmup(["reports"])
    task = next(iter(manager._warmups))
    await asyncio.gather(task, return_exceptions=True)

    assert manager._warmups == set()
    assert isinstance(task.exception(), RuntimeError)


async def test_query_workspace_warmup_cancels_siblings_on_failure(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    expected = RuntimeError("workspace failed")

    async def get_service(workspace: str) -> None:
        if workspace == "failed":
            await sibling_started.wait()
            raise expected
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    manager._get_service = AsyncMock(side_effect=get_service)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as raised:
        await manager._warm_query_services(["failed", "sibling"])

    assert raised.value is expected
    assert sibling_cancelled.is_set()


async def test_private_generation_helper_hands_prepared_history_to_engine(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    engine = AsyncMock()
    engine.generate_stream.return_value = ({"chunks": []}, None)
    manager._answer_synthesizer = engine
    manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            QueryPlan(original_query="follow up", standalone_query="follow up"),
            SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
        )
    )
    manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
        return_value=RetrievalResult(contexts={"chunks": []})
    )
    history = [{"role": "user", "content": "Earlier"}]

    await manager.aanswer_stream("follow up", workspaces=["default"], history=history)

    assert engine.generate_stream.await_args.kwargs["conversation_history"].messages == history


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
        assert "ws_a" in manager._backoff

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

    @pytest.fixture(autouse=True)
    def _stub_planning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # aretrieve now always plans; stub the describe+plan helper so routing
        # tests do not invoke a real planner LLM.
        async def _fake(self_: object, query: str, **_kwargs: object) -> tuple[object, object]:
            return (
                QueryPlan(original_query=query, standalone_query=query),
                SimpleNamespace(descriptions=[]),
            )

        monkeypatch.setattr(RAGServiceManager, "_describe_and_plan", _fake)

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
        manager.alist_workspaces = AsyncMock(return_value=["finance"])

        with pytest.raises(ValueError, match="all_workspaces"):
            await getattr(manager, method_name)(
                "query",
                all_workspaces=True,
                **explicit_selection,
            )
        manager.alist_workspaces.assert_not_awaited()

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

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aretrieve_forwards_bm25_query(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspace="ws_a", bm25_query="alpha beta")
        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["bm25_query"] == "alpha beta"

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aretrieve_threads_query_images_to_backend(self, mock_create, test_cfg) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock()
        mock_create.return_value = mock_svc
        blocks = [_image_block()]
        manager = RAGServiceManager(config=test_cfg)
        await manager.aretrieve("query", workspace="ws_a", query_images=blocks)
        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["query_image_blocks"] == blocks

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aretrieve_rejects_images_beyond_the_runtime_limit(
        self, mock_create, test_cfg
    ) -> None:
        from dlightrag.core.answer.errors import CurrentImagePayloadError

        test_cfg.query_images.max_current_images = 1
        manager = RAGServiceManager(config=test_cfg)

        with pytest.raises(CurrentImagePayloadError, match="at most 1 current images"):
            await manager.aretrieve(
                "query",
                workspace="ws_a",
                query_images=[_image_block(), _image_block()],
            )
            mock_create.assert_not_awaited()

    async def test_query_images_are_current_request_only(self, test_cfg) -> None:
        describer = AsyncMock()
        describer.describe = AsyncMock(return_value={"1": "Image 1: chart"})
        current = [_image_block()]

        prepared = await prepare_query_images(
            query_images=current,
            describer=describer,
        )

        assert prepared.descriptions == ["Image 1: chart"]
        assert prepared.descriptions_by_ordinal == {"1": "Image 1: chart"}
        describer.describe.assert_awaited_once_with(current)

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_calls_aretrieve_then_engine(self, mock_create, test_cfg) -> None:
        """aanswer() routes through aretrieve() then AnswerSynthesizer.generate()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        mock_engine.generate.return_value = RetrievalResult(answer="a", contexts=mock_contexts)

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        await manager.aanswer("query", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )


class TestAnswerViaEngine:
    """aanswer and aanswer_stream route through AnswerSynthesizer."""

    @pytest.fixture(autouse=True)
    def _stub_planning_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dlightrag.storage.pg_metadata_index.PGMetadataIndex.get_field_schema",
            AsyncMock(return_value={"columns": [], "custom_keys": []}),
        )

        async def warm_query_services(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(
            RAGServiceManager,
            "_warm_query_services",
            warm_query_services,
        )

    async def test_aplan_query_emits_query_planning_observation(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def llm_func(*, messages, **kwargs) -> str:
            return (
                '{"standalone_query":"rewritten query","bm25_query":"rewritten query",'
                '"filters":{},'
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

        plan = await manager._aplan_query_prepared(
            "raw query", text_history=None, workspaces=["ws_a"]
        )

        assert plan.standalone_query == "rewritten query"
        assert trace_calls == [
            {
                "name": "query_planning",
                "as_type": "chain",
                "input": {"query": "raw query"},
                "metadata": {
                    "workspaces": ["ws_a"],
                    "history_messages": 0,
                },
                "updates": [
                    {
                        "output": {
                            "standalone_query": "rewritten query",
                            "has_metadata_filter": False,
                            "planner_outcome": "planned",
                        }
                    }
                ],
            }
        ]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_calls_retrieve_then_engine(
        self, mock_create, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aanswer() calls aretrieve() then AnswerSynthesizer.generate()."""
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
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("what is X?", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "what is X?",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result is expected_result
        retrieve = next(call for call in trace_calls if call["name"] == "retrieve")
        orchestration = next(call for call in trace_calls if call["name"] == "answer_orchestration")
        assert retrieve["input"] == {"query": "what is X?"}
        assert "query" not in retrieve["metadata"]
        assert retrieve["metadata"] == {
            "workspaces": ["ws_a"],
            "top_k": test_cfg.top_k,
            "chunk_top_k": test_cfg.chunk_top_k,
            "has_filters": False,
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
        assert orchestration["input"] == {"query": "what is X?"}
        assert "query" not in orchestration["metadata"]
        assert orchestration["metadata"]["stream"] is False
        assert orchestration["updates"] == [
            {
                "output": {
                    "answer_len": len(answer_text),
                    "source_count": 0,
                    "context_chunk_count": 0,
                    "answer": answer_text,
                }
            }
        ]

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_derives_candidate_and_context_limits(
        self, mock_create, test_cfg
    ) -> None:
        """Answer over-fetches retrieval candidates for the final prompt."""
        cfg = test_cfg
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        expected_result = RetrievalResult(answer="a", contexts=mock_contexts)
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=cfg)
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", workspace="ws_a")

        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == test_cfg.chunk_top_k
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result is expected_result

    @patch("dlightrag.core.servicemanager.RAGService.acreate", new_callable=AsyncMock)
    async def test_aanswer_uses_chunk_top_k_as_candidate_override(
        self, mock_create, test_cfg
    ) -> None:
        """Answer chunk_top_k remains the explicit retrieval candidate override."""
        cfg = test_cfg
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        expected_result = RetrievalResult(answer="a", contexts=mock_contexts)
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=cfg)
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", workspace="ws_a", chunk_top_k=7)

        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == 7
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result is expected_result

    async def test_aanswer_threads_history_to_planning_and_generation(self, test_cfg) -> None:
        """Caller history reaches retrieval planning and answer generation."""
        manager = RAGServiceManager(config=test_cfg)
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="follow up", standalone_query="standalone"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts={"chunks": []})
        )
        mock_engine = AsyncMock()
        mock_engine.generate.return_value = RetrievalResult(answer="a", contexts={"chunks": []})
        manager._answer_synthesizer = mock_engine
        history = [{"role": "user", "content": "Earlier"}]

        await manager.aanswer("follow up", workspace="ws_a", history=history)

        plan_call = manager._describe_and_plan.await_args
        assert plan_call is not None
        assert plan_call.kwargs["text_history"].messages == history
        generate_call = mock_engine.generate.await_args
        assert generate_call is not None
        assert generate_call.kwargs["conversation_history"].messages == history

    async def test_aanswer_stream_threads_history_to_prepared_turn(self, test_cfg) -> None:
        """Streaming answer carries caller history on the prepared turn."""
        manager = RAGServiceManager(config=test_cfg)
        manager._aanswer_stream_prepared = AsyncMock(  # type: ignore[method-assign]
            return_value=({"chunks": []}, None)
        )
        history = [{"role": "user", "content": "Earlier"}]

        await manager.aanswer_stream("follow up", workspaces=["ws_a"], history=history)

        stream_call = manager._aanswer_stream_prepared.await_args
        assert stream_call is not None
        turn = stream_call.args[0]
        assert turn.current_query == "follow up"
        assert turn.text_history == ({"role": "user", "content": "Earlier"},)

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
            return '{"items": [{"id": "0", "phrases": ["market growth"], "confidence": 1.0}]}'

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
                        source_uri="local://ws_a/report.pdf",
                        workspace="ws_a",
                        document_id="doc-report",
                        download_locator="/docs/report.pdf",
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
        manager._answer_synthesizer = AsyncMock()
        manager._answer_synthesizer.generate.side_effect = [_answer_result(), _answer_result()]
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
        """aanswer_stream() calls aretrieve() then AnswerSynthesizer.generate_stream()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_retrieval = MagicMock(contexts=mock_contexts)
        mock_svc.aretrieve.return_value = mock_retrieval
        mock_create.return_value = mock_svc

        mock_engine = AsyncMock()
        mock_stream = AsyncMock()
        mock_engine.generate_stream.return_value = (mock_contexts, mock_stream)

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        contexts, stream = await manager.aanswer_stream("what is X?", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "what is X?",
            mock_contexts,
            conversation_history=ANY,
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
        mock_fed_retrieve.return_value = MagicMock(contexts=mock_contexts, trace={})

        mock_engine = AsyncMock()
        expected_result = RetrievalResult(answer="a", contexts=mock_contexts)
        mock_engine.generate.return_value = expected_result

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        result = await manager.aanswer("query", workspaces=["ws_a", "ws_b"])
        mock_fed_retrieve.assert_awaited_once()
        mock_engine.generate.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
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
        manager._get_schema = AsyncMock(return_value={})  # type: ignore[method-assign]
        manager._answer_synthesizer = engine
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
        manager._answer_synthesizer = mock_engine
        manager._query_planner = QueryPlanner(llm_func=None)

        contexts, stream = await manager.aanswer_stream("query", workspaces=["ws_a", "ws_b"])
        mock_fed_retrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
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
        manager._get_schema = AsyncMock(return_value={})  # type: ignore[method-assign]
        manager._answer_synthesizer = engine
        manager._query_planner = QueryPlanner(llm_func=None)

        resolved_contexts, stream = await manager.aanswer_stream(
            "query",
            all_workspaces=True,
        )

        manager.alist_workspaces.assert_awaited_once()
        assert mock_federated.await_args.args[1] == ["ws_a", "ws_b"]
        assert resolved_contexts is contexts
        assert stream is not None

    def test_get_answer_synthesizer_lazy_creates(self, test_cfg) -> None:
        """_get_answer_synthesizer() lazily creates an AnswerSynthesizer instance."""
        manager = RAGServiceManager(config=test_cfg)
        assert manager._answer_synthesizer is None
        with patch("dlightrag.models.llm.get_query_model_func") as mock_llm:
            mock_llm.return_value = MagicMock()
            engine = manager._get_answer_synthesizer()
            assert engine is not None
            # Second call returns same instance
            engine2 = manager._get_answer_synthesizer()
            assert engine2 is engine

    def test_get_answer_synthesizer_threads_pixel_limit_to_one_image_budget(
        self,
        test_cfg,
        monkeypatch,
    ) -> None:
        test_cfg.answer.image_max_pixels = 123
        manager = RAGServiceManager(config=test_cfg)

        with patch("dlightrag.models.llm.get_query_model_func", return_value=MagicMock()):
            engine = manager._get_answer_synthesizer()

        budgets = []
        new_image_budget = engine._new_image_budget

        def capture_image_budget():
            budget = new_image_budget()
            budgets.append(budget)
            return budget

        monkeypatch.setattr(engine, "_new_image_budget", capture_image_budget)

        engine._prepare_model_call(
            "query",
            {"chunks": [], "entities": [], "relationships": []},
        )

        # One evidence image budget carries the configured pixel ceiling; there
        # are no separate composer/rag lanes.
        assert len(budgets) == 1
        assert budgets[0].max_pixels == 123

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
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="q", standalone_query="q"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts=contexts)
        )

        mock_engine = AsyncMock()
        mock_engine.generate_stream = AsyncMock(return_value=(contexts, _AttrStream(["token"])))
        manager._answer_synthesizer = mock_engine

        _, first_stream = await manager.aanswer_stream("q1", workspace="ws_a")
        second = asyncio.create_task(manager.aanswer_stream("q2", workspace="ws_a"))
        await asyncio.sleep(0)

        assert not second.done()
        assert first_stream is not None
        async for _ in first_stream:
            pass

        _, second_stream = await asyncio.wait_for(second, timeout=1.0)
        assert second_stream is not None

    async def test_a_saturated_service_says_so_instead_of_queueing_forever(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"max_async": 1, "answer_acquire_timeout": 0.01})
        manager = RAGServiceManager(config=cfg)
        contexts = {"chunks": [], "entities": [], "relationships": []}
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="q", standalone_query="q"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts=contexts)
        )

        mock_engine = AsyncMock()
        mock_engine.generate_stream = AsyncMock(return_value=(contexts, _AttrStream(["token"])))
        manager._answer_synthesizer = mock_engine

        await manager.aanswer_stream("q1", workspace="ws_a")

        with pytest.raises(RAGServiceUnavailableError):
            await manager.aanswer_stream("q2", workspace="ws_a")


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
        download_uri_for_key = lambda key: f"https://cdn.example.com/{key}"  # noqa: E731

        result = await manager.aingest_source(
            "ws_a",
            source,
            source_type="bynder",
            documents=[SourceDocument(key="asset.pdf")],
            source_uri_for_key=lambda key: f"bynder://assets/{key}",
            download_uri_for_key=download_uri_for_key,
            retain_source_file=True,
        )

        assert result == {"processed": 1}
        mock_svc.aingest_source.assert_awaited_once()
        assert mock_svc.aingest_source.await_args.kwargs["documents"] == [
            SourceDocument(key="asset.pdf")
        ]
        assert (
            mock_svc.aingest_source.await_args.kwargs["download_uri_for_key"]
            is download_uri_for_key
        )
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

    async def test_source_download_does_not_initialize_cold_workspace(self, test_cfg) -> None:
        from dlightrag.core.source_download import RedirectDownloadTarget

        manager = RAGServiceManager(config=test_cfg)
        manager._get_service = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("source download must not initialize services")
        )
        metadata_index = AsyncMock()
        service = AsyncMock()
        target = RedirectDownloadTarget(url="https://cdn.example.com/report.pdf")
        service.prepare.return_value = target

        with (
            patch(
                "dlightrag.storage.pg_metadata_index.PGMetadataIndex",
                return_value=metadata_index,
            ) as index_type,
            patch(
                "dlightrag.core.source_download.SourceDownloadService",
                return_value=service,
            ) as service_type,
        ):
            result = await manager.aprepare_source_download("Finance-Team", "doc-1")

        assert result is target
        index_type.assert_called_once_with(workspace="finance_team")
        service_type.assert_called_once_with(
            config=test_cfg,
            metadata_index=metadata_index,
            workspace="finance_team",
        )
        service.prepare.assert_awaited_once_with("doc-1")
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
    model_kwargs = {"reasoning_effort": "none"}
    test_cfg = test_cfg.model_copy(
        update={
            "llm": LLMConfig(
                default=ModelConfig(
                    model="gpt-5.4-mini",
                    api_key="test",
                    model_kwargs=model_kwargs,
                )
            ),
            "rerank": RerankConfig(strategy="chat_llm_reranker"),
        }
    )
    from dlightrag.core.vision_probe import ImageProbeOutcome

    first = RAGServiceManager(config=test_cfg)
    first._rerank_supports_vision = False
    second = RAGServiceManager(config=test_cfg)
    provider = SimpleNamespace(aclose=AsyncMock())
    probe = AsyncMock(return_value=ImageProbeOutcome(status="supported"))

    monkeypatch.setattr("dlightrag.models.providers.get_provider", MagicMock(return_value=provider))
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_image_capability", probe)

    await second._probe_vision_support()

    assert first._rerank_supports_vision is False
    assert second._rerank_supports_vision is True
    probe.assert_awaited_once_with(
        provider, model="gpt-5.4-mini", ceiling=1, model_kwargs=model_kwargs
    )


async def test_rerank_vision_probe_does_not_borrow_default_key(
    monkeypatch: pytest.MonkeyPatch,
    test_cfg: DlightragConfig,
) -> None:
    from dlightrag.core.vision_probe import ImageProbeOutcome

    config = test_cfg.model_copy(
        update={
            "llm": LLMConfig(default=ModelConfig(model="default-model", api_key="default-key")),
            "rerank": RerankConfig(
                strategy="chat_llm_reranker",
                provider="openai",
                model="local-reranker",
                api_key=None,
                base_url="http://host.docker.internal:9999/v1",
            ),
        }
    )
    manager = RAGServiceManager(config=config)
    provider = SimpleNamespace(aclose=AsyncMock())
    provider_factory = MagicMock(return_value=provider)

    monkeypatch.setattr("dlightrag.models.providers.get_provider", provider_factory)
    monkeypatch.setattr(
        "dlightrag.core.vision_probe.probe_image_capability",
        AsyncMock(return_value=ImageProbeOutcome(status="supported")),
    )

    await manager._probe_vision_support()

    provider_factory.assert_called_once_with(
        "openai",
        api_key=None,
        base_url="http://host.docker.internal:9999/v1",
        timeout=240.0,
        max_retries=3,
    )


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
    async def test_create_eagerly_initializes_query_planner(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()

        manager = await RAGServiceManager.acreate(config=test_cfg)

        assert manager._query_planner is not None

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

        async def fake_initialize_workspace_registry(self):  # noqa: ANN001, ANN202
            return None

        async def fake_get_service(self, workspace: str):  # noqa: ANN001, ANN202
            created.append(workspace)
            self._services[workspace] = workspace
            return workspace

        monkeypatch.setattr(
            RAGServiceManager,
            "_initialize_workspace_registry",
            fake_initialize_workspace_registry,
        )
        monkeypatch.setattr(RAGServiceManager, "_get_service", fake_get_service)
        monkeypatch.setattr("dlightrag.observability.init_tracing", lambda config: None)

        manager = await RAGServiceManager.acreate(config=cfg)

        assert manager.is_ready()
        assert created == ["default"]


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
        manager._get_schema = AsyncMock(return_value={})  # type: ignore[method-assign]
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
    async def test_aplan_query_uses_schema_for_requested_workspace(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        schemas = {
            ("reports",): {
                "columns": [{"name": "filename", "type": "character varying"}],
                "custom_keys": ["department"],
            },
            ("legal",): {
                "columns": [{"name": "filename", "type": "character varying"}],
                "custom_keys": ["jurisdiction"],
            },
        }

        async def get_field_schema(_index, *, workspaces):  # noqa: ANN001, ANN202
            return schemas[workspaces]

        monkeypatch.setattr(
            "dlightrag.storage.pg_metadata_index.PGMetadataIndex.get_field_schema",
            get_field_schema,
        )

        llm = AsyncMock(return_value='{"standalone_query": "q", "filters": {}}')
        manager._query_planner = QueryPlanner(llm_func=llm)

        await manager._aplan_query_prepared("q", text_history=None, workspaces=["reports"])
        await manager._aplan_query_prepared("q", text_history=None, workspaces=["legal"])

        first_payload = json.loads(llm.await_args_list[0].kwargs["messages"][1]["content"])
        second_payload = json.loads(llm.await_args_list[1].kwargs["messages"][1]["content"])
        assert "department" in first_payload["metadata_schema"]
        assert "jurisdiction" not in first_payload["metadata_schema"]
        assert "jurisdiction" in second_payload["metadata_schema"]
        assert "department" not in second_payload["metadata_schema"]

    async def test_partial_schema_failure_does_not_poison_cache(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        get_field_schema = AsyncMock(return_value={"columns": [], "custom_keys": ["department"]})
        monkeypatch.setattr(
            "dlightrag.storage.pg_metadata_index.PGMetadataIndex.get_field_schema",
            get_field_schema,
        )

        # A transient lookup failure must not be cached as a degraded schema.
        get_field_schema.side_effect = RuntimeError("db down")
        degraded = await manager._get_schema(["reports"])
        assert degraded == {}
        assert manager._schema_cache == {}

        # The next call retries and caches the recovered schema.
        get_field_schema.side_effect = None
        recovered = await manager._get_schema(["reports"])
        assert recovered["custom_keys"] == ["department"]
        assert ("reports",) in manager._schema_cache

    async def test_cold_workspace_schema_uses_one_set_query_without_service_warmup(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._get_service = AsyncMock(side_effect=AssertionError("must not warm services"))
        get_field_schema = AsyncMock(
            return_value={"columns": [], "custom_keys": ["department", "jurisdiction"]}
        )
        monkeypatch.setattr(
            "dlightrag.storage.pg_metadata_index.PGMetadataIndex.get_field_schema",
            get_field_schema,
        )

        schema = await manager._get_schema(["reports", "legal"])

        assert schema["custom_keys"] == ["department", "jurisdiction"]
        get_field_schema.assert_awaited_once_with(workspaces=("legal", "reports"))
        manager._get_service.assert_not_awaited()
        assert ("legal", "reports") in manager._schema_cache

    async def test_schema_cache_key_ignores_workspace_order(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        get_field_schema = AsyncMock(return_value={"columns": [], "custom_keys": []})
        monkeypatch.setattr(
            "dlightrag.storage.pg_metadata_index.PGMetadataIndex.get_field_schema",
            get_field_schema,
        )

        first = await manager._get_schema(["reports", "legal"])
        second = await manager._get_schema(["legal", "reports"])

        assert second is first
        get_field_schema.assert_awaited_once()


class TestWebSearchCapability:
    """A key present is the capability; without one the path does not exist."""

    def test_without_a_key_there_is_no_web_search_to_reach(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)

        assert manager._get_web_search() is None

    def test_with_a_key_one_client_is_shared_by_every_turn(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)

        assert manager._get_web_search() is not None
        assert manager._get_web_search() is manager._get_web_search()

    async def test_closing_the_manager_closes_the_web_client(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        search = manager._get_web_search()
        assert search is not None

        await manager.aclose()

        assert search._client.is_closed


class TestExaContentsFallback:
    """Manager composition root adapts Exa Contents into the registry fallback."""

    async def test_contents_passages_become_one_deterministic_text(self) -> None:
        from dlightrag.core.retrieval.web_search import (
            ExaSearch,
            WebSearchHit,
            WebSearchResult,
        )
        from dlightrag.core.servicemanager import _exa_contents_text

        class _FakeExa:
            def __init__(self) -> None:
                self.urls: list[str] = []

            async def contents(self, url: str) -> WebSearchResult:
                self.urls.append(url)
                page = {"url": url, "title": "The Page"}
                return WebSearchResult(
                    hits=(
                        WebSearchHit(text="first passage", **page),
                        WebSearchHit(text="second passage", **page),
                    ),
                    cost_dollars=0.0,
                )

        exa = _FakeExa()
        fallback = _exa_contents_text(cast(ExaSearch, exa))

        text = await fallback("https://example.org/page")

        assert exa.urls == ["https://example.org/page"]
        assert text is not None
        assert "first passage" in text
        assert "second passage" in text
        assert "The Page" in text

    async def test_contents_unavailable_yields_no_text(self) -> None:
        from dlightrag.core.retrieval.web_search import ExaSearch, WebSearchUnavailable
        from dlightrag.core.servicemanager import _exa_contents_text

        class _FakeExa:
            async def contents(self, url: str):
                raise WebSearchUnavailable("timeout")

        fallback = _exa_contents_text(cast(ExaSearch, _FakeExa()))

        assert await fallback("https://example.org/page") is None

    def test_registry_receives_fallback_only_when_web_search_present(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        resources = [ResourceInput(content=b"payload")]

        registry, _tools = manager._build_resource_context(
            resources, web_search=manager._get_web_search()
        )
        assert registry is not None
        assert registry._url_text_fallback is not None

        plain = RAGServiceManager(config=test_cfg)
        registry2, _t2 = plain._build_resource_context(resources, web_search=None)
        assert registry2 is not None
        assert registry2._url_text_fallback is None


class TestAgenticAnswerCapability:
    async def test_without_exa_fast_path_never_builds_a_tool_model(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = MagicMock(side_effect=AssertionError("tool model must stay absent"))
        monkeypatch.setattr("dlightrag.models.tool_model.create_query_tool_model", create)
        manager = RAGServiceManager(config=test_cfg)
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="q", standalone_query="q"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts={"chunks": []})
        )
        engine = AsyncMock()
        engine.generate.return_value = RetrievalResult(answer="a", contexts={"chunks": []})
        manager._answer_synthesizer = engine

        await manager.aanswer("q", workspace="alpha")

        # No Exa and no resources means the fast path -- no control tool model.
        create.assert_not_called()

    def test_with_exa_one_tool_model_is_shared(self, test_cfg, monkeypatch) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        model = MagicMock()
        create = MagicMock(return_value=model)
        monkeypatch.setattr("dlightrag.models.tool_model.create_query_tool_model", create)
        manager = RAGServiceManager(config=cfg)

        assert manager._get_query_tool_model() is model
        assert manager._get_query_tool_model() is model
        create.assert_called_once_with(cfg)

    async def test_closing_manager_closes_tool_model(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        model = AsyncMock()
        manager._query_tool_model = model

        await manager.aclose()

        model.aclose.assert_awaited_once()

    async def test_with_exa_aanswer_uses_agentic_path(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="question", standalone_query="question"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager._answer_synthesizer = MagicMock()
        monkeypatch.setattr(
            "dlightrag.core.servicemanager.AnswerOrchestrator", _CapturingOrchestrator
        )

        await manager.aanswer("question", workspace="alpha")

        # An Exa key makes the request research, and the fast-path synthesizer is
        # never invoked directly by the manager.
        assert _CapturingOrchestrator.last["init"]["search_web"] is not None
        init = _CapturingOrchestrator.last["init"]
        assert init["resource_manifest"] == ()
        assert {tool.name for tool in init["resource_tools"]} >= {"read_resource"}
        assert callable(init["register_web_source"])
        assert "answer" in _CapturingOrchestrator.last
        manager._answer_synthesizer.generate.assert_not_called()

    async def test_image_attachment_without_exa_keeps_agentic_inspection(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.core.answer.capability import AnswerImageCapability

        manager = RAGServiceManager(config=test_cfg)
        manager._answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=2,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="vision-test",
            failure_kind=None,
        )
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="inspect", standalone_query="inspect"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager._answer_synthesizer = MagicMock()
        monkeypatch.setattr(
            "dlightrag.models.llm.get_vlm_model_func",
            MagicMock(return_value=AsyncMock(return_value="visual evidence")),
        )
        inspector = MagicMock()
        monkeypatch.setattr("dlightrag.core.resources.visual.ResourceInspector", inspector)
        monkeypatch.setattr(
            "dlightrag.core.servicemanager.AnswerOrchestrator", _CapturingOrchestrator
        )

        await manager.aanswer(
            "inspect",
            workspace="alpha",
            resources=[
                ResourceInput(
                    filename="chart.png",
                    content=_png_bytes(),
                    declared_mime="image/png",
                )
            ],
        )

        init = _CapturingOrchestrator.last["init"]
        assert init["search_web"] is None
        assert {tool.name for tool in init["resource_tools"]} == {
            "read_resource",
            "inspect_resource",
        }
        assert len(init["resource_manifest"]) == 1
        assert init["resource_manifest"][0].filename == "chart.png"
        assert init["image_budget"].count == 1
        image_blocks = _CapturingOrchestrator.last["answer"]["query_images"]
        assert image_blocks[0]["text"] == (
            f"[current image 1 | resource: {init['resource_manifest'][0].resource_id}]"
        )
        assert image_blocks[1]["type"] == "image_url"
        assert inspector.call_args.kwargs["max_images"] == 2

    async def test_with_exa_raw_retrieve_remains_knowledge_base_only(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager._open_query_workspaces = AsyncMock(  # type: ignore[method-assign]
            return_value=["alpha"]
        )
        service = AsyncMock()
        service.aretrieve.return_value = RetrievalResult()
        manager._get_service = AsyncMock(return_value=service)  # type: ignore[method-assign]
        manager._query_planner = QueryPlanner(llm_func=None)
        manager._query_tool_model = AsyncMock()
        manager._web_search = AsyncMock()

        await manager.aretrieve("question", workspace="alpha")

        service.aretrieve.assert_awaited_once()
        manager._query_tool_model.assert_not_awaited()
        manager._web_search.search.assert_not_awaited()

    async def test_with_exa_prepared_stream_uses_agentic_path(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager._describe_and_plan = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                QueryPlan(original_query="question", standalone_query="question"),
                SimpleNamespace(descriptions=[], descriptions_by_ordinal={}),
            )
        )
        manager._answer_synthesizer = MagicMock()
        monkeypatch.setattr(
            "dlightrag.core.servicemanager.AnswerOrchestrator", _CapturingOrchestrator
        )
        answer_turn = importlib.import_module("dlightrag.core.answer.turn")
        turn = answer_turn.PreparedAnswerTurn.stateless("question")

        await manager._aanswer_stream_prepared(turn, workspace="alpha")

        assert _CapturingOrchestrator.last["init"]["search_web"] is not None
        assert "answer_stream" in _CapturingOrchestrator.last
        manager._describe_and_plan.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_agentic_kb_tool_plans_the_agent_query_lazily(self, test_cfg) -> None:
        from dlightrag.core.answer.synthesizer import AnswerSynthesizer
        from dlightrag.models.tool_turn import AssistantTurn, ToolCall

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager._open_query_workspaces = AsyncMock(  # type: ignore[method-assign]
            return_value=["alpha"]
        )
        agent_plan = QueryPlan(
            original_query="agent chosen terms",
            standalone_query="agent chosen terms",
            bm25_query="agent terms",
        )
        manager._aplan_query_prepared = AsyncMock(return_value=agent_plan)  # type: ignore[method-assign]
        manager._describe_and_plan = AsyncMock()  # type: ignore[method-assign]
        manager.aretrieve = AsyncMock(return_value=RetrievalResult())  # type: ignore[method-assign]
        manager._warm_query_services = AsyncMock()  # type: ignore[method-assign]
        manager._web_search = AsyncMock()
        model = AsyncMock(
            side_effect=[
                AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id="kb",
                            name="search_knowledge_base",
                            arguments={"query": "agent chosen terms"},
                        ),
                    ),
                    stop_reason="tool_use",
                ),
                AssistantTurn(text="ready", tool_calls=(), stop_reason="stop"),
            ]
        )
        model.complete_text = AsyncMock(return_value="Answer.")
        manager._query_tool_model = model
        manager._answer_synthesizer = AnswerSynthesizer(
            image_max_pixels=40_000_000,
            model_func=None,
        )

        await manager.aanswer(
            "Original request",
            workspace="alpha",
            history=[
                {"role": "user", "content": "Earlier context"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
        )

        manager._describe_and_plan.assert_not_awaited()  # type: ignore[attr-defined]
        manager._aplan_query_prepared.assert_awaited_once_with(  # type: ignore[attr-defined]
            "agent chosen terms",
            text_history=None,
            current_image_descriptions=None,
            workspaces=["alpha"],
            preserve_query=True,
        )
        retrieve_call = manager.aretrieve.await_args  # type: ignore[attr-defined]
        assert retrieve_call is not None
        assert retrieve_call.args[0] == "agent chosen terms"
        retrieval_plan = retrieve_call.kwargs["plan"]
        assert retrieval_plan.original_query == "agent chosen terms"
        assert retrieval_plan.standalone_query == "agent chosen terms"
        assert retrieval_plan.bm25_query == agent_plan.bm25_query

    async def test_agentic_answer_plans_once_and_runs_both_evidence_sources(self, test_cfg) -> None:
        from dlightrag.core.answer.synthesizer import AnswerSynthesizer
        from dlightrag.core.retrieval.web_search import WebSearchHit, WebSearchResult
        from dlightrag.models.tool_turn import AssistantTurn, ToolCall

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        plan = QueryPlan(original_query="What about it?", standalone_query="inflation 2026")
        manager._aplan_query_prepared = AsyncMock(return_value=plan)  # type: ignore[method-assign]
        manager._describe_and_plan = AsyncMock()  # type: ignore[method-assign]
        manager._warm_query_services = AsyncMock()  # type: ignore[method-assign]
        manager._open_query_workspaces = AsyncMock(  # type: ignore[method-assign]
            return_value=["alpha"]
        )
        corpus = RetrievalResult(
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "reference_id": "upstream",
                        "full_doc_id": "doc-1",
                        "file_path": "report.pdf",
                        "content": "corpus fact",
                        "_workspace": "alpha",
                        "metadata": {
                            "source_type": "file",
                            "source_uri": "file:///alpha/report.pdf",
                            "source_download_locator": "file:///alpha/report.pdf",
                        },
                    }
                ],
                "entities": [],
                "relationships": [],
            }
        )
        manager.aretrieve = AsyncMock(return_value=corpus)  # type: ignore[method-assign]
        web = AsyncMock()
        web.search.return_value = WebSearchResult(
            hits=(
                WebSearchHit(
                    url="https://example.com/current",
                    title="Current page",
                    text="current fact",
                ),
            ),
            cost_dollars=0.007,
        )
        manager._web_search = web
        model = AsyncMock(
            side_effect=[
                AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id="kb",
                            name="search_knowledge_base",
                            arguments={"query": "inflation 2026"},
                        ),
                        ToolCall(
                            id="web",
                            name="search_web",
                            arguments={"query": "inflation 2026"},
                        ),
                    ),
                    stop_reason="tool_use",
                ),
                AssistantTurn(
                    text="draft control turn text",
                    tool_calls=(),
                    stop_reason="stop",
                ),
            ]
        )
        model.complete_text = AsyncMock(return_value="Answer [1-1][2-1].")
        manager._query_tool_model = model
        # A real synthesizer owns the tools-disabled final call for research too.
        manager._answer_synthesizer = AnswerSynthesizer(
            image_max_pixels=40_000_000, model_func=None
        )

        result = await manager.aanswer("What about it?", workspace="alpha")

        assert result.answer == "Answer [1-1][2-1]."
        assert [source.id for source in result.sources] == ["1", "2"]
        model.complete_text.assert_awaited_once()
        manager._describe_and_plan.assert_not_awaited()  # type: ignore[attr-defined]
        manager._aplan_query_prepared.assert_awaited_once_with(  # type: ignore[attr-defined]
            "inflation 2026",
            text_history=None,
            current_image_descriptions=None,
            workspaces=["alpha"],
            preserve_query=True,
        )
        retrieve_call = manager.aretrieve.await_args  # type: ignore[attr-defined]
        assert retrieve_call is not None
        assert retrieve_call.args[0] == "inflation 2026"
        assert retrieve_call.kwargs["plan"].standalone_query == "inflation 2026"
        web.search.assert_awaited_once_with("inflation 2026")

    async def test_agentic_stream_keeps_slots_until_disconnect(self, test_cfg) -> None:
        from dlightrag.citations.streaming import aclose_answer_stream
        from dlightrag.core.retrieval.web_search import WebSearchHit, WebSearchResult
        from dlightrag.models.tool_turn import AssistantTurn, ToolCall

        class StreamingToolModel:
            def __init__(self) -> None:
                self.turns = [
                    AssistantTurn(
                        text="",
                        tool_calls=(
                            ToolCall(
                                id="kb",
                                name="search_knowledge_base",
                                arguments={"query": "planned question"},
                            ),
                            ToolCall(
                                id="web",
                                name="search_web",
                                arguments={"query": "planned question"},
                            ),
                        ),
                        stop_reason="tool_use",
                    ),
                    AssistantTurn(
                        text="control draft",
                        tool_calls=(),
                        stop_reason="stop",
                    ),
                ]
                self.closed = asyncio.Event()

            async def __call__(self, **_kwargs: Any) -> AssistantTurn:
                return self.turns.pop(0)

            def stream_text(self, **_kwargs: Any) -> AsyncIterator[str]:
                async def tokens() -> AsyncIterator[str]:
                    try:
                        yield "first token"
                        await asyncio.Event().wait()
                    finally:
                        self.closed.set()

                return tokens()

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        plan = QueryPlan(original_query="Question", standalone_query="planned question")
        manager._aplan_query_prepared = AsyncMock(return_value=plan)  # type: ignore[method-assign]
        manager._warm_query_services = AsyncMock()  # type: ignore[method-assign]
        manager._open_query_workspaces = AsyncMock(  # type: ignore[method-assign]
            return_value=["alpha"]
        )
        manager.aretrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(
                contexts={
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "reference_id": "upstream",
                            "full_doc_id": "doc-1",
                            "file_path": "report.pdf",
                            "content": "corpus fact",
                            "_workspace": "alpha",
                            "metadata": {
                                "source_type": "file",
                                "source_uri": "file:///alpha/report.pdf",
                                "source_download_locator": "file:///alpha/report.pdf",
                            },
                        }
                    ],
                    "entities": [],
                    "relationships": [],
                }
            )
        )
        web = AsyncMock()
        web.search.return_value = WebSearchResult(
            hits=(
                WebSearchHit(
                    url="https://example.com/current",
                    title="Current",
                    text="web fact",
                ),
            ),
            cost_dollars=0.007,
        )
        manager._web_search = web
        model = StreamingToolModel()
        manager._query_tool_model = model  # type: ignore[assignment]
        manager._answer_stream_sem = asyncio.Semaphore(1)
        manager._direct_llm_sem = asyncio.Semaphore(1)

        contexts, stream = await manager.aanswer_stream("Question", workspace="alpha")

        assert len(contexts["chunks"]) == 2
        assert stream is not None
        assert manager._answer_stream_sem.locked()
        assert not manager._direct_llm_sem.locked()
        assert await stream.__anext__() == "first token"
        assert manager._direct_llm_sem.locked()

        await aclose_answer_stream(stream)

        assert model.closed.is_set()
        assert not manager._answer_stream_sem.locked()
        assert not manager._direct_llm_sem.locked()


# ---------------------------------------------------------------------------
# _ScopedAnswerStream must await its cleanup exactly once, never fire-and-forget.
# ---------------------------------------------------------------------------


async def test_scoped_answer_stream_awaits_on_close_on_natural_exhaustion() -> None:
    from dlightrag.core.servicemanager import _ScopedAnswerStream

    calls = 0
    closed = asyncio.Event()

    async def on_close() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        closed.set()

    async def inner() -> AsyncIterator[str]:
        yield "a"
        yield "b"

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    stream = _ScopedAnswerStream(inner(), sem, on_close=on_close)

    tokens = [token async for token in stream]

    assert tokens == ["a", "b"]
    # on_close was awaited before StopAsyncIteration propagated -- not scheduled.
    assert closed.is_set()
    assert calls == 1
    # The semaphore slot is released exactly once.
    assert not sem.locked()
    # No background task was left running.
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == []
    # Cleanup is idempotent: a later aclose does not run on_close again.
    await stream.aclose()
    assert calls == 1
    assert not sem.locked()


async def test_scoped_answer_stream_releases_slot_when_on_close_raises() -> None:
    from dlightrag.core.servicemanager import _ScopedAnswerStream

    async def on_close() -> None:
        raise RuntimeError("cleanup boom")

    async def inner() -> AsyncIterator[str]:
        yield "only"

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    stream = _ScopedAnswerStream(inner(), sem, on_close=on_close)

    with pytest.raises(RuntimeError, match="cleanup boom"):
        async for _ in stream:
            pass

    # Even when cleanup raises, the slot is released and no task lingers.
    assert not sem.locked()
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == []


async def test_scoped_answer_stream_awaits_on_close_on_explicit_aclose() -> None:
    from dlightrag.core.servicemanager import _ScopedAnswerStream

    calls = 0

    async def on_close() -> None:
        nonlocal calls
        calls += 1

    async def inner() -> AsyncIterator[str]:
        yield "a"
        yield "b"

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    stream = _ScopedAnswerStream(inner(), sem, on_close=on_close)

    assert await stream.__anext__() == "a"
    await stream.aclose()

    assert calls == 1
    assert not sem.locked()
    # Idempotent across a second aclose.
    await stream.aclose()
    assert calls == 1
