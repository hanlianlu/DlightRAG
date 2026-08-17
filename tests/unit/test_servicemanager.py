# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RAGServiceManager: workspace pool, routing, health tracking."""

import asyncio
import io
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from dlightrag_ai.capacity import CONTEXT_POLICY_REVISION, ModelCapabilityError, ModelProfile
from dlightrag_ai.catalog import MODEL_CATALOG_REVISION, UnknownModelProfileError
from dlightrag_ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag_rag.pool import WorkspaceUnavailableError
from dlightrag_rag.retrieval import (
    RetrievalPlanner,
    RetrievalResult,
)
from PIL import Image

from dlightrag.answer.capabilities import AnswerCapabilityCoordinator
from dlightrag.answer.executor import AnswerExecutor
from dlightrag.answer.model_runtime import AnswerModelRuntimeClosedError
from dlightrag.answer.resources.images import prepare_query_images
from dlightrag.answer.resources.models import ResourceInput, TextWindowBudget
from dlightrag.answer.runs.execution import AnswerRunInput, AnswerRunRequest
from dlightrag.answer.runs.results import AnswerResult
from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    ModelCapacityOverrideConfig,
    ModelConfig,
    WebSearchConfig,
    set_config,
)
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError
from dlightrag.model_settings import answer_executor_settings
from dlightrag.services.corpora import CorpusAdmin
from tests.unit.conftest import answer_model_profile

_TEST_PLANNER_PROFILE = ModelProfile(
    context_window_tokens=1_200_000,
    max_input_tokens=1_000_000,
)


def test_manager_applies_product_image_decode_ceiling(test_cfg, monkeypatch) -> None:
    from dlightrag_ai.media import MAX_DECODE_IMAGE_PIXELS

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    RAGServiceManager(config=test_cfg)

    assert Image.MAX_IMAGE_PIXELS == MAX_DECODE_IMAGE_PIXELS


def _image_block(url: str = "data:image/png;base64,abc") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (20, 80, 160)).save(output, "PNG")
    return output.getvalue()


def _text_window_budget() -> TextWindowBudget:
    return TextWindowBudget(tokens=1_000)


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


def _scripted_stream(text: str) -> Any:
    """Return a tools-disabled final generation that yields ``text`` once."""

    def _stream_text(**_kwargs: Any) -> AsyncIterator[str]:
        async def _tokens() -> AsyncIterator[str]:
            yield text

        return _tokens()

    return _stream_text


def _synthesizer_mock() -> AsyncMock:
    engine = AsyncMock()
    engine.history_input_measure = MagicMock(
        return_value=lambda messages: (
            100 + sum(len(str(message.get("content") or "")) for message in messages)
        )
    )
    return engine


def _install_answer_synthesizer(
    manager: RAGServiceManager,
    synthesizer: Any,
    profile: ModelProfile | None = None,
) -> None:
    manager._answer_models._answer_synthesizers[
        profile or manager._capabilities.model_profile("query")
    ] = synthesizer


def _install_retrieval_planner(
    manager: RAGServiceManager,
    planner: RetrievalPlanner,
    profile: ModelProfile | None = None,
) -> None:
    selected = profile or manager._capabilities.model_profile("extract")
    current = manager.retrieval.planner_for
    manager.retrieval.planner_for = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda requested=None: (
            planner if (requested or selected) == selected else current(requested)
        )
    )


class _MemoryRunSession:
    """The fenced session one durable run writes through, without PostgreSQL."""

    def __init__(self, request: dict[str, Any]) -> None:
        self.owner_id = "owner"
        self.run_id = "run-1"
        self.request = request
        self.checkpoint = None
        self.completed_turns = 0
        self.phases: list[str] = []
        self.tokens: list[str] = []

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, phase: str) -> None:
        self.phases.append(phase)

    async def emit_token(self, text: str) -> None:
        self.tokens.append(text)

    async def flush_tokens(self) -> None:
        return None

    async def commit_checkpoint(self, envelope: Any) -> None:
        self.completed_turns += 1

    async def attach_artifacts(self, **kwargs: Any) -> None:
        return None


class _MemoryArtifactStore:
    """The artifact reads a durable execution performs, backed by a dict."""

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self._blobs = dict(blobs or {})

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None:
        return self._blobs.get(digest)

    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        return ()


def _answer_executor(manager: RAGServiceManager) -> AnswerExecutor:
    return AnswerExecutor(
        store=cast(Any, manager._answer_run_store),
        pool=manager._workspace_pool,
        retrieve=manager.retrieval.retrieve_result,
        models=manager._answer_models,
        capabilities=manager._capabilities,
        resources=manager._answer_resources,
        settings=answer_executor_settings(manager.config),
        telemetry=manager._telemetry,
    )


def _execution(manager: RAGServiceManager) -> AnswerExecutor:
    executor = manager._answer_executor
    if executor is None:
        executor = _answer_executor(manager)
        manager._answer_executor = executor
    return executor


async def _durable_answer(manager: Any, query: str, **kwargs: Any) -> AnswerResult:
    """Execute one durable answer run in process and restore its canonical result."""
    from dlightrag.answer.runs.results import restore_answer_result
    from dlightrag.runtime import artifact_digest

    resources = kwargs.pop("resources", None)
    manager._answer_run_store = _MemoryArtifactStore(
        {
            artifact_digest(resource.content): resource.content
            for resource in resources or ()
            if resource.content is not None
        }
    )
    executor = _execution(manager)
    executor._store = manager._answer_run_store
    request = await manager._normalized_answer_run_request(
        query,
        workspace=kwargs.pop("workspace", None),
        workspaces=kwargs.pop("workspaces", None),
        all_workspaces=kwargs.pop("all_workspaces", False),
        top_k=kwargs.pop("top_k", None),
        chunk_top_k=kwargs.pop("chunk_top_k", None),
        filters=kwargs.pop("filters", None),
        history=kwargs.pop("history", None),
        semantic_highlights=kwargs.pop("semantic_highlights", False),
        resources=resources,
    )
    run_input = await manager.aprepare_answer_run_input(
        request,
        resources=resources,
        idempotency_fingerprint="test-public-input",
    )
    assert not kwargs, f"unexpected answer options: {sorted(kwargs)}"
    stored = await executor._execute(cast(Any, _MemoryRunSession(run_input.as_request())))
    return restore_answer_result(stored)


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

    def prepare_run(self, query: str, **kwargs: Any) -> Any:
        _CapturingOrchestrator.last["prepare_run"] = {"query": query, **kwargs}
        return MagicMock()

    async def answer_stream(self, query: str, **kwargs: Any) -> Any:
        _CapturingOrchestrator.last["answer_stream"] = {"query": query, **kwargs}
        return {"chunks": []}, None


async def test_private_generation_helper_hands_prepared_history_to_engine(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    engine = _synthesizer_mock()
    engine.generate_stream.return_value = ({"chunks": []}, _AttrStream(["a"]))
    _install_answer_synthesizer(manager, engine)
    manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
        return_value=RetrievalResult(contexts={"chunks": []})
    )
    history = [
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Earlier answer"},
    ]

    await _durable_answer(manager, "follow up", workspaces=["default"], history=history)

    assert engine.generate_stream.await_args.kwargs["conversation_history"].messages == history


@pytest.fixture()
def test_cfg(tmp_path) -> DlightragConfig:
    cfg = DlightragConfig(
        working_dir=str(tmp_path / "dlightrag_storage"),
        llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
        model_capacity_overrides=[
            ModelCapacityOverrideConfig(
                provider="openai",
                model="gpt-5.4-mini",
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                supports_images=True,
                supports_tools=True,
                supports_reasoning=True,
            )
        ],
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="test",
            startup_probe=False,
        ),
    )
    set_config(cfg)
    return cfg


class TestAnswerWorkspaceSelection:
    """Durable Answer keeps its in-process workspace-selection contract."""

    @pytest.mark.parametrize(
        "explicit_selection",
        [
            {"workspace": "finance"},
            {"workspaces": ["finance"]},
        ],
    )
    async def test_all_workspaces_conflicts_with_explicit_selection(
        self,
        explicit_selection: dict[str, Any],
        test_cfg,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager.corpora.list_workspaces = AsyncMock(return_value=["finance"])  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="all_workspaces"):
            await manager.aanswer(
                "query",
                all_workspaces=True,
                **explicit_selection,
            )
        manager.corpora.list_workspaces.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_streaming_convenience_rejects_conflicting_selection(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager.corpora.list_workspaces = AsyncMock(return_value=["finance"])  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="all_workspaces"):
            async for _ in manager.aanswer_stream(
                "query", all_workspaces=True, workspace="finance"
            ):
                pass
        manager.corpora.list_workspaces.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_query_images_are_current_request_only(self, test_cfg) -> None:
        describer = AsyncMock()
        describer.describe = AsyncMock(return_value=["Image 1: chart"])
        current = [_image_block()]

        descriptions = await prepare_query_images(
            query_images=current,
            describer=describer,
        )

        assert descriptions == ["Image 1: chart"]
        describer.describe.assert_awaited_once_with(current)

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_calls_aretrieve_then_engine(self, mock_create, test_cfg) -> None:
        """A durable run routes through aretrieve() then generate_stream()."""
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = _synthesizer_mock()
        mock_engine.generate_stream.return_value = (mock_contexts, _AttrStream(["a"]))

        manager = RAGServiceManager(config=test_cfg)
        _install_answer_synthesizer(manager, mock_engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        await _durable_answer(manager, "query", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )


class TestAnswerViaEngine:
    """aanswer and aanswer_stream route through AnswerSynthesizer."""

    @pytest.fixture(autouse=True)
    def _stub_planning_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "dlightrag.adapters.postgres.pg_metadata_index.PGMetadataIndex.get_field_schema",
            AsyncMock(return_value={"columns": [], "custom_keys": []}),
        )

        async def warm_workspaces(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(
            "dlightrag_rag.pool.WorkspacePool.warm",
            warm_workspaces,
        )

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_calls_retrieve_then_engine(self, mock_create, test_cfg) -> None:
        """A durable run calls aretrieve() then AnswerSynthesizer.generate_stream()."""
        trace_calls: list[dict[str, Any]] = []
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_retrieval = MagicMock(contexts=mock_contexts, trace={})
        mock_svc.aretrieve.return_value = mock_retrieval
        mock_create.return_value = mock_svc

        mock_engine = _synthesizer_mock()
        answer_text = "Answer [1-1]."
        mock_engine.generate_stream.return_value = (mock_contexts, _AttrStream([answer_text]))

        manager = RAGServiceManager(config=test_cfg)
        manager._telemetry = cast(
            Any,
            SimpleNamespace(
                capture_sensitive_data=True,
                observe=_record_trace_calls(trace_calls),
            ),
        )
        manager.retrieval._telemetry = manager._telemetry
        _install_answer_synthesizer(manager, mock_engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        result = await _durable_answer(manager, "what is X?", workspace="ws_a")
        mock_svc.aretrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "what is X?",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result.answer == answer_text
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
                    "chunk_count": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "standalone_query": "what is X?",
                    "query_image_description_count": 0,
                }
            }
        ]
        assert orchestration["input"] == {"query": "what is X?"}
        assert "query" not in orchestration["metadata"]
        assert orchestration["metadata"]["research"] is False
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

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_derives_candidate_and_context_limits(
        self, mock_create, test_cfg
    ) -> None:
        """Answer over-fetches retrieval candidates for the final prompt."""
        cfg = test_cfg
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = _synthesizer_mock()
        mock_engine.generate_stream.return_value = (mock_contexts, _AttrStream(["a"]))

        manager = RAGServiceManager(config=cfg)
        _install_answer_synthesizer(manager, mock_engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        result = await _durable_answer(manager, "query", workspace="ws_a")

        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == test_cfg.chunk_top_k
        mock_engine.generate_stream.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result.answer == "a"

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_uses_chunk_top_k_as_candidate_override(
        self, mock_create, test_cfg
    ) -> None:
        """Answer chunk_top_k remains the explicit retrieval candidate override."""
        cfg = test_cfg
        mock_svc = AsyncMock()
        mock_contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        mock_engine = _synthesizer_mock()
        mock_engine.generate_stream.return_value = (mock_contexts, _AttrStream(["a"]))

        manager = RAGServiceManager(config=cfg)
        _install_answer_synthesizer(manager, mock_engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        result = await _durable_answer(manager, "query", workspace="ws_a", chunk_top_k=7)

        retrieve_kwargs = mock_svc.aretrieve.await_args.kwargs
        assert retrieve_kwargs["top_k"] == test_cfg.top_k
        assert retrieve_kwargs["chunk_top_k"] == 7
        mock_engine.generate_stream.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result.answer == "a"

    async def test_aanswer_threads_history_to_planning_and_generation(self, test_cfg) -> None:
        """Fast answer gives one bounded history to retrieval and final synthesis."""
        manager = RAGServiceManager(config=test_cfg)
        manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts={"chunks": []})
        )
        mock_engine = _synthesizer_mock()
        mock_engine.generate_stream.return_value = ({"chunks": []}, _AttrStream(["a"]))
        mock_engine.history_input_measure = MagicMock(
            return_value=lambda messages: (
                100 + sum(len(str(message.get("content") or "")) for message in messages)
            )
        )
        _install_answer_synthesizer(manager, mock_engine)
        history = [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Earlier answer"},
        ]

        await _durable_answer(manager, "follow up", workspace="ws_a", history=history)

        retrieve_call = manager.retrieval.retrieve_result.await_args  # type: ignore[attr-defined]
        assert retrieve_call is not None
        assert retrieve_call.args[0] == "follow up"
        assert retrieve_call.kwargs["conversation_history"] == history
        generate_call = mock_engine.generate_stream.await_args
        assert generate_call is not None
        assert generate_call.args[0] == "follow up"
        assert generate_call.kwargs["conversation_history"].messages == history

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_semantic_highlights_are_opt_in(
        self, mock_create, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK answer highlights are disabled by default and enabled per call."""
        mock_contexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "full_doc_id": "doc-report",
                    "file_path": "report.pdf",
                    "content": "The report says market growth improved in 2025.",
                    "_workspace": "ws_a",
                    "metadata": {
                        "source_uri": "local://ws_a/report.pdf",
                        "source_download_locator": "/docs/report.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        }
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = MagicMock(contexts=mock_contexts, trace={})
        mock_create.return_value = mock_svc

        async def llm_func(*, messages, **kwargs) -> str:
            return '{"items": [{"id": "0", "phrases": ["market growth"], "confidence": 1.0}]}'

        highlight_model = AsyncMock(side_effect=llm_func)
        highlight_model.aclose = AsyncMock()
        monkeypatch.setattr(
            "dlightrag.answer.model_runtime.CompletionModel",
            MagicMock(return_value=highlight_model),
        )
        trace_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "dlightrag.observability.tracing.trace_observation",
            _record_trace_calls(trace_calls),
        )

        manager = RAGServiceManager(config=test_cfg)
        synthesizer = _synthesizer_mock()
        synthesizer.generate_stream.side_effect = [
            (mock_contexts, _AttrStream(["Market growth improved [1-1]."])),
            (mock_contexts, _AttrStream(["Market growth improved [1-1]."])),
        ]
        _install_answer_synthesizer(manager, synthesizer)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        plain = await _durable_answer(manager, "query", workspace="ws_a")
        highlighted = await _durable_answer(
            manager,
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
        highlight_model.aclose.assert_awaited_once()

    @patch("dlightrag.services.retrieval.federated_retrieve", new_callable=AsyncMock)
    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_aanswer_multi_workspace_uses_federated_retrieve(
        self, mock_create, mock_fed_retrieve, test_cfg
    ) -> None:
        """A run over multiple workspaces federates retrieval, then uses the engine."""
        mock_contexts = {"chunks": [{"text": "ctx"}], "entities": [], "relationships": []}
        mock_fed_retrieve.return_value = MagicMock(contexts=mock_contexts, trace={})

        mock_engine = _synthesizer_mock()
        mock_engine.generate_stream.return_value = (mock_contexts, _AttrStream(["a"]))

        manager = RAGServiceManager(config=test_cfg)
        _install_answer_synthesizer(manager, mock_engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        result = await _durable_answer(manager, "query", workspaces=["ws_a", "ws_b"])
        mock_fed_retrieve.assert_awaited_once()
        mock_engine.generate_stream.assert_awaited_once_with(
            "query",
            mock_contexts,
            conversation_history=ANY,
        )
        assert result.answer == "a"

    @patch("dlightrag.services.retrieval.federated_retrieve", new_callable=AsyncMock)
    async def test_aanswer_all_workspaces_uses_federated_retrieve(
        self,
        mock_federated,
        test_cfg,
    ) -> None:
        contexts = {"chunks": [], "entities": [], "relationships": []}
        mock_federated.return_value = RetrievalResult(contexts=contexts)
        engine = _synthesizer_mock()
        engine.generate_stream.return_value = (contexts, _AttrStream(["answer"]))
        manager = RAGServiceManager(config=test_cfg)
        manager.corpora.list_workspaces = AsyncMock(  # type: ignore[method-assign]
            return_value=["ws_a", "ws_b"]
        )
        manager.retrieval.schema_for = AsyncMock(return_value={})  # type: ignore[method-assign]
        _install_answer_synthesizer(manager, engine)
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(
                llm_func=None,
                model_profile=_TEST_PLANNER_PROFILE,
            ),
        )

        result = await _durable_answer(manager, "query", all_workspaces=True)

        manager.corpora.list_workspaces.assert_awaited_once()  # type: ignore[attr-defined]
        assert mock_federated.await_args.args[1] == ["ws_a", "ws_b"]
        assert result.answer == "answer"

    def test_composed_answer_runtime_uses_configured_image_policy(
        self,
        test_cfg,
    ) -> None:
        test_cfg.answer.image_max_pixels = 123
        manager = RAGServiceManager(config=test_cfg)

        with patch(
            "dlightrag.answer.model_runtime.CompletionModel",
            return_value=MagicMock(),
        ):
            engine = manager._answer_models.answer_synthesizer(
                manager._capabilities.model_profile("query")
            )

        policy = engine._image_policy
        assert policy.max_pixels == 123
        assert engine._model_profile.context_window_tokens == 400_000
        # One immutable policy, one fresh budget per call -- never shared state.
        first = policy.new_budget()
        first.count = 1
        assert policy.new_budget().count == 0
        assert policy.new_budget().max_pixels == 123


class TestDegradedMode:
    @pytest.fixture(autouse=True)
    def _isolate_workspace_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CorpusAdmin, "initialize", AsyncMock())
        monkeypatch.setattr(CorpusAdmin, "start_recovery", AsyncMock())
        monkeypatch.setattr(RAGServiceManager, "_initialize_answer_run_store", AsyncMock())
        monkeypatch.setattr(AnswerCapabilityCoordinator, "probe_all", AsyncMock())

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_create_sets_ready_on_success(self, mock_create, test_cfg) -> None:
        mock_create.return_value = AsyncMock()
        manager = await RAGServiceManager.acreate(config=test_cfg)
        assert manager.health.is_ready
        assert not manager.health.is_degraded
        # Warnings may include "Workspace registry unavailable" in tests
        # without a running PostgreSQL — that's expected and non-fatal.

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_create_eagerly_initializes_retrieval_planner(
        self, mock_create, test_cfg
    ) -> None:
        mock_create.return_value = AsyncMock()

        manager = await RAGServiceManager.acreate(config=test_cfg)

        assert manager.retrieval.planner_for() is manager.retrieval.planner_for()

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_transient_workspace_registry_failure_only_warns(
        self,
        mock_create,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg,
    ) -> None:
        mock_create.return_value = AsyncMock()
        monkeypatch.setattr(
            CorpusAdmin,
            "initialize",
            AsyncMock(side_effect=RuntimeError("registry unavailable")),
        )

        manager = await RAGServiceManager.acreate(config=test_cfg)

        assert manager.health.is_ready
        assert manager.health.warnings == ("Workspace registry unavailable",)

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_transient_ingest_recovery_failure_only_warns(
        self,
        mock_create,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg,
    ) -> None:
        mock_create.return_value = AsyncMock()
        monkeypatch.setattr(
            CorpusAdmin,
            "start_recovery",
            AsyncMock(side_effect=RuntimeError("recovery unavailable")),
        )

        manager = await RAGServiceManager.acreate(config=test_cfg)

        assert manager.health.is_ready
        assert manager.health.warnings == ("Ingest job recovery unavailable",)

    async def test_create_closes_after_capability_probe_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg: DlightragConfig,
    ) -> None:
        probe_all = AsyncMock(side_effect=RuntimeError("probe failed"))
        close = AsyncMock()
        monkeypatch.setattr(AnswerCapabilityCoordinator, "probe_all", probe_all)
        monkeypatch.setattr(RAGServiceManager, "aclose", close)

        with pytest.raises(RuntimeError, match="probe failed"):
            await RAGServiceManager.acreate(config=test_cfg)

        probe_all.assert_awaited_once()
        close.assert_awaited_once()

    @pytest.mark.parametrize("role", ["extract", "keyword", "query", "vlm"])
    async def test_startup_resolves_every_reachable_model_profile(
        self,
        role: str,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg: DlightragConfig,
    ) -> None:
        unknown = ModelConfig(model=f"unknown-{role}", api_key="test")
        roles = test_cfg.llm.roles.model_copy(update={role: unknown})
        cfg = test_cfg.model_copy(update={"llm": test_cfg.llm.model_copy(update={"roles": roles})})
        initialize_registry = AsyncMock()
        monkeypatch.setattr(CorpusAdmin, "initialize", initialize_registry)

        with pytest.raises(UnknownModelProfileError):
            await RAGServiceManager.acreate(config=cfg)

        initialize_registry.assert_not_awaited()


async def test_keyed_sdk_replay_bypasses_resolved_input_preparation(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    existing = MagicMock(owner_id="owner", run_id="run-1", status="running")
    manager.areplay_answer_run = AsyncMock(return_value=existing)  # type: ignore[method-assign]
    manager.aprepare_answer_run_input = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("replay must not prepare model input")
    )
    manager._workspace_pool.warm = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("replay must not warm workspaces")
    )

    creation = await manager.acreate_answer_run(
        "question",
        workspaces=["default"],
        idempotency_key="key-1",
        owner_id="owner",
    )

    assert creation.replayed is True
    assert creation.run is existing
    manager.aprepare_answer_run_input.assert_not_awaited()  # type: ignore[attr-defined]
    manager._workspace_pool.warm.assert_not_awaited()


async def test_sdk_acceptance_rebuilds_current_attachments_from_durable_mime(test_cfg) -> None:
    manager = RAGServiceManager(config=test_cfg)
    prepared = MagicMock()
    creation = MagicMock()
    manager.aprepare_answer_run_input = AsyncMock(  # type: ignore[method-assign]
        return_value=prepared
    )
    manager.astart_answer_run = AsyncMock(return_value=creation)  # type: ignore[method-assign]
    payload = b"\x89PNG\r\n\x1a\nnot-promoted-without-image-mime"

    result = await manager.acreate_answer_run(
        "question",
        resources=[ResourceInput(filename="chart.png", content=payload)],
    )

    assert result is creation
    prepare_call = manager.aprepare_answer_run_input.await_args  # type: ignore[attr-defined]
    assert prepare_call is not None
    (resource,) = prepare_call.kwargs["resources"]
    assert resource.declared_mime == "application/octet-stream"
    assert resource.content is None
    assert resource.loader is not None
    assert await resource.loader() == payload
    start_call = manager.astart_answer_run.await_args  # type: ignore[attr-defined]
    assert start_call is not None
    assert start_call.kwargs["request"] is prepared
    assert start_call.kwargs["attachment_bytes"] == [payload]

    async def test_startup_rejects_web_research_without_query_tool_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg: DlightragConfig,
    ) -> None:
        profile = test_cfg.model_capacity_overrides[0].model_copy(update={"supports_tools": False})
        cfg = test_cfg.model_copy(
            update={
                "model_capacity_overrides": [profile],
                "web_search": WebSearchConfig(api_key="k"),
            }
        )
        initialize_registry = AsyncMock()
        monkeypatch.setattr(CorpusAdmin, "initialize", initialize_registry)

        with pytest.raises(ModelCapabilityError, match="tool calling"):
            await RAGServiceManager.acreate(config=cfg)

        initialize_registry.assert_not_awaited()

    async def test_startup_aborts_before_workspace_init_for_an_active_old_policy_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        test_cfg: DlightragConfig,
    ) -> None:
        seed = RAGServiceManager(config=test_cfg)
        seed._capabilities.resolve_profiles()
        profiles: dict[ModelRole, ModelProfile] = {
            role: seed._capabilities.model_profile(role) for role in MODEL_ROLE_NAMES
        }
        request = AnswerRunInput(
            query="accepted",
            pinned_models=seed._pin_model_profiles(profiles),
            context_policy_revision="old-policy",
            model_catalog_revision=MODEL_CATALOG_REVISION,
            idempotency_fingerprint="public-input",
        )
        store = AsyncMock()
        store.list_active_run_requirements.return_value = (
            {
                "context_policy_revision": request.context_policy_revision,
                "pinned_models": [item.as_json() for item in request.pinned_models],
            },
        )

        async def initialize_store(manager: RAGServiceManager) -> None:
            manager._answer_run_store = store

        initialize_registry = AsyncMock()
        monkeypatch.setattr(
            RAGServiceManager,
            "_initialize_answer_run_store",
            initialize_store,
        )
        monkeypatch.setattr(CorpusAdmin, "initialize", initialize_registry)

        with pytest.raises(RuntimeError, match="drain or owner-cancel"):
            await RAGServiceManager.acreate(config=test_cfg)

        initialize_registry.assert_not_awaited()

    def test_worker_uses_persisted_profiles_when_live_catalog_facts_change(
        self,
        test_cfg: DlightragConfig,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities.resolve_profiles()
        profiles: dict[ModelRole, ModelProfile] = {
            role: manager._capabilities.model_profile(role) for role in MODEL_ROLE_NAMES
        }
        pinned = manager._pin_model_profiles(profiles)
        request = AnswerRunInput(
            query="accepted",
            pinned_models=pinned,
            context_policy_revision=CONTEXT_POLICY_REVISION,
            model_catalog_revision="older-catalog",
            idempotency_fingerprint="public-input",
        )
        manager._capabilities._profiles["query"] = ModelProfile(context_window_tokens=10_000)

        resolved = AnswerExecutor.validate_pinned_model_profiles(request)

        assert resolved == {item.role: item.profile for item in pinned}

    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_create_sets_degraded_on_failure(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = RuntimeError("DB down")
        manager = await RAGServiceManager.acreate(config=test_cfg)
        assert not manager.health.is_ready
        assert manager.health.is_degraded
        assert any("DB down" in w for w in manager.health.warnings)

    async def test_create_warms_default_workspace_only(
        self, monkeypatch: pytest.MonkeyPatch, test_cfg
    ) -> None:
        cfg = test_cfg.model_copy(update={"max_async": 3})
        created: list[str] = []

        async def fake_build_workspace(  # noqa: ANN001, ANN202
            self, workspace_id: str, settings, backend
        ):
            created.append(workspace_id)
            return AsyncMock()

        monkeypatch.setattr(CorpusAdmin, "initialize", AsyncMock())
        monkeypatch.setattr(RAGServiceManager, "_build_workspace", fake_build_workspace)
        monkeypatch.setattr("dlightrag.observability.init_tracing", lambda config: None)

        manager = await RAGServiceManager.acreate(config=cfg)

        assert manager.health.is_ready
        assert created == ["default"]


class TestActionableErrors:
    @patch("dlightrag.core.servicemanager.WorkspaceRag.acreate", new_callable=AsyncMock)
    async def test_connection_refused_gets_hint(self, mock_create, test_cfg) -> None:
        mock_create.side_effect = ConnectionRefusedError("Connection refused")
        manager = RAGServiceManager(config=test_cfg)
        with pytest.raises(WorkspaceUnavailableError, match="Check.*DLIGHTRAG_POSTGRES"):
            await manager._workspace_pool.acquire("ws_a")

    def test_actionable_error_default(self) -> None:
        exc = ValueError("something broke")
        result = RAGServiceManager._actionable_error(exc)
        assert result == "ValueError: something broke"

    def test_actionable_error_timeout(self) -> None:
        exc = TimeoutError("request timed out")
        result = RAGServiceManager._actionable_error(exc)
        assert "overloaded" in result


async def test_unsupported_image_link_is_rejected_before_materialization(test_cfg) -> None:
    from dlightrag.answer.capability import AnswerImageCapability
    from dlightrag.answer.errors import AnswerImageError

    manager = RAGServiceManager(config=test_cfg)
    manager._capabilities._answer_image_capability = AnswerImageCapability(
        status="unsupported",
        configured_ceiling=2,
        effective_max_images=0,
        provider="test",
        base_url=None,
        model="text-only",
        failure_kind=None,
    )
    manager._answer_resources.materialize_link_image = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("image link was fetched before capability rejection")
    )

    with pytest.raises(AnswerImageError):
        await _durable_answer(
            manager,
            "inspect",
            workspace="default",
            resources=[
                ResourceInput(
                    filename="chart.png",
                    url="https://example.com/chart.png",
                    declared_mime="image/png",
                )
            ],
        )

    manager._answer_resources.materialize_link_image.assert_not_awaited()  # type: ignore[attr-defined]


class TestClose:
    """Test cleanup."""

    async def test_close_delegates_workspace_shutdown_to_pool(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager._workspace_pool.aclose = AsyncMock()  # type: ignore[method-assign]
        manager.health.mark_ready()

        await manager.aclose()

        manager._workspace_pool.aclose.assert_awaited_once()
        assert not manager.health.is_ready

    async def test_close_prevents_recreating_vlm_provider(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        factory = MagicMock()
        monkeypatch.setattr("dlightrag.answer.model_runtime.CompletionModel", factory)

        await manager.aclose()

        with pytest.raises(AnswerModelRuntimeClosedError):
            manager._answer_models.vlm_func()
        factory.assert_not_called()

    async def test_close_repeatedly_delegates_to_idempotent_owned_runtimes(
        self,
        test_cfg,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        answer_model = AsyncMock()
        vlm_model = AsyncMock()
        tool_model = AsyncMock()
        web_search = AsyncMock()
        manager.retrieval.aclose = AsyncMock()  # type: ignore[method-assign]
        manager._answer_models._answer_model = answer_model
        manager._answer_models._vlm_model = vlm_model
        manager._answer_models._query_tool_model = tool_model
        manager._answer_models._web_search = web_search

        await manager.aclose()
        await manager.aclose()

        assert manager.retrieval.aclose.await_count == 2
        for component in (answer_model, vlm_model, tool_model, web_search):
            component.aclose.assert_awaited_once()
        with pytest.raises(AnswerModelRuntimeClosedError):
            manager._answer_models.answer_synthesizer(manager._capabilities.model_profile("query"))
        with pytest.raises(AnswerModelRuntimeClosedError):
            manager._answer_models.query_tool_model()
        with pytest.raises(AnswerModelRuntimeClosedError):
            manager._answer_models.web_search()
        with pytest.raises(AnswerModelRuntimeClosedError):
            manager._answer_models.vlm_func()


class TestWebSearchCapability:
    """A key present is the capability; without one the path does not exist."""

    def test_without_a_key_there_is_no_web_search_to_reach(self, test_cfg) -> None:
        manager = RAGServiceManager(config=test_cfg)

        assert manager._answer_models.web_search() is None

    def test_with_a_key_one_client_is_shared_by_every_turn(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)

        assert manager._answer_models.web_search() is not None
        assert manager._answer_models.web_search() is manager._answer_models.web_search()

    async def test_closing_the_manager_closes_the_web_client(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        search = manager._answer_models.web_search()
        assert search is not None

        await manager.aclose()

        assert search._client.is_closed


class TestExaContentsFallback:
    """Exa adapts its Contents response into the registry fallback."""

    async def test_contents_passages_become_one_deterministic_text(self) -> None:
        from dlightrag.answer.tools.web import (
            ExaSearch,
            WebSearchHit,
            WebSearchResult,
        )

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
        text = await ExaSearch.contents_text(cast(ExaSearch, exa), "https://example.org/page")

        assert exa.urls == ["https://example.org/page"]
        assert text is not None
        assert "first passage" in text
        assert "second passage" in text
        assert "The Page" in text

    async def test_contents_unavailable_yields_no_text(self) -> None:
        from dlightrag.answer.tools.web import ExaSearch, WebSearchUnavailable

        class _FakeExa:
            async def contents(self, url: str):
                raise WebSearchUnavailable("timeout")

        assert (
            await ExaSearch.contents_text(cast(ExaSearch, _FakeExa()), "https://example.org/page")
            is None
        )

    def test_registry_receives_fallback_only_when_web_search_present(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        resources = [ResourceInput(content=b"payload")]

        registry, _tools = manager._answer_resources.build_resource_context(
            resources,
            text_window_budget=_text_window_budget(),
            web_search=manager._answer_models.web_search(),
            vlm_profile=manager._capabilities.model_profile("vlm"),
        )
        assert registry is not None
        assert registry._url_text_fallback is not None

        plain = RAGServiceManager(config=test_cfg)
        registry2, _t2 = plain._answer_resources.build_resource_context(
            resources,
            text_window_budget=_text_window_budget(),
            web_search=None,
            vlm_profile=plain._capabilities.model_profile("vlm"),
        )
        assert registry2 is not None
        assert registry2._url_text_fallback is None


class TestAgenticAnswerCapability:
    def test_private_host_resource_link_is_an_answer_input_error(self, test_cfg) -> None:
        from dlightrag.answer.errors import AnswerResourceAdmissionError

        manager = RAGServiceManager(config=test_cfg)

        with pytest.raises(AnswerResourceAdmissionError):
            manager._answer_resources.build_resource_context(
                [ResourceInput(url="https://127.0.0.1/private.pdf")],
                text_window_budget=_text_window_budget(),
                web_search=None,
                vlm_profile=manager._capabilities.model_profile("vlm"),
            )

    async def test_without_exa_fast_path_never_builds_a_tool_model(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = MagicMock(side_effect=AssertionError("tool model must stay absent"))
        monkeypatch.setattr("dlightrag.answer.model_runtime.ToolModel", create)
        manager = RAGServiceManager(config=test_cfg)
        manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult(contexts={"chunks": []})
        )
        engine = _synthesizer_mock()
        engine.generate_stream.return_value = ({"chunks": []}, _AttrStream(["a"]))
        _install_answer_synthesizer(manager, engine)

        await _durable_answer(manager, "q", workspace="alpha")

        # No Exa and no resources means the fast path -- no control tool model.
        create.assert_not_called()

    async def test_resources_fail_before_preparation_without_query_tool_support(
        self,
        test_cfg: DlightragConfig,
    ) -> None:
        from dlightrag.answer.errors import AnswerModelCapabilityError

        profile = test_cfg.model_capacity_overrides[0].model_copy(update={"supports_tools": False})
        cfg = test_cfg.model_copy(update={"model_capacity_overrides": [profile]})
        manager = RAGServiceManager(config=cfg)
        manager.retrieval.warm = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("workspace initialization is unreachable")
        )
        manager._answer_resources.prepare_current_images = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("resource preparation is unreachable")
        )

        with pytest.raises(AnswerModelCapabilityError, match="cannot use the tools"):
            await manager.aprepare_answer_run_input(
                AnswerRunRequest(query="question", workspaces=("alpha",)),
                resources=[ResourceInput(filename="notes.txt", content=b"notes")],
                idempotency_fingerprint="public-input",
            )

        manager._answer_resources.prepare_current_images.assert_not_awaited()
        manager.retrieval.warm.assert_not_called()

    def test_with_exa_one_tool_model_is_shared(self, test_cfg, monkeypatch) -> None:
        from dlightrag.model_settings import model_settings_for_role

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        model = MagicMock()
        create = MagicMock(return_value=model)
        monkeypatch.setattr("dlightrag.answer.model_runtime.ToolModel", create)
        manager = RAGServiceManager(config=cfg)

        assert manager._answer_models.query_tool_model() is model
        assert manager._answer_models.query_tool_model() is model
        create.assert_called_once_with(
            model_settings_for_role(cfg, "query"),
            scheduler=manager._model_scheduler,
            telemetry=ANY,
        )

    async def test_closing_manager_closes_tool_model(self, test_cfg) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        model = AsyncMock()
        manager._answer_models._query_tool_model = model

        await manager.aclose()

        model.aclose.assert_awaited_once()

    async def test_describer_and_inspector_share_one_closed_vlm_callable(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.answer.capability import AnswerImageCapability

        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities._answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=2,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="vision-test",
            failure_kind=None,
        )
        manager._capabilities._vlm_image_status = "supported"
        model = AsyncMock()
        model.aclose = AsyncMock()
        model_factory = MagicMock(return_value=model)
        inspector = MagicMock()
        monkeypatch.setattr("dlightrag.answer.model_runtime.CompletionModel", model_factory)
        monkeypatch.setattr("dlightrag.answer.executor.ResourceInspector", inspector)

        describer = manager._answer_models.query_image_describer()
        registry, _tools = manager._answer_resources.build_resource_context(
            [ResourceInput(content=b"payload")],
            text_window_budget=_text_window_budget(),
            web_search=None,
            vlm_profile=manager._capabilities.model_profile("vlm"),
        )
        await manager.aclose()

        assert registry is not None
        model_factory.assert_called_once()
        assert inspector.call_args.kwargs["vlm_func"] is describer._vlm_func
        model.aclose.assert_awaited_once()

    async def test_current_image_verification_and_encoding_run_off_event_loop(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop_thread = threading.get_ident()
        preparation_threads: list[int] = []

        def verify(_data: bytes, *, max_pixels: int) -> str:
            assert max_pixels == test_cfg.answer.image_max_pixels
            preparation_threads.append(threading.get_ident())
            return "image/png"

        def encode(_data: bytes, *, fallback_mime: str) -> str:
            assert fallback_mime == "image/png"
            preparation_threads.append(threading.get_ident())
            return "data:image/png;base64,AA=="

        monkeypatch.setattr("dlightrag_ai.media.verify_web_image_bytes", verify)
        monkeypatch.setattr("dlightrag_ai.media.image_bytes_to_data_uri", encode)
        manager = RAGServiceManager(config=test_cfg)

        (
            images,
            _resources,
            _image_resources,
        ) = await manager._answer_resources.prepare_current_images(
            [ResourceInput(filename="chart.png", content=b"raw", declared_mime="image/png")]
        )

        assert len(images) == 1
        assert preparation_threads
        assert all(thread_id != loop_thread for thread_id in preparation_threads)

    async def test_agent_image_budgeting_runs_off_event_loop(self, test_cfg) -> None:
        from dlightrag_ai.media import image_bytes_to_data_uri

        from dlightrag.answer.capability import AnswerImageCapability

        loop_thread = threading.get_ident()
        budget_threads: list[int] = []
        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities._answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=2,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="vision-test",
            failure_kind=None,
        )
        budget = manager._capabilities.answer_image_policy(
            manager._capabilities.model_profile("query")
        ).new_budget()
        add_user_image = budget.add_user_image

        def capture_budget(value, *, label):
            budget_threads.append(threading.get_ident())
            return add_user_image(value, label=label)

        budget.add_user_image = capture_budget  # type: ignore[method-assign]
        image = _image_block(image_bytes_to_data_uri(_png_bytes()))

        result = manager._answer_resources.budget_agent_images([image], budget)
        if asyncio.iscoroutine(result):
            await result

        assert budget_threads
        assert all(thread_id != loop_thread for thread_id in budget_threads)

    async def test_with_exa_aanswer_uses_agentic_path(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        synthesizer = MagicMock()
        _install_answer_synthesizer(manager, synthesizer)
        monkeypatch.setattr("dlightrag.answer.executor.AnswerOrchestrator", _CapturingOrchestrator)

        await _durable_answer(manager, "question", workspace="alpha")

        # An Exa key makes the request research, and the fast-path synthesizer is
        # never invoked directly by the manager.
        assert _CapturingOrchestrator.last["init"]["search_web"] is not None
        init = _CapturingOrchestrator.last["init"]
        assert init["resource_manifest"] == ()
        assert {tool.name for tool in init["resource_tools"]} >= {"read_resource"}
        assert callable(init["register_web_source"])
        assert "answer_stream" in _CapturingOrchestrator.last
        synthesizer.generate.assert_not_called()

    async def test_image_attachment_without_exa_keeps_agentic_inspection(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.answer.capability import AnswerImageCapability

        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities._answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=2,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="vision-test",
            failure_kind=None,
        )
        manager._capabilities._vlm_image_status = "supported"
        _install_answer_synthesizer(manager, MagicMock())
        monkeypatch.setattr(
            "dlightrag.answer.model_runtime.CompletionModel",
            MagicMock(return_value=AsyncMock(return_value="visual evidence")),
        )
        inspector = MagicMock()
        monkeypatch.setattr("dlightrag.answer.executor.ResourceInspector", inspector)
        monkeypatch.setattr("dlightrag.answer.executor.AnswerOrchestrator", _CapturingOrchestrator)

        await _durable_answer(
            manager,
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
        image_blocks = _CapturingOrchestrator.last["prepare_run"]["query_images"]
        assert image_blocks[0]["text"] == (
            f"[current image 1 | resource: {init['resource_manifest'][0].resource_id}]"
        )
        assert image_blocks[1]["type"] == "image_url"
        # Inspection rides the VLM role's own capability and the deployment
        # ceiling, not the answer model's narrower effective image count.
        assert inspector.call_args.kwargs["image_policy"].max_images == test_cfg.answer.max_images

    def test_supported_with_zero_image_ceiling_withholds_visual_inspection(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        test_cfg.answer.max_images = 0
        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities._vlm_image_status = "supported"
        monkeypatch.setattr(
            "dlightrag.answer.model_runtime.CompletionModel",
            MagicMock(return_value=AsyncMock(return_value="visual evidence")),
        )
        inspector = MagicMock()
        monkeypatch.setattr("dlightrag.answer.executor.ResourceInspector", inspector)

        registry, tools = manager._answer_resources.build_resource_context(
            [ResourceInput(filename="chart.png", content=_png_bytes(), declared_mime="image/png")],
            text_window_budget=_text_window_budget(),
            web_search=None,
            vlm_profile=manager._capabilities.model_profile("vlm"),
        )

        # A zero ceiling means no image block can ever be sent, so an inspector
        # built on that policy could only fail; the tool must not be advertised.
        assert registry is not None
        assert {tool.name for tool in tools} == {"read_resource"}
        inspector.assert_not_called()

    async def test_with_exa_a_prepared_run_uses_the_agentic_path(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        _install_answer_synthesizer(manager, MagicMock())
        monkeypatch.setattr("dlightrag.answer.executor.AnswerOrchestrator", _CapturingOrchestrator)
        profiles: dict[ModelRole, ModelProfile] = {
            role: manager._capabilities.model_profile(role) for role in MODEL_ROLE_NAMES
        }

        await _answer_executor(manager).prepare_orchestrated_run(
            workspaces=["alpha"],
            top_k=None,
            chunk_top_k=None,
            filters=None,
            resources=None,
            pinned_image_descriptions=(),
            projected_history=PriorTurns(),
            model_profiles=profiles,
        )

        assert _CapturingOrchestrator.last["init"]["search_web"] is not None

    async def test_prepared_run_does_not_reproject_persisted_history(
        self, test_cfg, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        _install_answer_synthesizer(manager, MagicMock())
        manager.retrieval.schema_for = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("schema lookup is unreachable")
        )
        monkeypatch.setattr("dlightrag.answer.executor.AnswerOrchestrator", _CapturingOrchestrator)
        monkeypatch.setattr(
            "dlightrag.core.servicemanager.project_history",
            MagicMock(side_effect=AssertionError("history projection is unreachable")),
        )
        projected = PriorTurns(
            [
                {"role": "user", "content": "persisted question"},
                {"role": "assistant", "content": "persisted answer"},
            ]
        )
        profiles: dict[ModelRole, ModelProfile] = {
            role: manager._capabilities.model_profile(role) for role in MODEL_ROLE_NAMES
        }

        run = await _answer_executor(manager).prepare_orchestrated_run(
            workspaces=["alpha"],
            top_k=None,
            chunk_top_k=None,
            filters=None,
            resources=None,
            pinned_image_descriptions=(),
            projected_history=projected,
            model_profiles=profiles,
        )

        assert run.history is projected

    async def test_acceptance_projection_does_not_build_the_execution_rig(
        self,
        test_cfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        manager.retrieval.schema_for = AsyncMock(return_value={})  # type: ignore[method-assign]
        manager._answer_models._web_search = AsyncMock()
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(llm_func=None, model_profile=_TEST_PLANNER_PROFILE),
        )
        monkeypatch.setattr(
            "dlightrag.answer.executor.AnswerOrchestrator",
            MagicMock(side_effect=AssertionError("orchestrator is execution-only")),
        )
        monkeypatch.setattr(
            "dlightrag.answer.model_runtime.ToolModel",
            MagicMock(side_effect=AssertionError("tool model is execution-only")),
        )

        prepared = await manager.aprepare_answer_run_input(
            AnswerRunRequest(query="question", workspaces=("alpha",)),
            resources=None,
            idempotency_fingerprint="public-input",
        )

        assert prepared.idempotency_fingerprint == "public-input"

    async def test_acceptance_closes_resources_when_schema_resolution_fails(
        self,
        test_cfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        registry = AsyncMock()
        models = manager._capabilities.request_model_context(None)
        manager._answer_resources.resolve = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                registry=registry,
                models=models,
                current_images=(),
                research=False,
            )
        )
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        manager.retrieval.schema_for = AsyncMock(  # type: ignore[method-assign]
            side_effect=WorkspaceUnavailableError("database unavailable")
        )

        with pytest.raises(WorkspaceUnavailableError, match="database unavailable"):
            await manager._project_answer_run_acceptance(
                AnswerRunRequest(query="question", workspaces=("alpha",)),
                resources=None,
            )

        registry.aclose.assert_awaited_once()

    async def test_run_preparation_closes_resources_before_ownership_transfer(
        self,
        test_cfg,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        registry = AsyncMock()
        models = manager._capabilities.request_model_context(None)
        manager._answer_resources.resolve = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(registry=registry, research=True, models=models)
        )
        manager._answer_models.query_tool_model = MagicMock(  # type: ignore[method-assign]
            side_effect=RAGServiceUnavailableError("manager closed")
        )
        profiles: dict[ModelRole, ModelProfile] = {
            role: manager._capabilities.model_profile(role) for role in MODEL_ROLE_NAMES
        }

        with pytest.raises(RAGServiceUnavailableError, match="manager closed"):
            await _answer_executor(manager).prepare_orchestrated_run(
                workspaces=["alpha"],
                top_k=None,
                chunk_top_k=None,
                filters=None,
                resources=None,
                pinned_image_descriptions=(),
                projected_history=PriorTurns(),
                model_profiles=profiles,
            )

        registry.aclose.assert_awaited_once()

    async def test_acceptance_pins_the_profile_snapshot_it_projected(
        self,
        test_cfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = RAGServiceManager(config=test_cfg)
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        manager.retrieval.schema_for = AsyncMock(return_value={})  # type: ignore[method-assign]
        _install_retrieval_planner(
            manager,
            RetrievalPlanner(llm_func=None, model_profile=_TEST_PLANNER_PROFILE),
        )
        project = manager._project_answer_run_acceptance

        async def project_then_narrow(*args: Any, **kwargs: Any):
            projection = await project(*args, **kwargs)
            manager._capabilities._profiles["query"] = ModelProfile(
                context_window_tokens=10_000,
                supports_images=False,
            )
            return projection

        monkeypatch.setattr(manager, "_project_answer_run_acceptance", project_then_narrow)

        prepared = await manager.aprepare_answer_run_input(
            AnswerRunRequest(query="question", workspaces=("alpha",)),
            resources=None,
            idempotency_fingerprint="public-input",
        )

        query_pin = next(item for item in prepared.pinned_models if item.role == "query")
        assert query_pin.profile.context_window_tokens == 400_000
        assert query_pin.profile.supports_images is True

    async def test_agentic_kb_tool_plans_the_agent_query_lazily(self, test_cfg) -> None:
        from dlightrag_ai.messages import AssistantTurn, ToolCall

        from dlightrag.answer.synthesizer import AnswerSynthesizer

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
            return_value=RetrievalResult()
        )
        manager._answer_models._web_search = AsyncMock()
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
        model.stream_text = _scripted_stream("Answer.")
        manager._answer_models._query_tool_model = model
        _install_answer_synthesizer(
            manager,
            AnswerSynthesizer(
                image_policy=manager._capabilities.answer_image_policy(answer_model_profile()),
                model_profile=answer_model_profile(),
                model_func=None,
            ),
        )

        await _durable_answer(
            manager,
            "Original request",
            workspace="alpha",
            history=[
                {"role": "user", "content": "Earlier context"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
        )

        retrieve_call = manager.retrieval.retrieve_result.await_args  # type: ignore[attr-defined]
        assert retrieve_call is not None
        assert retrieve_call.args[0] == "agent chosen terms"
        assert retrieve_call.kwargs["conversation_history"] == [
            {"role": "user", "content": "Earlier context"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        assert retrieve_call.kwargs["preserve_query"] is True
        assert "plan" not in retrieve_call.kwargs

    async def test_agentic_kb_searches_describe_current_images_once(self, test_cfg) -> None:
        from dlightrag_ai.messages import AssistantTurn, ToolCall

        from dlightrag.answer.capability import AnswerImageCapability
        from dlightrag.answer.synthesizer import AnswerSynthesizer

        manager = RAGServiceManager(config=test_cfg)
        manager._capabilities._answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=2,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="vision-test",
            failure_kind=None,
        )
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
        rows = [
            {
                "chunk_id": f"c{index}",
                "reference_id": "upstream",
                "full_doc_id": "doc-1",
                "file_path": "report.pdf",
                "content": f"fact {index}",
                "_workspace": "alpha",
                "metadata": {
                    "source_type": "file",
                    "source_uri": "file:///alpha/report.pdf",
                    "source_download_locator": "file:///alpha/report.pdf",
                },
            }
            for index in (1, 2)
        ]
        manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                RetrievalResult(contexts={"chunks": [rows[0]]}),
                RetrievalResult(contexts={"chunks": [rows[1]]}),
            ]
        )
        describer = AsyncMock()
        describer.describe.return_value = ["Image 1: chart"]
        manager._answer_models.query_image_describer = MagicMock(  # type: ignore[method-assign]
            return_value=describer
        )
        model = AsyncMock(
            side_effect=[
                AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id="kb-1", name="search_knowledge_base", arguments={"query": "one"}
                        ),
                    ),
                    stop_reason="tool_use",
                ),
                AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id="kb-2", name="search_knowledge_base", arguments={"query": "two"}
                        ),
                    ),
                    stop_reason="tool_use",
                ),
                AssistantTurn(text="ready", tool_calls=(), stop_reason="stop"),
            ]
        )
        model.complete_text = AsyncMock(return_value="Answer [1-1].")
        model.stream_text = _scripted_stream("Answer [1-1].")
        manager._answer_models._query_tool_model = model
        _install_answer_synthesizer(
            manager,
            AnswerSynthesizer(
                image_policy=manager._capabilities.answer_image_policy(answer_model_profile()),
                model_profile=answer_model_profile(),
                model_func=None,
            ),
        )

        await _durable_answer(
            manager,
            "Read this chart",
            workspace="alpha",
            resources=[
                ResourceInput(
                    filename="chart.png",
                    content=_png_bytes(),
                    declared_mime="image/png",
                )
            ],
        )

        describer.describe.assert_awaited_once()
        assert manager.retrieval.retrieve_result.await_count == 2  # type: ignore[attr-defined]
        for call in manager.retrieval.retrieve_result.await_args_list:  # type: ignore[attr-defined]
            assert call.kwargs["conversation_history"] == []
            assert call.kwargs["preserve_query"] is True
            assert call.kwargs["image_descriptions"] == ["Image 1: chart"]

    async def test_agentic_answer_plans_once_and_runs_both_evidence_sources(self, test_cfg) -> None:
        from dlightrag_ai.messages import AssistantTurn, ToolCall

        from dlightrag.answer.synthesizer import AnswerSynthesizer
        from dlightrag.answer.tools.web import WebSearchHit, WebSearchResult

        cfg = test_cfg.model_copy(update={"web_search": WebSearchConfig(api_key="k")})
        manager = RAGServiceManager(config=cfg)
        manager.retrieval.warm = MagicMock()  # type: ignore[method-assign]
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
        manager.retrieval.retrieve_result = AsyncMock(  # type: ignore[method-assign]
            return_value=corpus
        )
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
        manager._answer_models._web_search = web
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
        model.stream_text = _scripted_stream("Answer [1-1][2-1].")
        manager._answer_models._query_tool_model = model
        # A real synthesizer owns the tools-disabled final call for research too.
        _install_answer_synthesizer(
            manager,
            AnswerSynthesizer(
                image_policy=manager._capabilities.answer_image_policy(answer_model_profile()),
                model_profile=answer_model_profile(),
                model_func=None,
            ),
        )

        result = await _durable_answer(manager, "What about it?", workspace="alpha")

        assert result.answer == "Answer [1-1][2-1]."
        assert [source.id for source in result.sources] == ["1", "2"]
        model.complete_text.assert_not_awaited()
        retrieve_call = manager.retrieval.retrieve_result.await_args  # type: ignore[attr-defined]
        assert retrieve_call is not None
        assert retrieve_call.args[0] == "inflation 2026"
        assert retrieve_call.kwargs["conversation_history"] == []
        assert retrieve_call.kwargs["preserve_query"] is True
        assert "plan" not in retrieve_call.kwargs
        web.search.assert_awaited_once_with("inflation 2026")
