# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval-service query-image preparation cache and federation contracts."""

import asyncio
import base64
import io
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from dlightrag.application.retrieval import RetrievalService, RetrievalSettings
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.rag.retrieval.visual import (
    DirectVisualRetriever,
    PreparedVisualQuery,
    VisualEmbeddingDomain,
)

_MISSING = object()


def _image_block(color: tuple[int, int, int]) -> dict[str, Any]:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


class _Embedder:
    supports_images = True
    input_modality = "multimodal"

    def __init__(
        self,
        *,
        provider: str = "visual-provider",
        model: str = "visual-model",
        endpoint: str = "https://embed.example.test/v1/images",
        dim: int = 3,
        outcomes: Sequence[object] = (),
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.request_url = endpoint
        self.base_url = endpoint
        self.dim = dim
        self.api_key = "must-never-enter-domain-or-cache"
        self.calls = 0
        self.outcomes = list(outcomes)
        self.started = started
        self.release = release

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, list):
                return outcome
        return [[float(index + 1), 0.25, 0.5][: self.dim] for index in range(len(images))]


class _Stores:
    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.queries: list[list[float]] = []
        self.chunks_vdb = self

    async def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        vector = list(kwargs["query_embedding"])
        self.queries.append(vector)
        index = len(self.queries)
        return [
            {
                "id": f"{self.workspace}-visual-{index}",
                "content": f"visual {index}",
                "file_path": f"/{self.workspace}.pdf",
                "distance": float(index) / 10,
            }
        ]


class _Runtime:
    def __init__(
        self,
        workspace: str,
        *,
        embedder: _Embedder | None,
        semantic_chunks: Sequence[dict[str, Any]] = (),
    ) -> None:
        self.workspace = workspace
        self.stores = _Stores(workspace)
        self.visual = (
            DirectVisualRetriever(embedder=embedder, stores=self.stores, top_k=8)
            if embedder is not None
            else None
        )
        self.semantic_chunks = list(semantic_chunks)
        self.prepared_arguments: list[object] = []
        self.raw_blocks_forwarded: list[object] = []

    @property
    def visual_embedding_domain(self) -> VisualEmbeddingDomain | None:
        return self.visual.embedding_domain if self.visual is not None else None

    async def prepare_visual_query(
        self, query_image_blocks: list[dict[str, Any]]
    ) -> PreparedVisualQuery | None:
        assert self.visual is not None
        return await self.visual.prepare(query_image_blocks)

    async def aretrieve(
        self,
        query: str,
        *,
        prepared_visual_query: PreparedVisualQuery | None | object = _MISSING,
        **kwargs: Any,
    ) -> RetrievalResult:
        del query
        self.raw_blocks_forwarded.append(kwargs.get("query_image_blocks", _MISSING))
        self.prepared_arguments.append(prepared_visual_query)
        visual_chunks = (
            await self.visual.search_prepared(prepared_visual_query)
            if self.visual is not None and isinstance(prepared_visual_query, PreparedVisualQuery)
            else []
        )
        return RetrievalResult(
            contexts={
                "chunks": [*self.semantic_chunks, *visual_chunks],
                "entities": [],
                "relationships": [],
            }
        )


class _Pool:
    def __init__(self, runtimes: dict[str, _Runtime]) -> None:
        self.runtimes = runtimes

    async def acquire(self, workspace: str) -> _Runtime:
        return self.runtimes[workspace]

    async def warm(self, _workspaces: Sequence[str]) -> None:
        return None


class _Planner:
    async def plan(self, query: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )


class _Planners:
    def planner_for(self, model_profile: Any = None) -> Any:
        del model_profile
        return _Planner()

    async def aclose(self) -> None:
        return None


def _service(runtimes: dict[str, _Runtime]) -> RetrievalService:
    return RetrievalService(
        pool=_Pool(runtimes),  # type: ignore[arg-type]
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=128,
            timeout_seconds=30,
            query_image_limit=64,
            workspace_fanout_concurrency=3,
        ),
        telemetry=NoopTelemetry(),
    )


async def test_four_same_domain_workspaces_prepare_once_and_search_every_vdb() -> None:
    embedder = _Embedder()
    runtimes = {
        workspace: _Runtime(workspace, embedder=embedder)
        for workspace in ("ws-a", "ws-b", "ws-c", "ws-d")
    }
    service = _service(runtimes)
    images = (_image_block((255, 0, 0)), _image_block((0, 255, 0)))

    result = await service.retrieve_result("query", workspaces=tuple(runtimes), query_images=images)

    assert embedder.calls == 1
    assert [chunk["chunk_id"] for chunk in result.contexts["chunks"]] == [
        "ws-a-visual-1",
        "ws-b-visual-1",
        "ws-c-visual-1",
        "ws-d-visual-1",
        "ws-a-visual-2",
        "ws-b-visual-2",
        "ws-c-visual-2",
        "ws-d-visual-2",
    ]
    for runtime in runtimes.values():
        assert runtime.stores.queries == [[1.0, 0.25, 0.5], [2.0, 0.25, 0.5]]
        assert runtime.raw_blocks_forwarded == [_MISSING]
        assert isinstance(runtime.prepared_arguments[0], PreparedVisualQuery)
    assert result.trace["visual_preparation_domain_count"] == 1
    assert result.trace["visual_preparation_started_count"] == 1


async def test_two_domains_prepare_once_each_and_never_cross_domains() -> None:
    first = _Embedder(model="domain-a")
    second = _Embedder(model="domain-b")
    runtimes = {
        "ws-a1": _Runtime("ws-a1", embedder=first),
        "ws-a2": _Runtime("ws-a2", embedder=first),
        "ws-b1": _Runtime("ws-b1", embedder=second),
        "ws-b2": _Runtime("ws-b2", embedder=second),
    }
    service = _service(runtimes)

    await service.retrieve_result(
        "query", workspaces=tuple(runtimes), query_images=(_image_block((1, 2, 3)),)
    )

    assert first.calls == second.calls == 1
    assert all(len(runtime.stores.queries) == 1 for runtime in runtimes.values())
    for runtime in runtimes.values():
        prepared = runtime.prepared_arguments[0]
        assert isinstance(prepared, PreparedVisualQuery)
        assert prepared.domain == runtime.visual_embedding_domain


async def test_repeat_request_hits_cache_and_lru_evicts_oldest_without_raw_storage() -> None:
    embedder = _Embedder()
    runtime = _Runtime("ws", embedder=embedder)
    service = _service({"ws": runtime})
    first_image = _image_block((0, 0, 0))

    first = await service.retrieve_result("query", workspaces=("ws",), query_images=(first_image,))
    repeated = await service.retrieve_result(
        "query", workspaces=("ws",), query_images=(first_image,)
    )

    assert embedder.calls == 1
    assert first.trace["visual_preparation_started_count"] == 1
    assert repeated.trace["visual_preparation_cache_hit_count"] == 1

    for value in range(1, 33):
        await service.retrieve_result(
            "query",
            workspaces=("ws",),
            query_images=(_image_block((value, value, value)),),
        )
    assert embedder.calls == 33
    await service.retrieve_result("query", workspaces=("ws",), query_images=(first_image,))
    assert embedder.calls == 34
    assert len(service._visual_query_cache) == 32
    assert all(
        isinstance(prepared, PreparedVisualQuery)
        for prepared in service._visual_query_cache.values()
    )
    cache_repr = repr(service._visual_query_cache)
    assert "data:image" not in cache_repr
    assert embedder.api_key not in cache_repr


async def test_cancelled_waiter_does_not_cancel_visual_singleflight() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    embedder = _Embedder(started=started, release=release)
    runtime = _Runtime("ws", embedder=embedder)
    service = _service({"ws": runtime})
    image = _image_block((5, 6, 7))

    cancelled_waiter = asyncio.create_task(
        service.retrieve_result("query", workspaces=("ws",), query_images=(image,))
    )
    await started.wait()
    completing_waiter = asyncio.create_task(
        service.retrieve_result("query", workspaces=("ws",), query_images=(image,))
    )
    await asyncio.sleep(0)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    release.set()

    result = await completing_waiter
    await asyncio.sleep(0)

    assert result.contexts["chunks"]
    assert embedder.calls == 1
    assert result.trace["visual_preparation_singleflight_hit_count"] == 1
    assert service._visual_query_flights == {}
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "retrieval-visual-prepare" and not task.done()
    ]


@pytest.mark.parametrize("first_outcome", [RuntimeError("provider down"), []])
async def test_failed_or_empty_federated_preparation_is_shared_but_later_retries(
    first_outcome: object,
) -> None:
    embedder = _Embedder(outcomes=[first_outcome])
    runtimes = {
        workspace: _Runtime(
            workspace,
            embedder=embedder,
            semantic_chunks=[{"chunk_id": f"{workspace}-semantic"}],
        )
        for workspace in ("ws-a", "ws-b", "ws-c", "ws-d")
    }
    service = _service(runtimes)
    image = _image_block((8, 9, 10))

    degraded = await service.retrieve_result(
        "query", workspaces=tuple(runtimes), query_images=(image,)
    )

    assert embedder.calls == 1
    assert all("semantic" in chunk["chunk_id"] for chunk in degraded.contexts["chunks"])
    assert all(runtime.stores.queries == [] for runtime in runtimes.values())
    assert all(runtime.prepared_arguments == [None] for runtime in runtimes.values())
    assert degraded.trace["visual_preparation_failed_count"] == 1

    recovered = await service.retrieve_result(
        "query", workspaces=tuple(runtimes), query_images=(image,)
    )

    assert embedder.calls == 2
    assert any("visual" in chunk["chunk_id"] for chunk in recovered.contexts["chunks"])


async def test_no_images_and_visual_disabled_workspace_do_zero_preparation() -> None:
    embedder = _Embedder()
    enabled = _Runtime("enabled", embedder=embedder)
    disabled = _Runtime("disabled", embedder=None)
    service = _service({"enabled": enabled, "disabled": disabled})

    await service.retrieve_result("query", workspaces=("enabled",))
    await service.retrieve_result(
        "query", workspaces=("disabled",), query_images=(_image_block((11, 12, 13)),)
    )

    assert embedder.calls == 0
    assert enabled.prepared_arguments == [_MISSING]
    assert disabled.prepared_arguments == [_MISSING]


async def test_close_cancels_and_joins_inflight_visual_preparation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    embedder = _Embedder(started=started, release=asyncio.Event())
    original_embed = embedder.embed_query_images

    async def observe_cancel(images: list[Image.Image]) -> list[list[float]]:
        try:
            return await original_embed(images)
        finally:
            cancelled.set()

    embedder.embed_query_images = observe_cancel  # type: ignore[method-assign]
    runtime = _Runtime("ws", embedder=embedder)
    service = _service({"ws": runtime})
    retrieval = asyncio.create_task(
        service.retrieve_result(
            "query", workspaces=("ws",), query_images=(_image_block((14, 15, 16)),)
        )
    )
    await started.wait()

    await service.aclose()
    await cancelled.wait()
    outcome = await asyncio.gather(retrieval, return_exceptions=True)

    assert isinstance(outcome[0], asyncio.CancelledError)
    assert service._visual_query_flights == {}
    assert service._visual_query_cache == {}
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "retrieval-visual-prepare" and not task.done()
    ]
