# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the inline Retrieval application service."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from dlightrag.application.retrieval import (
    CorpusUnavailableError,
    ProjectedRetrieval,
    RetrievalService,
    RetrievalSettings,
    RetrievalTimeoutError,
    RetrieveProjection,
    RetrieveRequest,
)
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.telemetry import NoopTelemetry
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalResult
from dlightrag.engine.rag.retrieval.runtime import RetrievalPlannerRuntime

_PROJECTION = RetrieveProjection(
    downloadable_workspaces=frozenset(),
    visual_workspaces=frozenset(),
)


class _Planner:
    def __init__(self, plan=None) -> None:
        self._plan = plan

    async def plan(self, query: str, **_kwargs):
        return self._plan or SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )


class _Planners:
    def __init__(self, planner=None) -> None:
        self._planner = planner or _Planner()

    def planner_for(self, model_profile: Any | None = None) -> Any:
        del model_profile
        return self._planner

    async def aclose(self) -> None:
        return None


async def test_timeout_bounds_only_the_inline_retrieval_request() -> None:
    blocked = asyncio.Event()

    async def block(*_args, **_kwargs):
        await blocked.wait()

    runtime = AsyncMock()
    runtime.aretrieve.side_effect = block
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=0.01,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    with pytest.raises(RetrievalTimeoutError, match="timed out"):
        await service.retrieve(
            RetrieveRequest(
                query="report",
                workspaces=("finance",),
                projection=_PROJECTION,
            )
        )

    assert service.closed is False


async def test_explicit_filters_and_bm25_override_planner_inference() -> None:
    inferred = MetadataFilter(author="Planner")
    explicit = MetadataFilter(author="Caller")
    planner = _Planner(
        SimpleNamespace(
            standalone_query="standalone",
            metadata_filter=inferred,
            metadata_filter_source="llm_inferred",
            bm25_query="inferred lexical",
            outcome="planned",
        )
    )
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    await service.retrieve(
        RetrieveRequest(
            query="report",
            workspaces=("finance",),
            projection=_PROJECTION,
            top_k=9,
            chunk_top_k=6,
            bm25_query="caller lexical",
            filters=explicit,
        )
    )

    assert runtime.aretrieve.await_args.args == ("standalone",)
    kwargs = runtime.aretrieve.await_args.kwargs
    assert kwargs["top_k"] == 9
    assert kwargs["chunk_top_k"] == 6
    assert kwargs["filters"] is explicit
    assert kwargs["filter_source"] == "explicit"
    assert kwargs["bm25_query"] == "caller lexical"


async def test_schema_cache_is_set_keyed_bounded_and_uses_stale_on_refresh_failure() -> None:
    now = 0.0
    lookup = AsyncMock(return_value={"revision": "current"})
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
        clock=lambda: now,
    )

    first = await service.schema_for(("legal", "finance"))
    reordered = await service.schema_for(("finance", "legal"))
    assert first == reordered == {"revision": "current"}
    lookup.assert_awaited_once_with(("finance", "legal"))

    now = 301.0
    lookup.side_effect = RuntimeError("unavailable")
    assert await service.schema_for(("legal", "finance")) == {"revision": "current"}

    lookup.side_effect = None
    lookup.return_value = {}
    for index in range(129):
        await service.schema_for((f"workspace_{index}",))
    lookup.reset_mock()
    await service.schema_for(("workspace_0",))
    lookup.assert_awaited_once_with(("workspace_0",))


async def test_cold_schema_failure_is_retried_and_recovered() -> None:
    lookup = AsyncMock(side_effect=RuntimeError("database unavailable"))
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    assert await service.schema_for(("reports",)) == {}
    lookup.side_effect = None
    lookup.return_value = {"custom_keys": ["department"]}

    assert await service.schema_for(("reports",)) == {"custom_keys": ["department"]}
    assert lookup.await_count == 2


async def test_schema_lookup_is_single_flight_and_cache_is_not_mutable_by_callers() -> None:
    release = asyncio.Event()
    lookup_started = asyncio.Event()

    async def lookup(_workspaces):
        lookup_started.set()
        await release.wait()
        return {"custom_keys": ["department"]}

    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    first = asyncio.create_task(service.schema_for(("reports",)))
    second = asyncio.create_task(service.schema_for(("reports",)))
    await lookup_started.wait()
    release.set()
    first_schema, second_schema = await asyncio.gather(first, second)

    assert first_schema == second_schema == {"custom_keys": ["department"]}
    first_schema["poisoned"] = True
    first_schema["custom_keys"].append("poisoned")
    assert await service.schema_for(("reports",)) == {"custom_keys": ["department"]}


async def test_schema_ttl_starts_when_successful_refresh_enters_cache() -> None:
    now = 0.0

    async def slow_lookup(_workspaces):
        nonlocal now
        now = 299.0
        return {"revision": "current"}

    lookup = AsyncMock(side_effect=slow_lookup)
    service = RetrievalService(
        pool=AsyncMock(),
        planners=_Planners(),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
        clock=lambda: now,
    )

    assert await service.schema_for(("reports",)) == {"revision": "current"}
    now = 300.0
    assert await service.schema_for(("reports",)) == {"revision": "current"}
    lookup.assert_awaited_once()


async def test_raw_retrieval_uses_history_profile_without_inline_projection_or_timeout() -> None:
    profile = ModelProfile(context_window_tokens=10_000)
    planner = AsyncMock()
    planner.plan.return_value = SimpleNamespace(
        standalone_query="standalone",
        metadata_filter=None,
        metadata_filter_source=None,
        bm25_query=None,
        outcome="planned",
    )
    planners = Mock()
    planners.planner_for.return_value = planner
    planners.aclose = AsyncMock()
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.side_effect = AssertionError("raw retrieval must not project")
    service = RetrievalService(
        pool=pool,
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(side_effect=AssertionError("images are already prepared")),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=0.0001,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )
    history = ({"role": "user", "content": "earlier"},)

    result = await service.retrieve_result(
        "query",
        workspaces=("reports",),
        conversation_history=history,
        image_descriptions=("Image 1: chart",),
        preserve_query=True,
        model_profile=profile,
    )

    assert result is runtime.aretrieve.return_value
    planners.planner_for.assert_called_once_with(profile)
    assert planner.plan.await_args is not None
    assert planner.plan.await_args.kwargs["conversation_history"] == history
    assert planner.plan.await_args.kwargs["preserve_query"] is True
    projector.assert_not_called()


async def test_requested_workspace_schema_is_passed_to_planner() -> None:
    schemas = {
        ("reports",): {"custom_keys": ["department"]},
        ("legal",): {"custom_keys": ["jurisdiction"]},
    }
    lookup = AsyncMock(side_effect=lambda workspaces: schemas[workspaces])
    planner = AsyncMock()
    planner.plan.return_value = SimpleNamespace(
        standalone_query="query",
        metadata_filter=None,
        metadata_filter_source=None,
        bm25_query=None,
        outcome="planned",
    )
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=lookup,
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    await service.retrieve(
        RetrieveRequest(query="query", workspaces=("reports",), projection=_PROJECTION)
    )
    await service.retrieve(
        RetrieveRequest(query="query", workspaces=("legal",), projection=_PROJECTION)
    )

    assert planner.plan.await_args_list[0].kwargs["schema"] == schemas[("reports",)]
    assert planner.plan.await_args_list[1].kwargs["schema"] == schemas[("legal",)]


async def test_retrieve_starts_workspace_warmup_before_planning() -> None:
    warm_started = asyncio.Event()
    release_warm = asyncio.Event()
    plan_started = asyncio.Event()

    async def warm(_workspaces) -> None:
        warm_started.set()
        await release_warm.wait()

    async def plan(query: str, **_kwargs):
        await warm_started.wait()
        plan_started.set()
        return SimpleNamespace(
            standalone_query=query,
            metadata_filter=None,
            metadata_filter_source=None,
            bm25_query=None,
            outcome="planned",
        )

    planner = AsyncMock()
    planner.plan.side_effect = plan
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.warm.side_effect = warm
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(planner),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    task = asyncio.create_task(
        service.retrieve(
            RetrieveRequest(query="query", workspaces=("reports",), projection=_PROJECTION)
        )
    )
    try:
        await plan_started.wait()
    finally:
        release_warm.set()

    await task
    pool.warm.assert_awaited_once_with(("reports",))


async def test_close_cancels_warmups_and_closed_service_starts_no_new_warmup() -> None:
    warm_started = asyncio.Event()
    warm_cancelled = asyncio.Event()

    async def warm(_workspaces) -> None:
        warm_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            warm_cancelled.set()

    pool = AsyncMock()
    pool.warm.side_effect = warm
    planners = _Planners()
    service = RetrievalService(
        pool=pool,
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    service.warm(("reports",))
    await warm_started.wait()
    await service.aclose()
    await warm_cancelled.wait()
    service.warm(("legal",))

    pool.warm.assert_awaited_once_with(("reports",))

    request = RetrieveRequest(query="closed", workspaces=("reports",), projection=_PROJECTION)
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        await service.retrieve(request)
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        await service.retrieve_result("closed", workspaces=("reports",))
    with pytest.raises(CorpusUnavailableError, match="Retrieval service is closed"):
        service.planner_for()


async def test_concurrent_service_close_callers_join_the_same_cleanup() -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    planners = _Planners()

    async def close_planners() -> None:
        close_started.set()
        await release_close.wait()

    planners.aclose = close_planners  # type: ignore[method-assign]
    service = RetrievalService(
        pool=AsyncMock(),
        planners=planners,
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=Mock(),
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
        ),
        telemetry=NoopTelemetry(),
    )

    first = asyncio.create_task(service.aclose())
    await close_started.wait()
    second = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert not second.done()
    release_close.set()

    await asyncio.gather(first, second)


async def test_query_images_are_limited_described_and_forwarded() -> None:
    images = ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},)
    image_preparer = AsyncMock(return_value=["Image 1: chart"])
    runtime = AsyncMock()
    runtime.aretrieve.return_value = RetrievalResult()
    pool = AsyncMock()
    pool.acquire.return_value = runtime
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=image_preparer,
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=1,
        ),
        telemetry=NoopTelemetry(),
    )

    response = await service.retrieve(
        RetrieveRequest(
            query="query",
            workspaces=("reports",),
            projection=_PROJECTION,
            query_images=images,
        )
    )

    image_preparer.assert_awaited_once_with(images)
    assert runtime.aretrieve.await_args.kwargs["query_image_blocks"] == list(images)
    assert response.image_descriptions == ("Image 1: chart",)
    with pytest.raises(ValueError, match="at most 1 current images"):
        await service.retrieve(
            RetrieveRequest(
                query="query",
                workspaces=("reports",),
                projection=_PROJECTION,
                query_images=images * 2,
            )
        )


async def test_multiple_workspaces_use_federated_retrieval() -> None:
    pool = AsyncMock()
    projector = Mock()
    projector.return_value = ProjectedRetrieval(contexts={}, sources=())
    service = RetrievalService(
        pool=pool,
        planners=_Planners(),
        schema_lookup=AsyncMock(return_value={}),
        image_preparer=AsyncMock(return_value=[]),
        projector=projector,
        settings=RetrievalSettings(
            default_top_k=8,
            default_chunk_top_k=5,
            timeout_seconds=30,
            query_image_limit=4,
            workspace_fanout_concurrency=3,
        ),
        telemetry=NoopTelemetry(),
    )

    with patch(
        "dlightrag.application.retrieval.service.federated_retrieve",
        new=AsyncMock(return_value=RetrievalResult()),
    ) as federated:
        await service.retrieve(
            RetrieveRequest(
                query="query",
                workspaces=("reports", "legal"),
                projection=_PROJECTION,
            )
        )

    federated.assert_awaited_once()
    assert federated.await_args is not None
    assert federated.await_args.args[:2] == ("query", ["reports", "legal"])
    acquire = federated.await_args.args[2]
    assert acquire is not pool.acquire  # the service wraps acquire to translate pool errors
    assert federated.await_args.kwargs["max_concurrency"] == 3
    pool.acquire.assert_not_awaited()


async def test_planner_runtime_caches_by_profile_and_closes_its_model_once() -> None:
    default_profile = ModelProfile(context_window_tokens=200_000)
    pinned_profile = ModelProfile(context_window_tokens=125_000)
    model_settings = Mock()
    scheduler = Mock()
    model = AsyncMock()

    with patch(
        "dlightrag.engine.rag.retrieval.runtime.CompletionModel",
        return_value=model,
    ) as create_model:
        runtime = RetrievalPlannerRuntime(
            model_settings=model_settings,
            default_profile=lambda: default_profile,
            scheduler=scheduler,
            telemetry=NoopTelemetry(),
        )
        default_planner = runtime.planner_for()
        pinned_planner = runtime.planner_for(pinned_profile)

        assert runtime.planner_for() is default_planner
        assert runtime.planner_for(pinned_profile) is pinned_planner
        assert pinned_planner is not default_planner
        create_model.assert_called_once_with(
            model_settings,
            scheduler=scheduler,
            telemetry=ANY,
        )

        await runtime.aclose()
        await runtime.aclose()

    model.aclose.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.planner_for()


async def test_concurrent_planner_runtime_close_callers_join_model_cleanup() -> None:
    default_profile = ModelProfile(context_window_tokens=200_000)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    model = AsyncMock()

    async def close_model() -> None:
        close_started.set()
        await release_close.wait()

    model.aclose.side_effect = close_model
    with patch("dlightrag.engine.rag.retrieval.runtime.CompletionModel", return_value=model):
        runtime = RetrievalPlannerRuntime(
            model_settings=Mock(),
            default_profile=lambda: default_profile,
            scheduler=Mock(),
            telemetry=NoopTelemetry(),
        )
        runtime.planner_for()
        first = asyncio.create_task(runtime.aclose())
        await close_started.wait()
        second = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)
        assert not second.done()
        release_close.set()

        await asyncio.gather(first, second)

    model.aclose.assert_awaited_once()
