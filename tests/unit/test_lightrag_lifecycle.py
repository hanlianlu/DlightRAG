# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG worker-pool lifecycle helpers."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import pytest

from dlightrag.core.lightrag_lifecycle import shutdown_lightrag_worker_pools


def _shutdown_target() -> SimpleNamespace:
    return SimpleNamespace(shutdown=lambda *, graceful=True: None)


class TestShutdownLightRagWorkerPools:
    async def test_unwraps_and_deduplicates_shutdown_targets(self) -> None:
        shared = _shutdown_target()
        rerank = _shutdown_target()
        ignored = SimpleNamespace()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=shared),
            llm_model_func=SimpleNamespace(func=shared),
            rerank_model_func=rerank,
            role_llm_funcs={
                "query": SimpleNamespace(func=shared),
                "answer": SimpleNamespace(func=rerank),
                "extract": SimpleNamespace(func=ignored),
            },
        )

        with patch(
            "dlightrag.core.lightrag_lifecycle.shutdown_async_callable",
            new_callable=AsyncMock,
        ) as shutdown:
            count = await shutdown_lightrag_worker_pools(lightrag)

        assert count == 2
        shutdown.assert_has_awaits([call(shared), call(rerank)], any_order=True)
        assert shutdown.await_count == 2

    async def test_dry_run_counts_without_shutting_down(self) -> None:
        embedding = _shutdown_target()
        llm = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=embedding),
            llm_model_func=llm,
            rerank_model_func=SimpleNamespace(),
            role_llm_funcs={
                "query": SimpleNamespace(func=embedding),
                "answer": SimpleNamespace(func=llm),
            },
        )

        with patch(
            "dlightrag.core.lightrag_lifecycle.shutdown_async_callable",
            new_callable=AsyncMock,
        ) as shutdown:
            count = await shutdown_lightrag_worker_pools(lightrag, dry_run=True)

        assert count == 2
        shutdown.assert_not_awaited()

    async def test_dry_run_counts_discovered_targets_but_real_mode_counts_only_successful_shutdowns(
        self, caplog
    ) -> None:
        broken = _shutdown_target()
        healthy = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=broken),
            llm_model_func=None,
            rerank_model_func=healthy,
            role_llm_funcs={},
        )

        async def _shutdown(func: object) -> None:
            if func is broken:
                raise RuntimeError("boom")

        with (
            patch(
                "dlightrag.core.lightrag_lifecycle.shutdown_async_callable",
                new_callable=AsyncMock,
                side_effect=_shutdown,
            ) as shutdown,
            caplog.at_level(logging.DEBUG),
        ):
            dry_run_count = await shutdown_lightrag_worker_pools(lightrag, dry_run=True)
            real_count = await shutdown_lightrag_worker_pools(lightrag)

        assert dry_run_count == 2
        assert real_count == 1
        shutdown.assert_has_awaits([call(broken), call(healthy)])
        assert "Failed to shutdown embedding_func worker pool" in caplog.text

    async def test_preserves_cancellation(self) -> None:
        target = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=target),
            llm_model_func=None,
            rerank_model_func=None,
            role_llm_funcs={},
        )

        with patch(
            "dlightrag.core.lightrag_lifecycle.shutdown_async_callable",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ):
            with pytest.raises(asyncio.CancelledError):
                await shutdown_lightrag_worker_pools(lightrag)
