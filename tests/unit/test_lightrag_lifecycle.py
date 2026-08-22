# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG worker-pool lifecycle helpers."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.rag.lifecycle import await_shared_cleanup, shutdown_lightrag_worker_pools


def _shutdown_target() -> SimpleNamespace:
    return SimpleNamespace(shutdown=AsyncMock())


async def test_shared_cleanup_preserves_caller_cancellation_over_cleanup_failure() -> None:
    release = asyncio.Event()

    async def fail_cleanup() -> None:
        await release.wait()
        raise RuntimeError("cleanup failed")

    cleanup = asyncio.create_task(fail_cleanup())
    caller = asyncio.create_task(await_shared_cleanup(cleanup))
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await caller


class TestShutdownLightRagWorkerPools:
    async def test_unwraps_and_deduplicates_shutdown_targets(self) -> None:
        shared = _shutdown_target()
        rerank = _shutdown_target()
        ignored = SimpleNamespace()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=shared),
            _role_llm_states={
                "query": SimpleNamespace(wrapped=SimpleNamespace(func=shared)),
                "answer": SimpleNamespace(wrapped=SimpleNamespace(func=rerank)),
                "extract": SimpleNamespace(wrapped=SimpleNamespace(func=ignored)),
            },
        )

        count = await shutdown_lightrag_worker_pools(lightrag)

        assert count == 2
        shared.shutdown.assert_awaited_once_with(graceful=True)
        rerank.shutdown.assert_awaited_once_with(graceful=True)

    async def test_dry_run_counts_without_shutting_down(self) -> None:
        embedding = _shutdown_target()
        llm = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=embedding),
            _role_llm_states={
                "query": SimpleNamespace(wrapped=SimpleNamespace(func=embedding)),
                "answer": SimpleNamespace(wrapped=SimpleNamespace(func=llm)),
            },
        )

        count = await shutdown_lightrag_worker_pools(lightrag, dry_run=True)

        assert count == 2
        embedding.shutdown.assert_not_called()
        llm.shutdown.assert_not_called()

    async def test_dry_run_counts_discovered_targets_but_real_mode_counts_only_successful_shutdowns(
        self, caplog
    ) -> None:
        broken = _shutdown_target()
        healthy = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=broken),
            _role_llm_states={"query": SimpleNamespace(wrapped=healthy)},
        )

        broken.shutdown.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.DEBUG):
            dry_run_count = await shutdown_lightrag_worker_pools(lightrag, dry_run=True)
            real_count = await shutdown_lightrag_worker_pools(lightrag)

        assert dry_run_count == 2
        assert real_count == 1
        broken.shutdown.assert_awaited_once_with(graceful=True)
        healthy.shutdown.assert_awaited_once_with(graceful=True)
        assert "Failed to shutdown embedding_func worker pool" in caplog.text

    async def test_preserves_cancellation(self) -> None:
        target = _shutdown_target()
        lightrag = SimpleNamespace(
            embedding_func=SimpleNamespace(func=target),
            _role_llm_states={},
        )

        target.shutdown.side_effect = asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await shutdown_lightrag_worker_pools(lightrag)
