# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The open-web search result states a streak of searches that found nothing new."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytestmark = pytest.mark.asyncio


async def test_searches_that_add_nothing_say_so_after_a_streak_and_a_find_resets_it() -> None:
    """A live extract child ran one query family 29 times, each adding nothing, until its context
    passed half a million tokens. The result states the streak; it refuses nothing."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.tools.search import _BARREN_SEARCH_STREAK, _search_open_web

    new_chunks = {"value": 0}

    class _Ledger:
        def add_rows(self, rows: list[Any]) -> Any:
            return SimpleNamespace(new_chunks=new_chunks["value"], dropped_rows=0)

        async def aflush_images(self) -> None:
            return None

    async def search(_request: Any) -> Any:
        return SimpleNamespace(
            hits=[], cost_dollars=0.0, provider="stub", degradation=None, dropped_results=0
        )

    async def in_source_order() -> None:
        return None

    runtime = SimpleNamespace(in_source_order=in_source_order)
    trace: dict[str, Any] = {"web_search_cost_dollars": 0.0}

    async def call() -> str:
        result = await _search_open_web(
            cast(Any, search),
            cast(Any, SimpleNamespace()),
            cast(Any, _Ledger()),
            trace,
            None,
            cast(Any, runtime),
            browser_available=False,
        )
        return result.text_content

    texts = [await call() for _ in range(_BARREN_SEARCH_STREAK)]
    assert all("added nothing new" not in text for text in texts[:-1])
    assert f"The last {_BARREN_SEARCH_STREAK} searches added nothing new" in texts[-1]
    new_chunks["value"] = 2
    assert "added nothing new" not in await call()
    new_chunks["value"] = 0
    assert "added nothing new" not in await call()
