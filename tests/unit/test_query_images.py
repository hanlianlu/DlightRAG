# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VLM-assisted query image semantic enhancement."""

from __future__ import annotations

from unittest.mock import AsyncMock

from dlightrag.core.query_images import QueryImageEnhancer

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


async def test_query_image_enhancer_appends_descriptions() -> None:
    vlm = AsyncMock(return_value="a line chart about revenue")
    enhancer = QueryImageEnhancer(vlm_func=vlm, enabled=True, max_images=1)

    result = await enhancer.enhance("find similar pages", [_PNG_B64])

    assert result.query.startswith("find similar pages")
    assert "Visual context from user-supplied images" in result.query
    assert result.descriptions == ["Image 1: a line chart about revenue"]
    vlm.assert_awaited_once()
    content = vlm.await_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"


async def test_query_image_enhancer_is_best_effort() -> None:
    vlm = AsyncMock(side_effect=RuntimeError("vlm unavailable"))
    enhancer = QueryImageEnhancer(vlm_func=vlm, enabled=True, max_images=1)

    result = await enhancer.enhance("plain query", [_PNG_B64])

    assert result.query == "plain query"
    assert result.descriptions == []


async def test_query_image_enhancer_disabled_returns_original_query() -> None:
    vlm = AsyncMock()
    enhancer = QueryImageEnhancer(vlm_func=vlm, enabled=False, max_images=1)

    result = await enhancer.enhance("plain query", [_PNG_B64])

    assert result.query == "plain query"
    assert result.descriptions == []
    vlm.assert_not_called()
