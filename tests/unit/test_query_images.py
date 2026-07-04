# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VLM-assisted query image semantic enhancement."""

import asyncio
from unittest.mock import AsyncMock

from dlightrag.core.query_images import QueryImageEnhancer

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _image_block(payload: str = _PNG_B64) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


async def test_query_image_enhancer_appends_descriptions() -> None:
    vlm = AsyncMock(return_value="a line chart about revenue")
    enhancer = QueryImageEnhancer(vlm_func=vlm, max_images=1)

    result = await enhancer.enhance("find similar pages", [_image_block()])

    assert result.query.startswith("find similar pages")
    assert "Visual context from user-supplied images" in result.query
    assert result.descriptions == ["Image 1: a line chart about revenue"]
    vlm.assert_awaited_once()
    await_args = vlm.await_args
    assert await_args is not None
    content = await_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"


async def test_query_image_enhancer_is_best_effort() -> None:
    vlm = AsyncMock(side_effect=RuntimeError("vlm unavailable"))
    enhancer = QueryImageEnhancer(vlm_func=vlm, max_images=1)

    result = await enhancer.enhance("plain query", [_image_block()])

    assert result.query == "plain query"
    assert result.descriptions == []


async def test_query_image_enhancer_without_vlm_returns_original_query() -> None:
    vlm = AsyncMock()
    enhancer = QueryImageEnhancer(vlm_func=None, max_images=1)

    result = await enhancer.enhance("plain query", [_image_block()])

    assert result.query == "plain query"
    assert result.descriptions == []
    vlm.assert_not_called()


async def test_query_image_enhancer_describes_images_concurrently() -> None:
    active = 0
    peak = 0

    async def vlm(**kwargs) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "visual detail"

    enhancer = QueryImageEnhancer(vlm_func=vlm, max_images=3)

    result = await enhancer.enhance("query", [_image_block(), _image_block(), _image_block()])

    assert len(result.descriptions) == 3
    assert peak > 1
