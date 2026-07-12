# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VLM-assisted query image semantic enhancement."""

import asyncio
import base64
import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dlightrag.core.query_images import QueryImageEnhancer
from dlightrag.utils.images import decode_image_base64, validate_web_images

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _image_block(payload: str = _PNG_B64) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _enhancer(vlm, *, max_images: int, max_bytes: int = 10_000, max_px: int = 64):
    return QueryImageEnhancer(
        vlm_func=vlm,
        max_images=max_images,
        max_total_bytes=max_bytes,
        max_bytes_per_image=max_bytes,
        max_px=max_px,
        min_px=max_px,
        quality=89,
        min_quality=79,
    )


def _image_data_uri(image: Image.Image, *, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


async def test_query_image_enhancer_appends_descriptions() -> None:
    vlm = AsyncMock(return_value="a line chart about revenue")
    enhancer = _enhancer(vlm, max_images=1)

    result = await enhancer.enhance("find similar pages", [_image_block()])

    assert result.query.startswith("find similar pages")
    assert "Visual context from user-supplied images" in result.query
    assert result.descriptions == {"1": "Image 1: a line chart about revenue"}
    vlm.assert_awaited_once()
    await_args = vlm.await_args
    assert await_args is not None
    content = await_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"


async def test_query_image_enhancer_is_best_effort() -> None:
    vlm = AsyncMock(side_effect=RuntimeError("vlm unavailable"))
    enhancer = _enhancer(vlm, max_images=1)

    result = await enhancer.enhance("plain query", [_image_block()])

    assert result.query == "plain query"
    assert result.descriptions == {}


async def test_query_image_enhancer_without_vlm_returns_original_query() -> None:
    vlm = AsyncMock()
    enhancer = _enhancer(None, max_images=1)

    result = await enhancer.enhance("plain query", [_image_block()])

    assert result.query == "plain query"
    assert result.descriptions == {}
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

    enhancer = _enhancer(vlm, max_images=3)

    result = await enhancer.enhance("query", [_image_block(), _image_block(), _image_block()])

    assert len(result.descriptions) == 3
    assert peak > 1


async def test_query_image_descriptions_keep_sparse_ordinals() -> None:
    vlm = AsyncMock(side_effect=[RuntimeError("first failed"), "second image"])
    enhancer = _enhancer(vlm, max_images=2)

    result = await enhancer.enhance("query", [_image_block(), _image_block()])

    assert result.descriptions == {"2": "Image 2: second image"}


async def test_query_image_vlm_receives_bounded_transport_while_durable_bytes_remain() -> None:
    original_uri = _image_data_uri(Image.effect_noise((256, 192), 180).convert("RGB"))
    raw, _mime = decode_image_base64(original_uri)
    (durable,) = validate_web_images(
        [original_uri],
        max_images=1,
        max_bytes=len(raw),
    )
    original_bytes = durable.image_bytes
    vlm = AsyncMock(return_value="bounded image")
    enhancer = _enhancer(vlm, max_images=1, max_bytes=5_000, max_px=96)

    result = await enhancer.enhance("query", [durable.model_block])

    assert result.descriptions == {"1": "Image 1: bounded image"}
    await_args = vlm.await_args
    assert await_args is not None
    content = await_args.kwargs["messages"][0]["content"]
    bounded_uri = content[0]["image_url"]["url"]
    bounded_raw, _ = decode_image_base64(bounded_uri)
    assert len(bounded_raw) <= 5_000
    with Image.open(io.BytesIO(bounded_raw)) as bounded_image:
        assert max(bounded_image.size) <= 96
    assert durable.image_bytes == original_bytes
    assert durable.data_uri == original_uri


async def test_query_image_compression_skip_preserves_sparse_ordinal_and_sibling() -> None:
    too_large = _image_data_uri(Image.effect_noise((128, 128), 200).convert("RGB"), fmt="JPEG")
    small = _image_data_uri(Image.new("RGB", (1, 1), "white"))
    vlm = AsyncMock(return_value="small image")
    enhancer = _enhancer(vlm, max_images=2, max_bytes=100, max_px=128)

    result = await enhancer.enhance(
        "query",
        [_image_block(too_large.split(",", 1)[1]), _image_block(small.split(",", 1)[1])],
    )

    assert result.descriptions == {"2": "Image 2: small image"}
    vlm.assert_awaited_once()


async def test_query_image_https_url_is_validated_and_passed_to_vlm() -> None:
    vlm = AsyncMock(return_value="remote chart")
    enhancer = _enhancer(vlm, max_images=1)
    image = {
        "type": "image_url",
        "image_url": {"url": "https://example.test/chart.png", "detail": "high"},
    }

    result = await enhancer.enhance("query", [image])

    assert result.descriptions == {"1": "Image 1: remote chart"}
    await_args = vlm.await_args
    assert await_args is not None
    assert await_args.kwargs["messages"][0]["content"][0] == image


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/chart.png",
        "file:///tmp/chart.png",
        "https://127.0.0.1/chart.png",
        "https://[::1]/chart.png",
        "https://0x7f000001/chart.png",
    ],
)
async def test_query_image_unsafe_urls_are_not_sent_to_vlm(url: str) -> None:
    vlm = AsyncMock(return_value="must not be used")
    enhancer = _enhancer(vlm, max_images=1)

    result = await enhancer.enhance(
        "query",
        [{"type": "image_url", "image_url": {"url": url}}],
    )

    assert result.query == "query"
    assert result.descriptions == {}
    vlm.assert_not_awaited()
