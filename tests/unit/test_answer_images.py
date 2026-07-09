# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-model image budgeting."""

import base64
import io

from PIL import Image

from dlightrag.core.answer_images import AnswerImageBudget
from dlightrag.utils.images import (
    bounded_embedding_image_data_uri,
    bounded_image_data_uri,
    decode_image_base64,
)

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_answer_image_budget_bounds_base64_images() -> None:
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    block = budget.add_base64(_PNG_B64, label="chunk:c1")

    assert block is not None
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    assert budget.count == 1
    assert budget.add_base64(_PNG_B64, label="chunk:c2") is None


def test_answer_image_budget_passes_user_url_without_bytes_count() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=1,
        max_bytes_per_image=1,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    block = budget.add_user_image("https://example.com/chart.png", label="query:1")

    assert block == {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
    assert budget.count == 1
    assert budget.used_bytes == 0


def test_answer_image_budget_bounds_dict_data_uri_blocks() -> None:
    raw = _png_bytes((16, 16))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_px=1536,
        min_px=1024,
        quality=88,
        min_quality=72,
    )

    block = budget.add_user_image(
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
        label="query:dict",
    )

    assert block == {
        "type": "image_url",
        "image_url": {"url": data_uri, "detail": "high"},
    }
    assert budget.count == 1
    assert budget.used_bytes == len(raw)


def test_answer_image_budget_rejects_invalid_base64() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )

    assert budget.add_base64("not image data", label="bad") is None
    assert budget.count == 0


def test_bounded_image_data_uri_skips_instead_of_degrading_below_quality_floor() -> None:
    raw = _jpeg_bytes((768, 768), quality=95)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"

    bounded = bounded_image_data_uri(
        data_uri,
        max_bytes=1_000,
        max_px=768,
        min_px=768,
        quality=88,
        min_quality=72,
    )

    assert bounded is None


def test_bounded_image_data_uri_preserves_small_budgeted_png() -> None:
    raw = _png_bytes((32, 32))
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    bounded = bounded_image_data_uri(
        data_uri,
        max_bytes=10_000,
        max_px=1536,
        min_px=1024,
        quality=88,
        min_quality=72,
    )

    assert bounded is not None
    uri, byte_count = bounded
    assert uri == data_uri
    assert byte_count == len(raw)


def test_bounded_embedding_image_keeps_small_image_lossless() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (100, 100), "white"))

    assert uri.startswith("data:image/png;base64,")
    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.size == (100, 100)


def test_bounded_embedding_image_caps_long_edge_and_total_pixels() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (5000, 4000), "white"))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) <= 4096
        assert decoded.width * decoded.height <= 15_000_000


def test_bounded_embedding_image_caps_total_pixels_when_long_edge_fits() -> None:
    # 4000x4000 = 16MP: long edge is under 4096 but the total exceeds the 15MP cap.
    uri = bounded_embedding_image_data_uri(Image.new("RGB", (4000, 4000), "white"))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.width * decoded.height <= 15_000_000


def test_bounded_embedding_image_converges_under_tight_byte_budget() -> None:
    # Noise resists compression, so PNG is skipped and the JPEG/downscale ladder
    # must run to satisfy the byte budget; it always returns a payload.
    noise = Image.effect_noise((2000, 2000), 200).convert("RGB")

    uri = bounded_embedding_image_data_uri(noise, max_bytes=200_000)

    assert uri.startswith("data:image/jpeg;base64,")
    raw, _ = decode_image_base64(uri)
    assert len(raw) <= 200_000
    with Image.open(io.BytesIO(raw)) as decoded:
        assert max(decoded.size) >= 1


def test_bounded_embedding_image_converts_non_rgb_modes() -> None:
    uri = bounded_embedding_image_data_uri(Image.new("RGBA", (128, 128), (255, 0, 0, 128)))

    raw, _ = decode_image_base64(uri)
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.size == (128, 128)


def _jpeg_bytes(size: tuple[int, int], *, quality: int) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
