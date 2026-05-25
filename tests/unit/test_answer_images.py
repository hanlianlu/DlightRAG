# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-model image budgeting."""

from __future__ import annotations

import base64
import io

from PIL import Image

from dlightrag.core.answer_images import AnswerImageBudget
from dlightrag.utils.images import bounded_image_data_uri

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO"
    "+/p9sAAAAASUVORK5CYII="
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
