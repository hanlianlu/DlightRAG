# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for answer-model image budgeting."""

from __future__ import annotations

from dlightrag.core.answer_images import AnswerImageBudget

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
        quality=85,
    )

    block = budget.add_base64(_PNG_B64, label="chunk:c1")

    assert block is not None
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert budget.count == 1
    assert budget.add_base64(_PNG_B64, label="chunk:c2") is None


def test_answer_image_budget_passes_user_url_without_bytes_count() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=1,
        max_bytes_per_image=1,
        max_px=64,
        quality=85,
    )

    block = budget.add_user_image("https://example.com/chart.png", label="query:1")

    assert block == {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
    assert budget.count == 1
    assert budget.used_bytes == 0


def test_answer_image_budget_rejects_invalid_base64() -> None:
    budget = AnswerImageBudget(
        max_images=2,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_px=64,
        quality=85,
    )

    assert budget.add_base64("not image data", label="bad") is None
    assert budget.count == 0
