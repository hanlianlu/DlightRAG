# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the single adaptive answer-image transport assembler."""

import pytest

from dlightrag.core.answer_prompt import AnswerPromptAssembler, CurrentImagePayloadError


def _img(tag: str) -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tag}"}}


def _urls(blocks: list[dict[str, object]]) -> list[str]:
    return [b["image_url"]["url"] for b in blocks]  # type: ignore[index]


def test_allocation_order_current_history_rag() -> None:
    out = AnswerPromptAssembler().assemble(
        current_images=[_img("C1")],
        history_images=[_img("H1"), _img("H2")],
        rag_visual_blocks=[_img("R1")],
        effective_max_images=3,
    )
    assert _urls(out.blocks) == [
        "data:image/png;base64,C1",
        "data:image/png;base64,H1",
        "data:image/png;base64,H2",
    ]
    assert out.counts == {"current": 1, "history": 2, "rag": 0}


def test_rag_fills_remaining_after_current_and_history() -> None:
    out = AnswerPromptAssembler().assemble(
        current_images=[_img("C1")],
        history_images=[_img("H1")],
        rag_visual_blocks=[_img("R1"), _img("R2")],
        effective_max_images=3,
    )
    assert _urls(out.blocks) == [
        "data:image/png;base64,C1",
        "data:image/png;base64,H1",
        "data:image/png;base64,R1",
    ]
    assert out.counts == {"current": 1, "history": 1, "rag": 1}


def test_zero_effective_sends_no_raw_images() -> None:
    out = AnswerPromptAssembler().assemble(
        current_images=[],
        history_images=[_img("H1")],
        rag_visual_blocks=[_img("R1")],
        effective_max_images=0,
    )
    assert out.blocks == []
    assert out.counts == {"current": 0, "history": 0, "rag": 0}


def test_current_over_capacity_raises() -> None:
    with pytest.raises(CurrentImagePayloadError):
        AnswerPromptAssembler().assemble(
            current_images=[_img("C1"), _img("C2")],
            history_images=[],
            rag_visual_blocks=[],
            effective_max_images=1,
        )


def test_current_exactly_fills_capacity_leaves_no_history_or_rag() -> None:
    out = AnswerPromptAssembler().assemble(
        current_images=[_img("C1"), _img("C2")],
        history_images=[_img("H1")],
        rag_visual_blocks=[_img("R1")],
        effective_max_images=2,
    )
    assert _urls(out.blocks) == [
        "data:image/png;base64,C1",
        "data:image/png;base64,C2",
    ]
    assert out.counts == {"current": 2, "history": 0, "rag": 0}
