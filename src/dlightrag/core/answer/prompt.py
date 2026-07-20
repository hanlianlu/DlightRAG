# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer image transport allocation policy and accounting.

A single ``effective_max_images`` counter governs how many raw images reach the
answer model. Allocation is strictly ordered: current-turn images reserve slots
first, then selected history images, then reranked RAG visual chunks. This
computes the per-origin counts used for observability and enforces the
current-image invariant; the shared ``AnswerImageBudget`` applies the same
ordering when the engine places blocks into the final multi-turn messages.
Overflow history and RAG images keep their stored text descriptions; current
images have no silent fallback -- if they cannot all fit, the request fails and
names the overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ImageBlock = dict[str, Any]


class CurrentImagePayloadError(RuntimeError):
    """Current-turn images exceed the effective transport capacity."""


@dataclass(frozen=True, slots=True)
class AssembledImages:
    """Final ordered image blocks and the per-origin allocation counts."""

    blocks: list[ImageBlock]
    counts: dict[str, int]


class AnswerPromptAssembler:
    """Allocate the single answer-image transport budget across origins."""

    def assemble(
        self,
        *,
        current_images: list[ImageBlock],
        history_images: list[ImageBlock],
        rag_visual_blocks: list[ImageBlock],
        effective_max_images: int,
    ) -> AssembledImages:
        if len(current_images) > effective_max_images:
            raise CurrentImagePayloadError(
                f"{len(current_images)} current-turn images exceed the effective "
                f"answer-image capacity of {effective_max_images}"
            )
        blocks: list[ImageBlock] = list(current_images)
        remaining = effective_max_images - len(blocks)
        history = history_images[:remaining]
        blocks.extend(history)
        remaining -= len(history)
        rag = rag_visual_blocks[:remaining]
        blocks.extend(rag)
        return AssembledImages(
            blocks=blocks,
            counts={
                "current": len(current_images),
                "history": len(history),
                "rag": len(rag),
            },
        )


__all__ = ["AnswerPromptAssembler", "AssembledImages", "CurrentImagePayloadError"]
