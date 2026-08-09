# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single context/evidence capacity calculation for unified answers.

``FINAL_GENERATION_CAPACITY_RESERVE`` is input-packing headroom only: it is not
``max_output_tokens``, is not an output cap, and never forces an answer of that
size. Evidence is bounded to at most ``EVIDENCE_RATIO`` of the window, and each
tool observation is bounded to ``MAX_TOOL_OBSERVATION_TOKENS``.
"""

from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_RATIO = 0.60
MAX_TOOL_OBSERVATION_TOKENS = 16_000
FINAL_GENERATION_CAPACITY_RESERVE = 32_768


@dataclass(frozen=True)
class AnswerCapacity:
    """Context-window budget shared by planning, packing, and final synthesis."""

    context_window_tokens: int

    @property
    def final_generation_capacity_reserve(self) -> int:
        return FINAL_GENERATION_CAPACITY_RESERVE

    @property
    def observation_tokens(self) -> int:
        return MAX_TOOL_OBSERVATION_TOKENS

    def evidence_ceiling(self, *, fixed_input_tokens: int) -> int:
        ratio_limit = int(self.context_window_tokens * EVIDENCE_RATIO)
        available = (
            self.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE - fixed_input_tokens
        )
        return max(0, min(ratio_limit, available))


__all__ = [
    "EVIDENCE_RATIO",
    "FINAL_GENERATION_CAPACITY_RESERVE",
    "MAX_TOOL_OBSERVATION_TOKENS",
    "AnswerCapacity",
]
