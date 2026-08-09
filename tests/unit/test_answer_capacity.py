# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the single unified answer capacity calculation."""

from dlightrag.core.answer.capacity import (
    EVIDENCE_RATIO,
    FINAL_GENERATION_CAPACITY_RESERVE,
    MAX_TOOL_OBSERVATION_TOKENS,
    AnswerCapacity,
)
from dlightrag.core.answer.errors import (
    ANSWER_INPUT_OVERFLOW,
    AnswerInputError,
    AnswerInputOverflowError,
    classify_answer_error,
)


def test_default_capacity_reserves_generation_headroom() -> None:
    capacity = AnswerCapacity(260_000)
    assert capacity.evidence_ceiling(fixed_input_tokens=10_000) == 156_000
    assert capacity.final_generation_capacity_reserve == 32_768


def test_observation_tokens_are_bounded() -> None:
    assert AnswerCapacity(260_000).observation_tokens == MAX_TOOL_OBSERVATION_TOKENS
    assert MAX_TOOL_OBSERVATION_TOKENS == 16_000


def test_evidence_ceiling_never_goes_negative() -> None:
    capacity = AnswerCapacity(260_000)
    assert capacity.evidence_ceiling(fixed_input_tokens=1_000_000) == 0


def test_reserve_is_packing_headroom_not_the_evidence_cap() -> None:
    # A smaller window makes the reserve plus fixed input the binding limit,
    # proving the reserve only shapes packing math and is never the ceiling.
    capacity = AnswerCapacity(100_000)
    ratio_limit = int(100_000 * EVIDENCE_RATIO)
    available = 100_000 - FINAL_GENERATION_CAPACITY_RESERVE - 20_000
    assert available < ratio_limit
    assert capacity.evidence_ceiling(fixed_input_tokens=20_000) == available


def test_input_overflow_error_has_stable_kind() -> None:
    error = AnswerInputOverflowError("answer inputs exceed context capacity")
    assert error.error_kind == ANSWER_INPUT_OVERFLOW
    assert isinstance(error, AnswerInputError)
    assert classify_answer_error(error) == ANSWER_INPUT_OVERFLOW
