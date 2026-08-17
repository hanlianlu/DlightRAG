# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Regression tests for closing answer-token streams."""

from dlightrag.answer.citations.streaming import aclose_answer_stream


async def test_aclose_answer_stream_noop_for_none_and_str() -> None:
    await aclose_answer_stream(None)
    await aclose_answer_stream("plain string")
