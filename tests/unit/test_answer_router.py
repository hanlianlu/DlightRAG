# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AnswerModeRouter and durable resolve helpers."""

import pytest

from dlightrag.answer.router import AnswerModeRouter, RoutingFailedError
from dlightrag.answer.routing import decide_resolved_mode


def test_explicit_mode_resolves_without_a_router() -> None:
    assert (
        decide_resolved_mode(requested_mode="fast", valid_modes=frozenset({"fast", "research"}))
        == "fast"
    )
    assert decide_resolved_mode(requested_mode="research", valid_modes=frozenset({"research"})) == (
        "research"
    )


def test_auto_with_one_valid_mode_skips_the_router() -> None:
    assert decide_resolved_mode(requested_mode="auto", valid_modes=frozenset({"fast"})) == "fast"


def test_auto_with_both_modes_needs_the_router() -> None:
    assert (
        decide_resolved_mode(requested_mode="auto", valid_modes=frozenset({"fast", "research"}))
        is None
    )


async def test_router_accepts_structured_mode() -> None:
    async def llm(**_kwargs: object) -> str:
        return '{"mode":"research"}'

    chosen = await AnswerModeRouter(llm).choose(
        query="compare the filings",
        valid_modes=("fast", "research"),
    )
    assert chosen == "research"


async def test_router_rejects_invalid_structured_output() -> None:
    async def llm(**_kwargs: object) -> str:
        return "not-json"

    with pytest.raises(RoutingFailedError):
        await AnswerModeRouter(llm).choose(query="q", valid_modes=("fast", "research"))
