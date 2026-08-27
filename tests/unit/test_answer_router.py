# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""AnswerModeRouter and durable resolve helpers."""

import pytest

from dlightrag.application.answer_runs.routing import decide_resolved_mode
from dlightrag.engine.answer.router import AnswerModeRouter, RoutingFailedError


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


async def test_router_defaults_to_research_and_reads_full_context() -> None:
    captured: dict[str, object] = {}

    async def llm(**kwargs: object) -> str:
        captured.update(kwargs)
        return '{"mode":"fast"}'

    history = [{"role": "user", "content": "根据刚上传的年报"}]
    await AnswerModeRouter(llm).choose(
        query="净利润是多少",
        history=history,
        valid_modes=("fast", "research"),
    )
    messages = captured["messages"]
    assert isinstance(messages, list)
    system = messages[0]["content"]
    assert "Default research" in system
    assert "Unsure → research" in system
    assert messages[1] == history[0]
    assert "净利润是多少" in messages[-1]["content"]


async def test_router_accepts_a_bare_mode_token() -> None:
    async def llm(**_kwargs: object) -> str:
        return "research"

    chosen = await AnswerModeRouter(llm).choose(
        query="write a poem",
        valid_modes=("fast", "research"),
    )
    assert chosen == "research"


async def test_router_rejects_invalid_structured_output() -> None:
    async def llm(**_kwargs: object) -> str:
        return "not-json"

    with pytest.raises(RoutingFailedError):
        await AnswerModeRouter(llm).choose(query="q", valid_modes=("fast", "research"))
