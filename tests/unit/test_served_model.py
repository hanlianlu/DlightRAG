# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Each adapter hands up the model name its server stated (ADR 0043).

The providers run over their real SDK clients with only the HTTP transport replaced, so the name
read is the one a vendor's response and stream put on the wire. A tool turn carries it on the turn;
a text stream puts it in the wrapper's holder, beside the usage counters and never inside them.
"""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.engine.ai.contracts import ApiFamily
from dlightrag.engine.ai.providers import get_provider
from tests.unit.test_provider_attachment_contract import (
    _MODELS,
    _VIEW_TOOL,
    _HttpCapture,
    bind_mock_http,
)

#: The wire, the API family, what the canned responses of the contract tests name as served.
_WIRES = [
    pytest.param("openai", "chat_completion", "openai", _MODELS["openai"], id="chat-completions"),
    pytest.param("openai", "response", "openai_response", "gpt-5.4", id="responses"),
    pytest.param("anthropic", "chat_completion", "anthropic", _MODELS["anthropic"], id="anthropic"),
]
_MESSAGES = [{"role": "user", "content": "look"}]


def _provider(name: str, family: ApiFamily, wire: str) -> Any:
    provider = get_provider(name, api_key="test-key", api_family=family, max_retries=0)
    bind_mock_http(provider, _HttpCapture(wire).handler)
    return provider


async def _served_by(provider: Any, entrypoint: str) -> str | None:
    """The model name one entry point of ``provider`` hands up, after the checks it owes."""
    if entrypoint == "complete_tool_turn":
        turn = await provider.complete_tool_turn(_MESSAGES, "asked-for", tools=[_VIEW_TOOL])
        return turn.served_model
    if entrypoint == "complete_tool_turn_streaming":
        emitted: list[str] = []

        async def emit_text(text: str) -> None:
            emitted.append(text)

        turn = await provider.complete_tool_turn_streaming(
            _MESSAGES, "asked-for", tools=[_VIEW_TOOL], emit_text=emit_text
        )
        assert emitted
        return turn.served_model

    holder: dict[str, Any] = {}
    tokens = [
        token
        async for token in getattr(provider, entrypoint)(
            _MESSAGES, "asked-for", usage_holder=holder
        )
    ]
    assert tokens
    # The counters stay what they were: integers a sum can take, and nothing else among them.
    assert all(isinstance(value, int) for value in holder["usage_details"].values())
    return holder["served_model"]


@pytest.mark.parametrize(
    "entrypoint",
    ["complete_tool_turn", "complete_tool_turn_streaming", "stream", "stream_tool_text"],
)
@pytest.mark.parametrize(("name", "family", "wire", "served"), _WIRES)
async def test_an_entry_point_hands_up_the_model_the_server_stated(
    name: str, family: ApiFamily, wire: str, served: str, entrypoint: str
) -> None:
    provider = _provider(name, family, wire)

    try:
        # The model asked for is "asked-for"; what the server names is what served the request.
        assert await _served_by(provider, entrypoint) == served
    finally:
        await provider.aclose()
