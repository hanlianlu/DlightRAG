# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opaque provider replay is gated by the exact source model fingerprint."""

from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.replay import bind_provider_replay, messages_for_model

_SOURCE = ModelFingerprint("openai", "model-a", "endpoint-a")


def _turn() -> AssistantTurn:
    return AssistantTurn(
        text="",
        reasoning="private thought",
        tool_calls=(
            ToolCall(
                id="call-1",
                name="search",
                arguments={"q": "x"},
                thought_signature="opaque-tool-signature",
            ),
        ),
        stop_reason="tool_use",
        provider_state={"reasoning_details": [{"data": "opaque-reasoning"}]},
    )


def _message(turn: AssistantTurn) -> dict[str, object]:
    call = turn.tool_calls[0]
    return {
        "role": "assistant",
        "content": turn.text,
        "reasoning": turn.reasoning,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "thought_signature": call.thought_signature,
            }
        ],
        "provider_state": turn.provider_state,
    }


def test_same_fingerprint_replays_opaque_state_and_tool_signature() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)

    prepared = messages_for_model([_message(bound)], _SOURCE)[0]

    assert prepared["provider_state"] == {"reasoning_details": [{"data": "opaque-reasoning"}]}
    assert prepared["tool_calls"][0]["thought_signature"] == "opaque-tool-signature"


def test_cross_model_drops_opaque_reasoning_and_tool_signatures_but_keeps_plain_text() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)
    target = ModelFingerprint("openai", "model-b", "endpoint-a")

    prepared = messages_for_model([_message(bound)], target)[0]

    assert "provider_state" not in prepared
    assert "thought_signature" not in prepared["tool_calls"][0]
    assert prepared["reasoning"] == "private thought"


def test_same_model_name_at_a_different_endpoint_is_not_the_same_replay_identity() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)
    target = ModelFingerprint("openai", "model-a", "endpoint-b")

    prepared = messages_for_model([_message(bound)], target)[0]

    assert "provider_state" not in prepared
    assert "thought_signature" not in prepared["tool_calls"][0]


def test_legacy_unbound_provider_state_is_never_replayed() -> None:
    prepared = messages_for_model(
        [
            {
                "role": "assistant",
                "content": "answer",
                "provider_state": {"signature": "unbound"},
            }
        ],
        _SOURCE,
    )[0]

    assert prepared == {"role": "assistant", "content": "answer"}
