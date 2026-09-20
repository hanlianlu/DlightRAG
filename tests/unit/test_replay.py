# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Opaque provider replay is gated by the exact source model fingerprint."""

import pytest

from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall, tool_call_message
from dlightrag.engine.ai.providers.openai_response import response_input
from dlightrag.engine.ai.replay import ProviderReplayError, bind_provider_replay, messages_for_model
from dlightrag.engine.ai.response_policy import ResponseRequestError

_SOURCE = ModelInvocationFingerprint("openai", "model-a", "endpoint-a", "chat_completion")


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
    target = ModelInvocationFingerprint("openai", "model-b", "endpoint-a", "chat_completion")

    prepared = messages_for_model([_message(bound)], target)[0]

    assert "provider_state" not in prepared
    assert "thought_signature" not in prepared["tool_calls"][0]
    assert prepared["reasoning"] == "private thought"


def test_same_invocation_with_an_unknown_replay_envelope_version_fails() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)
    assert bound.provider_state is not None
    identity = bound.provider_state["_dlightrag_replay"]
    assert isinstance(identity, dict)
    identity["v"] = 999

    with pytest.raises(ProviderReplayError, match="version"):
        messages_for_model([_message(bound)], _SOURCE)


def test_same_model_name_at_a_different_endpoint_is_not_the_same_replay_identity() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)
    target = ModelInvocationFingerprint("openai", "model-a", "endpoint-b", "chat_completion")

    prepared = messages_for_model([_message(bound)], target)[0]

    assert "provider_state" not in prepared
    assert "thought_signature" not in prepared["tool_calls"][0]


def test_same_endpoint_with_a_different_api_family_is_not_the_same_replay_identity() -> None:
    bound = bind_provider_replay(_turn(), _SOURCE)
    target = ModelInvocationFingerprint(
        _SOURCE.provider,
        _SOURCE.model,
        _SOURCE.endpoint_fingerprint,
        "response",
    )

    prepared = messages_for_model([_message(bound)], target)[0]

    assert "provider_state" not in prepared
    assert "thought_signature" not in prepared["tool_calls"][0]


def test_cross_family_response_reconstructs_canonical_calls_after_stripping_chat_state() -> None:
    turn = AssistantTurn(
        text="working",
        reasoning="private thought",
        tool_calls=(ToolCall(id="call-1", name="search", arguments={"q": "x"}),),
        stop_reason="tool_use",
        provider_state={"reasoning_content": "private thought"},
    )
    bound = bind_provider_replay(turn, _SOURCE)
    target = ModelInvocationFingerprint(
        _SOURCE.provider,
        _SOURCE.model,
        _SOURCE.endpoint_fingerprint,
        "response",
    )
    prepared = messages_for_model(
        [
            {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [tool_call_message(turn.tool_calls[0])],
                "provider_state": bound.provider_state,
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search",
                "content": "result",
            },
        ],
        target,
    )

    assert response_input(prepared) == [
        {"role": "assistant", "content": "working"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "search",
            "arguments": '{"q":"x"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result",
        },
    ]


def test_same_family_response_replay_rejects_an_unknown_native_version() -> None:
    with pytest.raises(ResponseRequestError, match="version"):
        response_input(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call_message(ToolCall("call-1", "search", {"q": "x"}))],
                    "provider_state": {
                        "response_replay": {
                            "v": 999,
                            "items": [],
                        }
                    },
                }
            ]
        )


def test_same_family_response_replay_rejects_malformed_native_items() -> None:
    with pytest.raises(ResponseRequestError, match="items"):
        response_input(
            [
                {
                    "role": "assistant",
                    "content": "answer",
                    "provider_state": {
                        "response_replay": {
                            "v": 1,
                            "items": [],
                        }
                    },
                }
            ]
        )


def test_response_tool_output_requires_one_matching_prior_call_id() -> None:
    with pytest.raises(ResponseRequestError, match="matching function call"):
        response_input(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-missing",
                    "content": "orphan",
                }
            ]
        )


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
