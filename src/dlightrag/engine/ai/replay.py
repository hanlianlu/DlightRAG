# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-identity gate for opaque provider reasoning replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.messages import AssistantTurn

_ENVELOPE_KEY = "_dlightrag_replay"
_ENVELOPE_VERSION = 1


def bind_provider_replay(
    turn: AssistantTurn,
    fingerprint: ModelFingerprint,
) -> AssistantTurn:
    """Bind opaque response state and Tool signatures to their source model."""
    has_tool_signature = any(call.thought_signature is not None for call in turn.tool_calls)
    if turn.provider_state is None and not has_tool_signature:
        return turn
    return replace(
        turn,
        provider_state={
            _ENVELOPE_KEY: {
                "v": _ENVELOPE_VERSION,
                "provider": fingerprint.provider,
                "model": fingerprint.model,
                "endpoint_fingerprint": fingerprint.endpoint_fingerprint,
            },
            "payload": turn.provider_state,
        },
    )


def messages_for_model(
    messages: list[dict[str, Any]],
    fingerprint: ModelFingerprint,
) -> list[dict[str, Any]]:
    """Unwrap same-model provider state and strip every cross-model opaque value.

    Legacy unbound ``provider_state`` is also stripped: without a durable source
    identity it cannot be replayed safely.
    """
    prepared: list[dict[str, Any]] = []
    for source in messages:
        if source.get("role") != "assistant":
            prepared.append(source)
            continue
        message = dict(source)
        state = message.pop("provider_state", None)
        same_model = _is_same_model_state(state, fingerprint)
        if same_model and isinstance(state, Mapping):
            payload = state.get("payload")
            if payload is not None:
                message["provider_state"] = payload
        elif message.get("tool_calls"):
            message["tool_calls"] = [
                {key: value for key, value in dict(call).items() if key != "thought_signature"}
                if isinstance(call, Mapping)
                else call
                for call in message["tool_calls"]
            ]
        prepared.append(message)
    return prepared


def _is_same_model_state(state: object, fingerprint: ModelFingerprint) -> bool:
    if not isinstance(state, Mapping):
        return False
    identity = state.get(_ENVELOPE_KEY)
    if not isinstance(identity, Mapping) or identity.get("v") != _ENVELOPE_VERSION:
        return False
    return (
        identity.get("provider") == fingerprint.provider
        and identity.get("model") == fingerprint.model
        and identity.get("endpoint_fingerprint") == fingerprint.endpoint_fingerprint
    )


__all__ = ["bind_provider_replay", "messages_for_model"]
