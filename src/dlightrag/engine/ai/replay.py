# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-identity gate for opaque provider reasoning replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.messages import AssistantTurn

_ENVELOPE_KEY = "_dlightrag_replay"
_ENVELOPE_VERSION = 2


class ProviderReplayError(ValueError):
    """Opaque replay claims the current invocation but violates its envelope contract."""


def bind_provider_replay(
    turn: AssistantTurn,
    fingerprint: ModelInvocationFingerprint,
) -> AssistantTurn:
    """Bind opaque response state to its source model."""
    if turn.provider_state is None:
        return turn
    return replace(
        turn,
        provider_state={
            _ENVELOPE_KEY: {
                "v": _ENVELOPE_VERSION,
                "provider": fingerprint.provider,
                "model": fingerprint.model,
                "endpoint_fingerprint": fingerprint.endpoint_fingerprint,
                "api_family": fingerprint.api_family,
            },
            "payload": turn.provider_state,
        },
    )


def messages_for_model(
    messages: list[dict[str, Any]],
    fingerprint: ModelInvocationFingerprint,
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
        if _is_same_invocation_state(state, fingerprint) and isinstance(state, Mapping):
            payload = state.get("payload")
            if payload is not None:
                message["provider_state"] = payload
        prepared.append(message)
    return prepared


def _is_same_invocation_state(
    state: object,
    fingerprint: ModelInvocationFingerprint,
) -> bool:
    if not isinstance(state, Mapping):
        return False
    identity = state.get(_ENVELOPE_KEY)
    if not isinstance(identity, Mapping):
        return False
    same_invocation = (
        identity.get("provider") == fingerprint.provider
        and identity.get("model") == fingerprint.model
        and identity.get("endpoint_fingerprint") == fingerprint.endpoint_fingerprint
        and identity.get("api_family") == fingerprint.api_family
    )
    if same_invocation and identity.get("v") != _ENVELOPE_VERSION:
        raise ProviderReplayError("provider replay envelope version is unsupported")
    return same_invocation


__all__ = ["ProviderReplayError", "bind_provider_replay", "messages_for_model"]
