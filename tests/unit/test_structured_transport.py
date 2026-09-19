# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Invocation-scoped capability facts for structured model output."""

from dlightrag.engine.ai.fingerprints import (
    ModelEndpointFingerprint,
    ModelInvocationFingerprint,
)
from dlightrag.engine.ai.structured_transport import JsonSchemaTransportCache


def test_json_schema_rejection_does_not_cross_api_families() -> None:
    endpoint = ModelEndpointFingerprint("openai", "model-a", "endpoint-a")
    chat = ModelInvocationFingerprint(
        endpoint.provider,
        endpoint.model,
        endpoint.endpoint_fingerprint,
        "chat_completion",
    )
    response = ModelInvocationFingerprint(
        endpoint.provider,
        endpoint.model,
        endpoint.endpoint_fingerprint,
        "response",
    )
    cache = JsonSchemaTransportCache()

    cache.remember_rejected(chat)

    assert cache.rejected(chat) is True
    assert cache.rejected(response) is False
