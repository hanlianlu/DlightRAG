# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-owned request policy shared by Response API transports."""

from collections.abc import Mapping
from typing import Any


class ResponseRequestError(ValueError):
    """A canonical request cannot be represented by the supported Responses subset."""


RESPONSE_RESERVED_MODEL_KWARGS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tokens",
        "max_tool_calls",
        "messages",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "response_format",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "truncation",
    }
)


def validate_response_extensions(options: Mapping[str, Any]) -> None:
    """Reject raw extension fields that would take ownership from the product."""
    conflicts = sorted(RESPONSE_RESERVED_MODEL_KWARGS.intersection(options))
    if conflicts:
        raise ResponseRequestError(
            "raw model kwargs cannot override Response fields: " + ", ".join(conflicts)
        )


__all__ = [
    "RESPONSE_RESERVED_MODEL_KWARGS",
    "ResponseRequestError",
    "validate_response_extensions",
]
