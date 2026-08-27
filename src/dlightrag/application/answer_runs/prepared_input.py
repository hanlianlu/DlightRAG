# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical size bound for the accepted Answer execution envelope."""

from collections.abc import Mapping
from typing import Any

from dlightrag.engine.agent.session.effects import canonical_json

MAX_PREPARED_INPUT_BYTES = 8 * 1024 * 1024


class PreparedInputTooLargeError(ValueError):
    """The canonical prepared input exceeds the durable 8 MiB bound."""

    def __init__(self, *, encoded_bytes: int) -> None:
        self.encoded_bytes = encoded_bytes
        super().__init__(
            "prepared_input_too_large: "
            f"{encoded_bytes} bytes exceed the {MAX_PREPARED_INPUT_BYTES} byte bound"
        )


def require_prepared_input_bounds(prepared_input: Mapping[str, Any]) -> None:
    """Validate the one canonical envelope serializer used by acceptance and storage."""
    encoded = canonical_json(dict(prepared_input)).encode("utf-8")
    if len(encoded) > MAX_PREPARED_INPUT_BYTES:
        raise PreparedInputTooLargeError(encoded_bytes=len(encoded))


__all__ = [
    "MAX_PREPARED_INPUT_BYTES",
    "PreparedInputTooLargeError",
    "require_prepared_input_bounds",
]
