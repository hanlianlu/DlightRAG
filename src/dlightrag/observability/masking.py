# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Langfuse payload masking."""

from typing import Any

from dlightrag_ai.telemetry import bounded_telemetry_text

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "secret",
    "password",
    "token",
    "authorization",
    "connection_string",
    "account_key",
    "sas_token",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def mask_langfuse_payload(data: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Mask secrets, large text, and inline media before export."""
    if isinstance(data, dict):
        if data.get("type") == "image_url":
            return {"type": "image_url", "image_url": "[image omitted]"}
        return {
            key: "[redacted]" if _is_sensitive_key(str(key)) else mask_langfuse_payload(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [mask_langfuse_payload(item) for item in data]
    if isinstance(data, tuple):
        return [mask_langfuse_payload(item) for item in data]
    if isinstance(data, bytes):
        return f"[bytes omitted: {len(data)}]"
    if isinstance(data, str):
        return bounded_telemetry_text(data)
    return data


__all__ = ["mask_langfuse_payload"]
