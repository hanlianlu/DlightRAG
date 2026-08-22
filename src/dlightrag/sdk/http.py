# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Environment-backed HTTP client settings."""

import os

DEFAULT_API_URL = "http://localhost:8100"
DEFAULT_CLIENT_TIMEOUT = 120.0
CLIENT_TIMEOUT_ENV = "DLIGHTRAG_CLIENT_TIMEOUT"


def client_timeout() -> float:
    """Return the timeout for one caller-owned SDK HTTP client."""
    value = os.environ.get(CLIENT_TIMEOUT_ENV)
    return float(value) if value else DEFAULT_CLIENT_TIMEOUT


def api_url() -> str:
    """Return the configured DlightRAG API origin."""
    return os.environ.get("DLIGHTRAG_API_URL", DEFAULT_API_URL).rstrip("/")


def auth_token() -> str | None:
    """Resolve an API bearer from client or nested deployment environment."""
    token = os.environ.get("DLIGHTRAG_API_TOKEN") or os.environ.get("DLIGHTRAG_ACCESS__API_TOKEN")
    return token or None


def auth_headers() -> dict[str, str]:
    """Return bearer-only headers suitable for multipart or streaming requests."""
    token = auth_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def json_headers() -> dict[str, str]:
    """Return JSON content and optional bearer headers."""
    return {"Content-Type": "application/json", **auth_headers()}


__all__ = [
    "CLIENT_TIMEOUT_ENV",
    "DEFAULT_API_URL",
    "DEFAULT_CLIENT_TIMEOUT",
    "api_url",
    "auth_headers",
    "auth_token",
    "client_timeout",
    "json_headers",
]
