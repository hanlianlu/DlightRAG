# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The OAuth callback a deployment publishes is checked once, when configuration loads."""

import pytest
from pydantic import ValidationError

from dlightrag.application.connections import ConnectionPolicy


@pytest.mark.parametrize(
    "callback",
    [
        "https://app.example/web/oauth/connections/mcp/callback",
        "http://localhost:8100/web/oauth/connections/mcp/callback",
        "http://127.0.0.1:8100/web/oauth/connections/mcp/callback",
        "http://[::1]:8100/web/oauth/connections/mcp/callback",
    ],
)
def test_a_public_https_or_loopback_callback_is_accepted(callback: str) -> None:
    assert ConnectionPolicy(oauth_callback_url=callback).oauth_callback_url == callback


@pytest.mark.parametrize(
    "callback",
    [
        "http://app.example/web/oauth/connections/mcp/callback",
        "https://app.example/web/oauth/callback",
        "https://app.example/web/oauth/connections/mcp/callback?next=1",
        "https://app.example/web/oauth/connections/mcp/callback#top",
        "/web/oauth/connections/mcp/callback",
        "ftp://app.example/web/oauth/connections/mcp/callback",
    ],
)
def test_any_other_callback_fails_configuration_without_echoing_it(callback: str) -> None:
    with pytest.raises(ValidationError) as raised:
        ConnectionPolicy(oauth_callback_url=callback)

    assert callback not in str(raised.value)
