# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Client ID Metadata Document this deployment publishes for OAuth.

MCP's authorization spec orders client registration: a client already registered with the
authorization server, then a Client ID Metadata Document, then Dynamic Client Registration as the
compatibility fallback. This module owns the middle one, so a Connection can authorize at a server
it has no prior relationship with, without registering there and without a client secret anywhere.

The document is public by design: an authorization server fetches `client_id` over the network
without a credential, so it carries only what an authorization redirect already reveals -- the
client name, this deployment's own callback URL, and the fact that the client uses PKCE and no
secret.
"""

from typing import Any
from urllib.parse import urlsplit, urlunsplit

CLIENT_NAME = "DlightRAG personal Connection"
"""The name an authorization server shows on its consent screen."""

CLIENT_METADATA_PATH = "/web/oauth/connections/mcp/client-metadata"
"""Path component of the published `client_id`.

The document draft requires an https URL with a non-root path component, no userinfo, no
dot-segments, and no fragment, and discourages a query, so the path is fixed, short, and stable.
"""


def client_metadata_url(oauth_callback_url: str | None) -> str | None:
    """The `client_id` this deployment can publish, or None when it cannot publish one.

    The document must be served over https from the same public origin, and the locked SDK rejects
    every other shape, so a deployment whose public callback is plain http or carries userinfo
    keeps Dynamic Client Registration instead.
    """
    if not oauth_callback_url:
        return None
    parts = urlsplit(oauth_callback_url)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
        return None
    return urlunsplit((parts.scheme, parts.netloc, CLIENT_METADATA_PATH, "", ""))


def client_metadata_document(*, metadata_url: str, oauth_callback_url: str) -> dict[str, Any]:
    """The JSON served at `metadata_url`.

    `client_id` must equal the document URL exactly: this client and the authorization server both
    compare the two as plain strings.
    """
    return {
        "client_id": metadata_url,
        "client_name": CLIENT_NAME,
        "redirect_uris": [oauth_callback_url],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
