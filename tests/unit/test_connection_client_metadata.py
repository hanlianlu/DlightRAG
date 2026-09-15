# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The published Client ID Metadata Document stays a public, minimal, exact-URL identity."""

from urllib.parse import urlsplit

from dlightrag.application.connections.client_metadata import (
    CLIENT_METADATA_PATH,
    CLIENT_NAME,
    client_metadata_document,
    client_metadata_url,
)

CALLBACK = "https://app.example/web/oauth/connections/mcp/callback"


def test_a_public_https_callback_publishes_a_short_stable_path_on_that_origin():
    assert client_metadata_url(CALLBACK) == f"https://app.example{CLIENT_METADATA_PATH}"


def test_the_published_url_keeps_an_explicit_port_and_drops_callback_query_data():
    assert (
        client_metadata_url("https://app.example:8443/web/oauth/connections/mcp/callback?x=1#y")
        == f"https://app.example:8443{CLIENT_METADATA_PATH}"
    )


def test_a_deployment_that_cannot_serve_https_publishes_nothing():
    for callback in (
        None,
        "",
        "http://app.example/web/oauth/connections/mcp/callback",
        "https://user:secret@app.example/web/oauth/connections/mcp/callback",
        "https:///web/oauth/connections/mcp/callback",
        "/web/oauth/connections/mcp/callback",
    ):
        assert client_metadata_url(callback) is None


def test_the_published_url_satisfies_the_document_drafts_url_rules():
    metadata_url = client_metadata_url(CALLBACK)
    assert metadata_url is not None
    parts = urlsplit(metadata_url)
    assert parts.scheme == "https"
    assert not parts.username and not parts.password
    assert not parts.query and not parts.fragment
    assert parts.path.startswith("/")
    assert not any(segment in {".", ".."} for segment in parts.path.split("/"))


def test_the_document_holds_exactly_the_public_identity_and_no_secret():
    metadata_url = client_metadata_url(CALLBACK)
    assert metadata_url is not None
    document = client_metadata_document(metadata_url=metadata_url, oauth_callback_url=CALLBACK)
    assert document == {
        "client_id": metadata_url,
        "client_name": CLIENT_NAME,
        "redirect_uris": [CALLBACK],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    # Both sides compare client_id and the document URL as plain strings.
    assert document["client_id"] == metadata_url
    assert not any(
        "secret" in key or "token" in key for key in document if key != "token_endpoint_auth_method"
    )
