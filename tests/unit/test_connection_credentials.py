# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A credential carries the header it travels in, inside the one sealed envelope.

``bearer`` and ``oauth`` are the default instance of one rule -- a token in
``Authorization: Bearer`` -- and a ``header`` credential names its own. The name and the prefix
are part of the credential, so no column states them and no read of a Connection reveals them.
"""

import json
import time

import pytest
from pydantic import SecretStr

from dlightrag.application.connections.credentials import (
    DEFAULT_CREDENTIAL_HEADER,
    DEFAULT_CREDENTIAL_PREFIX,
    CredentialTransport,
    credential_access,
    credential_token,
    credential_transport,
    header_credential,
)


def test_a_bearer_and_an_oauth_credential_keep_the_one_format_they_had() -> None:
    """Neither kind is rewritten: their envelopes hold the token alone, exactly as before."""
    assert credential_transport(SecretStr("token"), authentication="bearer") == CredentialTransport(
        DEFAULT_CREDENTIAL_HEADER, DEFAULT_CREDENTIAL_PREFIX
    )
    assert credential_transport(SecretStr("{}"), authentication="oauth") == CredentialTransport(
        DEFAULT_CREDENTIAL_HEADER, DEFAULT_CREDENTIAL_PREFIX
    )
    assert credential_token(SecretStr("token"), authentication="bearer").get_secret_value() == (
        "token"
    )


def test_a_header_credential_carries_its_header_and_prefix_with_the_token() -> None:
    credential = header_credential("X-Lark-MCP-TAT", "", SecretStr("t-fixture"))

    assert credential_transport(credential, authentication="header") == CredentialTransport(
        "X-Lark-MCP-TAT", ""
    )
    assert credential_access(credential, authentication="header").token.get_secret_value() == (
        "t-fixture"
    )
    # The header name is not a column and not a separate field: it is inside the sealed payload.
    assert json.loads(credential.get_secret_value()) == {
        "header": "X-Lark-MCP-TAT",
        "prefix": "",
        "token": "t-fixture",
    }


def test_a_prefixed_header_credential_sends_the_prefix_it_was_given() -> None:
    credential = header_credential("X-Vendor-Token", "Token ", SecretStr("fixture"))

    access = credential_access(credential, authentication="header")

    assert (access.header, access.prefix) == ("X-Vendor-Token", "Token ")


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        "{}",
        json.dumps({"header": "X-Fixture"}),  # no prefix stated
        json.dumps({"header": "Host", "prefix": "", "token": "x"}),  # a transport header
        json.dumps({"header": "X Fixture", "prefix": "", "token": "x"}),  # not a field name
        json.dumps({"header": "X-Fixture", "prefix": "a\r\nb", "token": "x"}),
        json.dumps({"header": "X-Fixture", "prefix": "", "token": "a\nb"}),
        json.dumps({"header": "X-Fixture", "prefix": "", "token": ""}),
    ],
)
def test_a_credential_that_cannot_be_sent_needs_authorization(payload: str) -> None:
    """A malformed or unusable credential fails closed, and never falls back to plaintext."""
    from dlightrag.application.connections.models import ConnectionsError

    with pytest.raises(ConnectionsError) as raised:
        credential_access(SecretStr(payload), authentication="header")

    assert raised.value.status == 401
    assert payload not in str(raised.value)


def test_an_expired_oauth_token_needs_authorization_before_the_host_refreshes() -> None:
    from dlightrag.application.connections.models import ConnectionsError

    payload = SecretStr(
        json.dumps(
            {
                "expires_at": time.time() - 1,
                "tokens": {"access_token": "fixture", "token_type": "Bearer"},
            }
        )
    )
    with pytest.raises(ConnectionsError) as raised:
        credential_token(payload, authentication="oauth")

    assert raised.value.status == 401
