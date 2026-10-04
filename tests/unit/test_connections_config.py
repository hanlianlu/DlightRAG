# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Connections' view of the deployment key ring: Grant envelopes, and Connections' own errors."""

import json

import pytest
from pydantic import SecretStr

from dlightrag.application.connections import ConnectionsError
from dlightrag.application.connections.credentials import GRANT_LABEL, GrantCipher
from dlightrag.engine.credential_cipher import CredentialCipher

KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)


def _seal(cipher: GrantCipher) -> str:
    return cipher.encrypt(
        SecretStr("fixture-bearer"), owner_id="a", connection_id="c", grant_id="g"
    )[1]


def _open(cipher: GrantCipher, envelope: str, *, owner_id: str = "a") -> SecretStr:
    return cipher.decrypt(envelope, owner_id=owner_id, connection_id="c", grant_id="g")


def test_a_grant_is_sealed_under_its_label_and_its_owner_connection_and_grant() -> None:
    cipher = CredentialCipher(SecretStr(KEYRING))

    envelope = _seal(GrantCipher(cipher))

    opened = cipher.open(envelope, label=GRANT_LABEL, binding=("a", "c", "g"))
    assert opened == SecretStr("fixture-bearer")


def test_without_a_ring_credentials_fail_closed() -> None:
    reader = GrantCipher(CredentialCipher(None))

    with pytest.raises(ConnectionsError, match="keyring is missing") as raised:
        _seal(reader)
    assert raised.value.status == 503
    with pytest.raises(ConnectionsError, match="keyring is missing") as raised:
        _open(reader, "{}")
    assert raised.value.status == 503


def test_a_grant_no_key_opens_needs_authorization() -> None:
    envelope = _seal(GrantCipher(CredentialCipher(SecretStr(KEYRING))))
    ring = json.loads(KEYRING)
    ring["active"] = "next"
    ring["keys"]["next"] = "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="
    rotated = GrantCipher(CredentialCipher(SecretStr(json.dumps(ring))))
    lost = GrantCipher(
        CredentialCipher(
            SecretStr(json.dumps({"active": "next", "keys": {"next": ring["keys"]["next"]}}))
        )
    )

    assert rotated.retired_key_ids == ("test",)
    assert _open(rotated, envelope) == SecretStr("fixture-bearer")
    # Another owner's grant, or one whose key the ring lost, opens for nobody: authorize again.
    for cipher, owner_id in ((rotated, "b"), (lost, "a")):
        with pytest.raises(ConnectionsError, match="needs authorization") as raised:
            _open(cipher, envelope, owner_id=owner_id)
        assert raised.value.status == 401


def test_config_import_does_not_eagerly_load_answer_through_connection_policy():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dlightrag.application.config import DlightragConfig; assert not any(m.startswith('dlightrag.engine.answer') for m in sys.modules); from dlightrag.application.connections import Connections",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
