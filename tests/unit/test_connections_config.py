# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deployment's Connection key ring: one private file every worker reads."""

import json
import stat

import pytest
from pydantic import SecretStr

from dlightrag.application.connections import ConnectionsError
from dlightrag.application.connections.credentials import (
    KEYRING_FILE,
    CredentialCipher,
    deployment_cipher,
)

KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)


def _seal(cipher: CredentialCipher) -> str:
    return cipher.encrypt(
        SecretStr("fixture-bearer"), owner_id="a", connection_id="c", grant_id="g"
    )[1]


def _open(cipher: CredentialCipher, envelope: str, *, owner_id: str = "a") -> SecretStr:
    return cipher.decrypt(envelope, owner_id=owner_id, connection_id="c", grant_id="g")


def test_the_first_writer_creates_a_private_ring_every_worker_reads(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE

    envelope = _seal(deployment_cipher(path, create=True))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [entry.name for entry in tmp_path.iterdir()] == [KEYRING_FILE]
    for worker in (deployment_cipher(path, create=True), deployment_cipher(path, create=False)):
        assert _open(worker, envelope) == SecretStr("fixture-bearer")


def test_a_writer_adopts_the_ring_it_finds(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE
    path.write_text(KEYRING)

    envelope = _seal(CredentialCipher(SecretStr(KEYRING)))

    assert _open(deployment_cipher(path, create=True), envelope) == SecretStr("fixture-bearer")
    assert path.read_text() == KEYRING


def test_without_a_ring_credentials_fail_closed(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE
    reader = deployment_cipher(path, create=False)

    with pytest.raises(ConnectionsError, match="keyring is missing") as raised:
        _seal(reader)
    assert raised.value.status == 503
    assert not path.exists()


@pytest.mark.parametrize(
    "ring",
    [
        "fixture-key-must-not-echo",
        '{"active":"missing","keys":{}}',
        '{"active":"test","keys":{"test":"invalid!"}}',
        '{"active":"test","keys":{"test":"YQ","test":"YQ"}}',
    ],
)
def test_invalid_private_keyring_fails_closed_without_echo(ring):
    with pytest.raises(ConnectionsError, match="deployment") as error:
        CredentialCipher(SecretStr(ring))
    assert ring not in str(error.value)


def test_retired_keys_still_open_and_a_grant_no_key_opens_needs_authorization():
    envelope = _seal(CredentialCipher(SecretStr(KEYRING)))
    ring = json.loads(KEYRING)
    ring["active"] = "next"
    ring["keys"]["next"] = "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="
    rotated = CredentialCipher(SecretStr(json.dumps(ring)))
    lost = CredentialCipher(
        SecretStr(json.dumps({"active": "next", "keys": {"next": ring["keys"]["next"]}}))
    )

    assert rotated.retired_key_ids == ("test",)
    assert _open(rotated, envelope) == SecretStr("fixture-bearer")
    assert rotated.encrypt(SecretStr("new"), owner_id="a", connection_id="c", grant_id="g")[0] == (
        "next"
    )
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
