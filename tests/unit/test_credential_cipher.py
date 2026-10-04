# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deployment key ring and the envelopes sealed under it: one private file every worker reads."""

import json
import stat

import pytest
from pydantic import SecretStr

from dlightrag.application.connections.credentials import GRANT_LABEL
from dlightrag.engine.answer.agent_browser import ACCOUNT_LABEL
from dlightrag.engine.credential_cipher import (
    KEYRING_FILE,
    CredentialCipher,
    KeyRingError,
    UnreadableEnvelope,
    deployment_cipher,
)

KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)
LABEL = "dlightrag-fixture-v1"
BINDING = ("a", "c", "g")


def _seal(
    cipher: CredentialCipher, *, label: str = LABEL, binding: tuple[str, ...] = BINDING
) -> str:
    return cipher.seal(SecretStr("fixture-secret"), label=label, binding=binding)[1]


def _open(
    cipher: CredentialCipher,
    envelope: str,
    *,
    label: str = LABEL,
    binding: tuple[str, ...] = BINDING,
) -> SecretStr:
    return cipher.open(envelope, label=label, binding=binding)


def test_the_first_writer_creates_a_private_ring_every_worker_reads(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE

    envelope = _seal(deployment_cipher(path, create=True))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [entry.name for entry in tmp_path.iterdir()] == [KEYRING_FILE]
    for worker in (deployment_cipher(path, create=True), deployment_cipher(path, create=False)):
        assert _open(worker, envelope) == SecretStr("fixture-secret")


def test_a_writer_adopts_the_ring_it_finds(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE
    path.write_text(KEYRING)

    envelope = _seal(CredentialCipher(SecretStr(KEYRING)))

    assert _open(deployment_cipher(path, create=True), envelope) == SecretStr("fixture-secret")
    assert path.read_text() == KEYRING


def test_without_a_ring_nothing_is_sealed_or_opened(tmp_path) -> None:
    path = tmp_path / KEYRING_FILE
    reader = deployment_cipher(path, create=False)
    envelope = _seal(CredentialCipher(SecretStr(KEYRING)))

    with pytest.raises(KeyRingError, match="keyring is missing"):
        _seal(reader)
    with pytest.raises(KeyRingError, match="keyring is missing"):
        _open(reader, envelope)
    assert reader.active_key_id is None and not path.exists()


@pytest.mark.parametrize(
    "ring",
    [
        "fixture-key-must-not-echo",
        '{"active":"missing","keys":{}}',
        '{"active":"test","keys":{"test":"invalid!"}}',
        '{"active":"test","keys":{"test":"YQ","test":"YQ"}}',
    ],
)
def test_an_invalid_ring_fails_closed_without_echo(ring: str) -> None:
    with pytest.raises(KeyRingError, match="deployment") as error:
        CredentialCipher(SecretStr(ring))
    assert ring not in str(error.value)


def test_retired_keys_still_open_the_active_one_seals_and_a_lost_one_opens_for_nobody() -> None:
    envelope = _seal(CredentialCipher(SecretStr(KEYRING)))
    ring = json.loads(KEYRING)
    ring["active"] = "next"
    ring["keys"]["next"] = "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="
    rotated = CredentialCipher(SecretStr(json.dumps(ring)))
    lost = CredentialCipher(
        SecretStr(json.dumps({"active": "next", "keys": {"next": ring["keys"]["next"]}}))
    )

    assert rotated.retired_key_ids == ("test",)
    assert _open(rotated, envelope) == SecretStr("fixture-secret")
    key_id, resealed = rotated.seal(SecretStr("new"), label=LABEL, binding=BINDING)
    assert (key_id, json.loads(resealed)["key_id"]) == ("next", "next")
    # A binding that differs, or a key the ring lost, opens for nobody, and says nothing more.
    for cipher, binding in ((rotated, ("b", "c", "g")), (lost, BINDING)):
        with pytest.raises(UnreadableEnvelope) as raised:
            _open(cipher, envelope, binding=binding)
        assert str(raised.value) == ""


def test_an_account_envelope_never_opens_as_a_grant_nor_a_grant_as_an_account() -> None:
    cipher = CredentialCipher(SecretStr(KEYRING))
    binding = ("owner", "site", "id")
    account = _seal(cipher, label=ACCOUNT_LABEL, binding=binding)
    grant = _seal(cipher, label=GRANT_LABEL, binding=binding)

    assert _open(cipher, account, label=ACCOUNT_LABEL, binding=binding) == SecretStr(
        "fixture-secret"
    )
    assert _open(cipher, grant, label=GRANT_LABEL, binding=binding) == SecretStr("fixture-secret")
    for envelope, label in ((account, GRANT_LABEL), (grant, ACCOUNT_LABEL)):
        with pytest.raises(UnreadableEnvelope):
            _open(cipher, envelope, label=label, binding=binding)
