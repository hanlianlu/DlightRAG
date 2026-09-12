# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Secret-only Connections settings follow the existing source pipeline."""

import json

import pytest
from pydantic import SecretStr

from dlightrag.application.config import DlightragConfig, load_config

KEYRING = json.dumps(
    {"active": "test", "keys": {"test": "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="}}
)
VARIABLE = "DLIGHTRAG_ANSWER__AGENT__CONNECTIONS__CREDENTIAL_SECRET_KEYRING"


def test_keyring_env_file_and_process_precedence_are_secret_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    path = tmp_path / "fixture.env"
    path.write_text(f"{VARIABLE}='{KEYRING}'\n")
    config = load_config(env_file=path)
    assert config.answer.agent.connections.credential_secret_keyring == SecretStr(KEYRING)
    assert "credential_secret_keyring" not in config.model_dump_json()
    assert KEYRING not in repr(config)
    monkeypatch.setenv(VARIABLE, "replacement")
    config = load_config(env_file=path)
    assert config.answer.agent.connections.credential_secret_keyring == SecretStr("replacement")


def test_yaml_keyring_rejected_even_when_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    (tmp_path / "config.yaml").write_text(
        "answer:\n  agent:\n    connections:\n      credential_secret_keyring: forbidden-value\n"
    )
    monkeypatch.setenv(VARIABLE, KEYRING)
    with pytest.raises(ValueError, match="secret source") as error:
        load_config(_env_file=None)
    assert "forbidden-value" not in str(error.value)


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
    from dlightrag.application.connections import ConnectionsError
    from dlightrag.application.connections.credentials import CredentialCipher

    with pytest.raises(ConnectionsError, match="deployment") as error:
        CredentialCipher(SecretStr(ring))
    assert ring not in str(error.value)


def test_retained_key_rotation_and_owner_authentication():
    from dlightrag.application.connections import ConnectionsError
    from dlightrag.application.connections.credentials import CredentialCipher

    old = CredentialCipher(SecretStr(KEYRING))
    _, envelope = old.encrypt(
        SecretStr("fixture-bearer"), owner_id="a", connection_id="c", grant_id="g"
    )
    ring = json.loads(KEYRING)
    ring["active"] = "next"
    ring["keys"]["next"] = "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="
    rotated = CredentialCipher(SecretStr(json.dumps(ring)))
    assert rotated.decrypt(envelope, owner_id="a", connection_id="c", grant_id="g") == SecretStr(
        "fixture-bearer"
    )
    assert (
        rotated.encrypt(SecretStr("new"), owner_id="a", connection_id="c", grant_id="g")[0]
        == "next"
    )
    with pytest.raises(ConnectionsError, match="deployment"):
        rotated.decrypt(envelope, owner_id="b", connection_id="c", grant_id="g")


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
