# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned AES-256-GCM envelopes; key material arrives only by injection."""

import base64
import json
import os
import re

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from .models import ConnectionsError


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _decode(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", value):
        raise ValueError
    decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != value.rstrip("="):
        raise ValueError
    return decoded


class CredentialCipher:
    def __init__(self, keyring: SecretStr | None) -> None:
        self._keys: dict[str, bytes] = {}
        self._active: str | None = None
        if keyring is None:
            return
        try:
            raw = json.loads(keyring.get_secret_value(), object_pairs_hook=_unique)
            if set(raw) != {"active", "keys"} or not isinstance(raw["keys"], dict):
                raise ValueError
            for identity, encoded in raw["keys"].items():
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity):
                    raise ValueError
                key = _decode(encoded)
                if len(key) != 32:
                    raise ValueError
                self._keys[identity] = key
            self._active = raw["active"]
            if self._active not in self._keys:
                raise ValueError
        except Exception:
            raise ConnectionsError("Credential deployment keyring is invalid", 503) from None

    @property
    def active_key_id(self) -> str | None:
        return self._active

    @staticmethod
    def _aad(owner_id: str, connection_id: str, grant_id: str) -> bytes:
        return json.dumps(
            ["dlightrag-connection-v1", owner_id, connection_id, grant_id], separators=(",", ":")
        ).encode()

    def encrypt(
        self, secret: SecretStr, *, owner_id: str, connection_id: str, grant_id: str
    ) -> tuple[str, str]:
        if self._active is None:
            raise ConnectionsError("Credential deployment keyring is missing", 503)
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._keys[self._active]).encrypt(
            nonce, secret.get_secret_value().encode(), self._aad(owner_id, connection_id, grant_id)
        )
        envelope = json.dumps(
            {
                "version": 1,
                "key_id": self._active,
                "nonce": base64.urlsafe_b64encode(nonce).decode(),
                "ciphertext": base64.urlsafe_b64encode(ciphertext).decode(),
            }
        )
        return self._active, envelope

    def decrypt(
        self, envelope: str, *, owner_id: str, connection_id: str, grant_id: str
    ) -> SecretStr:
        try:
            raw = json.loads(envelope)
            if raw["version"] != 1:
                raise ValueError
            plaintext = AESGCM(self._keys[raw["key_id"]]).decrypt(
                _decode(raw["nonce"]),
                _decode(raw["ciphertext"]),
                self._aad(owner_id, connection_id, grant_id),
            )
            return SecretStr(plaintext.decode())
        except Exception:
            raise ConnectionsError(
                "Credential deployment keyring cannot read this grant", 503
            ) from None


def access_bearer(secret: SecretStr, *, authentication: str) -> SecretStr:
    """Read an unexpired authorized token; the host owns fenced refresh preflight."""
    import time

    if authentication == "bearer":
        return secret
    try:
        if authentication != "oauth":
            raise ValueError
        raw = json.loads(secret.get_secret_value())
        expiry = raw["expires_at"]
        if expiry is not None and (not isinstance(expiry, (int, float)) or expiry <= time.time()):
            raise ValueError
        tokens = raw["tokens"]
        token = tokens["access_token"]
        if (
            tokens["token_type"].lower() != "bearer"
            or not isinstance(token, str)
            or not token
            or len(token) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError
        return SecretStr(token)
    except Exception:
        raise ConnectionsError("OAuth credential needs authorization", 401) from None
