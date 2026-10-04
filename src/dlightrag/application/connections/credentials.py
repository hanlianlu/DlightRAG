# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Connections' view of the deployment key ring, and the reading of a Grant's bearer."""

import json

from pydantic import SecretStr

from dlightrag.engine.credential_cipher import CredentialCipher, KeyRingError, UnreadableEnvelope

from .models import ConnectionsError

#: A Grant's envelope is sealed under this label, so it never opens as an Agent Account.
GRANT_LABEL = "dlightrag-connection-v1"


class GrantCipher:
    """The deployment cipher as Connections use it: Grant and OAuth-flow envelopes, Connections' errors."""

    def __init__(self, cipher: CredentialCipher) -> None:
        self._cipher = cipher

    @property
    def retired_key_ids(self) -> tuple[str, ...]:
        """Keys the ring keeps only to open older grants until they are re-encrypted."""
        return self._cipher.retired_key_ids

    def encrypt(
        self, secret: SecretStr, *, owner_id: str, connection_id: str, grant_id: str
    ) -> tuple[str, str]:
        try:
            return self._cipher.seal(
                secret, label=GRANT_LABEL, binding=(owner_id, connection_id, grant_id)
            )
        except KeyRingError:
            raise ConnectionsError("Credential deployment keyring is missing", 503) from None

    def decrypt(
        self, envelope: str, *, owner_id: str, connection_id: str, grant_id: str
    ) -> SecretStr:
        try:
            return self._cipher.open(
                envelope, label=GRANT_LABEL, binding=(owner_id, connection_id, grant_id)
            )
        except KeyRingError:
            raise ConnectionsError("Credential deployment keyring is missing", 503) from None
        except UnreadableEnvelope:
            # Its key is gone (a lost ring, a retired key removed early): nobody can use
            # this grant again, so its owner authorizes the Connection anew.
            raise ConnectionsError("Connection needs authorization", 401) from None


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
