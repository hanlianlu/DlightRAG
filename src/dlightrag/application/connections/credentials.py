# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Connections' view of the deployment key ring, and the reading of a Grant's credential."""

import json
from dataclasses import dataclass

from pydantic import SecretStr

from dlightrag.engine.credential_cipher import CredentialCipher, KeyRingError, UnreadableEnvelope

from .models import ConnectionsError, CredentialAccess
from .policy import credential_header_problem

#: A Grant's envelope is sealed under this label, so it never opens as an Agent Account.
GRANT_LABEL = "dlightrag-connection-v1"

#: What ``bearer`` and ``oauth`` state: one instance of one rule, ``Authorization`` with ``Bearer``.
DEFAULT_CREDENTIAL_HEADER = "Authorization"
DEFAULT_CREDENTIAL_PREFIX = "Bearer "


@dataclass(frozen=True, slots=True)
class CredentialTransport:
    """Where a credential travels: the header it names, and the prefix its value carries."""

    header: str
    prefix: str


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


def header_credential(header: str, prefix: str, token: SecretStr) -> SecretStr:
    """Seal a credential that travels in a header of its own, name and prefix included.

    The header a credential travels in is part of the credential, not of the Connection: it is
    stored inside the sealed envelope, so no column states it. It is not a secret, so Settings
    reads it back to say where a stored credential travels -- the token is never read back --
    and a deployment that cannot open the envelope answers no name. ``bearer`` and ``oauth``
    keep the one format they always had -- the token alone, in ``Authorization: Bearer`` -- so
    their envelopes and every stored row stay as they are.
    """
    return SecretStr(
        json.dumps(
            {"header": header, "prefix": prefix, "token": token.get_secret_value()},
            separators=(",", ":"),
        )
    )


def credential_transport(secret: SecretStr, *, authentication: str) -> CredentialTransport:
    """Which header this credential travels in, and the prefix its value carries.

    ``bearer`` and ``oauth`` are one instance of one rule, and a ``header`` credential names its
    own, which is how an endpoint that wants a vendor header (``X-Lark-MCP-TAT``, no prefix) is
    spoken to without a special case anywhere in the transport.
    """
    if authentication != "header":
        return CredentialTransport(DEFAULT_CREDENTIAL_HEADER, DEFAULT_CREDENTIAL_PREFIX)
    try:
        raw = json.loads(secret.get_secret_value())
        header, prefix = raw["header"], raw["prefix"]
    except Exception:
        raise ConnectionsError("Credential needs authorization", 401) from None
    if credential_header_problem(header, prefix) is not None:
        raise ConnectionsError("Credential needs authorization", 401)
    return CredentialTransport(header, prefix)


def credential_token(secret: SecretStr, *, authentication: str) -> SecretStr:
    """Read an unexpired authorized token; the host owns fenced refresh preflight."""
    import time

    if authentication == "bearer":
        return secret
    try:
        if authentication not in {"header", "oauth"}:
            raise ValueError
        if authentication == "header":
            token = json.loads(secret.get_secret_value())["token"]
        else:
            raw = json.loads(secret.get_secret_value())
            expiry = raw["expires_at"]
            if expiry is not None and (
                not isinstance(expiry, (int, float)) or expiry <= time.time()
            ):
                raise ValueError
            tokens = raw["tokens"]
            token = tokens["access_token"]
            if tokens["token_type"].lower() != "bearer":
                raise ValueError
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError
        return SecretStr(token)
    except ConnectionsError:
        raise
    except Exception:
        message = (
            "Credential needs authorization"
            if authentication == "header"
            else "OAuth credential needs authorization"
        )
        raise ConnectionsError(message, 401) from None


def credential_access(secret: SecretStr, *, authentication: str) -> CredentialAccess:
    """One stored credential as what a request needs: which header, and which token."""
    transport = credential_transport(secret, authentication=authentication)
    return CredentialAccess(
        token=credential_token(secret, authentication=authentication),
        header=transport.header,
        prefix=transport.prefix,
    )
