# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned AES-256-GCM envelopes under the deployment's own key ring."""

import base64
import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from .models import ConnectionsError

logger = logging.getLogger(__name__)

#: The key ring's file in the deployment's working directory: beside the corpus,
#: never in the database whose backups hold the envelopes.
KEYRING_FILE = "connection-keyring.json"


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

    @property
    def retired_key_ids(self) -> tuple[str, ...]:
        """Keys the ring keeps only to open older grants until they are re-encrypted."""
        return tuple(sorted(key for key in self._keys if key != self._active))

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
        if self._active is None:
            raise ConnectionsError("Credential deployment keyring is missing", 503)
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
            # Its key is gone (a lost ring, a retired key removed early): nobody can use
            # this grant again, so its owner authorizes the Connection anew.
            raise ConnectionsError("Connection needs authorization", 401) from None


def deployment_cipher(path: Path, *, create: bool) -> CredentialCipher:
    """The deployment's key ring at ``path``, which the first writer to start creates.

    The ring is ``{"active": "<id>", "keys": {"<id>": "<base64url 32 bytes>"}}``. A
    new ring is written whole to a private temporary file and linked into place, so
    of writers starting together one wins and the rest read its ring. A reader never
    creates one; without it, Connections that hold credentials fail closed. To rotate,
    add a key, point ``active`` at it, and restart: writer maintenance re-encrypts
    live grants, after which the old key may go.
    """
    if create and not path.exists():
        try:
            _create_ring(path)
        except OSError as exc:
            logger.warning(
                "Cannot create the Connection key ring at %s (%s); Connections that "
                "hold credentials stay unavailable",
                path,
                type(exc).__name__,
            )
    try:
        return CredentialCipher(SecretStr(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return CredentialCipher(None)


def _create_ring(path: Path) -> None:
    key_id = datetime.now(UTC).strftime("%Y%m%d")
    key = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as ring:
            json.dump({"active": key_id, "keys": {key_id: key}}, ring)
            ring.flush()
            os.fsync(ring.fileno())
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(temporary)


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
