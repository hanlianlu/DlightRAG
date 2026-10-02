# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared sealed envelope and deterministic secret derivation for opaque cursors."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping, Set
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from dlightrag.application.errors import ApplicationInputError

_CURSOR_NONCE_BYTES = 12
_CURSOR_SECRET_DERIVATION_ITERATIONS = 600_000
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class OpaqueCursorError(ApplicationInputError):
    """A sealed opaque cursor is malformed, non-canonical, or fails to open."""


class CursorSecretBox:
    """Derive stable, domain-separated cursor secrets from deployment identity material."""

    __slots__ = ("_material",)

    def __init__(self, material: bytes) -> None:
        if not isinstance(material, bytes) or not material:
            raise ValueError("cursor secret material must be non-empty bytes")
        self._material = material

    def derive(self, domain: str) -> bytes:
        """Return the stable PBKDF2-derived secret for one named cursor domain.

        The material is deployment identity (including the database password),
        so a computationally expensive KDF raises the cost of offline
        dictionary attacks against that secret when an attacker holds a
        valid cursor token. Derivation runs once per domain at startup.
        """
        encoded_domain = _encoded_domain(domain)
        return hashlib.pbkdf2_hmac(
            "sha256",
            self._material,
            salt=encoded_domain,
            iterations=_CURSOR_SECRET_DERIVATION_ITERATIONS,
        )


class OpaqueCursorEnvelope:
    """Seal and open one canonical-JSON cursor shape.

    A cursor is opaque in fact, not only in name: its fields (catalog keys a
    caller may not be allowed to see) are encrypted with AES-256-GCM, so a
    holder learns nothing but whether two cursors are the same. The nonce is
    derived from the payload, which keeps each cursor's token stable.

    Concrete cursor modules retain responsibility for typed field conversion;
    this module owns secret requirements, domain separation, canonical encoding,
    authenticated encryption, scope pinning, and the exact payload shape.
    """

    def __init__(
        self,
        secret: bytes | None,
        *,
        domain: str,
        scope: str,
        fields: Set[str],
    ) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("an explicit non-empty cursor secret is required")
        self._domain = _encoded_domain(domain)
        self._nonce_key = _subkey(secret, self._domain, b"nonce")
        self._cipher = AESGCM(_subkey(secret, self._domain, b"seal"))
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("cursor scope must be a non-empty string without NUL")
        shape = frozenset(fields)
        if not shape or "scope" in shape:
            raise ValueError("cursor fields must be non-empty and exclude scope")
        self._scope = scope
        self._fields = shape

    def encode(self, fields: Mapping[str, Any]) -> str:
        """Seal the exact field shape as one opaque token."""
        if set(fields) != self._fields:
            raise ValueError("opaque cursor fields do not match its shape")
        payload = _canonical_json({**fields, "scope": self._scope})
        nonce = hmac.new(self._nonce_key, payload, hashlib.sha256).digest()[:_CURSOR_NONCE_BYTES]
        return _base64url_encode(nonce + self._cipher.encrypt(nonce, payload, self._domain))

    def decode(self, token: str) -> dict[str, Any]:
        """Open one token and return its fields."""
        try:
            if not isinstance(token, str):
                raise ValueError
            sealed = _base64url_decode(token)
            nonce, ciphertext = sealed[:_CURSOR_NONCE_BYTES], sealed[_CURSOR_NONCE_BYTES:]
            payload = self._cipher.decrypt(nonce, ciphertext, self._domain)
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or _canonical_json(decoded) != payload:
                raise ValueError
            if decoded.pop("scope", None) != self._scope or set(decoded) != self._fields:
                raise ValueError
            return decoded
        except (
            binascii.Error,
            InvalidTag,
            UnicodeDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OpaqueCursorError("invalid opaque cursor") from exc


def _subkey(secret: bytes, domain: bytes, purpose: bytes) -> bytes:
    """One 256-bit key per cursor domain and purpose."""
    return hmac.new(secret, domain + b"\0" + purpose, hashlib.sha256).digest()


def _encoded_domain(domain: str) -> bytes:
    if not isinstance(domain, str) or not domain or "\0" in domain:
        raise ValueError("cursor domain must be a non-empty string without NUL")
    return domain.encode("utf-8")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not value or any(character not in _BASE64URL_CHARACTERS for character in value):
        raise ValueError("invalid base64url")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded


__all__ = ["CursorSecretBox", "OpaqueCursorEnvelope", "OpaqueCursorError"]
