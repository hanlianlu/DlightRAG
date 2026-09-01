# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared integrity envelope and deterministic secret derivation for opaque cursors."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping, Set
from typing import Any

_CURSOR_MAC_BYTES = 16
_CURSOR_SECRET_DERIVATION_ITERATIONS = 600_000
_BASE64URL_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class OpaqueCursorError(ValueError):
    """A signed opaque cursor is malformed, non-canonical, or fails integrity checks."""


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
    """Sign and validate one versioned canonical-JSON cursor shape.

    Concrete cursor modules retain responsibility for typed field conversion;
    this module owns secret requirements, domain separation, canonical encoding,
    integrity verification, scope/version pinning, and exact payload shapes.
    """

    def __init__(
        self,
        secret: bytes | None,
        *,
        domain: str,
        scope: str,
        fields_by_version: Mapping[int, Set[str]],
        current_version: int,
    ) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("an explicit non-empty cursor secret is required")
        self._secret = secret
        self._domain = _encoded_domain(domain)
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("cursor scope must be a non-empty string without NUL")
        normalized_versions: dict[int, frozenset[str]] = {}
        for version, fields in fields_by_version.items():
            if type(version) is not int or version < 1:
                raise ValueError("cursor versions must be positive integers")
            normalized = frozenset(fields)
            if not normalized or {"scope", "v"}.intersection(normalized):
                raise ValueError("cursor fields must be non-empty and exclude scope/version")
            normalized_versions[version] = normalized
        if current_version not in normalized_versions:
            raise ValueError("current cursor version must have a declared field shape")
        self._scope = scope
        self._fields_by_version = normalized_versions
        self._current_version = current_version

    def encode(self, fields: Mapping[str, Any]) -> str:
        """Encode the current exact field shape as one signed opaque token."""
        expected = self._fields_by_version[self._current_version]
        if set(fields) != expected:
            raise ValueError("opaque cursor fields do not match the current version")
        payload = _canonical_json(
            {
                **fields,
                "scope": self._scope,
                "v": self._current_version,
            }
        )
        mac = self._mac(payload)
        return f"{_base64url_encode(payload)}.{_base64url_encode(mac)}"

    def decode(self, token: str) -> dict[str, Any]:
        """Verify one token and return its canonical decoded payload."""
        try:
            if not isinstance(token, str):
                raise ValueError
            encoded, encoded_mac = token.split(".")
            if not encoded or not encoded_mac:
                raise ValueError
            payload = _base64url_decode(encoded)
            supplied_mac = _base64url_decode(encoded_mac)
            expected_mac = self._mac(payload)
            if len(supplied_mac) != _CURSOR_MAC_BYTES or not hmac.compare_digest(
                supplied_mac, expected_mac
            ):
                raise ValueError
            decoded = json.loads(payload)
            if not isinstance(decoded, dict) or _canonical_json(decoded) != payload:
                raise ValueError
            keys = set(decoded)
            version = decoded.get("v")
            if type(version) is not int or decoded.get("scope") != self._scope:
                raise ValueError
            expected_fields = self._fields_by_version.get(version)
            if expected_fields is None or keys != expected_fields | {"scope", "v"}:
                raise ValueError
            return decoded
        except (
            binascii.Error,
            UnicodeDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OpaqueCursorError("invalid opaque cursor") from exc

    def _mac(self, payload: bytes) -> bytes:
        return hmac.new(
            self._secret,
            self._domain + b"\0" + payload,
            hashlib.sha256,
        ).digest()[:_CURSOR_MAC_BYTES]


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
