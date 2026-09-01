# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared opaque-cursor envelope and secret governance contracts."""

import base64
import hashlib
import hmac
import json

import pytest

from dlightrag.application.opaque_cursor import (
    _CURSOR_SECRET_DERIVATION_ITERATIONS,
    CursorSecretBox,
    OpaqueCursorEnvelope,
    OpaqueCursorError,
)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _signed(secret: bytes, domain: str, payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, domain.encode() + b"\0" + raw, hashlib.sha256).digest()[:16]
    return f"{_encode(raw)}.{_encode(mac)}"


def test_secret_box_is_stable_domain_separated() -> None:
    material = b"db-host\0db-name\0db-password"
    box = CursorSecretBox(material)

    assert box.derive("dlightrag-file-panel-cursor") == hashlib.pbkdf2_hmac(
        "sha256",
        material,
        salt=b"dlightrag-file-panel-cursor",
        iterations=_CURSOR_SECRET_DERIVATION_ITERATIONS,
    )
    assert box.derive("dlightrag-file-panel-cursor") != box.derive(
        "dlightrag-metadata-search-cursor"
    )
    assert "db-password" not in repr(box)


@pytest.mark.parametrize("material", [b"", None, "secret"])
def test_secret_box_rejects_missing_or_non_byte_material(material: object) -> None:
    with pytest.raises(ValueError, match="non-empty bytes"):
        CursorSecretBox(material)  # type: ignore[arg-type]


def test_envelope_requires_explicit_secret_and_declared_current_shape() -> None:
    with pytest.raises(ValueError, match="explicit non-empty"):
        OpaqueCursorEnvelope(
            None,
            domain="items",
            scope="items",
            fields_by_version={1: {"after"}},
            current_version=1,
        )
    with pytest.raises(ValueError, match="declared field shape"):
        OpaqueCursorEnvelope(
            b"secret",
            domain="items",
            scope="items",
            fields_by_version={1: {"after"}},
            current_version=2,
        )


def test_envelope_round_trips_current_and_accepts_declared_prior_versions() -> None:
    secret = b"opaque-cursor-test-secret"
    envelope = OpaqueCursorEnvelope(
        secret,
        domain="items",
        scope="item-list",
        fields_by_version={1: {"after"}, 2: {"after", "view"}},
        current_version=2,
    )

    current = envelope.encode({"after": "item-1", "view": "active"})
    prior = _signed(
        secret,
        "items",
        {"after": "item-0", "scope": "item-list", "v": 1},
    )

    assert envelope.decode(current) == {
        "after": "item-1",
        "scope": "item-list",
        "v": 2,
        "view": "active",
    }
    assert envelope.decode(prior) == {
        "after": "item-0",
        "scope": "item-list",
        "v": 1,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"after": "item", "scope": "wrong", "v": 2, "view": "active"},
        {"after": "item", "scope": "item-list", "v": 3, "view": "active"},
        {
            "after": "item",
            "extra": True,
            "scope": "item-list",
            "v": 2,
            "view": "active",
        },
    ],
)
def test_envelope_rejects_signed_scope_version_and_shape_drift(
    payload: dict[str, object],
) -> None:
    secret = b"opaque-cursor-test-secret"
    envelope = OpaqueCursorEnvelope(
        secret,
        domain="items",
        scope="item-list",
        fields_by_version={2: {"after", "view"}},
        current_version=2,
    )

    with pytest.raises(OpaqueCursorError):
        envelope.decode(_signed(secret, "items", payload))


def test_envelope_rejects_cross_domain_replay_and_tamper() -> None:
    secret = b"opaque-cursor-test-secret"
    first = OpaqueCursorEnvelope(
        secret,
        domain="first",
        scope="items",
        fields_by_version={1: {"after"}},
        current_version=1,
    )
    second = OpaqueCursorEnvelope(
        secret,
        domain="second",
        scope="items",
        fields_by_version={1: {"after"}},
        current_version=1,
    )
    token = first.encode({"after": "item-1"})

    with pytest.raises(OpaqueCursorError):
        second.decode(token)
    with pytest.raises(OpaqueCursorError):
        first.decode(token + "x")
