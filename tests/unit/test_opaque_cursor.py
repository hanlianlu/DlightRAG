# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared opaque-cursor envelope and secret governance contracts."""

import base64
import hashlib

import pytest

from dlightrag.application.opaque_cursor import (
    _CURSOR_SECRET_DERIVATION_ITERATIONS,
    CursorSecretBox,
    OpaqueCursorEnvelope,
    OpaqueCursorError,
)


def _envelope(
    fields: set[str], *, scope: str = "item-list", domain: str = "items"
) -> OpaqueCursorEnvelope:
    return OpaqueCursorEnvelope(
        b"opaque-cursor-test-secret", domain=domain, scope=scope, fields=fields
    )


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


def test_envelope_requires_explicit_secret_and_a_field_shape() -> None:
    with pytest.raises(ValueError, match="explicit non-empty"):
        OpaqueCursorEnvelope(None, domain="items", scope="items", fields={"after"})
    with pytest.raises(ValueError, match="non-empty and exclude scope"):
        OpaqueCursorEnvelope(b"secret", domain="items", scope="items", fields={"scope"})


def test_envelope_round_trips_its_fields() -> None:
    envelope = _envelope({"after", "view"})

    token = envelope.encode({"after": "item-1", "view": "active"})

    assert envelope.decode(token) == {"after": "item-1", "view": "active"}


@pytest.mark.parametrize(
    ("scope", "shape"),
    [("wrong", {"after", "view"}), ("item-list", {"after", "extra", "view"})],
    ids=["scope", "shape"],
)
def test_envelope_rejects_scope_and_shape_drift(scope: str, shape: set[str]) -> None:
    reader = _envelope({"after", "view"})
    issuer = _envelope(shape, scope=scope)

    with pytest.raises(OpaqueCursorError):
        reader.decode(issuer.encode(dict.fromkeys(shape, "item")))


def test_a_sealed_cursor_reveals_none_of_its_fields() -> None:
    """A catalog cursor names a row its holder may not see, so no field may show."""
    envelope = _envelope({"after"})

    token = envelope.encode({"after": "someone-elses-workspace"})
    sealed = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))

    assert b"someone-elses-workspace" not in sealed
    assert b"item-list" not in sealed
    assert envelope.encode({"after": "someone-elses-workspace"}) == token


def test_envelope_rejects_cross_domain_replay_and_tamper() -> None:
    first = _envelope({"after"}, domain="first")
    second = _envelope({"after"}, domain="second")
    token = first.encode({"after": "item-1"})
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")

    for forged in (token + "x", flipped, "not-a-token"):
        with pytest.raises(OpaqueCursorError):
            first.decode(forged)
    with pytest.raises(OpaqueCursorError):
        second.decode(token)
