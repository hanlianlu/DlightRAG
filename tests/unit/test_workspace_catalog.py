# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for bounded workspace-catalog pages and cursors."""

import pytest

from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogCursor,
    WorkspaceCatalogCursorCodec,
    WorkspaceCatalogCursorError,
    WorkspaceCatalogPageRequest,
)


def _codec() -> WorkspaceCatalogCursorCodec:
    return WorkspaceCatalogCursorCodec(b"catalog-test-secret")


def test_cursor_roundtrips_canonical_workspace_ordering_key() -> None:
    codec = _codec()
    cursor = WorkspaceCatalogCursor(after_workspace="finance")

    token = codec.encode(cursor)
    decoded = codec.decode(token)

    assert decoded == cursor
    assert decoded.after_workspace == "finance"


@pytest.mark.parametrize(
    "workspace",
    ["Finance", "finance-reports", "finance reports", "9lives", "finance!", ""],
)
def test_cursor_rejects_noncanonical_workspace(workspace: str) -> None:
    with pytest.raises(ValueError):
        WorkspaceCatalogCursor(after_workspace=workspace)


def test_decode_rejects_tampered_mac() -> None:
    codec = _codec()
    token = codec.encode(WorkspaceCatalogCursor(after_workspace="finance"))
    payload, mac = token.split(".")
    tampered = f"{payload}.{('A' + mac[1:])}"

    with pytest.raises(WorkspaceCatalogCursorError):
        codec.decode(tampered)


def test_decode_rejects_modified_payload() -> None:
    codec = _codec()
    token = codec.encode(WorkspaceCatalogCursor(after_workspace="finance"))
    payload, mac = token.split(".")
    tampered = f"{payload[:-2]}AA.{mac}"

    with pytest.raises(WorkspaceCatalogCursorError):
        codec.decode(tampered)


@pytest.mark.parametrize(
    "token",
    [
        "",
        "no-dot",
        ".mac-only",
        "payload.",
        "not-base64!....",
    ],
)
def test_decode_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(WorkspaceCatalogCursorError):
        _codec().decode(token)


def test_decode_rejects_wrong_scope_and_version() -> None:
    codec = WorkspaceCatalogCursorCodec(b"catalog-test-secret")
    import base64
    import hashlib
    import hmac
    import json

    def sign(payload: bytes) -> bytes:
        mac = hmac.new(b"catalog-test-secret", b"workspace-catalog\0" + payload, hashlib.sha256)
        return mac.digest()[:16]

    def encode(value: dict[str, object]) -> str:
        payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        mac = base64.urlsafe_b64encode(sign(payload)).rstrip(b"=").decode()
        return f"{encoded}.{mac}"

    with pytest.raises(WorkspaceCatalogCursorError):
        codec.decode(encode({"after_workspace": "finance", "scope": "other", "v": 1}))
    with pytest.raises(WorkspaceCatalogCursorError):
        codec.decode(encode({"after_workspace": "finance", "scope": "workspace-catalog", "v": 2}))
    with pytest.raises(WorkspaceCatalogCursorError):
        codec.decode(
            encode({"after_workspace": "finance", "scope": "workspace-catalog", "v": 1, "extra": 1})
        )


def test_page_request_defaults_and_bounds() -> None:
    default = WorkspaceCatalogPageRequest()
    assert default.limit == WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT
    assert default.cursor is None
    assert WorkspaceCatalogPageRequest(limit=1).limit == 1
    assert WorkspaceCatalogPageRequest(limit=WORKSPACE_CATALOG_PAGE_MAX_LIMIT).limit == 100


@pytest.mark.parametrize("limit", [0, 101, -1, True, 3.5, "50"])
def test_page_request_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(ValueError):
        WorkspaceCatalogPageRequest(limit=limit)  # type: ignore[arg-type]


def test_page_request_rejects_invalid_cursor() -> None:
    with pytest.raises(ValueError):
        WorkspaceCatalogPageRequest(cursor="finance")  # type: ignore[arg-type]
