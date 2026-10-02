# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for bounded workspace-catalog pages and cursors."""

import pytest

from dlightrag.application.corpus_admin import (
    WORKSPACE_CATALOG_PAGE_DEFAULT_LIMIT,
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    FilePanelCursor,
    FilePanelCursorCodec,
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


def test_decode_rejects_tampered_and_foreign_tokens() -> None:
    codec = _codec()
    token = codec.encode(WorkspaceCatalogCursor(after_workspace="finance"))
    foreign = [
        WorkspaceCatalogCursorCodec(b"other-secret").encode(
            WorkspaceCatalogCursor(after_workspace="finance")
        ),
        FilePanelCursorCodec(b"catalog-test-secret").encode(
            FilePanelCursor(workspace="finance", updated_at=None, doc_id="doc-1")
        ),
    ]

    for value in (("A" if token[0] != "A" else "B") + token[1:], token + "x", *foreign):
        with pytest.raises(WorkspaceCatalogCursorError):
            codec.decode(value)


@pytest.mark.parametrize("token", ["", "not-a-cursor", "not-base64!....", "A" * 64])
def test_decode_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(WorkspaceCatalogCursorError):
        _codec().decode(token)


def test_decode_rejects_invalid_sealed_values() -> None:
    """Defense in depth: a sealed token is trusted for its shape, never its values."""
    codec = _codec()

    for after_workspace in ("Finance", 7):
        with pytest.raises(WorkspaceCatalogCursorError):
            codec.decode(codec._envelope.encode({"after_workspace": after_workspace}))


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
