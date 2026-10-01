# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded file-panel cursor and the refusals the adapter settles before any read.

Page traversal and its indexes run against PostgreSQL in tests/integration/test_pg_storage.py.
"""

import base64
import datetime
import hashlib
import hmac
import json
from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
from dlightrag.application.corpus_admin import (
    FilePanelCursor,
    FilePanelCursorCodec,
    FilePanelCursorError,
    FilePanelPageRequest,
)


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Pool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    """Records every read, so a refusal can show it reached no storage."""

    def __init__(self) -> None:
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        return []


def _signed_token(secret: bytes, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, b"file-panel\0" + raw, hashlib.sha256).digest()[:16]

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    return f"{encode(raw)}.{encode(mac)}"


def test_file_panel_cursor_round_trips_null_and_naive_microseconds() -> None:
    codec = FilePanelCursorCodec(b"cursor-secret")
    timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)

    for cursor in (
        FilePanelCursor(workspace="finance", updated_at=None, doc_id="n" * 255),
        FilePanelCursor(workspace="finance", updated_at=timestamp, doc_id="doc-2"),
        FilePanelCursor(
            workspace="finance",
            updated_at=timestamp,
            doc_id="failed-1",
            view="failed",
        ),
    ):
        assert codec.decode(codec.encode(cursor)) == cursor

    legacy = _signed_token(
        b"cursor-secret",
        {
            "doc_id": "legacy-processed",
            "scope": "file-panel",
            "updated_at": None,
            "v": 1,
            "workspace": "finance",
        },
    )
    assert codec.decode(legacy) == FilePanelCursor(
        workspace="finance",
        updated_at=None,
        doc_id="legacy-processed",
    )


def test_file_panel_cursor_rejects_tamper_malformed_scope_version_and_noncanonical() -> None:
    secret = b"cursor-secret"
    codec = FilePanelCursorCodec(secret)
    cursor = FilePanelCursor(workspace="finance", updated_at=None, doc_id="doc-1")
    token = codec.encode(cursor)
    encoded, mac = token.split(".")

    invalid = [
        token + "x",
        "not-a-token",
        f"{encoded}=.{mac}",
        _signed_token(
            secret,
            {
                "doc_id": "doc-1",
                "scope": "conversation-history",
                "updated_at": None,
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "doc_id": "doc-1",
                "scope": "file-panel",
                "updated_at": None,
                "v": 2,
                "workspace": "finance",
            },
        ),
    ]
    for value in invalid:
        with pytest.raises(FilePanelCursorError):
            codec.decode(value)


def test_file_panel_page_validation_and_cursor_invariants() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        FilePanelPageRequest(limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        FilePanelPageRequest(limit=101)
    with pytest.raises(ValueError, match="integer"):
        FilePanelPageRequest(limit=True)
    with pytest.raises(ValueError, match="timezone"):
        FilePanelCursor(
            workspace="finance",
            updated_at=datetime.datetime.now(datetime.UTC),
            doc_id="doc",
        )
    with pytest.raises(ValueError, match="non-empty"):
        FilePanelCursor(workspace="finance", updated_at=None, doc_id="")


async def test_file_panel_store_rejects_cross_view_cursor_before_fetch() -> None:
    conn = _Conn()
    store = PGFilePanelStore(pool=_Pool(conn))
    failed_cursor = FilePanelCursor(
        workspace="finance",
        updated_at=None,
        doc_id="failed-1",
        view="failed",
    )

    with pytest.raises(ValueError, match="another view"):
        await store.list_processed_files(
            "finance",
            page=FilePanelPageRequest(cursor=failed_cursor),
        )
    with pytest.raises(ValueError, match="another view"):
        await store.list_failed_files(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(
                    workspace="finance",
                    updated_at=None,
                    doc_id="processed-1",
                )
            ),
        )

    assert conn.fetches == []


async def test_file_panel_store_rejects_cross_workspace_before_fetch() -> None:
    conn = _Conn()
    store = PGFilePanelStore(pool=_Pool(conn))

    with pytest.raises(ValueError, match="another workspace"):
        await store.list_processed_files(
            "finance",
            page=FilePanelPageRequest(
                cursor=FilePanelCursor(workspace="legal", updated_at=None, doc_id="doc")
            ),
        )

    assert conn.fetches == []
