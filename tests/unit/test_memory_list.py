# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded Profile Memory listing: the cursor codec and the REST and Web routes.

Service paging over a real store runs in tests/integration/test_memory_pg.py.
"""

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from dlightrag_memory import MemoryProvenance, MemoryRecord
from dlightrag_memory.errors import MemoryUnavailableError
from fastapi import HTTPException

from dlightrag.adapters.http.browser.routes.memory import list_memories as web_list_memories
from dlightrag.adapters.http.rest.routes.memory import list_memories as rest_list_memories
from dlightrag.application.access import UserContext, owner_id_from_user
from dlightrag.application.config import DlightragConfig
from dlightrag.application.memory import (
    MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    MEMORY_LIST_PAGE_MAX_LIMIT,
    MemoryDisabledError,
    MemoryListCursor,
    MemoryListCursorCodec,
    MemoryListCursorError,
    MemoryListPage,
    MemoryListPageRequest,
)
from tests.support.application_double import application_double

_UTC = datetime.UTC


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(origin_kind="management", origin_id="request-1")


def _record(
    *,
    owner: str = "alpha",
    body: str = "No email.",
    memory_id: str | None = None,
    updated_at: datetime.datetime | None = None,
) -> MemoryRecord:
    now = updated_at or datetime.datetime.now(_UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id or str(uuid.uuid4()),
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Cursor codec and page request contracts
# ---------------------------------------------------------------------------


def _codec() -> MemoryListCursorCodec:
    return MemoryListCursorCodec(b"memory-list-tests")


def test_cursor_roundtrip_is_canonical() -> None:
    codec = _codec()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    token = codec.encode(cursor)
    assert codec.decode(token) == cursor


@pytest.mark.parametrize(
    "token",
    [
        "",
        "only-one-part",
        "aaa.bbb",
        "x.y.z",
    ],
)
def test_malformed_tokens_are_rejected(token: str) -> None:
    with pytest.raises(MemoryListCursorError):
        _codec().decode(token)


def test_tampered_payload_fails_integrity() -> None:
    codec = _codec()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    token = codec.encode(cursor)
    encoded, encoded_mac = token.split(".")
    tampered = encoded[:-1] + ("A" if encoded[-1] != "A" else "B")
    with pytest.raises(MemoryListCursorError):
        codec.decode(f"{tampered}.{encoded_mac}")


def test_wrong_scope_and_version_are_rejected() -> None:
    import base64
    import hashlib
    import hmac
    import json

    secret = b"memory-list-tests"

    def make(scope: str, version: int) -> str:
        payload = json.dumps(
            {
                "memory_id": "12345678-1234-5678-1234-567812345678",
                "scope": scope,
                "updated_at": "2026-03-04T05:06:07.123456Z",
                "v": version,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        mac = hmac.new(secret, b"memory-list\0" + payload, hashlib.sha256).digest()[:16]
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        encoded_mac = base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
        return f"{encoded}.{encoded_mac}"

    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("other-scope", 1))
    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("memory-list", 2))
    with pytest.raises(MemoryListCursorError):
        _codec().decode(make("memory-list", True))  # type: ignore[arg-type]


def test_noncanonical_uuid_and_timestamp_are_rejected() -> None:
    codec = _codec()

    def token(overrides: dict[str, Any]) -> str:
        import base64
        import hashlib
        import hmac
        import json

        payload = json.dumps(
            {
                "memory_id": "12345678-1234-5678-1234-567812345678",
                "scope": "memory-list",
                "updated_at": "2026-03-04T05:06:07.123456Z",
                "v": 1,
                **overrides,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        mac = hmac.new(b"memory-list-tests", b"memory-list\0" + payload, hashlib.sha256).digest()[
            :16
        ]
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        encoded_mac = base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
        return f"{encoded}.{encoded_mac}"

    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"memory_id": "not-a-uuid"}))
    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"memory_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab".upper()}))
    with pytest.raises(MemoryListCursorError):
        codec.decode(token({"updated_at": "2026-03-04 05:06:07"}))


def test_page_request_validation() -> None:
    assert MemoryListPageRequest().limit == MEMORY_LIST_PAGE_DEFAULT_LIMIT
    assert MemoryListPageRequest(limit=1).limit == 1
    assert MemoryListPageRequest(limit=MEMORY_LIST_PAGE_MAX_LIMIT).limit == 100
    for bad in (0, -1, 101, True, "50"):
        with pytest.raises(ValueError):
            MemoryListPageRequest(limit=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MemoryListPageRequest(cursor="not-a-cursor")  # type: ignore[arg-type]


def test_cursor_rejects_bad_field_types() -> None:
    with pytest.raises(ValueError):
        MemoryListCursor(updated_at="now", memory_id=uuid.uuid4())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MemoryListCursor(
            updated_at=datetime.datetime.now(),  # naive
            memory_id=uuid.uuid4(),
        )
    with pytest.raises(ValueError):
        MemoryListCursor(
            updated_at=datetime.datetime.now(_UTC),
            memory_id="12345678-1234-5678-1234-567812345678",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# REST route
# ---------------------------------------------------------------------------


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    """The strict Application double; the routes list through its autospecced MemoryService."""
    application = application_double(test_config)
    application.memory.memory_list_cursor_codec = MemoryListCursorCodec(b"memory-list-tests")
    return application


def _request(application: Any) -> Any:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(application=application)),
        state=SimpleNamespace(user_context=_user()),
    )


def _user() -> UserContext:
    return UserContext(user_id="u-1", auth_mode="jwt")


@pytest.fixture(params=["rest", "web"])
def list_memories(request):
    from functools import partial

    return (
        partial(rest_list_memories, user=_user()) if request.param == "rest" else web_list_memories
    )


async def test_http_returns_page_with_next_cursor(list_memories, application: Any) -> None:
    memory = application.memory
    record = _record(owner="deployment")
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    memory.list_active_page.return_value = MemoryListPage(
        records=(record,),
        next_cursor=cursor,
    )

    response = await list_memories(_request(application))

    assert response["memories"] == [
        {"memory_id": record.memory_id, "kind": record.kind, "body": record.body}
    ]
    assert response["next_cursor"] == memory.memory_list_cursor_codec.encode(cursor)
    memory.list_active_page.assert_awaited_once()
    forwarded = memory.list_active_page.await_args
    assert forwarded is not None
    assert forwarded.kwargs["owner_id"] == owner_id_from_user(_user())
    assert forwarded.kwargs["auth_mode"] == "jwt"
    assert forwarded.kwargs["page"].limit == MEMORY_LIST_PAGE_DEFAULT_LIMIT
    assert forwarded.kwargs["page"].cursor is None


async def test_http_decodes_cursor_and_passes_it_through(list_memories, application: Any) -> None:
    memory = application.memory
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    memory.list_active_page.return_value = MemoryListPage(records=(), next_cursor=None)

    response = await list_memories(
        _request(application),
        limit=1,
        cursor=memory.memory_list_cursor_codec.encode(cursor),
    )

    assert response["next_cursor"] is None
    forwarded_call = memory.list_active_page.await_args
    assert forwarded_call is not None
    forwarded = forwarded_call.kwargs["page"]
    assert forwarded.limit == 1
    assert forwarded.cursor == cursor


async def test_http_rejects_tampered_cursor_before_service(list_memories, application: Any) -> None:
    memory = application.memory

    with pytest.raises(HTTPException) as exc:
        await list_memories(_request(application), cursor="tampered.token")
    assert exc.value.status_code == 422
    memory.list_active_page.assert_not_awaited()


async def test_http_leaves_memory_refusals_to_the_shared_error_handlers(
    list_memories, application: Any
) -> None:
    """Routes no longer translate Memory refusals; `install_error_handlers` answers them."""
    memory = application.memory

    memory.list_active_page.side_effect = MemoryDisabledError()
    with pytest.raises(MemoryDisabledError):
        await list_memories(_request(application))

    memory.list_active_page.side_effect = MemoryUnavailableError()
    with pytest.raises(MemoryUnavailableError):
        await list_memories(_request(application))
