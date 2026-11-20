# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded Profile Memory listing: the cursor codec and the REST and Web routes.

Service paging over a real store runs in tests/integration/test_memory_pg.py.
"""

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from dlightrag_memory import (
    MemoryOperationReceipt,
    MemoryProvenance,
    MemoryRecord,
    MemoryRunChange,
)
from fastapi import HTTPException

from dlightrag.adapters.http.browser.routes.memory import list_memories as web_list_memories
from dlightrag.adapters.http.browser.routes.memory import run_memory_changes as web_run_changes
from dlightrag.adapters.http.rest.routes.memory import list_memories as rest_list_memories
from dlightrag.adapters.http.rest.routes.memory import run_memory_changes as rest_run_changes
from dlightrag.application.access.principal import (
    UserContext,
    owner_id_from_user,
)
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.memory.errors import MemoryDisabledError
from dlightrag.application.memory.memory_list import (
    MEMORY_LIST_PAGE_DEFAULT_LIMIT,
    MEMORY_LIST_PAGE_MAX_LIMIT,
    MemoryListCursor,
    MemoryListCursorCodec,
    MemoryListCursorError,
    MemoryListPage,
    MemoryListPageRequest,
)
from dlightrag.application.memory.projections import memory_receipt_payload
from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex
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


def test_tampered_token_fails_integrity() -> None:
    codec = _codec()
    cursor = MemoryListCursor(
        updated_at=datetime.datetime(2026, 3, 4, 5, 6, 7, 123456, tzinfo=_UTC),
        memory_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    token = codec.encode(cursor)
    with pytest.raises(MemoryListCursorError):
        codec.decode(("A" if token[0] != "A" else "B") + token[1:])


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
        {
            "memory_id": record.memory_id,
            "kind": record.kind,
            "body": record.body,
            "origin_kind": record.provenance.origin_kind,
        }
    ]
    assert response["next_cursor"] == memory.memory_list_cursor_codec.encode(cursor)
    memory.list_active_page.assert_awaited_once()
    forwarded = memory.list_active_page.await_args
    assert forwarded is not None
    assert forwarded.kwargs["owner_id"] == owner_id_from_user(_user())
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


# ---------------------------------------------------------------------------
# What one Run's writers left
# ---------------------------------------------------------------------------


@pytest.fixture(params=["rest", "web"])
def run_changes(request):
    from functools import partial

    return partial(rest_run_changes, user=_user()) if request.param == "rest" else web_run_changes


def _change(
    action: str, origin_kind: str, *, body: str = "Answer in Chinese.", kind: str = "preference"
) -> MemoryRunChange:
    record = MemoryRecord(
        owner_id="deployment",
        memory_id=str(uuid.uuid4()),
        kind=kind,  # type: ignore[arg-type]
        body=body,
        provenance=MemoryProvenance(
            origin_kind=origin_kind,  # type: ignore[arg-type]
            origin_id="origin",
            run_id="run-1",
            session_id="session-1",
        ),
    )
    return MemoryRunChange(
        record=record,
        change_id=str(uuid.uuid4()),
        action=action,  # type: ignore[arg-type]
    )


async def test_http_names_each_change_with_the_writer_that_made_it(
    run_changes, application: Any
) -> None:
    memory = application.memory
    by_agent = _change("add", "answer_run", body="Prefers tables first.")
    by_pass = _change("update", "session_pass", body="Lives in Munich.", kind="fact")
    merged = _change("fuse", "session_pass", body="Likes terse answers.")
    memory.run_changes.return_value = (by_agent, by_pass, merged)

    response = await run_changes("run-1", _request(application))

    assert response == {
        "changes": [
            {
                "memory_id": change.record.memory_id,
                "kind": change.record.kind,
                "body": change.record.body,
                "action": change.action,
                "change_id": change.change_id,
                "writer": writer,
            }
            for change, writer in (
                (by_agent, "agent"),
                (by_pass, "conversation"),
                (merged, "conversation"),
            )
        ]
    }
    memory.run_changes.assert_awaited_once_with(
        owner_id=owner_id_from_user(_user()), run_id="run-1"
    )


async def test_http_answers_a_run_with_nothing_left_with_an_empty_list(
    run_changes, application: Any
) -> None:
    application.memory.run_changes.return_value = ()

    assert await run_changes("run-1", _request(application)) == {"changes": []}


# ---------------------------------------------------------------------------
# The reader content gate
# ---------------------------------------------------------------------------

_PROTECTED_PARAGRAPH = (
    "The protected method reconciles the ledger in two passes: first it groups every "
    "receipt by its audit window, then it compares the grouped totals against the "
    "settlement extract before writing a single adjustment."
)


def _protected_index() -> ProtectedTextIndex:
    """A one-document fingerprint index, standing in for the process's Skill protection."""
    return ProtectedTextIndex.build(
        [("charts", "references/method.md", _PROTECTED_PARAGRAPH.encode("utf-8"))]
    )


def _gate_application(application: Any) -> None:
    """Give the double a populated reader gate, the way a composed process carries one."""
    application.skill_protection = SimpleNamespace(mask=_protected_index().mask)


async def test_http_masks_protected_text_in_a_memory_body(list_memories, application: Any) -> None:
    """A Memory row is a reader surface: its body carries the same bytes as any other carrier."""
    memory = application.memory
    memory.list_active_page.return_value = MemoryListPage(
        records=(_record(owner="deployment", body=f"Kept:\n{_PROTECTED_PARAGRAPH}"),),
        next_cursor=None,
    )
    _gate_application(application)

    response = await list_memories(_request(application))

    body = response["memories"][0]["body"]
    assert _PROTECTED_PARAGRAPH not in body
    assert body == f"Kept:\n{MASK_PLACEHOLDER}"


async def test_http_masks_protected_text_in_a_run_change_body(
    run_changes, application: Any
) -> None:
    """The change feed repeats each record's body, so it passes the same gate."""
    application.memory.run_changes.return_value = (
        _change("add", "answer_run", body=f"Copied:\n{_PROTECTED_PARAGRAPH}"),
    )
    _gate_application(application)

    response = await run_changes("run-1", _request(application))

    body = response["changes"][0]["body"]
    assert _PROTECTED_PARAGRAPH not in body
    assert body == f"Copied:\n{MASK_PLACEHOLDER}"


def test_a_receipt_projection_masks_protected_text_in_the_body() -> None:
    """A mutation's response is a reader surface too; Memory itself keeps the original."""
    receipt = MemoryOperationReceipt(
        change_id="change-1",
        action="remember",
        outcome="changed",
        memory_ids=("memory-1",),
        provenance=_provenance(),
        kind="fact",
        body=_PROTECTED_PARAGRAPH,
    )

    assert memory_receipt_payload(receipt)["body"] == _PROTECTED_PARAGRAPH

    masked = memory_receipt_payload(receipt, mask_reader_text=_protected_index().mask)
    assert masked["body"] == MASK_PLACEHOLDER
    assert masked["memory_ids"] == ["memory-1"]
