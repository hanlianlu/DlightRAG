# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for what an answer says the Run left in Profile Memory."""

from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route, expect

from tests.e2e.test_source_panel import _open_ready_page

_CHANGES = [
    {
        "memory_id": "memory-1",
        "kind": "preference",
        "body": "Answers open with a table.",
        "action": "add",
        "change_id": "change-1",
        "writer": "conversation",
    },
    {
        "memory_id": "memory-2",
        "kind": "fact",
        "body": "Lives in Munich.",
        "action": "update",
        "change_id": "change-2",
        "writer": "conversation",
    },
]


def _receipt(**fields: Any) -> dict[str, Any]:
    return {
        "action": "forget",
        "outcome": "changed",
        "change_id": "forget-1",
        "memory_ids": ["memory-1"],
        "kind": "preference",
        "body": "Answers open with a table.",
        "supersedes_id": None,
        "target_change_id": None,
        **fields,
    }


def _install_memory_routes(page: Page, requests: list[tuple[str, str]]) -> None:
    """The Run left two changes; forgetting or undoing one is answered as the server does."""

    def memory(route: Route) -> None:
        method, path = route.request.method, urlparse(route.request.url).path
        requests.append((method, path))
        if path.endswith("/changes") and method == "GET":
            route.fulfill(json={"changes": _CHANGES})
        elif method == "DELETE":
            route.fulfill(json=_receipt())
        elif path.endswith("/undo"):
            route.fulfill(
                json=_receipt(
                    action="undo",
                    change_id="undo-1",
                    memory_ids=["restored"],
                    kind="fact",
                    body="Lives in Berlin.",
                    target_change_id="change-2",
                )
            )
        else:
            route.fulfill(status=404, json={})

    page.route("**/web/api/memory/**", memory)


def _answer(page: Page) -> None:
    page.wait_for_selector("[aria-current='page']", timeout=10000)
    page.locator(".composer-input").fill("what did you learn about me")
    page.click(".composer-send")
    page.get_by_role("button", name="Fork").last.wait_for(timeout=10000)


@pytest.mark.e2e
def test_an_answer_says_what_the_run_left_in_memory_and_the_reader_takes_it_back(
    page: Page,
) -> None:
    requests: list[tuple[str, str]] = []
    _install_memory_routes(page, requests)
    _open_ready_page(page)
    _answer(page)

    # The pass has had no time when the Run is done, so the notice comes with the first look after it.
    notice = page.locator("dl-memory-notice").last.get_by_role("button").first
    expect(notice).to_have_text("Remembered 2", timeout=10000)
    # The reader was here for the answer, so it is said once, politely, to a screen reader.
    expect(page.get_by_role("status").filter(has_text="Remembered 2")).to_have_count(1)
    expect(notice).to_have_attribute("aria-expanded", "false")
    notice.click()
    popover = page.get_by_role("dialog", name="Remembered from this conversation")
    expect(popover).to_be_visible()
    expect(popover.get_by_role("listitem")).to_have_count(2)

    popover.get_by_role("button", name="Undo: Answers open with a table.").click()

    expect(popover.get_by_role("listitem").first).to_contain_text("Undone")
    expect(notice).to_have_text("Remembered 1")
    expect(page.get_by_text("Forgot: Answers open with a table.")).to_be_visible()
    assert ("DELETE", "/web/api/memory/memory-1") in requests

    popover.get_by_role("button", name="Undo: Lives in Munich.").click()
    expect(notice).to_have_text("Undone")
    assert ("POST", "/web/api/memory/changes/change-2/undo") in requests

    page.keyboard.press("Escape")
    expect(notice).to_have_attribute("aria-expanded", "false")
    expect(notice).to_be_focused()


@pytest.mark.e2e
def test_the_notice_is_a_sheet_from_the_bottom_edge_on_a_phone(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    _install_memory_routes(page, [])
    _open_ready_page(page)
    _answer(page)

    notice = page.get_by_role("button", name="Remembered 2")
    expect(notice).to_be_visible(timeout=10000)
    notice.click()

    popover = page.get_by_role("dialog", name="Remembered from this conversation")
    expect(popover).to_be_visible()
    box = popover.bounding_box()
    assert box is not None
    assert box["x"] == pytest.approx(0, abs=1)
    assert box["width"] == pytest.approx(390, abs=1)
    assert box["y"] + box["height"] == pytest.approx(844, abs=1)
    for control in (
        popover.get_by_role("button", name="Undo: Lives in Munich."),
        popover.get_by_role("button", name="Manage memory in Settings"),
    ):
        target = control.bounding_box()
        assert target is not None
        assert target["height"] >= 44, target


@pytest.mark.e2e
def test_the_notice_opens_settings_on_profile_memory(page: Page) -> None:
    _install_memory_routes(page, [])
    _open_ready_page(page)
    _answer(page)
    page.get_by_role("button", name="Remembered 2").click(timeout=10000)

    page.get_by_role("button", name="Manage memory in Settings").click()

    expect(page.locator("#settings-dialog")).to_be_visible()
    expect(page.locator('#settings-dialog [data-section="memory"]')).to_have_attribute(
        "aria-current", "page"
    )
