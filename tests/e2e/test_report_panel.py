# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for opening a Primary Report in the document panel."""

from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route

pytestmark = pytest.mark.e2e

_TIMESTAMP = "2026-08-20T12:00:00Z"
_CONVERSATION = {
    "conversation_id": "report-history",
    "title": "Report",
    "created_at": _TIMESTAMP,
    "updated_at": _TIMESTAMP,
}
_RUN_ID = "report-run"


def _turn(*, answer: str, html: str, primary_report: str | None) -> dict[str, object]:
    return {
        "turn_id": "report-turn",
        "turn_number": 1,
        "answer_run_id": _RUN_ID,
        "submission_id": "report-submission",
        "status": "succeeded",
        "cancel_requested": False,
        "user_text": "Write the report",
        "assistant_text": answer,
        "user_attachments": [],
        "answer_html": html,
        "primary_report": primary_report,
        "error_kind": None,
        "error_message": None,
        "created_at": _TIMESTAMP,
    }


def _install_routes(page: Page, *, turn: dict[str, object], report_html: str | None) -> None:
    def handle_conversations(route: Route) -> None:
        path = urlparse(route.request.url).path
        if path == "/web/conversations":
            route.fulfill(json=[_CONVERSATION])
            return
        if path == f"/web/conversations/{_CONVERSATION['conversation_id']}/history":
            route.fulfill(json={"conversation": _CONVERSATION, "turns": [turn]})
            return
        route.continue_()

    def handle_report(route: Route) -> None:
        if report_html is None:
            route.fulfill(status=404, json={"detail": "Primary report not found"})
            return
        route.fulfill(status=200, content_type="text/html; charset=utf-8", body=report_html)

    page.route("**/web/conversations**", handle_conversations)
    page.route(f"**/web/answer/{_RUN_ID}/report", handle_report)


def _open_ready_page(page: Page) -> None:
    with page.expect_response(
        lambda response: (
            response.request.method == "GET" and response.url.endswith("/history") and response.ok
        ),
        timeout=10000,
    ) as history_response:
        page.goto("/web/")
    history_response.value.finished()
    page.wait_for_selector(".composer-input", timeout=10000)


def test_view_report_opens_the_panel_on_click(page: Page) -> None:
    _install_routes(
        page,
        turn=_turn(
            answer="Delivery note.",
            html='<div id="answer-content"><p>Delivery note.</p></div>',
            primary_report="primary_report",
        ),
        report_html='<div id="answer-content"><h1>Quarterly review</h1><p>Long body.</p></div>',
    )
    _open_ready_page(page)
    control = page.get_by_role("button", name="View report")
    control.wait_for(timeout=10000)
    panel = page.locator("#panel")
    assert "open" not in (panel.get_attribute("class") or "")
    control.click()
    page.wait_for_function(
        "document.getElementById('panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert page.locator("#panel-title").inner_text() == "Report"
    assert "Quarterly review" in page.locator("#panel-content").inner_text()
    assert "Long body." in page.locator("#panel-content").inner_text()


def test_no_report_handle_has_no_control(page: Page) -> None:
    _install_routes(
        page,
        turn=_turn(
            answer="Just chat.",
            html='<div id="answer-content"><p>Just chat.</p></div>',
            primary_report=None,
        ),
        report_html=None,
    )
    _open_ready_page(page)
    page.wait_for_selector('[class*="aiMessageContent"]', timeout=10000)
    assert page.get_by_role("button", name="View report").count() == 0


def test_empty_chat_still_shows_the_report_control(page: Page) -> None:
    _install_routes(
        page,
        turn=_turn(
            answer="",
            html='<div id="answer-content"></div>',
            primary_report="primary_report",
        ),
        report_html='<div id="answer-content"><p>Only the report.</p></div>',
    )
    _open_ready_page(page)
    control = page.get_by_role("button", name="View report")
    control.wait_for(timeout=10000)
    assert "Only the report." not in page.locator("#chat-messages").inner_text()
    control.click()
    page.wait_for_function(
        "document.getElementById('panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert "Only the report." in page.locator("#panel-content").inner_text()
