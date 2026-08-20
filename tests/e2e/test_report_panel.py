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
        if path == "/web/api/conversations":
            route.fulfill(json=[_CONVERSATION])
            return
        if path == f"/web/api/conversations/{_CONVERSATION['conversation_id']}/history":
            route.fulfill(json={"conversation": _CONVERSATION, "turns": [turn]})
            return
        route.continue_()

    def handle_report(route: Route) -> None:
        if report_html is None:
            route.fulfill(status=404, json={"detail": "Primary report not found"})
            return
        route.fulfill(status=200, content_type="text/html; charset=utf-8", body=report_html)

    page.route("**/web/api/conversations**", handle_conversations)
    page.route(f"**/web/api/answer/{_RUN_ID}/report", handle_report)


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
    assert "open" not in (page.locator("#report-panel").get_attribute("class") or "")
    control.click()
    page.wait_for_function(
        "document.getElementById('report-panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert page.locator("#report-panel-title").inner_text() == "Report"
    assert "Quarterly review" in page.locator("#report-panel-content").inner_text()
    assert "Long body." in page.locator("#report-panel-content").inner_text()


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
        "document.getElementById('report-panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert "Only the report." in page.locator("#report-panel-content").inner_text()


def test_report_citation_opens_sources_beside_the_report(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    _install_routes(
        page,
        turn=_turn(
            answer="See the report.",
            html='<div id="answer-content"><p>See the report.</p></div>',
            primary_report="primary_report",
        ),
        report_html="""
          <div id="answer-content">
            <p>See
              <span class="citation-badge" data-ref="1" data-chunk="1"
                    data-action="filter-source" role="button" tabindex="0">1</span>.
            </p>
          </div>
          <div class="source-data hidden">
            <div class="source-doc" data-ref="1">
              <div class="source-doc-header">
                <button class="source-doc-toggle" type="button" data-action="toggle-doc">
                  <span class="source-doc-title">paper.pdf</span>
                </button>
              </div>
              <div class="source-doc-chunks" hidden>
                <div class="source-chunk" data-ref="1" data-chunk="1">Evidence chunk</div>
              </div>
            </div>
          </div>
        """,
    )
    _open_ready_page(page)
    page.get_by_role("button", name="View report").click()
    page.wait_for_function(
        "document.getElementById('report-panel')?.classList.contains('open')",
        timeout=10000,
    )
    page.locator("#report-panel-content").get_by_role("button", name="1").click()
    page.wait_for_function(
        "document.getElementById('panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert page.locator("#report-panel").evaluate("el => el.classList.contains('open')")
    assert "See" in page.locator("#report-panel-content").inner_text()
    assert "paper.pdf" in page.locator("#panel-content").inner_text()
