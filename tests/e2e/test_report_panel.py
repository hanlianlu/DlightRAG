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


def _presentation(
    *,
    answer: str,
    html: str,
    primary_report: str | None = None,
    sources: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "answer_text": answer,
        "answer_html": html,
        "sources": sources or [],
        "answer_images": [],
        "primary_report": primary_report,
    }


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
        "presentation": _presentation(
            answer=answer,
            html=html,
            primary_report=primary_report,
        ),
        "error_kind": None,
        "error_message": None,
        "created_at": _TIMESTAMP,
    }


def _install_routes(
    page: Page,
    *,
    turn: dict[str, object],
    report: dict[str, object] | None,
) -> None:
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
        if report is None:
            route.fulfill(status=404, json={"detail": "Primary report not found"})
            return
        route.fulfill(status=200, json=report)

    page.route("**/web/api/conversations**", handle_conversations)
    page.route(f"**/web/api/answer/{_RUN_ID}/report", handle_report)


def _open_ready_page(page: Page) -> None:
    with page.expect_response(
        lambda response: (
            response.request.method == "GET" and response.url.endswith("/history") and response.ok
        ),
        timeout=10000,
    ) as history_response:
        page.goto(f"/web/conversations/{_CONVERSATION['conversation_id']}")
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
        report=_presentation(
            answer="# Quarterly review\n\nLong body.",
            html="<h1>Quarterly review</h1><p>Long body.</p>",
        ),
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
        report=None,
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
        report=_presentation(answer="Only the report.", html="<p>Only the report.</p>"),
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
        report=_presentation(
            answer="See [1-1].",
            html=(
                '<p>See <cite class="citation-badge" data-ref="1" data-chunk="1" '
                'role="button" tabindex="0" aria-label="Source 1, chunk 1">1</cite>.</p>'
            ),
            sources=[
                {
                    "id": "1",
                    "title": "paper.pdf",
                    "source_url": None,
                    "download_url": None,
                    "chunks": [
                        {
                            "chunk_idx": 1,
                            "page_number": 1,
                            "content_html": "<p>Evidence chunk</p>",
                            "image_url": None,
                            "thumbnail_url": None,
                        }
                    ],
                }
            ],
        ),
    )
    _open_ready_page(page)
    page.get_by_role("button", name="View report").click()
    page.wait_for_function(
        "document.getElementById('report-panel')?.classList.contains('open')",
        timeout=10000,
    )
    page.locator("#report-panel-content .citation-badge").click()
    page.wait_for_function(
        "document.getElementById('panel')?.classList.contains('open')",
        timeout=10000,
    )
    assert page.locator("#report-panel").evaluate("el => el.classList.contains('open')")
    assert "See" in page.locator("#report-panel-content").inner_text()
    assert "paper.pdf" in page.locator("#panel-content").inner_text()
