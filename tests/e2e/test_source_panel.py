# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for source panel: citation click, expand/collapse."""

from urllib.parse import urlparse

import pytest


def _open_ready_page(page) -> None:
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    conversation = page.evaluate(
        """async () => {
            const token = document.cookie.split('; ')
                .find(value => value.startsWith('dlightrag_web_csrf='))?.split('=')[1] || '';
            const response = await fetch('/web/api/conversations', {
                method: 'POST',
                headers: {'X-CSRF-Token': decodeURIComponent(token)},
            });
            return await response.json();
        }"""
    )
    conversation_id = str(conversation["conversation_id"])
    with page.expect_response(
        lambda response: (
            response.request.method == "GET" and response.url.endswith("/history") and response.ok
        ),
        timeout=10000,
    ) as history_response:
        page.goto(f"/web/conversations/{conversation_id}")
    history_response.value.finished()
    page.wait_for_selector(".composer-input", timeout=10000)


_CITED_HTML = (
    "<p>DlightRAG cited answer "
    '<cite class="citation-badge" data-ref="1" data-chunk="1" '
    'role="button" tabindex="0" aria-label="Source 1, chunk 1">1-1</cite>.</p>'
)


def _source_presentation_domain(
    *,
    source_url: str | None = None,
    image_url: str | None = None,
) -> dict:
    """Presentation in camelCase domain shape, injected into element properties."""
    return {
        "answerText": "DlightRAG cited answer [1-1].",
        "parts": [
            {
                "type": "markdown",
                "text": "DlightRAG cited answer [1-1].",
                "html": _CITED_HTML,
                "artifact": None,
                "evidenceImage": None,
                "inline": False,
            }
        ],
        "sources": [
            {
                "id": "1",
                "title": "report.pdf",
                "sourceUrl": source_url,
                "downloadUrl": "/web/api/files/raw/doc-report?workspace=default",
                "chunks": [
                    {
                        "chunkIdx": 1,
                        "pageNumber": 1,
                        "contentHtml": "<p>Evidence text</p>",
                        "imageUrl": image_url,
                        "thumbnailUrl": image_url,
                    }
                ],
            }
        ],
        "evidenceImages": [],
        "artifacts": [],
        "artifactOutcome": {"status": "complete", "issues": []},
    }


def _source_presentation_wire(
    *,
    source_url: str | None = None,
    image_url: str | None = None,
) -> dict:
    """Presentation in snake_case wire shape, served by the mocked server."""
    return {
        "answer_text": "DlightRAG cited answer [1-1].",
        "parts": [
            {
                "type": "markdown",
                "text": "DlightRAG cited answer [1-1].",
                "html": _CITED_HTML,
                "artifact": None,
                "evidence_image": None,
                "inline": False,
            }
        ],
        "sources": [
            {
                "id": "1",
                "title": "report.pdf",
                "source_url": source_url,
                "download_url": "/web/api/files/raw/doc-report?workspace=default",
                "chunks": [
                    {
                        "chunk_idx": 1,
                        "page_number": 1,
                        "content_html": "<p>Evidence text</p>",
                        "image_url": image_url,
                        "thumbnail_url": image_url,
                    }
                ],
            }
        ],
        "evidence_images": [],
        "artifacts": [],
        "artifact_outcome": {"status": "complete", "issues": []},
    }


def _inject_answer_with_sources(page, *, image_url: str | None = None) -> None:
    page.wait_for_selector("[aria-current='page']", timeout=10000)
    page.locator(".composer-input").fill("show cited source")
    page.click(".composer-send")
    page.get_by_role("button", name="Follow up").last.wait_for(timeout=10000)
    page.locator("dl-answer-presentation").last.evaluate(
        """(element, presentation) => {
          element.presentation = presentation;
          return element.updateComplete;
        }""",
        _source_presentation_domain(image_url=image_url),
    )
    page.locator("[data-answer-ref]").last.wait_for()


@pytest.mark.e2e
def test_reference_item_keyboard_opens_source_panel(page):
    """Keyboard activation on a reference item opens its expanded source."""
    _open_ready_page(page)
    _inject_answer_with_sources(page)

    page.locator("[data-answer-ref]").press("Enter")

    page.wait_for_selector('#panel-content [data-ref="1"][data-expanded]', timeout=10000)
    assert page.locator("#panel-title").text_content() == "Sources"
    assert page.locator("#panel-content [data-source-content]").text_content() == "Evidence text"
    assert "Wrong decoy" not in page.locator("#panel-content").text_content()


@pytest.mark.e2e
def test_mobile_source_panel_is_full_bleed_and_touch_reachable(page):
    page.set_viewport_size({"width": 390, "height": 844})
    _open_ready_page(page)
    _inject_answer_with_sources(page)
    page.locator("[data-answer-ref]").press("Enter")

    panel = page.locator('#panel.open[data-panel-kind="sources"]')
    panel.wait_for()
    details = panel.evaluate(
        """element => {
            const rect = element.getBoundingClientRect();
            const top = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
            return {
                rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
                topmost: top === element || (top !== null && element.contains(top)),
                overflow: document.documentElement.scrollWidth - innerWidth,
            };
        }"""
    )
    assert details["rect"]["x"] == pytest.approx(0, abs=1)
    assert details["rect"]["width"] == pytest.approx(390, abs=1)
    assert details["rect"]["height"] == pytest.approx(844, abs=1)
    assert details["topmost"] is True, details
    assert details["overflow"] <= 1, details

    for control in (
        panel.get_by_role("button", name="Close panel"),
        panel.locator("[data-source-toggle]").first,
        panel.get_by_role("link", name="Download source"),
    ):
        box = control.bounding_box()
        assert box is not None
        assert box["width"] >= 44, box
        assert box["height"] >= 44, box


@pytest.mark.e2e
def test_conversation_route_change_closes_sources_panel(page):
    _open_ready_page(page)
    _inject_answer_with_sources(page)
    page.locator("[data-answer-ref]").last.click()
    page.wait_for_selector('#panel.open[data-panel-kind="sources"]')

    page.get_by_role("button", name="New chat").click()

    page.wait_for_url("**/web/")
    assert page.locator("#panel").evaluate("element => !element.classList.contains('open')")


@pytest.mark.e2e
def test_composer_attachment_picker_keeps_sources_panel_open(page):
    _open_ready_page(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    _inject_answer_with_sources(page)
    page.locator("[data-answer-ref]").last.click()

    panel = page.locator("#panel")
    page.wait_for_selector('#panel-content [data-ref="1"][data-expanded]')
    with page.expect_file_chooser() as chooser_info:
        page.get_by_role("button", name="Attach files").click()
    chooser_info.value.set_files(
        {
            "name": "notes.pdf",
            "mimeType": "application/pdf",
            "buffer": b"%PDF-1.4 selected attachment",
        }
    )

    assert panel.evaluate("element => element.classList.contains('open')") is True
    assert panel.get_attribute("data-panel-kind") == "sources"


@pytest.mark.e2e
def test_theme_menu_and_selection_keep_sources_panel_open(page):
    _open_ready_page(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    _inject_answer_with_sources(page)
    page.locator("[data-answer-ref]").last.click()
    panel = page.locator("#panel")
    page.wait_for_selector('#panel-content [data-ref="1"][data-expanded]')

    page.get_by_role("button", name="Appearance").click()
    assert panel.evaluate("element => element.classList.contains('open')") is True
    page.locator("#theme-menu [data-theme-value='dark']").click()

    assert page.locator("html").get_attribute("data-color-mode") == "dark"
    assert panel.evaluate("element => element.classList.contains('open')") is True
    assert panel.get_attribute("data-panel-kind") == "sources"


@pytest.mark.e2e
def test_source_download_is_persistent_sibling_and_keyboard_reachable(page):
    _open_ready_page(page)
    _inject_answer_with_sources(page)
    page.locator("[data-answer-ref]").press("Enter")

    header = page.locator('#panel-content [data-ref="1"] [data-source-header]')
    download = header.locator(":scope > a[download]")

    assert download.count() == 1
    assert header.locator(":scope > [data-source-toggle]").count() == 1
    assert download.get_attribute("href").endswith(
        "/web/api/files/raw/doc-report?workspace=default"
    )
    assert download.get_attribute("aria-label") == "Download source"
    assert download.get_attribute("download") == ""
    download.focus()
    assert download.evaluate("element => element === document.activeElement") is True


@pytest.mark.e2e
def test_public_source_link_opens_new_tab_from_source_panel(page):
    timestamp = "2026-08-10T12:00:00Z"
    conversation = {
        "conversation_id": "public-source-history",
        "title": "Public source",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    source_url = "http://www.sgas.ruc.edu.cn/xwgg/yjyxw/f1a3ff59a5894391b7b0db77951c08b4.htm"
    presentation_wire = _source_presentation_wire(source_url=source_url)

    def handle_conversations(route):
        path = urlparse(route.request.url).path
        if path == "/web/api/conversations":
            route.fulfill(json={"items": [conversation], "next_cursor": None})
            return
        if path == "/web/api/conversations/public-source-history/history":
            route.fulfill(
                json={
                    "conversation": conversation,
                    "turns": [
                        {
                            "turn_id": "source-turn",
                            "turn_number": 1,
                            "answer_run_id": "source-run",
                            "submission_id": "source-submission",
                            "status": "succeeded",
                            "cancel_requested": False,
                            "user_text": "Show the source",
                            "assistant_text": "Cited answer.",
                            "user_attachments": [],
                            "presentation": presentation_wire,
                            "usage": {},
                            "evidence": {},
                            "error_kind": None,
                            "error_message": None,
                            "created_at": timestamp,
                        }
                    ],
                }
            )
            return
        route.continue_()

    page.route("**/web/api/conversations**", handle_conversations)
    with page.expect_response(
        lambda response: response.url.endswith("/public-source-history/history") and response.ok
    ) as history_response:
        page.goto("/web/conversations/public-source-history")
    history_response.value.finished()
    page.locator("[data-answer-ref]").press("Enter")

    link = page.get_by_role("link", name="Open source")
    assert link.get_attribute("target") == "_blank"
    assert link.get_attribute("rel") == "noopener noreferrer"
    with page.context.expect_page() as popup_info:
        link.click()
    popup = popup_info.value
    assert popup.url.startswith("http://www.sgas.ruc.edu.cn/")
    popup.close()


@pytest.mark.e2e
def test_escape_closes_source_lightbox_only_and_restores_image_focus(page):
    _open_ready_page(page)
    image_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    _inject_answer_with_sources(page, image_url=image_url)
    page.locator("[data-answer-ref]").press("Enter")
    page.wait_for_selector('#panel-content [data-ref="1"][data-expanded]')

    image = page.get_by_role("button", name="Open page image")
    image.click()
    page.locator("#image-lightbox[aria-hidden='false']").wait_for()
    page.keyboard.press("Escape")

    page.wait_for_function(
        "document.querySelector('#image-lightbox')?.getAttribute('aria-hidden') === 'true'"
    )
    assert page.locator("#panel").get_attribute("aria-hidden") is None
    assert page.locator("#panel").evaluate("element => element.classList.contains('open')") is True
    assert image.evaluate("element => document.activeElement === element") is True
