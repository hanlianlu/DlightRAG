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


def _source_presentation(*, source_url: str | None = None, image_url: str | None = None) -> dict:
    return {
        "answer_text": "DlightRAG cited answer [1-1].",
        "parts": [
            {
                "type": "markdown",
                "text": "DlightRAG cited answer [1-1].",
                "html": (
                    "<p>DlightRAG cited answer "
                    '<cite class="citation-badge" data-ref="1" data-chunk="1" '
                    'role="button" tabindex="0" aria-label="Source 1, chunk 1">1-1</cite>.</p>'
                ),
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


def _inject_answer_with_sources(page) -> None:
    page.wait_for_selector("[aria-current='page']", timeout=10000)
    page.locator(".composer-input").fill("show cited source")
    page.click(".composer-send")
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    page.locator("dl-answer-presentation").last.evaluate(
        """(element, presentation) => {
          element.presentation = presentation;
          return element.updateComplete;
        }""",
        _source_presentation(),
    )


def _inject_static_source_answer(page) -> None:
    page.evaluate(
        r"""(presentation) => {
          const aiMessageClass = Array.from(document.styleSheets)
            .flatMap((sheet) => Array.from(sheet.cssRules))
            .flatMap((rule) => Array.from(rule.selectorText?.matchAll(/\.([\w-]*aiMessage[\w-]*)/g) ?? []))
            .map((match) => match[1])
            .find((className) => !className.includes('Content') && !className.includes('Header'));
          const contentClass = Array.from(document.styleSheets)
            .flatMap((sheet) => Array.from(sheet.cssRules))
            .flatMap((rule) => Array.from(rule.selectorText?.matchAll(/\.([\w-]*aiMessageContent[\w-]*)/g) ?? []))
            .map((match) => match[1])[0];
          if (!aiMessageClass || !contentClass) throw new Error('AI message classes not found');
          const answer = document.createElement('div');
          answer.className = aiMessageClass;
          const content = document.createElement('div');
          content.className = contentClass;
          const element = document.createElement('dl-answer-presentation');
          element.presentation = presentation;
          content.appendChild(element);
          answer.appendChild(content);
          document.querySelector('#chat-messages')?.appendChild(answer);
        }""",
        _source_presentation(),
    )


@pytest.mark.e2e
def test_reference_item_keyboard_opens_source_panel(page):
    """Keyboard activation on a reference item opens its expanded source."""
    _open_ready_page(page)
    _inject_answer_with_sources(page)

    page.locator(".answer-ref-item").press("Enter")

    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]', timeout=10000)
    assert page.locator("#panel-title").text_content() == "Sources"
    assert page.locator("#panel-content .source-chunk-content").text_content() == "Evidence text"
    assert "Wrong decoy" not in page.locator("#panel-content").text_content()


@pytest.mark.e2e
def test_conversation_route_change_closes_sources_panel(page):
    _open_ready_page(page)
    _inject_static_source_answer(page)
    page.locator(".answer-ref-item").last.click()
    page.wait_for_selector('#panel.open[data-panel-kind="sources"]')

    page.get_by_role("button", name="New chat").click()

    page.wait_for_url("**/web/")
    assert page.locator("#panel").evaluate("element => !element.classList.contains('open')")


@pytest.mark.e2e
def test_composer_attachment_picker_keeps_sources_panel_open(page):
    _open_ready_page(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    _inject_static_source_answer(page)
    page.locator(".answer-ref-item").last.click()

    panel = page.locator("#panel")
    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]')
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
    _inject_static_source_answer(page)
    page.locator(".answer-ref-item").last.click()
    panel = page.locator("#panel")
    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]')

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
    page.locator(".answer-ref-item").press("Enter")

    header = page.locator('#panel-content .source-doc[data-ref="1"] .source-doc-header')
    download = header.locator(":scope > .source-action-icon[download]")

    assert download.count() == 1
    assert header.locator(":scope > .source-doc-toggle").count() == 1
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
    presentation = _source_presentation(source_url=source_url)

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
                            "presentation": presentation,
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
    page.locator(".answer-ref-item").press("Enter")

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
    page.locator(".composer-input").fill("show source image")
    page.click(".composer-send")
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    page.locator("dl-answer-presentation").last.evaluate(
        """(element, presentation) => {
          element.presentation = presentation;
          return element.updateComplete;
        }""",
        _source_presentation(image_url=image_url),
    )
    page.locator(".answer-ref-item").press("Enter")
    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]')

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
