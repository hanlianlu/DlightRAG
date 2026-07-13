# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for source panel: citation click, expand/collapse."""

import pytest


def _open_ready_page(page) -> None:
    with page.expect_response(
        lambda response: (
            response.request.method == "GET" and response.url.endswith("/history") and response.ok
        ),
        timeout=10000,
    ) as history_response:
        page.goto("/web/")
    history_response.value.finished()
    page.wait_for_selector(".composer-input", timeout=10000)


def _inject_answer_with_sources(page) -> None:
    page.wait_for_selector("[aria-current='page']", timeout=10000)
    page.wait_for_function(
        "!document.querySelector('#chat-messages [role=\"status\"]')",
        timeout=10000,
    )
    conversation_id = page.locator("[aria-current='page']").get_attribute("data-conversation-id")
    assert conversation_id
    ai_selector = '[class*="aiMessage"]:not([class*="Content"]):not([class*="Header"])'
    initial_ai_count = page.locator(ai_selector).count()
    page.locator(".composer-input").fill("show cited source")
    with page.expect_response(
        lambda response: (
            response.request.method == "GET"
            and response.url.endswith(f"/web/conversations/{conversation_id}/history")
            and response.ok
        ),
        timeout=10000,
    ) as history_response:
        page.click(".composer-send")
        page.wait_for_function(
            "([selector, count]) => document.querySelectorAll(selector).length > count",
            arg=[ai_selector, initial_ai_count],
            timeout=10000,
        )
    history_response.value.finished()
    page.evaluate(
        """
        () => new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        })
        """
    )
    page.wait_for_function(
        """
        () => {
          const messages = document.querySelectorAll('[class*="aiMessageContent"]');
          const latest = messages[messages.length - 1];
          return latest?.textContent.includes('DlightRAG');
        }
        """,
        timeout=10000,
    )
    page.evaluate(
        """
        () => {
          const messages = document.querySelectorAll(
            '[class*="aiMessage"]:not([class*="Content"]):not([class*="Header"])'
          );
          const ai = messages[messages.length - 1];
          const content = ai?.querySelector('[class*="aiMessageContent"]');
          if (!ai || !content) throw new Error('AI message not rendered');
          content.insertAdjacentHTML('beforeend', `
            <div class="source-data hidden">
              <div class="source-doc" data-ref="1">
                <div class="source-doc-chunks">
                  <div class="source-chunk-content">Wrong decoy</div>
                </div>
              </div>
            </div>
            <div class="answer-references">
              <div class="answer-references-title">References</div>
              <div class="answer-ref-item" data-action="open-ref-source" data-ref="1" role="button" tabindex="0">
                <span class="answer-ref-id">1</span>
                <span class="answer-ref-title">report.pdf</span>
              </div>
            </div>
          `);
          let sourceData = Array.from(ai.children).find((child) => child.classList.contains('source-data'));
          if (!sourceData) {
            sourceData = document.createElement('div');
            sourceData.className = 'source-data hidden';
            ai.appendChild(sourceData);
          }
          sourceData.innerHTML = `
              <div class="source-doc" data-ref="1">
                <div class="source-doc-header">
                  <button class="source-doc-toggle" type="button" data-action="toggle-doc">
                    <span class="collapse-icon">▶</span>
                    <span class="source-doc-title">report.pdf</span>
                    <span class="source-doc-badge">1</span>
                    <span class="source-doc-count">1</span>
                  </button>
                  <a href="/web/files/raw/doc-report?workspace=default"
                     class="source-dl-icon"
                     title="Download source"
                     aria-label="Download source"
                     download>
                    <svg class="source-dl-icon-svg" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
                         stroke-linejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                  </a>
                </div>
                <div class="source-doc-chunks" hidden>
                  <div class="source-chunk" data-ref="1" data-chunk="1">
                    <div class="source-chunk-content">Evidence text</div>
                  </div>
                </div>
              </div>
              <button class="show-all-btn" type="button" data-action="show-all-sources" hidden>Show all sources</button>
          `;
        }
        """
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
def test_source_download_is_persistent_sibling_and_keyboard_reachable(page):
    _open_ready_page(page)
    _inject_answer_with_sources(page)
    page.locator(".answer-ref-item").press("Enter")

    header = page.locator('#panel-content .source-doc[data-ref="1"] .source-doc-header')
    download = header.locator(":scope > .source-dl-icon")

    assert download.count() == 1
    assert header.locator(":scope > .source-doc-toggle").count() == 1
    assert download.get_attribute("href").endswith("/web/files/raw/doc-report?workspace=default")
    assert download.get_attribute("aria-label") == "Download source"
    assert download.get_attribute("download") == ""
    download.focus()
    assert download.evaluate("element => element === document.activeElement") is True


@pytest.mark.e2e
def test_escape_closes_source_lightbox_only_and_restores_image_focus(page):
    _open_ready_page(page)
    _inject_answer_with_sources(page)
    page.locator(".answer-ref-item").press("Enter")
    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]')
    page.evaluate(
        """
        () => {
          const button = document.createElement('button');
          button.id = 'stacked-source-image';
          button.type = 'button';
          button.setAttribute('aria-label', 'Open source image');
          button.setAttribute('data-action', 'open-lightbox');
          button.setAttribute(
            'data-src',
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
          );
          document.querySelector('#panel-content')?.appendChild(button);
        }
        """
    )

    image = page.get_by_role("button", name="Open source image")
    image.click()
    page.locator("#image-lightbox[aria-hidden='false']").wait_for()
    page.keyboard.press("Escape")

    page.wait_for_function(
        "document.querySelector('#image-lightbox')?.getAttribute('aria-hidden') === 'true'"
    )
    assert page.locator("#panel").get_attribute("aria-hidden") is None
    assert page.locator("#panel").evaluate("element => element.classList.contains('open')") is True
    assert image.evaluate("element => document.activeElement === element") is True
