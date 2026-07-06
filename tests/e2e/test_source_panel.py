# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for source panel: citation click, expand/collapse."""

import pytest


def _inject_answer_with_sources(page) -> None:
    page.locator(".composer-input").fill("show cited source")
    page.click(".composer-send")
    page.wait_for_function(
        """
        () => {
          const messages = document.querySelectorAll('[class*="aiMessageContent"]');
          return Array.from(messages).some((message) => message.textContent.includes('DlightRAG'));
        }
        """,
        timeout=10000,
    )
    page.evaluate(
        """
        () => {
          const ai = document.querySelector('[class*="aiMessage"]:not([class*="Content"]):not([class*="Header"])');
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
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    _inject_answer_with_sources(page)

    page.locator(".answer-ref-item").press("Enter")

    page.wait_for_selector('#panel-content .source-doc.expanded[data-ref="1"]', timeout=10000)
    assert page.locator("#panel-title").text_content() == "SOURCES"
    assert page.locator("#panel-content .source-chunk-content").text_content() == "Evidence text"
    assert "Wrong decoy" not in page.locator("#panel-content").text_content()
