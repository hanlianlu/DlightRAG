# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the composer Answer Mode menu."""

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


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


def test_mode_menu_selects_and_remembers_research(page: Page) -> None:
    _open_ready_page(page)
    trigger = page.locator("#composer-mode")
    menu = page.locator("#composer-mode-menu")
    assert trigger.inner_text() == "Auto"
    assert menu.get_attribute("hidden") is not None

    trigger.click()
    assert menu.get_attribute("hidden") is None
    menu.get_by_role("menuitemradio", name="Research").click()

    assert trigger.inner_text() == "Research"
    assert menu.get_attribute("hidden") is not None
    assert trigger.get_attribute("aria-expanded") == "false"
    chosen = page.evaluate("localStorage.getItem('dlightrag.answerMode')")
    assert chosen == "research"

    # A reload restores the stored choice.
    _open_ready_page(page)
    assert page.locator("#composer-mode").inner_text() == "Research"
