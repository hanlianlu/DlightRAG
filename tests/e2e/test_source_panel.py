# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for source panel: citation click, expand/collapse."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_source_panel_available(page):
    """Verify the source panel element exists in the DOM after page load."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    # The panel (which doubles as source panel) should exist
    panel = page.locator("#panel")
    assert panel.count() >= 1


@pytest.mark.e2e
def test_panelToggles_on_repeated_clicks(page):
    """Click the panel trigger multiple times — it shouldn't crash the UI."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")
    page.wait_for_function("document.querySelector('#panel').classList.contains('open')")
    page.click("#files-btn")

    assert page.locator(".app").is_visible()
    assert page.locator("#panel").evaluate("el => el.classList.contains('open')")
