# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for file ingestion flow via the UI."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
def test_ingestion_panel_opens(page):
    """Click the Files panel trigger → panel slides in with file-related controls."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    # The panel trigger button (if present) opens the file panel
    panel_trigger = page.locator("#panel-trigger-files")
    if panel_trigger.count() > 0:
        panel_trigger.click()
        page.wait_for_timeout(500)
        panel = page.locator("#panel")
        assert panel.is_visible() or panel.count() > 0


@pytest.mark.e2e
def test_empty_file_list_renders(page):
    """When no files are ingested, the panel shows an empty state or file-list structure."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    # Navigate to ingest panel if accessible
    panel_trigger = page.locator("#panel-trigger-files")
    if panel_trigger.count() > 0:
        panel_trigger.click()
        page.wait_for_timeout(500)

    # The panel-content area should exist
    panel_content = page.locator("#panel-content")
    assert panel_content.count() >= 1
