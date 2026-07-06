# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for file ingestion flow via the UI."""

import pytest


@pytest.mark.e2e
def test_ingestion_panel_opens(page):
    """Click the Files panel trigger → panel slides in with file-related controls."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")

    page.wait_for_function("document.querySelector('#panel').classList.contains('open')")
    assert page.locator("#panel-title").text_content() == "FILES"


@pytest.mark.e2e
def test_empty_file_list_renders(page):
    """When no files are ingested, the panel shows an empty state or file-list structure."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)


@pytest.mark.e2e
def test_file_panel_request_state_does_not_show_upload_indicator(page):
    """Non-upload panel requests must not surface the upload-only indicator."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")
    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)

    page.locator("#panel-content").evaluate("el => el.classList.add('htmx-request')")
    assert not page.locator("#upload-spinner").is_visible()

    page.locator("#upload-zone").evaluate("el => el.classList.add('is-uploading')")
    assert page.locator("#upload-spinner").is_visible()


@pytest.mark.e2e
def test_file_panel_workspace_switch_replaces_loading_state(page):
    """Selecting another Files workspace should replace the loading state with file controls."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")
    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)

    page.click(".ingest-target-pill")
    page.locator(".ui-popover--ingest .ui-popover-item", has_text="Research").click()

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert "Loading files..." not in page.locator("#panel-content").text_content()
    assert page.locator(".ingest-target-name").text_content() == "research"
