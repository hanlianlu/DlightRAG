# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for file ingestion flow via the UI."""

import json
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import expect


@pytest.mark.e2e
def test_ingestion_panel_opens(page):
    """Click the Files panel trigger → panel slides in with file-related controls."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")

    page.wait_for_function("document.querySelector('#panel').classList.contains('open')")
    expect(page.get_by_role("heading", name="Files", exact=True)).to_be_visible()
    expect(
        page.get_by_role("button", name="Files in Default; choose file workspace")
    ).to_be_visible()
    assert not page.locator("#files-btn").is_visible()
    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    upload_zone = page.locator("#upload-zone")
    assert upload_zone.get_attribute("role") is None
    assert upload_zone.locator(":scope > [data-upload-file-action]").count() == 1


@pytest.mark.e2e
def test_file_panel_loads_older_keyset_page_through_accessible_control(page):
    def route_files(route):
        query = parse_qs(urlparse(route.request.url).query)
        cursor = query.get("cursor", [None])[0]
        workspace = query.get("workspace", ["default"])[0]
        files = (
            [
                {"file_name": f"File {index:02d}", "file_path": f"/files/{index:02d}"}
                for index in range(50)
            ]
            if cursor is None
            else [
                {"file_name": "File 50", "file_path": "/files/50"},
                {"file_name": "File 51", "file_path": "/files/51"},
            ]
        )
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "workspace": workspace,
                    "files": files,
                    "ingest": {
                        "busy": False,
                        "message": "",
                        "progress_percent": None,
                        "current_batch": None,
                        "total_batches": None,
                        "documents": None,
                        "pending_enqueues": 0,
                    },
                    "next_cursor": "older-page" if cursor is None else None,
                }
            ),
        )

    page.route("**/web/api/files?**", route_files)
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)
    page.click("#files-btn")

    file_list = page.get_by_role("list", name="Processed files")
    expect(file_list.get_by_role("listitem")).to_have_count(50)
    load_older = page.get_by_role("button", name="Load older files")
    expect(load_older).to_be_enabled()
    load_older.click()

    expect(file_list.get_by_role("listitem")).to_have_count(52)
    expect(load_older).to_have_count(0)


@pytest.mark.e2e
def test_file_panel_request_state_does_not_show_upload_indicator(page):
    """Only an upload in progress may surface the upload indicator."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#files-btn")
    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)

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

    page.click("[data-ingest-pill]")
    page.locator(".dl-popover--ingest .dl-popover-item", has_text="Research").click()

    page.wait_for_function(
        "!document.querySelector('#panel-content')?.textContent.includes('Loading files...')"
    )
    assert "Loading files..." not in page.locator("#panel-content").text_content()
    assert page.locator("[data-ingest-name]").text_content() == "Research"


@pytest.mark.e2e
def test_file_panel_uses_last_selected_topbar_workspace(page):
    """Files defaults to the last explicitly selected topbar workspace."""
    page.goto("/web/")
    page.wait_for_selector(".app", timeout=10000)

    page.click("#workspace-selector")
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Default").click()
    page.click("#files-btn")

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert page.locator("[data-ingest-name]").text_content() == "Research"
