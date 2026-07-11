# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for workspace CRUD: create, open, delete workspaces."""

import pytest


@pytest.mark.e2e
def test_workspace_popover_opens(page):
    """Click the workspace selector → popover appears with workspace items."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    selector = page.locator("#workspace-selector")
    selector.click()

    # Popover should be visible with at least one workspace item
    page.wait_for_selector(".ui-popover--workspace", timeout=5000)
    popover = page.locator(".ui-popover--workspace")
    assert popover.is_visible()

    # "Default" workspace should be listed
    items = popover.locator(".ui-popover-item")
    assert items.count() >= 1


@pytest.mark.e2e
def test_workspace_popover_closes(page):
    """Click outside the popover or press Escape → popover closes."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.wait_for_selector(".ui-popover--workspace", timeout=5000)
    assert page.locator(".ui-popover--workspace").is_visible()

    # Click outside — the popover should go away
    page.locator(".app").click(position={"x": 10, "y": 10})
    page.wait_for_timeout(300)
    assert page.locator(".ui-popover--workspace").count() == 0


@pytest.mark.e2e
def test_workspace_create_input_visible(page):
    """Verify the new-workspace input row exists inside the popover."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.wait_for_selector(".ui-popover--workspace", timeout=5000)

    # The create-row with input and button should be present
    create_row = page.locator(".ui-popover-create")
    assert create_row.is_visible()
    assert create_row.locator(".ui-popover-input").is_visible()
    assert create_row.locator(".ui-popover-create-btn").is_visible()


@pytest.mark.e2e
def test_workspace_selector_labels_all_when_scope_covers_every_workspace(page):
    """Selecting every workspace individually promotes the topbar scope to All."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".ui-popover--workspace .ui-popover-item", has_text="Research").click()

    assert page.locator("#workspace-label").text_content() == "All Workspaces (2)"
    assert (
        page.locator(
            ".ui-popover--workspace .ui-popover-item", has_text="All Workspaces"
        ).get_attribute("aria-selected")
        == "true"
    )


@pytest.mark.e2e
def test_workspace_selector_all_sets_default_primary(page):
    """Selecting All is explicit and resets single-workspace surfaces to default."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".ui-popover--workspace .ui-popover-item", has_text="Research").click()
    page.locator(".ui-popover--workspace .ui-popover-item", has_text="All Workspaces").click()
    page.locator("#files-btn").click()

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert page.locator("#workspace-label").text_content() == "All Workspaces (2)"
    assert page.locator(".ingest-target-name").text_content() == "default"


@pytest.mark.e2e
def test_workspace_selector_auto_all_keeps_last_explicit_primary(page):
    """An all-workspace query scope still keeps the last explicit workspace primary."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".ui-popover--workspace .ui-popover-item", has_text="Research").click()
    page.locator("#files-btn").click()

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert page.locator("#workspace-label").text_content() == "All Workspaces (2)"
    assert page.locator(".ingest-target-name").text_content() == "research"
