# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for workspace CRUD: create, open, delete workspaces."""

from __future__ import annotations

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
