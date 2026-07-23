# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for web appearance menu and runtime theme behavior."""

import pytest
from playwright.sync_api import Browser, Page, expect


def _open_theme_menu(page: Page) -> None:
    trigger = page.locator("#theme-trigger")
    trigger.click()
    expect(trigger).to_have_attribute("aria-expanded", "true")
    expect(page.locator("#theme-menu")).to_be_visible()


def _choose_theme(page: Page, value: str) -> None:
    _open_theme_menu(page)
    page.locator(f"#theme-menu [data-theme-value='{value}']").click()


def _wait_ui_ready(page: Page) -> None:
    expect(page.locator("#theme-trigger")).to_be_visible()
    expect(page.locator("#workspace-label")).not_to_have_text("")


@pytest.mark.e2e
def test_system_light_initial_then_select_dark_and_reload_persists(page: Page) -> None:
    page.emulate_media(color_scheme="light")
    page.goto("/web/")
    _wait_ui_ready(page)

    root = page.locator("html")
    expect(root).to_have_attribute("data-theme", "system")
    expect(root).to_have_attribute("data-color-mode", "light")

    _choose_theme(page, "dark")
    expect(root).to_have_attribute("data-theme", "dark")
    expect(root).to_have_attribute("data-color-mode", "dark")
    expect(page.locator("#theme-menu [data-theme-value='dark']")).to_have_attribute(
        "aria-checked", "true"
    )

    page.reload()
    expect(root).to_have_attribute("data-theme", "dark")
    expect(root).to_have_attribute("data-color-mode", "dark")


@pytest.mark.e2e
def test_system_mode_media_change_updates_effective_only(page: Page) -> None:
    page.emulate_media(color_scheme="dark")
    page.goto("/web/")
    _wait_ui_ready(page)

    root = page.locator("html")
    expect(root).to_have_attribute("data-theme", "system")
    expect(root).to_have_attribute("data-color-mode", "dark")

    page.emulate_media(color_scheme="light")
    expect(root).to_have_attribute("data-theme", "system")
    expect(root).to_have_attribute("data-color-mode", "light")


@pytest.mark.e2e
def test_theme_menu_keyboard_navigation_and_escape_focus_restore(page: Page) -> None:
    page.goto("/web/")
    _wait_ui_ready(page)

    trigger = page.locator("#theme-trigger")
    trigger.focus()

    page.keyboard.press("Enter")
    menu = page.locator("#theme-menu")
    expect(menu).to_be_visible()
    checked = menu.locator("[role='menuitemradio'][aria-checked='true']")
    expect(checked).to_be_focused()
    expect(checked).to_have_attribute("data-theme-value", "system")

    page.keyboard.press("Escape")
    expect(trigger).to_be_focused()
    expect(trigger).to_have_attribute("aria-expanded", "false")
    expect(menu).to_be_hidden()

    trigger.focus()
    page.keyboard.press("ArrowDown")

    expect(menu).to_be_visible()
    expect(checked).to_be_focused()

    page.keyboard.press("End")
    dark_item = menu.locator("[data-theme-value='dark']")
    expect(dark_item).to_be_focused()

    page.keyboard.press("Escape")
    expect(trigger).to_be_focused()
    expect(trigger).to_have_attribute("aria-expanded", "false")
    expect(menu).to_be_hidden()


@pytest.mark.e2e
def test_theme_selection_works_when_local_storage_unavailable(
    browser: Browser,
    e2e_base_url: str,
) -> None:
    context = browser.new_context(base_url=e2e_base_url)
    context.add_init_script(
        """
        Object.defineProperty(window, 'localStorage', {
          get() { throw new Error('blocked storage'); }
        });
        """
    )
    page = context.new_page()
    page.goto("/web/")
    _wait_ui_ready(page)

    _choose_theme(page, "light")
    root = page.locator("html")
    expect(root).to_have_attribute("data-theme", "light")
    expect(root).to_have_attribute("data-color-mode", "light")

    context.close()


@pytest.mark.e2e
def test_theme_storage_event_syncs_across_tabs(browser: Browser, e2e_base_url: str) -> None:
    context = browser.new_context(base_url=e2e_base_url)
    first = context.new_page()
    second = context.new_page()

    first.goto("/web/")
    second.goto("/web/")
    _wait_ui_ready(first)
    _wait_ui_ready(second)

    _choose_theme(first, "light")
    second_root = second.locator("html")
    expect(second_root).to_have_attribute("data-theme", "light")
    expect(second_root).to_have_attribute("data-color-mode", "light")

    context.close()


@pytest.mark.e2e
def test_theme_attrs_stable_when_opening_files_panel(page: Page) -> None:
    page.goto("/web/")
    _wait_ui_ready(page)
    _choose_theme(page, "light")

    root = page.locator("html")
    expect(root).to_have_attribute("data-theme", "light")
    expect(root).to_have_attribute("data-color-mode", "light")

    page.click("#files-btn")
    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    expect(root).to_have_attribute("data-theme", "light")
    expect(root).to_have_attribute("data-color-mode", "light")
