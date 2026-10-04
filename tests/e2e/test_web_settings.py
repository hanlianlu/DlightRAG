# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the Settings dialog: its navigation, its phone layout, and Agent Accounts."""

import pytest
from playwright.sync_api import Locator, Page, expect

from tests.e2e.test_web_conversations import (
    AgentAccountsRouteState,
    _agent_account,
    _assert_surface_owns_viewport_layer,
    _assert_touch_target,
    _install_agent_accounts_routes,
    _install_conversation_routes,
    _open_settings,
    _open_settings_page,
)

PAGES = ["Connections", "Agent Accounts", "Profile Memory", "Conversation Sessions", "Language"]


def _three_accounts() -> AgentAccountsRouteState:
    return AgentAccountsRouteState(
        accounts=[
            _agent_account(
                "discourse.org",
                email="agent@discourse.org",
                username="dlr20261004k7q2",
                registered_days_ago=2,
                last_used_days_ago=0,
            ),
            _agent_account(
                "huggingface.co",
                email="agent-2c9e@example.test",
                username="dlight-agent-2c9",
                registered_days_ago=9,
                last_used_days_ago=3,
            ),
            _agent_account(
                "ycombinator.com",
                username="dlight_reader",
                registered_days_ago=30,
            ),
        ]
    )


def _agent_accounts_page(page: Page, state: AgentAccountsRouteState) -> Locator:
    """Open Settings on Agent Accounts, with the routes answering from ``state``."""
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, state)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    return _open_settings_page(page, "Agent Accounts")


def _names(rows: Locator) -> list[str]:
    """The accessible names of navigation rows, which their aria-labelledby points at."""
    return rows.evaluate_all(
        "rows => rows.map(row => "
        "document.getElementById(row.getAttribute('aria-labelledby')).textContent.trim())"
    )


@pytest.mark.e2e
def test_settings_is_a_centered_dialog_with_a_navigation_beside_its_page(page: Page) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings(page)
    box = settings.bounding_box()
    assert box is not None
    assert box["width"] == pytest.approx(880, abs=1)
    assert box["height"] == pytest.approx(640, abs=1)
    assert box["x"] + box["width"] / 2 == pytest.approx(720, abs=1)
    assert box["y"] + box["height"] / 2 == pytest.approx(450, abs=1)
    _assert_surface_owns_viewport_layer(page, ".settings-dialog")
    assert settings.evaluate("element => getComputedStyle(element).borderRadius") == "22px"
    assert settings.evaluate("element => getComputedStyle(element).borderTopWidth") == "1px"
    assert settings.evaluate("element => element.matches(':modal')") is True

    # Both themes paint it from the surface token.
    for color_mode in ("light", "dark"):
        page.locator("html").evaluate(
            "(element, mode) => { element.dataset.colorMode = mode; }", color_mode
        )
        assert settings.evaluate(
            """element => {
                const probe = document.createElement('div');
                probe.style.backgroundColor = 'var(--color-bg-surface)';
                document.body.append(probe);
                const expected = getComputedStyle(probe).backgroundColor;
                probe.remove();
                return getComputedStyle(element).backgroundColor === expected;
            }"""
        )
    # Nothing inside the dialog casts a shadow: the dialog is the one overlapping surface.
    assert (
        settings.evaluate(
            """element => [...element.querySelectorAll('*')].filter(node =>
            node.getClientRects().length && getComputedStyle(node).boxShadow !== 'none').length"""
        )
        == 0
    )

    navigation = settings.get_by_role("navigation", name="Settings")
    assert _names(navigation.get_by_role("button")) == PAGES
    assert _names(navigation.get_by_role("group")) == ["Agent", "Data", "General"]
    expect(navigation.get_by_role("button", name="Connections")).to_have_attribute(
        "aria-current", "page"
    )
    expect(navigation.locator("[aria-current='page']")).to_have_count(1)
    expect(settings.get_by_role("region", name="Connections")).to_be_visible()
    expect(navigation.get_by_role("button", name="Agent Accounts")).to_contain_text("3")


@pytest.mark.e2e
def test_settings_navigation_moves_with_the_keyboard_and_gives_focus_back(page: Page) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings(page)
    navigation = settings.get_by_role("navigation", name="Settings")
    row = navigation.get_by_role("button")
    expect(row.nth(0)).to_be_focused()

    page.keyboard.press("ArrowDown")
    expect(row.nth(1)).to_be_focused()
    page.keyboard.press("End")
    expect(row.nth(4)).to_be_focused()
    page.keyboard.press("ArrowDown")
    expect(row.nth(0)).to_be_focused()
    page.keyboard.press("ArrowUp")
    expect(row.nth(4)).to_be_focused()
    page.keyboard.press("Home")
    expect(row.nth(0)).to_be_focused()
    # Moving focus opens nothing; Enter opens the page the focus is on.
    page.keyboard.press("ArrowDown")
    expect(row.nth(0)).to_have_attribute("aria-current", "page")
    page.keyboard.press("Enter")
    expect(row.nth(1)).to_have_attribute("aria-current", "page")
    expect(settings.get_by_role("region", name="Agent Accounts")).to_be_visible()
    expect(row.nth(0)).not_to_have_attribute("aria-current", "page")
    expect(row.nth(1)).to_be_focused()

    page.keyboard.press("Escape")
    settings.wait_for(state="hidden")
    expect(page.locator("#settings-btn")).to_be_focused()


@pytest.mark.e2e
def test_settings_is_two_levels_on_a_phone_with_touch_sized_controls(page: Page) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings(page)
    box = settings.bounding_box()
    assert box is not None
    assert (box["x"], box["y"], box["width"], box["height"]) == (0, 0, 390, 844)
    assert settings.evaluate("element => getComputedStyle(element).borderRadius") == "0px"
    _assert_surface_owns_viewport_layer(page, ".settings-dialog")

    # The first level is the section list: every row names its page and says what it holds.
    navigation = settings.get_by_role("navigation", name="Settings")
    rows = navigation.get_by_role("button")
    assert _names(rows) == PAGES
    expect(settings.get_by_role("region")).to_have_count(0)
    expect(rows.nth(0)).to_be_focused()
    expect(navigation.locator("[aria-current='page']")).to_have_count(0)
    expect(rows.nth(1)).to_contain_text("3 websites")
    expect(rows.nth(2)).to_contain_text("Off")
    for control in [*rows.all(), settings.get_by_role("button", name="Close settings")]:
        _assert_touch_target(control)

    rows.nth(1).click()
    region = settings.get_by_role("region", name="Agent Accounts")
    expect(region).to_be_visible()
    expect(navigation).to_be_hidden()
    back = settings.get_by_role("button", name="Back")
    expect(back).to_be_visible()
    for control in [
        back,
        settings.get_by_role("button", name="Close settings"),
        *region.get_by_role("button", name="Remove the account for").all(),
    ]:
        _assert_touch_target(control)
    # The page is a list of three-line rows, not a table.
    expect(region.get_by_role("table")).to_have_count(0)
    expect(region.get_by_role("listitem").nth(0)).to_contain_text("Signed in today")
    expect(region.get_by_role("listitem").nth(2)).to_contain_text("not signed in since")
    # The switch itself is small; the card it lives in is the target a finger finds.
    _assert_touch_target(region.locator("label").filter(has=page.get_by_role("switch")))

    back.click()
    expect(navigation).to_be_visible()
    expect(settings.get_by_role("region")).to_have_count(0)
    expect(rows.nth(1)).to_be_focused()

    page.keyboard.press("Escape")
    settings.wait_for(state="hidden")
    expect(page.locator("#settings-btn")).to_be_visible()


@pytest.mark.e2e
def test_agent_accounts_page_lists_what_the_agent_registered(page: Page) -> None:
    state = _three_accounts()
    settings = _agent_accounts_page(page, state)

    table = settings.get_by_role("table")
    headers = table.get_by_role("columnheader")
    assert [header.inner_text().strip() for header in headers.all()] == [
        "Website",
        "Sign-in",
        "Registered",
        "Last sign-in",
        "Remove",
    ]
    rows = table.locator("tbody tr")
    expect(rows).to_have_count(3)
    first, second, third = rows.nth(0), rows.nth(1), rows.nth(2)
    expect(first.get_by_role("rowheader")).to_contain_text("discourse.org")
    expect(first.locator("td").nth(0)).to_contain_text("agent@discourse.org")
    expect(first.locator("td").nth(0)).to_contain_text("dlr20261004k7q2")
    expect(first.locator("td").nth(2)).to_have_text("Today")
    expect(second.locator("td").nth(2)).to_have_text("3 days ago")
    expect(third.locator("td").nth(0)).to_have_text("dlight_reader")
    expect(third.locator("td").nth(2)).to_have_text("—Never")
    # The monogram is a letter, never an image: the page asks no website for anything.
    expect(first.locator("[aria-hidden='true']").first).to_have_text("D")
    expect(settings.locator("dl-settings-agent-accounts img")).to_have_count(0)
    # Nothing on the page takes input, so a secret has nowhere to go.
    expect(
        settings.locator("dl-settings-agent-accounts input, dl-settings-agent-accounts textarea")
    ).to_have_count(0)
    assert {method for method, _, _ in state.requests} == {"GET"}


@pytest.mark.e2e
def test_agent_accounts_switch_turns_with_one_request_and_tells_what_it_does(page: Page) -> None:
    state = _three_accounts()
    settings = _agent_accounts_page(page, state)

    switch = settings.get_by_role("switch", name="Allow new sign-ups")
    expect(switch).to_have_attribute("aria-checked", "true")
    expect(
        settings.get_by_text("The agent may register on a website when it needs to")
    ).to_be_visible()

    switch.click()
    expect(switch).to_have_attribute("aria-checked", "false")
    expect(
        settings.get_by_text("Off: the agent only signs in with the accounts below")
    ).to_be_visible()
    assert ("PUT", "/web/api/agent-accounts/settings", {"registration_enabled": False}) in (
        state.requests
    )
    # The whole card is the switch's label, so a tap on its words turns it back on.
    settings.get_by_text("Off: the agent only signs in with the accounts below").click()
    expect(switch).to_have_attribute("aria-checked", "true")


@pytest.mark.e2e
def test_agent_accounts_remove_asks_first_then_deletes_one_account(page: Page) -> None:
    state = _three_accounts()
    settings = _agent_accounts_page(page, state)
    remove = settings.get_by_role("button", name="Remove the account for huggingface.co")

    remove.click()
    dialog = page.get_by_role("dialog", name="Remove the account for huggingface.co?")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("The account itself stays on the website.")
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_be_hidden()
    expect(remove).to_be_focused()
    assert not [request for request in state.requests if request[0] == "DELETE"]

    remove.click()
    dialog.get_by_role("button", name="Remove account").click()
    expect(settings.get_by_role("table").locator("tbody tr")).to_have_count(2)
    assert ("DELETE", "/web/api/agent-accounts/huggingface.co", None) in state.requests
    # The row that was there is gone: focus is on the one that took its place.
    expect(
        settings.get_by_role("button", name="Remove the account for ycombinator.com")
    ).to_be_focused()
    expect(settings.get_by_role("button", name="Agent Accounts")).to_contain_text("2")


@pytest.mark.e2e
def test_agent_accounts_page_states(page: Page) -> None:
    # Nothing registered yet.
    state = AgentAccountsRouteState()
    settings = _agent_accounts_page(page, state)
    expect(settings.get_by_text("No accounts yet")).to_be_visible()
    expect(settings.get_by_role("table")).to_have_count(0)
    page.keyboard.press("Escape")
    settings.wait_for(state="hidden")

    # The read fails, then a retry reads it.
    state.accounts = _three_accounts().accounts
    state.read_status = 503
    settings = _open_settings_page(page, "Agent Accounts")
    expect(settings.get_by_role("alert")).to_have_text("Could not load agent accounts.")
    state.read_status = 200
    settings.get_by_role("button", name="Retry").click()
    expect(settings.get_by_role("table").locator("tbody tr")).to_have_count(3)
    page.keyboard.press("Escape")
    settings.wait_for(state="hidden")

    # The deployment does not allow sign-ups: the switch reads off and cannot be turned.
    state.registration = {"allowed": False, "enabled": True}
    settings = _open_settings_page(page, "Agent Accounts")
    switch = settings.get_by_role("switch", name="Allow new sign-ups")
    expect(switch).to_be_disabled()
    expect(switch).to_have_attribute("aria-checked", "false")
    expect(
        settings.get_by_text(
            "This deployment does not allow sign-ups, so the agent only signs in with the accounts below."
        )
    ).to_be_visible()
    page.keyboard.press("Escape")
    settings.wait_for(state="hidden")

    # The deployment has not enabled Agent Accounts at all; stored ones can still be removed.
    state.available = False
    settings = _open_settings_page(page, "Agent Accounts")
    expect(settings.get_by_role("switch", name="Allow new sign-ups")).to_be_disabled()
    expect(
        settings.get_by_text(
            "This deployment has not enabled Agent Accounts. Stored accounts can still be removed here."
        )
    ).to_be_visible()
    expect(
        settings.get_by_role("button", name="Remove the account for discourse.org")
    ).to_be_enabled()


@pytest.mark.e2e
def test_settings_speaks_chinese_and_remembers_the_language(page: Page) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    settings = _open_settings_page(page, "Language")

    settings.get_by_role("radio", name="中文").check()
    # The dialog is named in the new language the moment it loads, so it is found by that name.
    settings = page.get_by_role("dialog", name="设置")
    expect(settings.get_by_role("navigation", name="设置")).to_be_visible()
    expect(settings.get_by_role("button", name="代理账号")).to_be_visible()
    settings.get_by_role("button", name="代理账号").click()
    expect(settings.get_by_role("region", name="代理账号")).to_be_visible()
    expect(settings.get_by_role("switch", name="允许注册新账号")).to_be_visible()
    assert page.evaluate("localStorage.getItem('dlightrag-lang')") == "zh"
    page.reload()
    page.locator("[aria-current='page']").wait_for()
    expect(page.locator("html")).to_have_attribute("lang", "zh")
