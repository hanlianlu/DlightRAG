# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the Settings dialog: its navigation, its phone layout, and Agent Accounts."""

from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Locator, Page, Route, expect

from tests.e2e.test_web_conversations import (
    AgentAccountsRouteState,
    _agent_account,
    _assert_surface_owns_viewport_layer,
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


def _install_memory_routes(
    page: Page, bodies: list[str], *, next_cursor: str | None = None
) -> None:
    """Profile Memory on, holding one memory per body, with a next page when there is a cursor."""

    def memory(route: Route) -> None:
        if urlparse(route.request.url).path == "/web/api/memory/settings":
            route.fulfill(json={"enabled": True, "active_count": len(bodies)})
            return
        route.fulfill(
            json={
                "memories": [
                    {"memory_id": f"memory-{index}", "kind": "preference", "body": body}
                    for index, body in enumerate(bodies)
                ],
                "next_cursor": next_cursor,
            }
        )

    page.route("**/web/api/memory**", memory)


def _install_busy_settings_routes(page: Page) -> None:
    """Connections with presets, and Profile Memory on with a next page, so every page has controls."""

    def connections(route: Route) -> None:
        def connection(connection_id: str, label: str, authentication: str, enabled: bool) -> Any:
            return {
                "connection_id": connection_id,
                "label": label,
                "endpoint": f"https://{connection_id}.example/mcp",
                "enabled": enabled,
                "activation_epoch": 1,
                "generation": 1,
                "authentication": authentication,
                "authorization_status": None,
                "status": "ready" if enabled else "disabled",
            }

        route.fulfill(
            json={
                "revision": "1",
                "presets": [
                    {
                        "preset_id": "notion",
                        "label": "Notion",
                        "endpoint": "https://mcp.notion.com/mcp",
                        "default_authentication": "oauth",
                    },
                    {
                        "preset_id": "hugging-face",
                        "label": "Hugging Face",
                        "endpoint": "https://huggingface.co/mcp",
                        "default_authentication": "none",
                    },
                ],
                "connections": [
                    connection("notion", "Notion", "oauth", True),
                    connection("wiki", "Team wiki", "none", False),
                ],
            }
        )

    page.route("**/web/api/connections/mcp", connections)
    page.route(
        "**/web/api/connections/mcp/*/oauth",
        lambda route: route.fulfill(json={"authorization_url": "https://auth.example/authorize"}),
    )
    _install_memory_routes(
        page,
        ["Use concise answers", "Works on the ingestion service"],
        next_cursor="more",
    )


# What a finger meets inside the dialog, and how large the area is that answers to it. The browser is
# asked what lies under each point around a control, so a target that reaches past its drawn box
# counts, and one that is covered or clipped does not. A radio or checkbox is turned by tapping its
# row, so the row is its target. Disabled controls take no tap and are left out.
_SMALL_TARGETS = """root => {
    const controls = root.querySelectorAll(
        "button, a[href], input:not([type='hidden']), select, textarea, dl-icon-button");
    const nameOf = control => control.getAttribute('aria-label')
        || document.getElementById(control.getAttribute('aria-labelledby'))?.textContent.trim()
        || control.textContent.trim() || control.getAttribute('placeholder') || control.tagName;
    const small = [];
    for (const control of controls) {
        if (control.disabled || control.hasAttribute('disabled')) continue;
        if (control.closest('[inert], [hidden]') || !control.getClientRects().length) continue;
        const target = control.matches("input[type='radio'], input[type='checkbox']")
            ? control.closest('label') ?? control : control;
        target.scrollIntoView({block: 'center'});
        const box = target.getBoundingClientRect();
        const x = box.left + box.width / 2;
        const y = box.top + box.height / 2;
        const owns = (px, py) => {
            const hit = document.elementFromPoint(px, py);
            return hit !== null && (hit === target || target.contains(hit));
        };
        const name = nameOf(control);
        if (!owns(x, y)) {
            small.push({name, problem: 'covered at its centre'});
            continue;
        }
        const reach = (dx, dy) => {
            let steps = 0;
            while (steps < 80 && owns(x + dx * (steps + 1), y + dy * (steps + 1))) steps += 1;
            return steps;
        };
        const width = reach(-1, 0) + reach(1, 0) + 1;
        const height = reach(0, -1) + reach(0, 1) + 1;
        if (width < 44 || height < 44) small.push({name, width, height});
    }
    return small;
}"""


def _assert_fingers_are_served(settings: Locator, where: str) -> None:
    """Every control in view answers to a 44px square, or says which do not."""
    small = settings.evaluate(_SMALL_TARGETS)
    assert not small, f"{where}: " + "; ".join(
        f"{target['name']!r} is {target.get('width')} by {target.get('height')}" for target in small
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
    # Docked surfaces cast no shadow: the dialog is the one overlapping surface, and the notice
    # is a toast that floats over it.
    assert (
        settings.evaluate(
            """element => [...element.querySelectorAll('*:not(.toast, .toast *)')].filter(node =>
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
def test_settings_fills_a_phone_and_every_control_a_finger_meets_is_44px(page: Page) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    _install_busy_settings_routes(page)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings(page)
    box = settings.bounding_box()
    assert box is not None
    assert (box["x"], box["y"], box["width"], box["height"]) == (0, 0, 390, 844)
    assert settings.evaluate("element => getComputedStyle(element).borderRadius") == "0px"
    _assert_surface_owns_viewport_layer(page, ".settings-dialog")

    navigation = settings.get_by_role("navigation", name="Settings")
    _assert_fingers_are_served(settings, "the section list")

    def visit(name: str) -> Locator:
        navigation.get_by_role("button", name=name, exact=True).click()
        return settings.get_by_role("region", name=name)

    def leave() -> None:
        settings.get_by_role("button", name="Back").click()
        expect(navigation).to_be_visible()

    # Connections, in every state that shows a control: the cards, an open card on each
    # authentication, the endpoint being changed, and the form that adds one.
    connections = visit("Connections")
    expect(connections.get_by_role("switch")).to_have_count(2)
    _assert_fingers_are_served(settings, "Connections")
    connections.get_by_role("button", name="Team wiki").click()
    connections.get_by_role("button", name="Change endpoint").click()
    _assert_fingers_are_served(settings, "an open Connection whose endpoint is being changed")
    connections.get_by_role("button", name="Bearer", exact=True).click()
    expect(connections.get_by_label("Personal bearer (write-only)")).to_be_visible()
    _assert_fingers_are_served(settings, "a Connection on bearer authentication")
    connections.get_by_role("button", name="OAuth", exact=True).click()
    connections.get_by_role("button", name="Authorize with OAuth").click()
    expect(
        connections.get_by_role("link", name="Continue to provider authorization")
    ).to_be_visible()
    _assert_fingers_are_served(settings, "a Connection on OAuth with its provider's link")
    connections.get_by_role("button", name="Add MCP connection").click()
    expect(connections.get_by_role("button", name="Use the Notion preset")).to_be_visible()
    _assert_fingers_are_served(settings, "the form that adds a Connection")
    leave()

    accounts = visit("Agent Accounts")
    expect(accounts.get_by_role("listitem")).to_have_count(3)
    _assert_fingers_are_served(settings, "Agent Accounts")
    leave()

    memory = visit("Profile Memory")
    expect(memory.get_by_role("button", name="Load more")).to_be_visible()
    _assert_fingers_are_served(settings, "Profile Memory")
    leave()

    visit("Conversation Sessions")
    _assert_fingers_are_served(settings, "Conversation Sessions")
    leave()

    visit("Language")
    _assert_fingers_are_served(settings, "Language")


@pytest.mark.e2e
def test_profile_memory_shows_the_whole_text_of_a_memory(page: Page) -> None:
    text = (
        "Reports go to the investment committee and use tables and short bullets, never long "
        "paragraphs, and every figure carries the page it came from.\n"
        "Ask before sending anything outside the firm."
    )
    _install_conversation_routes(page)
    _install_memory_routes(page, [text])
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings_page(page, "Profile Memory")
    body = settings.get_by_role("region", name="Profile Memory").locator("li p")
    expect(body).to_have_text(text)
    # Nothing is cut off, and the memory's own line break is kept: far more than the two lines it
    # was once clamped to.
    shape = body.evaluate(
        """element => ({
            clipped: element.scrollHeight > element.clientHeight,
            lines: element.getBoundingClientRect().height
                / Number.parseFloat(getComputedStyle(element).lineHeight),
        })"""
    )
    assert shape["clipped"] is False
    assert shape["lines"] >= 3.5


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
