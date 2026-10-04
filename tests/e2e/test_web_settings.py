# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What only the built app shows of the Settings dialog: its geometry, what a finger and an eye
meet on every page, and how the shell opens it. Behaviour lives in the browser tests."""

from collections.abc import Callable
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
    """Connections with presets and one that is failing, and Profile Memory on with a next page."""

    def connections(route: Route) -> None:
        def connection(
            connection_id: str, label: str, authentication: str, enabled: bool, status: str
        ) -> Any:
            return {
                "connection_id": connection_id,
                "label": label,
                "endpoint": f"https://{connection_id}.example/mcp",
                "enabled": enabled,
                "activation_epoch": 1,
                "generation": 1,
                "authentication": authentication,
                "authorization_status": None,
                "status": status,
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
                    connection("notion", "Notion", "oauth", True, "ready"),
                    # Enabled but not answering, so its card opens on the alert that says so.
                    connection("wiki", "Team wiki", "none", True, "degraded"),
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


# The text of the dialog that is too faint to read: its colour against what lies under it (every
# translucent layer between it and the first opaque one, composited), read from what the browser
# painted. Small text needs 4.5 to 1. A disabled control is exempt, as WCAG has it.
_FAINT_TEXT = """root => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    const rgba = color => {
        context.clearRect(0, 0, 1, 1);
        context.fillStyle = '#000';
        context.fillStyle = color;
        context.fillRect(0, 0, 1, 1);
        const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data;
        return [r, g, b, a / 255];
    };
    const over = (top, below) => [0, 1, 2].map(i => top[i] * top[3] + below[i] * (1 - top[3])).concat([1]);
    const luminance = color => {
        const channel = value => {
            const unit = value / 255;
            return unit <= 0.03928 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
        };
        return 0.2126 * channel(color[0]) + 0.7152 * channel(color[1]) + 0.0722 * channel(color[2]);
    };
    const background = element => {
        const layers = [];
        for (let node = element; node; node = node.parentElement) {
            const color = rgba(getComputedStyle(node).backgroundColor);
            if (color[3] > 0) layers.push(color);
            if (color[3] === 1) break;
        }
        if (!layers.length || layers.at(-1)[3] !== 1) layers.push([255, 255, 255, 1]);
        return layers.reverse().reduce((below, layer) => over(layer, below));
    };
    const faint = [];
    for (const element of root.querySelectorAll('*')) {
        const hasText = [...element.childNodes].some(
            node => node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '');
        if (!hasText || !element.getClientRects().length) continue;
        if (element.closest('.dl-sr-only, :disabled, [inert], [hidden]')) continue;
        const style = getComputedStyle(element);
        if (style.visibility === 'hidden' || Number(style.opacity) < 1) continue;
        const behind = background(element);
        const lighter = Math.max(luminance(over(rgba(style.color), behind)), luminance(behind));
        const darker = Math.min(luminance(over(rgba(style.color), behind)), luminance(behind));
        const ratio = (lighter + 0.05) / (darker + 0.05);
        if (ratio < 4.5) {
            faint.push({text: element.textContent.trim().slice(0, 40), ratio: Math.round(ratio * 100) / 100});
        }
    }
    return faint;
}"""


def _assert_text_reads(settings: Locator, where: str) -> None:
    """Every word in view reads against its background at 4.5 to 1, or say which do not."""
    faint = settings.evaluate(_FAINT_TEXT)
    assert not faint, f"{where}: " + "; ".join(
        f"{item['text']!r} is {item['ratio']} to 1" for item in faint
    )


def _walk_settings(page: Page, settings: Locator, check: Callable[[str], None]) -> None:
    """Visit every page of an open Settings in the states that show the most, and hand each to ``check``.

    On a phone a page is reached from the section list and left by Back; beside a pointer the
    navigation is always in view. The states are the cards of Connections with one open on each
    authentication, the form that adds one, a Memory list with a next page, and the other pages.
    """
    phone = (page.viewport_size or {"width": 0})["width"] <= 720
    navigation = settings.get_by_role("navigation", name="Settings")

    def visit(name: str) -> Locator:
        navigation.get_by_role("button", name=name, exact=True).click()
        return settings.get_by_role("region", name=name)

    def leave() -> None:
        if phone:
            settings.get_by_role("button", name="Back").click()
            expect(navigation).to_be_visible()

    if phone:
        check("the section list")

    connections = visit("Connections")
    expect(connections.get_by_role("switch")).to_have_count(2)
    check("Connections")
    connections.get_by_role("button", name="Team wiki").click()
    connections.get_by_role("button", name="Change endpoint").click()
    check("an open Connection whose endpoint is being changed")
    connections.get_by_role("button", name="Bearer", exact=True).click()
    expect(connections.get_by_label("Personal bearer (write-only)")).to_be_visible()
    check("a Connection on bearer authentication")
    connections.get_by_role("button", name="OAuth", exact=True).click()
    connections.get_by_role("button", name="Authorize with OAuth").click()
    expect(
        connections.get_by_role("link", name="Continue to provider authorization")
    ).to_be_visible()
    check("a Connection on OAuth with its provider's link")
    connections.get_by_role("button", name="Add MCP connection").click()
    expect(connections.get_by_role("button", name="Use the Notion preset")).to_be_visible()
    check("the form that adds a Connection")
    leave()

    accounts = visit("Agent Accounts")
    expect(accounts.get_by_role("switch", name="Allow new sign-ups")).to_be_visible()
    expect(
        accounts.get_by_role("button", name="Remove the account for discourse.org")
    ).to_be_visible()
    check("Agent Accounts")
    leave()

    memory = visit("Profile Memory")
    expect(memory.get_by_role("button", name="Load more")).to_be_visible()
    check("Profile Memory")
    leave()

    visit("Conversation Sessions")
    check("Conversation Sessions")
    leave()

    visit("Language")
    check("Language")


@pytest.mark.e2e
def test_settings_is_a_centered_dialog_with_a_navigation_column_beside_its_page(page: Page) -> None:
    _install_conversation_routes(page)
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

    # The navigation is a 15rem column with the page to its right, and no page resizes the dialog.
    navigation = settings.get_by_role("navigation", name="Settings")
    nav_box = navigation.bounding_box()
    pane_box = settings.get_by_role("region", name="Connections").bounding_box()
    assert nav_box is not None and pane_box is not None
    assert nav_box["width"] == pytest.approx(240, abs=1)
    assert nav_box["x"] + nav_box["width"] <= pane_box["x"] + 1
    assert nav_box["y"] == pytest.approx(pane_box["y"], abs=1)
    navigation.get_by_role("button", name="Language", exact=True).click()
    after = settings.bounding_box()
    assert after is not None
    assert (after["width"], after["height"]) == (box["width"], box["height"])


@pytest.mark.e2e
def test_settings_opens_from_its_button_and_escape_gives_focus_back_to_it(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    settings = _open_settings(page)
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

    _walk_settings(page, settings, lambda where: _assert_fingers_are_served(settings, where))


@pytest.mark.e2e
@pytest.mark.parametrize("viewport", [(1440, 900), (390, 844)], ids=["desktop", "phone"])
@pytest.mark.parametrize("color_mode", ["light", "dark"])
def test_every_word_in_settings_reads_at_4_5_to_1_in_both_themes(
    page: Page, viewport: tuple[int, int], color_mode: str
) -> None:
    _install_conversation_routes(page)
    _install_agent_accounts_routes(page, _three_accounts())
    _install_busy_settings_routes(page)
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.locator("html").evaluate(
        "(element, mode) => { element.dataset.colorMode = mode; }", color_mode
    )

    settings = _open_settings(page)
    _walk_settings(page, settings, lambda where: _assert_text_reads(settings, where))


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
def test_agent_accounts_are_a_table_where_the_page_has_room_and_three_lines_where_it_has_not(
    page: Page,
) -> None:
    settings = _agent_accounts_page(page, _three_accounts())
    region = settings.get_by_role("region", name="Agent Accounts")

    # The 880px dialog leaves its page room for every column, the date of registering included.
    table = region.get_by_role("table")
    expect(table.get_by_role("columnheader", name="Registered")).to_be_visible()

    # A narrower window narrows the dialog and with it the page, which is what the page measures:
    # every account becomes three lines, and the one that never signed in still says when it registered.
    page.set_viewport_size({"width": 800, "height": 800})
    expect(table).to_have_count(0)
    rows = region.get_by_role("listitem")
    expect(rows).to_have_count(3)
    expect(rows.nth(2)).to_contain_text("not signed in since")
    expect(rows.nth(2)).to_contain_text("Registered")

    page.set_viewport_size({"width": 1440, "height": 900})
    expect(table.get_by_role("columnheader", name="Registered")).to_be_visible()


@pytest.mark.e2e
def test_the_language_chosen_in_settings_is_the_one_the_app_starts_in(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    settings = _open_settings_page(page, "Language")

    settings.get_by_role("radio", name="中文").check()
    assert page.evaluate("localStorage.getItem('dlightrag-lang')") == "zh"

    page.reload()
    page.locator("[aria-current='page']").wait_for()
    expect(page.locator("html")).to_have_attribute("lang", "zh")
    expect(page.get_by_role("button", name="设置", exact=True)).to_be_visible()
