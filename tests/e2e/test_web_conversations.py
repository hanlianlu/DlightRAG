# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the Web conversation lifecycle shell."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from playwright.sync_api import Locator, Page, Route, expect


@dataclass
class ConversationRouteState:
    conversations: list[dict[str, str | None]] = field(default_factory=list)
    delete_status: int = 204


def _install_conversation_routes(page: Page) -> ConversationRouteState:
    now = datetime.now(UTC)
    state = ConversationRouteState()

    def summary(conversation_id: str, title: str | None = None) -> dict[str, str | None]:
        offset = timedelta(seconds=len(state.conversations))
        timestamp = (now + offset).isoformat().replace("+00:00", "Z")
        return {
            "conversation_id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    initial = summary(str(uuid4()))
    state.conversations.append(initial)
    page.add_init_script(
        f"""if (location.pathname === '/web/' || location.pathname === '/web') {{
            history.replaceState(
                history.state,
                '',
                '/web/conversations/{initial["conversation_id"]}',
            );
        }}"""
    )

    def handle(route: Route) -> None:
        request = route.request
        path = urlparse(request.url).path
        method = request.method
        if path == "/web/api/conversations" and method == "GET":
            route.fulfill(json=state.conversations)
            return
        if path == "/web/api/conversations" and method == "POST":
            item = summary(str(uuid4()))
            state.conversations.insert(0, item)
            route.fulfill(status=201, json=item)
            return
        if path == "/web/api/conversations" and method == "DELETE":
            if state.delete_status != 204:
                route.fulfill(
                    status=state.delete_status,
                    json={"detail": "Deletion failed"},
                )
                return
            state.conversations.clear()
            route.fulfill(status=204, body="")
            return

        parts = path.split("/")
        conversation_id = parts[4] if len(parts) > 4 else ""
        item = next(
            (row for row in state.conversations if row["conversation_id"] == conversation_id),
            None,
        )
        if item is None:
            route.fulfill(status=404, json={"detail": "Conversation not found"})
            return
        if path.endswith("/history") and method == "GET":
            route.fulfill(json={"conversation": item, "turns": []})
            return
        if method == "PATCH":
            item["title"] = str((request.post_data_json or {})["title"])
            item["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            route.fulfill(json=item)
            return
        if method == "DELETE":
            if state.delete_status != 204:
                route.fulfill(
                    status=state.delete_status,
                    json={"detail": "Deletion failed"},
                )
                return
            state.conversations.remove(item)
            route.fulfill(status=204, body="")
            return
        route.abort()

    page.route("**/web/api/conversations**", handle)
    return state


def _active_id(page: Page) -> str:
    active = page.locator("[aria-current='page']")
    active.wait_for()
    conversation_id = active.get_attribute("data-conversation-id")
    assert conversation_id
    return conversation_id


def _new_conversation(page: Page) -> tuple[str, str]:
    """Create server test data, then exercise direct route navigation to it."""
    previous_id = _active_id(page)
    created = page.evaluate(
        """async () => {
            const response = await fetch('/web/api/conversations', {method: 'POST'});
            return await response.json();
        }"""
    )
    conversation_id = str(created["conversation_id"])
    page.goto(f"/web/conversations/{conversation_id}")
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId === id",
        arg=conversation_id,
    )
    page.locator("#new-conversation-btn:not([disabled])").wait_for()
    return previous_id, conversation_id


def _open_settings(page: Page) -> Locator:
    trigger = page.get_by_role("button", name="Settings", exact=True)
    if not trigger.is_visible():
        page.get_by_role("button", name="Open conversations").click()
        page.locator("#chat-sidebar").wait_for(state="visible")
    trigger.focus()
    page.keyboard.press("Enter")
    dialog = page.get_by_role("dialog", name="Settings")
    dialog.wait_for()
    return dialog


def _add_draft_with_image(page: Page, text: str) -> None:
    page.get_by_role("textbox", name="Message").fill(text)
    page.locator("#attachment-input").set_input_files(
        files={
            "name": "draft.png",
            "mimeType": "image/png",
            "buffer": b"\x89PNG\r\n\x1a\n",
        }
    )
    page.wait_for_function("document.querySelector('#thumbnail-strip')?.children.length === 1")


def _open_delete_dialog_with_keyboard(page: Page, conversation_id: str) -> None:
    row = page.locator(f'[data-conversation-id="{conversation_id}"]')
    actions = row.get_by_role("button", name="Conversation actions")
    actions.focus()
    page.keyboard.press("Enter")
    page.get_by_role("menuitem", name="Rename").wait_for()
    page.wait_for_function(
        "document.activeElement?.getAttribute('role') === 'menuitem'"
        " && document.activeElement?.textContent === 'Rename'"
    )
    page.keyboard.press("ArrowDown")
    page.wait_for_function("document.activeElement?.textContent === 'Delete'")
    page.keyboard.press("Enter")
    page.get_by_role("dialog", name="Delete conversation").wait_for()


def _delete_dialog_accessible_description(page: Page) -> str:
    session = page.context.new_cdp_session(page)
    try:
        tree: dict[str, Any] = session.send("Accessibility.getFullAXTree")
    finally:
        session.detach()
    dialog = next(
        node
        for node in tree["nodes"]
        if node.get("role", {}).get("value") == "dialog"
        and node.get("name", {}).get("value") == "Delete conversation"
    )
    return str(dialog.get("description", {}).get("value", ""))


def _text_bottom(page: Page, selector: str) -> float:
    return float(
        page.locator(selector).evaluate(
            """element => {
                const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
                let node = walker.nextNode();
                while (node && !node.textContent?.trim()) node = walker.nextNode();
                if (!node) throw new Error(`No text node in ${element.id || element.className}`);
                const range = document.createRange();
                range.selectNodeContents(node);
                return range.getBoundingClientRect().bottom;
            }"""
        )
    )


@pytest.mark.e2e
def test_new_select_rename_delete_survive_reload(page: Page) -> None:
    state = _install_conversation_routes(page)
    page.goto("/web/")

    active = page.locator("[aria-current='page']")
    active.wait_for()
    initial_id = active.get_attribute("data-conversation-id")
    assert initial_id
    previous_id, first_id = _new_conversation(page)
    assert previous_id == initial_id

    active.get_by_role("button", name="Conversation actions").click()
    page.get_by_role("menuitem", name="Rename").click()
    page.get_by_role("textbox", name="Conversation title").fill("Research notes")
    page.keyboard.press("Enter")
    page.get_by_text("Research notes", exact=True).wait_for()

    _, second_id = _new_conversation(page)
    assert second_id != first_id
    page.get_by_role("button", name="Research notes", exact=True).click()
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId === id",
        arg=first_id,
    )
    assert page.locator("[aria-current='page']").get_attribute("data-conversation-id") == first_id

    page.reload()
    page.get_by_text("Research notes", exact=True).wait_for()
    assert page.locator("[aria-current='page']").get_attribute("data-conversation-id") == first_id

    page.locator("[aria-current='page']").get_by_role("button", name="Conversation actions").click()
    page.get_by_role("menuitem", name="Delete").click()
    page.get_by_role("dialog", name="Delete conversation").get_by_role(
        "button", name="Delete"
    ).click()
    page.wait_for_function(
        'id => !document.querySelector(`[data-conversation-id="${id}"]`)',
        arg=first_id,
    )
    assert all(row["conversation_id"] != first_id for row in state.conversations)
    assert page.locator("[aria-current='page']").get_attribute("data-conversation-id") == second_id


@pytest.mark.e2e
def test_new_chat_is_an_unpersisted_root_route(page: Page) -> None:
    state = _install_conversation_routes(page)
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    page.get_by_role("button", name="New chat").click()

    page.wait_for_url("**/web/")
    assert page.locator("[aria-current='page']").count() == 0
    assert len(state.conversations) == 1
    page.get_by_text("Ask anything about your documents").wait_for()


@pytest.mark.e2e
def test_conversation_routes_drive_back_forward_and_direct_reload(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    first_id = _active_id(page)
    created = page.evaluate(
        """async () => {
            const response = await fetch('/web/api/conversations', {method: 'POST'});
            return await response.json();
        }"""
    )
    second_id = str(created["conversation_id"])
    page.reload()
    page.locator(f'[data-conversation-id="{second_id}"] .conversation-select').click()
    page.wait_for_url(f"**/web/conversations/{second_id}")
    page.locator(f'[data-conversation-id="{first_id}"] .conversation-select').click()
    page.wait_for_url(f"**/web/conversations/{first_id}")

    page.go_back()
    page.wait_for_url(f"**/web/conversations/{second_id}")
    assert _active_id(page) == second_id
    page.go_forward()
    page.wait_for_url(f"**/web/conversations/{first_id}")
    assert _active_id(page) == first_id

    page.reload()
    assert _active_id(page) == first_id


@pytest.mark.e2e
def test_unavailable_route_stays_visible_and_offers_a_recent_conversation(page: Page) -> None:
    state = _install_conversation_routes(page)
    recent_id = str(state.conversations[0]["conversation_id"])

    page.goto("/web/conversations/not-owned")

    page.get_by_text("Conversation unavailable.").wait_for()
    assert page.url.endswith("/web/conversations/not-owned")
    page.get_by_role("button", name="Open recent conversation").click()
    page.wait_for_url(f"**/web/conversations/{recent_id}")
    assert _active_id(page) == recent_id


@pytest.mark.e2e
def test_browser_back_respects_the_shared_draft_guard(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    conversation_id = _active_id(page)
    page.get_by_role("button", name="New chat").click()
    page.locator(f'[data-conversation-id="{conversation_id}"] .conversation-select').click()
    page.get_by_role("textbox", name="Message").fill("keep this route")

    page.go_back()
    dialog = page.get_by_role("dialog", name="Discard draft?")
    dialog.get_by_role("button", name="Keep editing").click()
    page.wait_for_url(f"**/web/conversations/{conversation_id}")
    page.wait_for_timeout(50)  # let the router consume its restoration popstate
    assert page.get_by_role("textbox", name="Message").input_value() == "keep this route"

    page.go_back()
    dialog.get_by_role("button", name="Discard and continue").click()
    page.wait_for_url("**/web/")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.get_by_role("textbox", name="Message").input_value() == ""


@pytest.mark.e2e
def test_workspace_files_panel_survives_conversation_route_changes(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    page.get_by_role("button", name="Files", exact=True).click()
    page.locator("#panel.open").wait_for()

    page.get_by_role("button", name="New chat").click()

    page.wait_for_url("**/web/")
    assert page.locator("#panel").get_attribute("data-panel-kind") == "files"
    assert page.locator("#panel").evaluate("element => element.classList.contains('open')")


@pytest.mark.e2e
def test_draft_confirmation_and_sidebar_toggle_preserve_the_composer(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    page.get_by_role("textbox", name="Message").fill("Unsent draft")

    composer = page.locator("#composer")
    composer_handle = composer.evaluate_handle("node => node")
    page.get_by_role("button", name="Collapse conversations").click()
    assert composer.evaluate("(node, original) => node === original", composer_handle)
    assert page.get_by_role("textbox", name="Message").input_value() == "Unsent draft"

    page.get_by_role("button", name="Open conversations").click()
    page.get_by_role("button", name="New chat").click()
    dialog = page.get_by_role("dialog", name="Discard draft?")
    dialog.get_by_role("button", name="Keep editing").click()
    page.wait_for_function("document.activeElement?.id === 'new-conversation-btn'")
    assert page.get_by_role("textbox", name="Message").input_value() == "Unsent draft"

    page.get_by_role("button", name="New chat").click()
    dialog.get_by_role("button", name="Discard and continue").click()
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.get_by_role("textbox", name="Message").input_value() == ""


@pytest.mark.e2e
def test_active_delete_cancel_preserves_draft_selection_and_keyboard_focus(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    active_id = _active_id(page)
    _add_draft_with_image(page, "draft stays on cancel")

    _open_delete_dialog_with_keyboard(page, active_id)
    dialog = page.get_by_role("dialog", name="Delete conversation")
    dialog.get_by_text("Your unsent draft and attachments will also be discarded.").wait_for()
    assert "Your unsent draft and attachments will also be discarded." in (
        _delete_dialog_accessible_description(page)
    )
    dialog.get_by_role("button", name="Cancel").click()

    page.wait_for_function(
        "id => document.activeElement?.closest('[data-conversation-id]')?.dataset.conversationId === id"
        " && document.activeElement?.getAttribute('aria-label') === 'Conversation actions'",
        arg=active_id,
    )
    assert _active_id(page) == active_id
    assert page.get_by_role("textbox", name="Message").input_value() == "draft stays on cancel"
    assert page.locator("#thumbnail-strip").locator(":scope > *").count() == 1


@pytest.mark.e2e
def test_active_clean_delete_accessible_description_omits_draft_warning(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    active_id = _active_id(page)

    _open_delete_dialog_with_keyboard(page, active_id)
    description = _delete_dialog_accessible_description(page)
    assert "This conversation and its history will be permanently deleted." in description
    assert "Your unsent draft and attachments will also be discarded." not in description
    page.get_by_role("dialog", name="Delete conversation").get_by_role(
        "button", name="Cancel"
    ).click()


@pytest.mark.e2e
def test_active_delete_success_discards_draft_then_focuses_fallback(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    fallback_id, deleted_id = _new_conversation(page)
    _add_draft_with_image(page, "draft must not migrate")

    _open_delete_dialog_with_keyboard(page, deleted_id)
    page.get_by_role("dialog", name="Delete conversation").get_by_role(
        "button", name="Delete"
    ).click()

    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId === id",
        arg=fallback_id,
    )
    page.wait_for_function("document.querySelector('#thumbnail-strip')?.children.length === 0")
    page.wait_for_function(
        "id => document.activeElement?.closest('[data-conversation-id]')?.dataset.conversationId === id",
        arg=fallback_id,
    )
    assert page.get_by_role("textbox", name="Message").input_value() == ""
    assert page.locator(f'[data-conversation-id="{deleted_id}"]').count() == 0


@pytest.mark.e2e
def test_active_delete_failure_preserves_draft_selection_and_focus(page: Page) -> None:
    state = _install_conversation_routes(page)
    state.delete_status = 500
    page.goto("/web/")
    active_id = _active_id(page)
    _add_draft_with_image(page, "draft survives failure")

    _open_delete_dialog_with_keyboard(page, active_id)
    page.get_by_role("dialog", name="Delete conversation").get_by_role(
        "button", name="Delete"
    ).click()

    page.get_by_text("Could not delete the conversation.").wait_for()
    page.wait_for_function(
        "id => document.activeElement?.closest('[data-conversation-id]')?.dataset.conversationId === id"
        " && document.activeElement?.getAttribute('aria-label') === 'Conversation actions'",
        arg=active_id,
    )
    assert _active_id(page) == active_id
    assert page.get_by_role("textbox", name="Message").input_value() == "draft survives failure"
    assert page.locator("#thumbnail-strip").locator(":scope > *").count() == 1


@pytest.mark.e2e
def test_inactive_delete_never_touches_active_draft(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    inactive_id, active_id = _new_conversation(page)
    _add_draft_with_image(page, "active draft remains")

    _open_delete_dialog_with_keyboard(page, inactive_id)
    dialog = page.get_by_role("dialog", name="Delete conversation")
    assert not dialog.get_by_text(
        "Your unsent draft and attachments will also be discarded."
    ).is_visible()
    description = _delete_dialog_accessible_description(page)
    assert "This conversation and its history will be permanently deleted." in description
    assert "Your unsent draft and attachments will also be discarded." not in description
    dialog.get_by_role("button", name="Delete").click()

    page.wait_for_function(
        'id => !document.querySelector(`[data-conversation-id="${id}"]`)',
        arg=inactive_id,
    )
    assert _active_id(page) == active_id
    assert page.get_by_role("textbox", name="Message").input_value() == "active draft remains"
    assert page.locator("#thumbnail-strip").locator(":scope > *").count() == 1


@pytest.mark.e2e
def test_delete_all_conversations_is_quiet_accessible_and_returns_to_new_chat(
    page: Page,
) -> None:
    state = _install_conversation_routes(page)
    page.goto("/web/")
    _new_conversation(page)
    _new_conversation(page)
    assert len(state.conversations) == 3

    settings = _open_settings(page)
    trigger = settings.get_by_role("button", name="Delete all conversations")
    assert settings.get_by_text("Conversations retain 365 days", exact=True).is_visible()
    assert settings.get_by_text("3 conversations", exact=True).is_visible()
    trigger.click()

    dialog = page.get_by_role("dialog", name="Delete all conversations?")
    title = dialog.get_by_role("heading", name="Delete all conversations?")
    actions = dialog.locator(".ui-dialog-actions")
    assert dialog.evaluate(
        """element => {
            const bounds = element.getBoundingClientRect();
            return Math.abs(bounds.left + bounds.width / 2 - innerWidth / 2) < 1
                && Math.abs(bounds.top + bounds.height / 2 - innerHeight / 2) < 1;
        }"""
    )
    assert title.evaluate("element => getComputedStyle(element).textAlign") == "center"
    assert actions.evaluate("element => getComputedStyle(element).justifyContent") == "center"
    assert dialog.locator("p:visible").count() == 0
    dialog.get_by_role("button", name="Cancel").click()
    assert settings.is_visible()
    assert len(state.conversations) == 3
    settings.get_by_role("button", name="Close settings").click()

    _add_draft_with_image(page, "discard this draft")
    settings = _open_settings(page)
    settings.get_by_role("button", name="Delete all conversations").click()
    dialog.get_by_text("Draft and attachments will also be deleted.").wait_for()
    dialog.get_by_role("button", name="Delete all").click()

    page.wait_for_function("() => document.querySelectorAll('[data-conversation-id]').length === 0")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    page.wait_for_url("**/web/")
    assert len(state.conversations) == 0
    assert page.locator("[aria-current='page']").count() == 0
    assert page.locator("#thumbnail-strip").locator(":scope > *").count() == 0


@pytest.mark.e2e
def test_delete_all_failure_preserves_conversations_draft_and_theme_tokens(page: Page) -> None:
    state = _install_conversation_routes(page)
    state.delete_status = 500
    page.goto("/web/")
    _new_conversation(page)
    _add_draft_with_image(page, "keep this draft")

    settings = _open_settings(page)
    settings.get_by_role("button", name="Delete all conversations").click()
    dialog = page.get_by_role("dialog", name="Delete all conversations?")
    danger = dialog.get_by_role("button", name="Delete all")
    page.add_style_tag(content=".ui-dialog-actions button { transition: none !important; }")
    for color_mode in ("light", "dark"):
        page.locator("html").evaluate(
            "(element, mode) => { element.dataset.colorMode = mode; }",
            color_mode,
        )
        assert dialog.evaluate(
            """element => {
                const probe = document.createElement('div');
                probe.style.backgroundColor = 'var(--color-bg-surface)';
                document.body.append(probe);
                const expected = getComputedStyle(probe).backgroundColor;
                probe.remove();
                return getComputedStyle(element).backgroundColor === expected;
            }"""
        )
        assert danger.evaluate(
            """element => {
                const probe = document.createElement('div');
                probe.style.backgroundColor = 'var(--color-danger-bg)';
                document.body.append(probe);
                const expected = getComputedStyle(probe).backgroundColor;
                probe.remove();
                return getComputedStyle(element).backgroundColor === expected;
            }"""
        )

    danger.click()
    page.get_by_text("Could not delete conversations.").wait_for()
    assert settings.is_visible()
    assert len(state.conversations) == 2
    assert page.locator("[data-conversation-id]").count() == 2
    assert page.get_by_role("textbox", name="Message").input_value() == "keep this draft"
    assert page.locator("#thumbnail-strip").locator(":scope > *").count() == 1


@pytest.mark.e2e
def test_delete_all_is_keyboard_accessible_and_centered_on_mobile(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto("/web/")
    settings = _open_settings(page)

    trigger = settings.get_by_role("button", name="Delete all conversations")
    trigger.focus()
    page.keyboard.press("Enter")
    dialog = page.get_by_role("dialog", name="Delete all conversations?")
    dialog.wait_for()
    bounds = dialog.bounding_box()
    assert bounds is not None
    assert abs(bounds["x"] + bounds["width"] / 2 - 195) < 1
    assert abs(bounds["y"] + bounds["height"] / 2 - 422) < 1
    title = dialog.get_by_role("heading", name="Delete all conversations?")
    assert title.evaluate("element => element.scrollWidth <= element.clientWidth")
    assert (
        dialog.locator(".ui-dialog-actions").evaluate(
            "element => getComputedStyle(element).justifyContent"
        )
        == "center"
    )

    page.keyboard.press("Escape")
    dialog.wait_for(state="hidden")
    page.wait_for_function("document.activeElement?.id === 'delete-all-btn'")
    assert settings.is_visible()


@pytest.mark.e2e
def test_desktop_scope_baseline_and_two_panel_geometry(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    search_scope = page.locator("#workspace-selector")
    closed_panel = page.locator("#panel")
    assert closed_panel.get_attribute("aria-hidden") == "true"
    assert closed_panel.evaluate("element => element.inert") is True
    assert page.get_by_text("Search in:", exact=True).is_visible()
    assert "All workspaces (2)" in search_scope.inner_text()
    active_scope = search_scope.inner_text()

    files_trigger = page.get_by_role("button", name="Files", exact=True)
    files_trigger.click()
    page.locator("#upload-zone").wait_for()
    assert closed_panel.get_attribute("aria-hidden") is None
    assert closed_panel.evaluate("element => element.inert") is False
    assert page.get_by_text("Files in:", exact=True).is_visible()
    assert page.locator("#panel-title").inner_text() == ""
    assert page.get_by_role("button", name="Choose folder").is_visible()
    assert page.get_by_role("button", name="Upload files").count() == 0
    assert not files_trigger.is_visible()
    page.wait_for_timeout(220)

    new_bottom = _text_bottom(page, "#new-conversation-btn")
    search_bottom = _text_bottom(page, ".topbar-scope-label")
    files_bottom = _text_bottom(page, ".ingest-target-label")
    assert (
        max(new_bottom, search_bottom, files_bottom) - min(new_bottom, search_bottom, files_bottom)
        <= 1.5
    )

    composer = page.locator("#composer").bounding_box()
    sidebar = page.locator("#chat-sidebar").bounding_box()
    panel = page.locator("#panel").bounding_box()
    assert composer is not None and sidebar is not None and panel is not None
    assert composer["x"] >= sidebar["x"] + sidebar["width"]
    assert composer["x"] + composer["width"] <= panel["x"]

    composer_leading_geometry = page.locator("#query-form").evaluate(
        """form => {
            const button = form.querySelector('.composer-plus');
            const icon = form.querySelector('.composer-plus-icon');
            const formRect = form.getBoundingClientRect();
            const buttonRect = button.getBoundingClientRect();
            const iconRect = icon.getBoundingClientRect();
            const styles = getComputedStyle(form);
            return {
                buttonInset: buttonRect.left - formRect.left,
                expectedButtonInset:
                    parseFloat(styles.borderLeftWidth) + parseFloat(styles.paddingLeft),
                iconInset: iconRect.left - buttonRect.left,
                expectedIconInset: (buttonRect.width - iconRect.width) / 2,
            };
        }"""
    )
    assert composer_leading_geometry["buttonInset"] == pytest.approx(
        composer_leading_geometry["expectedButtonInset"], abs=1
    )
    assert composer_leading_geometry["iconInset"] == pytest.approx(
        composer_leading_geometry["expectedIconInset"], abs=1
    )

    page.get_by_role("button", name="New chat").click()
    page.wait_for_url("**/web/")
    assert search_scope.inner_text() == active_scope


@pytest.mark.e2e
def test_compact_files_panel_and_visible_conversation_scroll_independently(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1000, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.evaluate(
        """async () => {
            const app = document.querySelector('dl-app');
            app.hasMessages = true;
            await app.updateComplete;
            const filler = document.createElement('div');
            filler.style.height = '3000px';
            filler.textContent = 'Scroll probe';
            document.querySelector('#chat-messages')?.append(filler);
            document.querySelector('#chat-area').scrollTop = 0;
        }"""
    )
    chat = page.locator("#chat-area")
    assert chat.evaluate("element => element.scrollHeight > element.clientHeight") is True

    page.get_by_role("button", name="Files", exact=True).click()
    page.locator("#panel.open").wait_for()
    panel = page.locator("#panel-content")
    panel.evaluate(
        """element => {
            const filler = document.createElement('div');
            filler.style.height = '3000px';
            filler.textContent = 'Panel scroll probe';
            element.append(filler);
            element.scrollTop = 0;
        }"""
    )
    assert panel.evaluate("element => element.scrollHeight > element.clientHeight") is True
    panel_bounds = panel.bounding_box()
    assert panel_bounds is not None
    page.mouse.move(panel_bounds["x"] + panel_bounds["width"] / 2, 450)
    page.mouse.wheel(0, 600)
    page.wait_for_function("document.querySelector('#panel-content').scrollTop > 0")

    bounds = chat.bounding_box()
    assert bounds is not None
    page.mouse.move(bounds["x"] + 100, bounds["y"] + bounds["height"] / 2)
    page.mouse.wheel(0, 600)
    page.wait_for_function("document.querySelector('#chat-area').scrollTop > 0")
    page.mouse.click(bounds["x"] + 100, bounds["y"] + bounds["height"] / 2)
    page.wait_for_function("!document.querySelector('#panel').classList.contains('open')")


@pytest.mark.e2e
def test_escape_closes_files_workspace_popover_without_closing_panel(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Files", exact=True).click()
    page.locator("#upload-zone").wait_for()

    ingest_target = page.get_by_role("button", name="Files in Default; choose file workspace")
    ingest_target.click()
    page.get_by_role("dialog", name="Select ingest workspace").wait_for()
    page.keyboard.press("Escape")

    assert page.get_by_role("dialog", name="Select ingest workspace").count() == 0
    expect(ingest_target).to_be_focused()
    assert page.locator("#panel").evaluate("element => element.classList.contains('open')") is True


@pytest.mark.e2e
def test_composer_attachment_picker_keeps_files_panel_open(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Files", exact=True).click()
    page.locator("#upload-zone").wait_for()

    panel = page.locator("#panel")
    with page.expect_file_chooser() as chooser_info:
        page.get_by_role("button", name="Attach files").click()
    chooser_info.value.set_files([])

    assert panel.evaluate("element => element.classList.contains('open')") is True
    assert panel.get_attribute("data-panel-kind") == "files"


@pytest.mark.e2e
def test_only_the_conversation_area_dismisses_an_open_panel(page: Page) -> None:
    """Top-bar chrome must not dismiss the panel; the conversation area must."""
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Files", exact=True).click()
    page.locator("#upload-zone").wait_for()

    panel = page.locator("#panel")
    is_open = "element => element.classList.contains('open')"

    page.locator("#workspace-selector").click()
    page.get_by_role("dialog", name="Workspaces").wait_for()
    assert panel.evaluate(is_open) is True
    page.keyboard.press("Escape")

    page.locator("#chat-area").click(position={"x": 400, "y": 300})
    assert panel.evaluate(is_open) is False


@pytest.mark.e2e
@pytest.mark.parametrize("viewport", [(900, 800), (390, 844)])
def test_compact_drawers_are_modal_mutually_exclusive_and_restore_focus(
    page: Page, viewport: tuple[int, int]
) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    notification_offer = page.locator("#notify-offer")
    notification_offer.evaluate("element => { element.hidden = false; }")
    assert notification_offer.is_visible()
    open_conversations = page.get_by_role("button", name="Open conversations")
    open_conversations.click()
    sidebar = page.locator("#chat-sidebar")
    assert sidebar.get_attribute("role") == "dialog"
    assert sidebar.get_attribute("aria-modal") == "true"
    assert page.locator("dl-chat-feature").evaluate("element => element.inert") is True
    assert notification_offer.evaluate("element => element.inert") is True
    assert (
        notification_offer.evaluate("element => getComputedStyle(element).visibility") == "hidden"
    )
    page.keyboard.press("Escape")
    page.wait_for_function("document.activeElement?.id === 'conversation-sidebar-open'")
    assert notification_offer.evaluate("element => element.inert") is False
    assert notification_offer.is_visible()

    files = page.get_by_role("button", name="Files", exact=True)
    files.click()
    panel = page.locator("#panel")
    assert panel.get_attribute("role") == "dialog"
    assert panel.get_attribute("aria-modal") == "true"
    assert page.locator("dl-chat-feature").evaluate("element => element.inert") is False
    assert page.locator("#chat-messages").evaluate("element => element.inert") is True
    assert page.locator("dl-chat-composer").evaluate("element => element.inert") is True
    assert notification_offer.evaluate("element => element.inert") is True
    assert (
        notification_offer.evaluate("element => getComputedStyle(element).visibility") == "hidden"
    )
    new_bottom = _text_bottom(page, "#new-conversation-btn")
    search_bottom = _text_bottom(page, ".topbar-scope-label")
    files_bottom = _text_bottom(page, ".ingest-target-label")
    assert (
        max(new_bottom, search_bottom, files_bottom) - min(new_bottom, search_bottom, files_bottom)
        <= 1.5
    )
    page.keyboard.press("Escape")
    page.wait_for_function("document.activeElement?.id === 'files-btn'")


@pytest.mark.e2e
def test_resizing_open_files_panel_to_compact_locks_controls_but_not_scroll(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Files", exact=True).click()

    page.set_viewport_size({"width": 900, "height": 800})
    page.wait_for_function(
        "document.querySelector('#panel')?.getAttribute('aria-modal') === 'true'"
    )

    assert page.locator("#panel").get_attribute("aria-modal") == "true"
    assert page.locator("dl-chat-feature").evaluate("element => element.inert") is False
    assert page.locator("#chat-messages").evaluate("element => element.inert") is True
    assert page.locator("dl-chat-composer").evaluate("element => element.inert") is True
    outer_split = page.locator("#panel-split").bounding_box()
    primary_app = page.locator(".app-shell").bounding_box()
    panel = page.locator("#panel").bounding_box()
    assert outer_split is not None
    assert primary_app is not None
    assert panel is not None
    assert primary_app["width"] == pytest.approx(outer_split["width"], abs=1)
    assert panel["width"] == pytest.approx(420, abs=1)
    assert panel["x"] > outer_split["x"]


@pytest.mark.e2e
def test_wide_panel_effective_width_tracks_sidebar_and_viewport_transitions(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    page.get_by_role("button", name="Collapse conversations").click()
    page.get_by_role("button", name="Files", exact=True).click()
    page.wait_for_timeout(220)
    assert page.locator(".panel-resize-handle").count() == 0
    split = page.locator("#panel-split")
    handle = split.get_by_role("separator", name="Resize Files or Sources")
    handle_box = handle.bounding_box()
    assert handle_box is not None
    page.mouse.move(handle_box["x"] + handle_box["width"] / 2, 180)
    page.mouse.down()
    page.wait_for_function("document.body.hasAttribute('data-resizing')")
    page.mouse.move(480, 180, steps=8)
    page.mouse.up()
    page.wait_for_function("!document.body.hasAttribute('data-resizing')")

    def shell_geometry() -> dict[str, float]:
        return page.evaluate(
            """() => {
                const rect = selector => document.querySelector(selector).getBoundingClientRect();
                const sidebar = rect('#chat-sidebar');
                const composer = rect('#composer');
                const panel = rect('#panel');
                return {
                    sidebarWidth: sidebar.width,
                    composerX: composer.x,
                    composerWidth: composer.width,
                    composerRight: composer.right,
                    panelX: panel.x,
                    panelWidth: panel.width,
                    effectiveWidth: parseFloat(
                        getComputedStyle(document.documentElement).getPropertyValue('--panel-width')
                    ),
                };
            }"""
        )

    collapsed = shell_geometry()
    assert collapsed["panelWidth"] == pytest.approx(920, abs=1)
    assert collapsed["effectiveWidth"] == pytest.approx(collapsed["panelWidth"], abs=1)
    assert collapsed["composerWidth"] == pytest.approx(520, abs=1)
    assert collapsed["composerRight"] == pytest.approx(collapsed["panelX"], abs=1)
    assert page.evaluate("localStorage.getItem('dlightrag-panel-width')") == "920"

    page.get_by_role("button", name="Open conversations").click()
    page.wait_for_timeout(220)
    expanded = shell_geometry()
    assert expanded["composerX"] == pytest.approx(expanded["sidebarWidth"], abs=1)
    assert expanded["composerWidth"] == pytest.approx(520, abs=1)
    assert expanded["panelWidth"] == pytest.approx(
        1440 - expanded["sidebarWidth"] - expanded["composerWidth"], abs=1
    )
    assert expanded["composerRight"] == pytest.approx(expanded["panelX"], abs=1)
    assert expanded["effectiveWidth"] == pytest.approx(expanded["panelWidth"], abs=1)
    assert page.evaluate("localStorage.getItem('dlightrag-panel-width')") == "920"

    page.set_viewport_size({"width": 1280, "height": 820})
    page.wait_for_timeout(220)
    narrower = shell_geometry()
    # The one-pixel WA divider shares a half pixel with each adjacent track.
    assert narrower["composerWidth"] >= 519
    assert narrower["composerRight"] == pytest.approx(narrower["panelX"], abs=1)
    assert narrower["effectiveWidth"] == pytest.approx(narrower["panelWidth"], abs=1)

    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(220)
    restored = shell_geometry()
    assert restored["composerWidth"] >= 519
    assert restored["composerRight"] == pytest.approx(restored["panelX"], abs=1)
    assert restored["effectiveWidth"] == pytest.approx(restored["panelWidth"], abs=1)

    page.get_by_role("button", name="Collapse conversations").click()
    page.wait_for_timeout(220)
    recollapsed = shell_geometry()
    assert recollapsed["panelWidth"] == pytest.approx(920, abs=1)
    assert recollapsed["composerWidth"] == pytest.approx(520, abs=1)
    assert recollapsed["composerRight"] == pytest.approx(recollapsed["panelX"], abs=1)

    handle.focus()
    handle.press("ArrowRight")
    page.wait_for_function("localStorage.getItem('dlightrag-panel-width') !== '920'")
    keyboard_resized = shell_geometry()
    assert keyboard_resized["panelWidth"] < recollapsed["panelWidth"]
    persisted_width = int(page.evaluate("localStorage.getItem('dlightrag-panel-width')"))
    assert persisted_width == pytest.approx(keyboard_resized["effectiveWidth"], abs=1)

    handle.press("Enter")
    page.wait_for_timeout(50)
    after_enter = shell_geometry()
    assert after_enter["panelWidth"] == pytest.approx(keyboard_resized["panelWidth"], abs=1)
    assert page.evaluate("localStorage.getItem('dlightrag-panel-width')") == str(persisted_width)

    page.get_by_role("button", name="Close panel").click()
    page.get_by_role("button", name="Files", exact=True).click()
    page.wait_for_timeout(50)
    reopened = shell_geometry()
    assert reopened["panelWidth"] == pytest.approx(persisted_width, abs=1)


@pytest.mark.e2e
def test_split_panel_supports_touch_resize(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Collapse conversations").click()
    page.get_by_role("button", name="Files", exact=True).click()

    divider = page.locator("#panel-split").get_by_role("separator", name="Resize Files or Sources")
    divider.evaluate(
        """element => {
            const touch = new Touch({
                identifier: 1, target: element, clientX: 1020, clientY: 180,
            });
            element.dispatchEvent(new TouchEvent('touchstart', {
                bubbles: true, cancelable: true,
                touches: [touch], changedTouches: [touch],
            }));
        }"""
    )
    page.wait_for_function("document.body.hasAttribute('data-resizing')")
    page.evaluate(
        """() => document.dispatchEvent(new PointerEvent('pointermove', {
            bubbles: true, clientX: 480, clientY: 180,
            pointerId: 1, pointerType: 'touch',
        }))"""
    )
    page.wait_for_timeout(50)
    divider.evaluate(
        """element => {
            document.dispatchEvent(new PointerEvent('pointerup', {
                bubbles: true, clientX: 480, clientY: 180,
                pointerId: 1, pointerType: 'touch',
            }));
            const touch = new Touch({
                identifier: 1, target: element, clientX: 480, clientY: 180,
            });
            window.dispatchEvent(new TouchEvent('touchend', {
                bubbles: true, cancelable: true,
                touches: [], changedTouches: [touch],
            }));
        }"""
    )
    page.wait_for_function("!document.body.hasAttribute('data-resizing')")
    panel = page.locator("#panel").bounding_box()
    assert panel is not None
    assert panel["width"] > 800
    assert int(page.evaluate("localStorage.getItem('dlightrag-panel-width')")) > 800


@pytest.mark.e2e
def test_attachment_only_does_not_enable_send(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    send = page.get_by_role("button", name="Send")
    assert send.is_disabled()
    page.locator("#attachment-input").set_input_files(
        files={
            "name": "only-image.png",
            "mimeType": "image/png",
            "buffer": b"\x89PNG\r\n\x1a\n",
        }
    )
    page.wait_for_function("document.querySelector('#thumbnail-strip')?.children.length === 1")
    assert send.is_disabled()
    page.get_by_role("textbox", name="Message").fill("  ")
    assert send.is_disabled()
    page.get_by_role("textbox", name="Message").fill("Question")
    assert send.is_enabled()


@pytest.mark.e2e
def test_offscreen_history_loads_lazy_thumbnails_and_original_only_on_lightbox(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    turn_count = 40
    conversation_id = e2e_conversation_service.seed_image_history(turn_count=turn_count)
    thumbnail_requests: list[str] = []
    original_requests: list[str] = []

    def record_image_request(request) -> None:
        path = urlparse(request.url).path
        # Uploads live with the run that accepted them, not with the conversation.
        if not path.startswith("/web/api/runs/") or "/attachments/" not in path:
            return
        if path.endswith("/thumbnail"):
            thumbnail_requests.append(path)
        else:
            original_requests.append(path)

    page.on("request", record_image_request)
    page.set_viewport_size({"width": 1440, "height": 720})
    page.goto(f"/web/conversations/{conversation_id}")
    history_buttons = page.locator(
        "#chat-messages button[aria-label^='Open Turn '][aria-label$=', attachment 1']"
    )
    history_images = history_buttons.locator("img")
    page.wait_for_function(
        "count => Array.from(document.querySelectorAll('#chat-messages button'))"
        ".filter(button => button.getAttribute('aria-label')?.startsWith('Open Turn '))"
        ".filter(button => button.getAttribute('aria-label')?.endsWith(', attachment 1')).length === count",
        arg=turn_count,
    )
    page.wait_for_timeout(500)

    assert original_requests == []
    assert 0 < len(thumbnail_requests) < turn_count
    assert (
        history_images.evaluate_all(
            "images => images.filter(image => image.naturalWidth > 0).length"
        )
        > 0
    )
    history_buttons.last.click()
    page.locator("#image-lightbox[aria-hidden='false']").wait_for()
    page.wait_for_function("document.querySelector('#image-lightbox img')?.naturalWidth > 0")

    assert len(original_requests) == 1


@pytest.mark.e2e
def test_pending_document_is_an_unsaved_draft_and_clears_on_switch(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    # Attaching a document (no text) shows a compact chip in the shared strip.
    page.locator("#attachment-input").set_input_files(
        files={
            "name": "notes.pdf",
            "mimeType": "application/pdf",
            "buffer": b"%PDF-1.4 draft document",
        }
    )
    chip = page.locator('#thumbnail-strip [data-document-attachment="true"]')
    chip.wait_for()
    assert chip.count() == 1

    # Switching conversations must treat the pending document as an unsaved draft.
    page.get_by_role("button", name="New chat").click()
    dialog = page.get_by_role("dialog", name="Discard draft?")
    dialog.wait_for()
    dialog.get_by_role("button", name="Discard and continue").click()

    # Discarding the draft clears the pending document from the strip.
    page.wait_for_function(
        "document.querySelectorAll("
        "'#thumbnail-strip [data-document-attachment=\"true\"]').length === 0"
    )
    assert chip.count() == 0
