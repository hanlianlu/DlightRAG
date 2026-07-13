# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the Web conversation lifecycle shell."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from playwright.sync_api import Page, Route


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

    def handle(route: Route) -> None:
        request = route.request
        path = urlparse(request.url).path
        method = request.method
        if path == "/web/conversations" and method == "GET":
            route.fulfill(json=state.conversations)
            return
        if path == "/web/conversations" and method == "POST":
            item = summary(str(uuid4()))
            state.conversations.insert(0, item)
            route.fulfill(status=201, json=item)
            return

        parts = path.split("/")
        conversation_id = parts[3] if len(parts) > 3 else ""
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

    page.route("**/web/conversations**", handle)
    return state


def _active_id(page: Page) -> str:
    active = page.locator("[aria-current='page']")
    active.wait_for()
    conversation_id = active.get_attribute("data-conversation-id")
    assert conversation_id
    return conversation_id


def _new_conversation(page: Page) -> tuple[str, str]:
    previous_id = _active_id(page)
    page.locator("#new-conversation-btn").click()
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId !== id",
        arg=previous_id,
    )
    page.locator("#new-conversation-btn:not([disabled])").wait_for()
    return previous_id, _active_id(page)


def _add_draft_with_image(page: Page, text: str) -> None:
    page.get_by_role("textbox", name="Message").fill(text)
    page.locator("#image-input").set_input_files(
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
                const node = walker.nextNode();
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
    page.locator("#new-conversation-btn").click()
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId !== id",
        arg=initial_id,
    )
    active = page.locator("[aria-current='page']")
    first_id = active.get_attribute("data-conversation-id")
    assert first_id

    active.get_by_role("button", name="Conversation actions").click()
    page.get_by_role("menuitem", name="Rename").click()
    page.get_by_role("textbox", name="Conversation title").fill("Research notes")
    page.keyboard.press("Enter")
    page.get_by_text("Research notes", exact=True).wait_for()

    page.locator("#new-conversation-btn").click()
    page.wait_for_function(
        "id => document.querySelector('[aria-current=\"page\"]')?.dataset.conversationId !== id",
        arg=first_id,
    )
    second_id = page.locator("[aria-current='page']").get_attribute("data-conversation-id")
    assert second_id and second_id != first_id
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
    active_scope = search_scope.get_attribute("data-active")

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

    composer_edge_inset = page.locator("#query-form").evaluate(
        "form => form.querySelector('.composer-plus-icon').getBoundingClientRect().left"
        " - form.getBoundingClientRect().left"
    )
    assert 6 <= composer_edge_inset <= 12

    page.get_by_role("button", name="New chat").click()
    page.wait_for_function(
        "active => document.querySelector('#workspace-selector')?.getAttribute('data-active') === active",
        arg=active_scope,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("viewport", [(900, 800), (390, 844)])
def test_compact_drawers_are_modal_mutually_exclusive_and_restore_focus(
    page: Page, viewport: tuple[int, int]
) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    open_conversations = page.get_by_role("button", name="Open conversations")
    open_conversations.click()
    sidebar = page.locator("#chat-sidebar")
    assert sidebar.get_attribute("role") == "dialog"
    assert sidebar.get_attribute("aria-modal") == "true"
    assert page.locator("#composer").evaluate("element => element.inert") is True
    page.keyboard.press("Escape")
    page.wait_for_function("document.activeElement?.id === 'conversation-sidebar-open'")

    files = page.get_by_role("button", name="Files", exact=True)
    files.click()
    panel = page.locator("#panel")
    assert panel.get_attribute("role") == "dialog"
    assert panel.get_attribute("aria-modal") == "true"
    assert page.locator("#composer").evaluate("element => element.inert") is True
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
def test_resizing_open_files_panel_to_compact_keeps_background_inert(page: Page) -> None:
    _install_conversation_routes(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()
    page.get_by_role("button", name="Files", exact=True).click()

    page.set_viewport_size({"width": 900, "height": 800})

    assert page.locator("#panel").get_attribute("aria-modal") == "true"
    assert page.locator("#composer").evaluate("element => element.inert") is True


@pytest.mark.e2e
def test_attachment_only_does_not_enable_send(page: Page) -> None:
    _install_conversation_routes(page)
    page.goto("/web/")
    page.locator("[aria-current='page']").wait_for()

    send = page.get_by_role("button", name="Send")
    assert send.is_disabled()
    page.locator("#image-input").set_input_files(
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
