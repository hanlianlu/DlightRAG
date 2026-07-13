# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser coverage for the Web conversation lifecycle shell."""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from playwright.sync_api import Page, Route


def _install_conversation_routes(page: Page) -> list[dict[str, str | None]]:
    now = datetime.now(UTC)
    conversations: list[dict[str, str | None]] = []

    def summary(conversation_id: str, title: str | None = None) -> dict[str, str | None]:
        offset = timedelta(seconds=len(conversations))
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
            route.fulfill(json=conversations)
            return
        if path == "/web/conversations" and method == "POST":
            item = summary(str(uuid4()))
            conversations.insert(0, item)
            route.fulfill(status=201, json=item)
            return

        parts = path.split("/")
        conversation_id = parts[3] if len(parts) > 3 else ""
        item = next(
            (row for row in conversations if row["conversation_id"] == conversation_id),
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
            conversations.remove(item)
            route.fulfill(status=204, body="")
            return
        route.abort()

    page.route("**/web/conversations**", handle)
    return conversations


@pytest.mark.e2e
def test_new_select_rename_delete_survive_reload(page: Page) -> None:
    conversations = _install_conversation_routes(page)
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
    assert all(row["conversation_id"] != first_id for row in conversations)
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
    assert page.get_by_role("textbox", name="Message").input_value() == "Unsent draft"

    page.get_by_role("button", name="New chat").click()
    dialog.get_by_role("button", name="Discard and continue").click()
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.get_by_role("textbox", name="Message").input_value() == ""
