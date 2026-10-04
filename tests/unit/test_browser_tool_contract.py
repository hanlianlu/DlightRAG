# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a model is shown of the ``browser`` tool, and what its arguments accept."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.session.plan import AgentToolPlan
from dlightrag.engine.answer.tools.browser import (
    BROWSER_ACTIONS,
    BrowserArgs,
    BrowserRequest,
    browser_declaration,
)


def schema(*, upload: bool = True) -> dict[str, Any]:
    return browser_declaration(upload=upload).definition.parameters


def parse(arguments: dict[str, Any], *, upload: bool = True) -> BrowserRequest:
    model = browser_declaration(upload=upload).input_model
    return BrowserRequest.from_args(cast(BrowserArgs, model.model_validate(arguments)))


def test_upload_is_offered_only_with_a_workspace() -> None:
    without, with_workspace = schema(upload=False), schema(upload=True)

    assert without["properties"]["action"]["enum"] == [
        action for action in BROWSER_ACTIONS if action != "upload"
    ]
    assert len(BROWSER_ACTIONS) == 13
    assert "files" not in without["properties"] and "files" in with_workspace["properties"]
    assert with_workspace["properties"]["action"]["enum"] == list(BROWSER_ACTIONS)
    assert "upload (ref, files)" not in without["properties"]["action"]["description"]
    assert "upload (ref, files)" in with_workspace["properties"]["action"]["description"]
    digests = {browser_declaration(upload=flag).input_schema_digest for flag in (False, True)}
    assert len(digests) == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"action": "navigate"}, "browser navigate requires url"),
        ({"action": "navigate", "url": "http://a.example/", "ref": "e1"}, "does not take ref"),
        ({"action": "click", "ref": "e1", "url": "http://a.example/"}, "does not take url"),
        ({"action": "click"}, "browser click requires ref"),
        ({"action": "type", "ref": "e1"}, "browser type requires text"),
        ({"action": "select", "ref": "e1", "values": []}, "at least 1 item"),
        ({"action": "wait", "text": "x", "seconds": 1}, "exactly one of text, text_gone"),
        ({"action": "wait"}, "exactly one of text, text_gone"),
        ({"action": "wait", "text": "   "}, "wait text cannot be blank"),
        ({"action": "wait", "seconds": 31}, "less than or equal to 30"),
        ({"action": "wait", "seconds": 0}, "greater than 0"),
        ({"action": "click", "ref": "button"}, "String should match pattern"),
        ({"action": "click", "ref": "e1; drop"}, "String should match pattern"),
        ({"action": "press", "key": "a b"}, "String should match pattern"),
        ({"action": "press"}, "browser press requires key"),
        ({"action": "scroll", "direction": "sideways"}, "Input should be 'up' or 'down'"),
        ({"action": "snapshot", "submit": True}, "does not take submit"),
        ({"action": "upload", "ref": "e1", "files": []}, "at least 1 item"),
        ({"action": "upload", "ref": "e1"}, "browser upload requires files"),
        ({"action": "teleport"}, "Input should be"),
        ({"action": "snapshot", "extra": 1}, "Extra inputs are not permitted"),
    ],
)
def test_each_action_takes_only_its_own_arguments_and_says_what_is_wrong(
    arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        parse(arguments)


def test_a_narrower_tool_refuses_what_only_a_wider_one_offers() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        parse({"action": "upload", "ref": "e1", "files": ["a.txt"]}, upload=False)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse({"action": "click", "ref": "e1", "files": ["a.txt"]}, upload=False)


def test_typed_text_keeps_its_spaces_and_can_clear_a_field_while_a_query_is_trimmed() -> None:
    assert parse({"action": "type", "ref": "e1", "text": "  a b  "}).text == "  a b  "
    assert parse({"action": "type", "ref": "f2e12", "text": ""}).text == ""
    assert parse({"action": "find", "query": "  Search  "}).query == "Search"
    assert parse({"action": "navigate", "url": "  http://a.example/  "}).url == "http://a.example/"


def test_a_request_carries_the_defaults_its_actions_assume() -> None:
    request = parse({"action": "scroll"})
    assert (request.direction, request.ref, request.submit, request.full_page) == (
        "down",
        None,
        False,
        False,
    )
    typed = parse({"action": "type", "ref": "e3", "text": "lamp", "submit": True})
    assert (typed.ref, typed.text, typed.submit) == ("e3", "lamp", True)
    chosen = parse({"action": "select", "ref": "e4", "values": ["a", "b"]})
    assert chosen.values == ("a", "b")
    assert parse({"action": "upload", "ref": "e1", "files": ["a.txt"]}).files == ("a.txt",)


def test_the_declaration_pins_like_every_tool_and_never_replays() -> None:
    declared = browser_declaration(upload=True)

    assert (declared.name, declared.replay_policy) == ("browser", "never")
    assert (declared.read_only, declared.contract_version) == (False, 1)
    assert AgentToolPlan.from_tool(declared) == AgentToolPlan.from_tool(
        browser_declaration(upload=True)
    )


def test_the_description_teaches_the_tiers_and_the_captcha_boundary() -> None:
    description = browser_declaration(upload=False).description

    assert "rendered=true" in description
    assert "use browser only for interaction" in description
    assert "CAPTCHA" in description and "never solve" in description
    # Nothing a deployment configures can change what a Run is pinned to.
    assert not any(character.isdigit() for character in description.replace("[ref=eN]", ""))
