# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a model is shown of the ``browser`` tool, and what its arguments accept."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from dlightrag.engine.answer.tools.browser import browser_declaration


def parse(
    arguments: dict[str, Any],
    *,
    upload: bool = True,
    accounts: bool = True,
    mailbox: bool = True,
) -> Any:
    """The arguments as the tool receives them, after the declared schema has validated them."""
    declaration = browser_declaration(upload=upload, accounts=accounts, mailbox=mailbox)
    model: type[BaseModel] = declaration.input_model
    return model.model_validate(arguments)


def properties(*, upload: bool, accounts: bool, mailbox: bool = False) -> dict[str, Any]:
    declaration = browser_declaration(upload=upload, accounts=accounts, mailbox=mailbox)
    return declaration.definition.parameters["properties"]


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
        ({"action": "register"}, "browser register requires password_refs"),
        ({"action": "register", "password_refs": []}, "at least 1 item"),
        ({"action": "register", "password_refs": ["e1", "e2", "e3"]}, "at most 2 items"),
        ({"action": "register", "password_refs": ["password"]}, "String should match pattern"),
        (
            {"action": "register", "password_refs": ["e1"], "url": "http://a.example/"},
            "does not take url",
        ),
        ({"action": "login"}, "browser login takes email_ref, username_ref, or password_refs"),
        ({"action": "login", "email_ref": "e1", "text": "x"}, "does not take text"),
        ({"action": "click", "ref": "e1", "email_ref": "e2"}, "does not take email_ref"),
        ({"action": "snapshot", "password_refs": ["e1"]}, "does not take password_refs"),
        ({"action": "inbox", "url": "http://a.example/"}, "does not take url"),
        ({"action": "inbox", "ref": "e1"}, "does not take ref"),
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


@pytest.mark.parametrize("upload", [False, True])
def test_agent_accounts_add_their_fields_and_their_action_lines_to_the_schema(
    upload: bool,
) -> None:
    plain, accounting = (properties(upload=upload, accounts=on) for on in (False, True))

    assert set(accounting) - set(plain) == {"password_refs", "email_ref", "username_ref"}
    assert (
        "register (password_refs, email_ref, username_ref)" in accounting["action"]["description"]
    )
    assert "register" not in plain["action"]["description"]


def test_an_agent_mailbox_adds_the_inbox_line_and_no_field() -> None:
    plain, mailing = (properties(upload=True, accounts=True, mailbox=on) for on in (False, True))

    assert set(mailing) == set(plain)
    assert "inbox: mail to this session's aliases" in mailing["action"]["description"]
    assert "inbox" not in plain["action"]["description"]


def test_a_registration_names_the_fields_it_fills_and_a_login_may_name_any_of_them() -> None:
    registered = parse(
        {
            "action": "register",
            "password_refs": ["e4", "f1e5"],
            "email_ref": "e2",
            "username_ref": "e3",
        }
    )
    assert (registered.password_refs, registered.email_ref, registered.username_ref) == (
        ["e4", "f1e5"],
        "e2",
        "e3",
    )
    for field, value in (("email_ref", "e1"), ("username_ref", "e1"), ("password_refs", ["e1"])):
        assert parse({"action": "login", field: value}).action == "login"


def test_typed_text_keeps_its_spaces_and_can_clear_a_field_while_a_query_is_trimmed() -> None:
    assert parse({"action": "type", "ref": "e1", "text": "  a b  "}).text == "  a b  "
    assert parse({"action": "type", "ref": "f2e12", "text": ""}).text == ""
    assert parse({"action": "find", "query": "  Search  "}).query == "Search"
    assert parse({"action": "navigate", "url": "  http://a.example/  "}).url == "http://a.example/"


def test_the_description_teaches_the_tiers_and_the_captcha_boundary() -> None:
    description = browser_declaration(upload=False, accounts=False, mailbox=False).description

    assert "rendered=true" in description
    assert "use browser only for interaction" in description
    assert "CAPTCHA" in description and "never solve" in description
    # Nothing a deployment configures can change what a Run is pinned to.
    assert not any(character.isdigit() for character in description.replace("[ref=eN]", ""))


def test_a_run_with_agent_accounts_is_told_whose_identity_it_acts_as_and_who_makes_passwords() -> (
    None
):
    plain = browser_declaration(upload=False, accounts=False, mailbox=False).description
    accounting = browser_declaration(upload=False, accounts=True, mailbox=True).description

    assert accounting.startswith(plain) and len(accounting) > len(plain)
    assert "never type the owner's email address, name, password" in accounting
    assert "you never see or type one" in accounting
    assert "A Child Session's registration lasts only for this Run" in accounting
    assert "register and login" not in plain
    assert not any(character.isdigit() for character in accounting.replace("[ref=eN]", ""))
