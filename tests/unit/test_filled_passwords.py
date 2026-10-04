# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The passwords filled into an Agent Session's pages, and how they are kept out of every text."""

from pydantic import SecretStr

from dlightrag.engine.answer.agent_browser import PASSWORD_MASK, FilledPasswords

# A fixture, not a generated password: this test is about the primitive's own behaviour.
FIRST = SecretStr("Fixture-Pass_123*")
SECOND = SecretStr("Another.Fixture-9*x")


def test_every_filled_password_is_replaced_by_the_mask_in_text_and_in_bytes() -> None:
    passwords = FilledPasswords()
    passwords.add(FIRST)
    passwords.add(SECOND)
    first, second = FIRST.get_secret_value(), SECOND.get_secret_value()
    text = f'textbox "Password": {first} / https://a.example/?p={second}&q={first}'

    redacted = passwords.redact(text)
    as_bytes = passwords.redact_bytes(f"<input value='{first}'>{second}".encode())

    assert (
        redacted
        == f'textbox "Password": {PASSWORD_MASK} / https://a.example/?p={PASSWORD_MASK}&q={PASSWORD_MASK}'
    )
    assert as_bytes == f"<input value='{PASSWORD_MASK}'>{PASSWORD_MASK}".encode()
    assert passwords.found_in(text) and not passwords.found_in(redacted)
    assert not passwords.found_in("a prefix of Fixture-Pass_12")


def test_an_empty_value_is_nothing_to_hide() -> None:
    # A cleanup fills a field back to empty, and redacting that would split every text.
    passwords = FilledPasswords()
    passwords.add(SecretStr(""))

    assert not passwords
    assert passwords.redact("a page") == "a page" and not passwords.found_in("a page")


def test_a_set_shows_how_many_passwords_it_holds_and_never_one() -> None:
    passwords = FilledPasswords()
    assert repr(passwords) == "FilledPasswords(0 filled)" and not passwords

    passwords.add(FIRST)

    assert repr(passwords) == "FilledPasswords(1 filled)" and bool(passwords)
    assert FIRST.get_secret_value() not in f"{passwords!r} {passwords}"
