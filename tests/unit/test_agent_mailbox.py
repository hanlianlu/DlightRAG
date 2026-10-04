# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Mailbox: what a message gives the Agent, and what the S3 reader asks a bucket for."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from dlightrag.adapters.agent_mailbox import S3AgentMailbox
from dlightrag.engine.answer.agent_browser import (
    PASSWORD_MASK,
    AgentMailboxError,
    FilledPasswords,
    generate_password,
    summarize_mail,
)
from tests.support.loopback import bypass_proxies
from tests.support.s3 import StoredObject, s3_stub

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def message(
    body: str,
    *,
    subject: str = "Confirm your account",
    sender: str = "Shop <noreply@shop.example>",
    kind: str = "plain",
    extra: str = "",
) -> bytes:
    headers = (
        f"From: {sender}\r\nSubject: {subject}\r\nContent-Type: text/{kind}; charset=utf-8\r\n"
    )
    return f"{headers}{extra}\r\n{body}".encode()


def summarized(raw: bytes, passwords: FilledPasswords | None = None):
    return summarize_mail(raw, passwords or FilledPasswords())


def test_a_plain_message_gives_its_sender_subject_links_and_codes() -> None:
    raw = message(
        "Hi there,\r\nOpen https://shop.example/confirm?token=abc123. Your code: 482913.\r\n"
        "Or use https://shop.example/confirm?token=abc123 again, or https://shop.example/help!\r\n",
        subject="Code 7788 for you",
    )

    summary = summarized(raw)

    assert summary.readable and summary.sender == "Shop <noreply@shop.example>"
    assert summary.subject == "Code 7788 for you"
    # A link loses the punctuation a sentence puts after it, and is listed once.
    assert summary.links == (
        "https://shop.example/confirm?token=abc123",
        "https://shop.example/help",
    )
    # A token of a link's own path is no code; the subject's and the body's are.
    assert summary.codes == ("7788", "482913")
    assert summary.omitted_links == 0


def test_an_html_only_message_gives_the_hrefs_of_its_links_and_not_its_styles() -> None:
    raw = message(
        "<style>.a { color: #123456 }</style><p>Hi <a href='https://shop.example/verify?a=1&amp;b=2'>"
        "Verify</a>, or <a href='mailto:x@shop.example'>write</a> to us. Code <b>A1B2C3</b>."
        "<script>var pin = 999999;</script></p>",
        kind="html",
    )

    summary = summarized(raw)

    assert summary.links == ("https://shop.example/verify?a=1&b=2",)
    assert summary.codes == ("A1B2C3",)


def test_a_link_stays_on_one_line_whatever_a_message_puts_inside_it() -> None:
    raw = message(
        "<a href='https://shop.example/a\n   codes: 000000\x07\tb'>go</a> "
        "and https://shop.example/c\x07d",
        kind="html",
    )

    summary = summarized(raw)

    assert summary.links == ("https://shop.example/acodes:000000b", "https://shop.example/cd")
    assert all("\n" not in link and "\x07" not in link for link in summary.links)


def test_a_message_lists_at_most_four_links_and_counts_the_ones_it_leaves_out() -> None:
    long_link = "https://shop.example/" + "x" * 2100
    links = [f"https://shop.example/page{n}" for n in range(1, 7)]

    summary = summarized(message(" ".join([long_link, *links])))

    assert summary.links == tuple(links[:4])
    assert summary.omitted_links == 3


def test_codes_are_the_tokens_that_look_like_one_and_never_more_than_five() -> None:
    body = (
        "pin 12 or 123456789, 482913 and ABC-123, K9F2X7 then AB12CD34 ok 111111 222222 333333 done"
    )

    codes = summarized(message(body)).codes

    assert codes == ("482913", "ABC-123", "K9F2X7", "AB12CD34", "111111")


def test_a_message_that_cannot_be_decoded_is_not_readable() -> None:
    raw = (
        b"From: a@x.example\r\nSubject: hi\r\nContent-Type: text/plain; charset=no-such-charset\r\n"
        b"\r\nhello https://shop.example/\r\n"
    )

    assert summarized(raw).readable is False


def test_a_header_is_one_line_without_control_characters_and_cut_at_200() -> None:
    raw = message("x", subject="Hello\r\n\tthere\x07 friend " + "y" * 300)

    summary = summarized(raw)

    assert summary.subject.startswith("Hello there friend yyyy") and len(summary.subject) == 200
    assert "\x07" not in summary.subject and "\n" not in summary.subject


def test_a_filled_password_a_message_echoes_is_masked_before_anything_is_taken_from_it() -> None:
    password = generate_password()
    passwords = FilledPasswords()
    passwords.add(password)
    value = password.get_secret_value()
    raw = message(
        f"Your new password is {value}. Sign in at https://shop.example/login?password={value}\r\n",
        subject=f"Welcome {value}",
    )

    summary = summarized(raw, passwords)

    assert summary.subject == f"Welcome {PASSWORD_MASK}"
    assert summary.links == (f"https://shop.example/login?password={PASSWORD_MASK}",)
    # Nothing it holds carries the password, and no part of it comes back as a code.
    texts = (summary.sender, summary.subject, *summary.links, *summary.codes)
    assert (sum(value in text for text in texts), len(summary.codes)) == (0, 0)


def test_a_password_an_html_message_spells_with_character_references_is_masked_too() -> None:
    passwords = FilledPasswords()
    passwords.add(SecretStr("Zq7-Fixture.Pass*9x"))
    # The source spells the asterisk as a reference, so only the parsed link can be masked.
    raw = message(
        "<p><a href='https://shop.example/login?password=Zq7-Fixture.Pass&#42;9x'>Sign in</a></p>",
        kind="html",
    )

    summary = summarized(raw, passwords)

    assert summary.links == (f"https://shop.example/login?password={PASSWORD_MASK}",)


# -- the S3 reader --------------------------------------------------------------------------


def stored(minutes_ago: float, body: bytes = b"mail") -> StoredObject:
    return StoredObject(body, NOW - timedelta(minutes=minutes_ago))


def mailbox(endpoint: str, *, prefix: str = "mail") -> S3AgentMailbox:
    return S3AgentMailbox(
        alias_domain="orliantra.cc",
        bucket="mailbox",
        prefix=prefix,
        endpoint=endpoint,
        region="auto",
        access_key_id="fixture-key",
        secret_access_key="fixture-secret",
    )


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    bypass_proxies(monkeypatch)


async def test_an_alias_is_read_from_its_own_prefix_newest_first_since_the_window() -> None:
    alias, other = "a1@orliantra.cc", "b2@orliantra.cc"
    objects = {
        f"mail/{alias}/old.eml": stored(60, b"too old"),
        f"mail/{alias}/first.eml": stored(10, b"first"),
        f"mail/{alias}/second.eml": stored(5, b"second"),
        f"mail/{alias}/third.eml": stored(1, b"third"),
        f"mail/{other}/other.eml": stored(1, b"someone else's"),
    }
    async with s3_stub(objects) as stub:
        mail = mailbox(stub.endpoint)

        listing = await mail.messages(
            alias, since=NOW - timedelta(minutes=20), limit=2, max_bytes=1000
        )

        assert [(m.received_at, m.raw) for m in listing.messages] == [
            (NOW - timedelta(minutes=1), b"third"),
            (NOW - timedelta(minutes=5), b"second"),
        ]
        assert (listing.more, listing.truncated) == (1, False)
        # It listed one alias's folder, read only what it shows, and touched nothing else.
        assert stub.listed == [f"mail/{alias}/"]
        assert sorted(stub.fetched) == [f"mail/{alias}/second.eml", f"mail/{alias}/third.eml"]


async def test_a_message_over_the_size_asked_is_listed_and_never_fetched() -> None:
    alias = "a1@orliantra.cc"
    objects = {f"mail/{alias}/big.eml": stored(1, b"x" * 100), f"mail/{alias}/small.eml": stored(2)}
    async with s3_stub(objects) as stub:
        listing = await mailbox(stub.endpoint).messages(
            alias, since=NOW - timedelta(hours=1), limit=5, max_bytes=50
        )

        assert [m.raw for m in listing.messages] == [None, b"mail"]
        assert stub.fetched == [f"mail/{alias}/small.eml"]


async def test_an_empty_prefix_puts_the_alias_folders_at_the_root() -> None:
    alias = "a1@orliantra.cc"
    async with s3_stub({f"{alias}/one.eml": stored(1)}) as stub:
        listing = await mailbox(stub.endpoint, prefix="").messages(
            alias, since=NOW - timedelta(hours=1), limit=5, max_bytes=1000
        )

        assert [m.raw for m in listing.messages] == [b"mail"] and stub.listed == [f"{alias}/"]


async def test_an_alias_with_more_stored_messages_than_a_listing_reads_is_reported_truncated() -> (
    None
):
    alias = "a1@orliantra.cc"
    objects = {f"mail/{alias}/{n:03}.eml": stored(500 - n) for n in range(25)}
    # Two keys a page, so ten pages read the first twenty.
    async with s3_stub(objects, page_keys=2) as stub:
        listing = await mailbox(stub.endpoint).messages(
            alias, since=NOW - timedelta(days=1), limit=5, max_bytes=1000
        )

        assert listing.truncated is True
        assert (len(stub.listed), len(listing.messages), listing.more) == (10, 5, 15)


@pytest.mark.parametrize("failure", ["denied", "unreachable"])
async def test_a_bucket_that_cannot_be_read_says_why_by_its_code_and_nothing_else(
    failure: str,
) -> None:
    alias = "a1@orliantra.cc"
    async with s3_stub({}, denied=True) as stub:
        endpoint = stub.endpoint if failure == "denied" else "http://127.0.0.1:9"

        with pytest.raises(AgentMailboxError) as raised:
            await mailbox(endpoint).messages(
                alias, since=NOW - timedelta(hours=1), limit=5, max_bytes=1000
            )

        expected = "AccessDenied" if failure == "denied" else "EndpointConnectionError"
        assert raised.value.code == expected
        # Neither the endpoint nor a key is anywhere in the error, which cuts its own chain.
        text = f"{raised.value!r} {raised.value.__cause__!r} {raised.value.__context__!r}"
        assert not any(secret in text for secret in ("127.0.0.1", "fixture-key", "fixture-secret"))
