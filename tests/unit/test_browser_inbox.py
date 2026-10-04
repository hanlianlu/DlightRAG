# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the ``inbox`` action asks the Agent Mailbox for, and how it reports what it gets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from pydantic import SecretStr

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.answer.agent_browser import (
    AgentMailboxError,
    MailListing,
    MailObject,
    RunAgentAccounts,
    StoredAgentAccount,
    owner_alias,
)
from dlightrag.engine.answer.tools.browser import browser_tool
from tests.support.agent_browser import inert_browser_host
from tests.tool_helpers import recording_tool_runtime

DOMAIN = "orliantra.cc"
OWNER = "owner"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def alias_of(site: str) -> str:
    """The mailbox alias the owner has for ``site``."""
    return owner_alias(OWNER, site, DOMAIN)


def account(site: str, email: str) -> StoredAgentAccount:
    return StoredAgentAccount(OWNER, site, "account", email, None, "key", "envelope")


def minted(site: str) -> StoredAgentAccount:
    """The owner's account on ``site`` as a registration with a mailbox leaves it."""
    return account(site, alias_of(site))


class ScriptedMailbox:
    """An ``AgentMailbox`` that answers each address with the listing it was given."""

    alias_domain = DOMAIN

    def __init__(self, listings: dict[str, MailListing] | AgentMailboxError) -> None:
        self.listings = listings
        self.asked: list[tuple[str, datetime, int, int]] = []

    async def messages(
        self, address: str, *, since: datetime, limit: int, max_bytes: int
    ) -> MailListing:
        self.asked.append((address, since, limit, max_bytes))
        if isinstance(self.listings, AgentMailboxError):
            raise self.listings
        return self.listings[address]


def text_mail(subject: str, body: str = "") -> bytes:
    return f"From: Shop <noreply@shop.example>\r\nSubject: {subject}\r\n\r\n{body}".encode()


def stored(minutes_ago: int, raw: bytes | None) -> MailObject:
    return MailObject(NOW - timedelta(minutes=minutes_ago), raw)


class Inbox:
    """The browser tool of a Run with an Agent Mailbox, and what a Session did in it."""

    def __init__(self, mailbox: ScriptedMailbox) -> None:
        host = inert_browser_host(accounts=True, mailbox=mailbox)
        self.browser = host.browser
        self.accounts = cast(RunAgentAccounts, host.accounts)
        self.tool = browser_tool(
            host,
            environment=None,
            scheduler=AccessScheduler(),
            spill=None,
            image_preparer=None,
            child=False,
        )
        self.subjects: list[str] = []

    def signed_in(self, *accounts: StoredAgentAccount, scope: str = "parent") -> None:
        for each in accounts:
            self.accounts.session(scope, child=False).signed_in(each)

    async def call(self, scope: str = "parent") -> ToolResult:
        updates: list[ToolResult] = []
        runtime = recording_tool_runtime(updates, tool_name="browser", execution_scope=scope)
        result = await self.tool.execute(
            self.tool.input_model.model_validate({"action": "inbox"}), runtime
        )
        self.subjects = [update.subject for update in updates if update.subject]
        return result


async def test_inbox_asks_each_alias_for_its_newest_mail_a_little_before_the_window() -> None:
    first, second = alias_of("a.example"), alias_of("b.example")
    mailbox = ScriptedMailbox({first: MailListing((), 0, False), second: MailListing((), 0, False)})
    inbox = Inbox(mailbox)
    inbox.signed_in(minted("a.example"), minted("b.example"))
    window = inbox.accounts.session("parent", child=False).inbox_window()
    assert window is not None

    empty = await inbox.call()

    assert [(address, limit, size) for address, _, limit, size in mailbox.asked] == [
        (first, 5, 1 << 20),
        (second, 5, 1 << 20),
    ]
    # The bucket stamps mail with its own clock, so the window opens two minutes early.
    assert {since for _, since, _, _ in mailbox.asked} == {window.since - timedelta(seconds=120)}
    assert not empty.is_error and empty.text_content.startswith(
        f"No mail has arrived for {first}, {second} since "
    )
    assert inbox.subjects == [f"{first}, {second}"]


async def test_inbox_merges_the_aliases_newest_first_and_says_what_it_does_not_show() -> None:
    first, second = alias_of("a.example"), alias_of("b.example")
    long_link = "https://shop.example/" + "z" * 2100
    mailbox = ScriptedMailbox(
        {
            first: MailListing(
                (
                    stored(1, text_mail("Newest 482913", f"Go https://shop.example/a {long_link}")),
                    stored(5, None),
                    stored(9, b"From: a@x\r\nContent-Type: text/plain; charset=nope\r\n\r\nhi"),
                ),
                more=4,
                truncated=True,
            ),
            second: MailListing(
                (stored(2, text_mail("Middle")), stored(7, text_mail("Oldest"))),
                more=0,
                truncated=False,
            ),
        }
    )
    inbox = Inbox(mailbox)
    inbox.signed_in(minted("a.example"), minted("b.example"))

    result = await inbox.call()

    lines = result.text_content.splitlines()
    assert lines[0].startswith(f"[browser: inbox | 5 message(s) for {first}, {second} since ")
    assert lines[2:] == [
        f"1. 2026-10-04T11:59:00Z · to {first} · from Shop <noreply@shop.example>",
        "   subject: Newest 482913",
        "   link: https://shop.example/a",
        "   (1 longer link(s) omitted)",
        "   codes: 482913",
        f"2. 2026-10-04T11:58:00Z · to {second} · from Shop <noreply@shop.example>",
        "   subject: Middle",
        f"3. 2026-10-04T11:55:00Z · to {first} · a message over 1 MiB, not read",
        f"4. 2026-10-04T11:53:00Z · to {second} · from Shop <noreply@shop.example>",
        "   subject: Oldest",
        f"5. 2026-10-04T11:51:00Z · to {first} · a message that could not be read",
        "4 more message(s) in this window are not shown.",
        f"{first} holds over 10,000 stored messages; only the first 10,000 were listed.",
    ]
    assert not result.is_error and result.effects.evidence_sources == ()


async def test_inbox_shows_the_newest_five_of_what_the_aliases_returned() -> None:
    first, second = alias_of("a.example"), alias_of("b.example")
    mailbox = ScriptedMailbox(
        {
            alias: MailListing(
                tuple(stored(10 * n + shift, text_mail(f"{alias} {n}")) for n in range(1, 5)),
                more=2,
                truncated=False,
            )
            for alias, shift in ((first, 0), (second, 5))
        }
    )
    inbox = Inbox(mailbox)
    inbox.signed_in(minted("a.example"), minted("b.example"))

    result = await inbox.call()

    subjects = [line.strip() for line in result.text_content.splitlines() if "subject:" in line]
    assert subjects == [
        f"subject: {first} 1",
        f"subject: {second} 1",
        f"subject: {first} 2",
        f"subject: {second} 2",
        f"subject: {first} 3",
    ]
    # Three the merge dropped, and two each that the aliases themselves did not return.
    assert result.text_content.splitlines()[-1] == "7 more message(s) in this window are not shown."


async def test_inbox_needs_a_sign_in_and_an_alias_and_reports_a_bucket_it_cannot_read() -> None:
    alias = alias_of("a.example")
    mailbox = ScriptedMailbox(AgentMailboxError("AccessDenied"))
    inbox = Inbox(mailbox)

    unopened = await inbox.call()
    inbox.signed_in(account("typed.example", f"info@{DOMAIN}"))
    unaliased = await inbox.call()
    inbox.signed_in(minted("a.example"))
    unreadable = await inbox.call()

    assert [r.is_error for r in (unopened, unaliased, unreadable)] == [True] * 3
    assert unopened.text_content.startswith("inbox shows mail only after register or login")
    assert unaliased.text_content.startswith("The accounts this Agent Session used have no Agent")
    assert unreadable.text_content == "The Agent Mailbox could not be read (AccessDenied)."
    # Only the call that had an alias asked the bucket anything.
    assert [address for address, *_ in mailbox.asked] == [alias]


async def test_a_session_reads_only_its_own_window() -> None:
    alias = alias_of("a.example")
    mailbox = ScriptedMailbox({alias: MailListing((), 0, False)})
    inbox = Inbox(mailbox)
    inbox.signed_in(minted("a.example"), scope="child")

    parents = await inbox.call("parent")
    childs = await inbox.call("child")

    assert parents.is_error and not childs.is_error
    assert [address for address, *_ in mailbox.asked] == [alias]


async def test_a_password_a_mail_echoes_never_reaches_the_inbox() -> None:
    alias = alias_of("a.example")
    password = SecretStr("Zq7-Fixture.Pass_9x")
    mailbox = ScriptedMailbox(
        {
            alias: MailListing(
                (stored(1, text_mail(f"Hello {password.get_secret_value()}")),), 0, False
            )
        }
    )
    inbox = Inbox(mailbox)
    inbox.signed_in(minted("a.example"))
    inbox.browser.filled_passwords("parent").add(password)

    result = await inbox.call()

    assert password.get_secret_value() not in result.text_content
    assert "subject: Hello ********" in result.text_content
