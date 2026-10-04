# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Agent Mailbox: mail sent to an Agent's aliases, read from a bucket the deployment fills.

DlightRAG reads whole RFC 822 messages and never writes or deletes one (ADR 0034). A message is
untrusted context: anyone who learns an alias can write to it, so what is read is summarized
into the sender, the subject, the links and the codes, and every filled password is redacted
from it before anything is extracted.
"""

from __future__ import annotations

import email
import email.policy
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Protocol

from dlightrag.engine.answer.agent_browser.passwords import FilledPasswords


@dataclass(frozen=True, slots=True)
class MailObject:
    """One stored message."""

    received_at: datetime
    """The object's ``LastModified``, which the bucket sets and no sender writes."""
    raw: bytes | None
    """The whole message, or None when it was over the size the call allowed."""


@dataclass(frozen=True, slots=True)
class MailListing:
    """The messages one alias received since a time, newest first."""

    messages: tuple[MailObject, ...]
    """At most the limit asked for."""
    more: int
    """How many later messages in the window were not returned."""
    truncated: bool
    """The alias holds more stored messages than one listing reads, so it may have missed some."""


class AgentMailboxError(Exception):
    """The bucket could not be read. ``code`` is an S3 error code or a client exception's name."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AgentMailbox(Protocol):
    """The mail a deployment delivers to the addresses it mints on one domain."""

    alias_domain: str

    async def messages(
        self, address: str, *, since: datetime, limit: int, max_bytes: int
    ) -> MailListing:
        """The messages ``address`` received at or after ``since``, newest first.

        A message over ``max_bytes`` is listed with no bytes and never fetched.
        """
        ...


@dataclass(frozen=True, slots=True)
class MailSummary:
    """What a message gives the Agent: who wrote it, what it says it is, and where to go."""

    readable: bool
    sender: str = ""
    subject: str = ""
    links: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    omitted_links: int = 0
    """How many distinct links are not in ``links``: too long, or after the first four."""


_HEADER_CHARS = 200
_MAX_LINKS = 4
_MAX_LINK_CHARS = 2048
_MAX_CODES = 5
#: Trailing punctuation a sentence puts after a link without it belonging to the link.
_LINK_END = ".,;:!?"
_LINK = re.compile(r"""https?://[^\s<>"'()\[\]]+""")
_TOKEN = re.compile(r"[A-Za-z0-9._-]+")
_CODE = re.compile(r"\d{4,8}|(?=.*\d)(?=.*[A-Z])[A-Z0-9]{6,8}|[A-Z0-9]{3,4}-[A-Z0-9]{3,4}")
#: A code has between 4 and 9 characters, so a longer token, such as a password, is none.
_CODE_CHARS = range(4, 10)


def summarize_mail(raw: bytes, passwords: FilledPasswords) -> MailSummary:
    """The sender, subject, links and codes of one message, with every filled password redacted
    before anything is extracted. A message that cannot be parsed is not readable."""
    try:
        message = email.message_from_bytes(raw, policy=email.policy.default)
        sender = _header(message["From"], passwords)
        subject = _header(message["Subject"], passwords)
        part = message.get_body(preferencelist=("plain", "html"))
        content = "" if part is None else passwords.redact(part.get_content())
        if part is not None and part.get_content_subtype() == "html":
            collector = _HtmlCollector()
            collector.feed(content)
            collector.close()
            # Parsing decodes character references, which can spell a password the source did not.
            found = [passwords.redact(link) for link in collector.links]
            text = passwords.redact(" ".join(collector.text))
        else:
            found, text = _text_links(content), content
    except Exception:
        return MailSummary(readable=False)
    links: list[str] = []
    omitted = 0
    for link in dict.fromkeys(found):
        if len(link) > _MAX_LINK_CHARS or len(links) == _MAX_LINKS:
            omitted += 1
        else:
            links.append(link)
    return MailSummary(
        readable=True,
        sender=sender,
        subject=subject,
        links=tuple(links),
        codes=_codes(f"{subject} {_LINK.sub(' ', text)}"),
        omitted_links=omitted,
    )


def _header(value: object, passwords: FilledPasswords) -> str:
    """One header as a line: no control character, whitespace collapsed, a password masked."""
    line = " ".join(str(value or "").split())
    line = "".join(c for c in line if unicodedata.category(c) != "Cc")
    return passwords.redact(line)[:_HEADER_CHARS]


def _text_links(text: str) -> list[str]:
    return [_link(link.rstrip(_LINK_END)) for link in _LINK.findall(text)]


def _link(url: str) -> str:
    """A link on one line: a browser drops whitespace and control characters from a URL, and a
    message must not use them to start a line of its own in what the model reads."""
    return "".join(c for c in url if not c.isspace() and unicodedata.category(c) != "Cc")


def _codes(text: str) -> tuple[str, ...]:
    found: dict[str, None] = {}
    for token in _TOKEN.findall(text):
        token = token.strip("._-")
        if len(token) in _CODE_CHARS and _CODE.fullmatch(token):
            found.setdefault(token)
    return tuple(found)[:_MAX_CODES]


class _HtmlCollector(HTMLParser):
    """The visible text of an HTML body and its links, in the order the body gives them."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.text: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._hidden += 1
        elif tag == "a":
            href = _link(dict(attrs).get("href") or "")
            if href.lower().startswith(("http://", "https://")):
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._hidden:
            self._hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.text.append(data)
            self.links.extend(_text_links(data))


__all__ = [
    "AgentMailbox",
    "AgentMailboxError",
    "MailListing",
    "MailObject",
    "MailSummary",
    "summarize_mail",
]
