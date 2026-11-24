# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Naming the page that was refused for being automated (ADR 0032).

A bot filter answers a browser it read as automated with a page of its own, and an ordinary
read of it is a page that "looked empty" rather than a refusal. The one place that reads the
markers such a page carries lives here, so a Rendered Read and a ``browser`` call name the
same refusal in the same words, and the evidence the verdict used goes on the failure.

Only the markers are read here: nothing defeats, evades, or solves a check, and a page whose
text merely discusses one of these names is not a refusal (see ``detect_block``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

#: The markers a bot filter's refusal page carries, as the page writes them, read
#: case-insensitively. The first is Ctrip's filter (``whaleguard block``).
BLOCK_MARKERS: tuple[str, ...] = (
    "whaleguard",
    "access denied",
    "unusual traffic",
    "checking your browser",
    "just a moment",
    "enable javascript and cookies to continue",
    "网络异常",
)

#: The markers that name a human-verification check the page is running, which the Agent
#: stops on rather than reads through (ADR 0032).
VERIFICATION_MARKERS: tuple[str, ...] = (
    "verify you are human",
    "are you a robot",
    "滑动验证",
    "安全验证",
)

#: Where a marker has to be to count: the document's own title or a heading (the page's
#: verdict on the caller), a preformatted block (where Ctrip's own refusal prints itself),
#: or the text of an element the page itself marks as a captcha, challenge, or slider
#: widget (the check it is running). Ordinary running text is none of these, which is what
#: keeps an article *about* one of these checks out.
_VERDICT = re.compile(
    r"<(?:title|h1|h2|h3|pre)\b[^>]*>(?P<signal>.*?)</(?:title|h1|h2|h3|pre)>", re.I | re.S
)
_WIDGET = re.compile(
    r"<[a-z][a-z0-9]*\b(?=[^>]*\b(?:id|class|data-testid)\s*=\s*[\"'][^\"']*"
    r"(?:captcha|challenge|verify|verification|slide|slider)[^\"']*[\"'])[^>]*>"
    r"(?P<widget>.*?)</",
    re.I | re.S,
)
#: The one marker no page says by accident: a vendor's own token, which counts wherever it
#: is written. The rest are phrases a page can also use in prose, so they count only inside
#: a verdict element above.
_UNMISTAKABLE = "whaleguard"
_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")

BlockKind = Literal["blocked", "verification_required"]


@dataclass(frozen=True, slots=True)
class Block:
    """A page that was refused for being automated, with the evidence that named it."""

    reason: BlockKind
    """``blocked`` for a refusal, ``verification_required`` for a check the page is running."""
    marker: str
    """The marker the page carried, as this module reads it."""
    status: int | None
    """The HTTP status of the refusal, when the browser saw one."""


def detect_block(*, status: int | None, html: bytes) -> Block | None:
    """The refusal a page is, or None when nothing in it names one.

    A marker counts inside the document's own ``<title>``, a heading, or a preformatted
    block, or inside an element the page itself marks as a captcha, challenge, or slider
    widget. A marker in ordinary text is not a verdict: an article about ``whaleguard``
    carries the word in its body and is not a refusal, while the page a filter answers with
    carries it in its title or in the one block of text it consists of. Nothing here
    guesses, defeats, or solves a check.
    """
    for signal in _signals(html):
        for markers, reason in (
            (VERIFICATION_MARKERS, "verification_required"),
            (BLOCK_MARKERS, "blocked"),
        ):
            for marker in markers:
                if marker in signal:
                    return Block(reason=cast(BlockKind, reason), marker=marker, status=status)
    # A vendor's own token is no page's prose: the refusal Ctrip answers with is 168 bytes
    # of markup and the words ``whaleguard block``, in no element at all.
    if _UNMISTAKABLE in _plain(html.decode("utf-8", errors="replace")):
        return Block(reason="blocked", marker=_UNMISTAKABLE, status=status)
    return None


def refused_on_navigation(status: int | None, html: bytes) -> Block | None:
    """The refusal a page the browser just navigated to is, or None when it is not one.

    Only a 4xx answer can refuse: a page that answers 200 was served, whatever its text
    mentions.
    """
    if status is None or status < 400:
        return None
    return detect_block(status=status, html=html)


def _signals(html: bytes) -> list[str]:
    """The text of each element a marker counts in, tags removed and whitespace collapsed."""
    decoded = html.decode("utf-8", errors="replace")
    # Each pattern carries exactly one named group; read the match by its own group, since
    # indexing a match by the other pattern's group raises ``IndexError: no such group``.
    return [
        _plain(match[group] or "")
        for pattern, group in ((_VERDICT, "signal"), (_WIDGET, "widget"))
        for match in pattern.finditer(decoded)
    ]


def _plain(markup: str) -> str:
    return _SPACE.sub(" ", _TAG.sub(" ", markup)).casefold().strip()


__all__ = [
    "BLOCK_MARKERS",
    "VERIFICATION_MARKERS",
    "Block",
    "BlockKind",
    "detect_block",
    "refused_on_navigation",
]
