# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Actual reference occurrences and target kinds shared by Answer consumers.

Markdown syntax belongs to ``answer_markdown``. These helpers inspect its tokens;
they never locate or rewrite matching text in the source. Classification names a
target's kind, not permission to publish it, fetch it, or display it.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urlsplit

from markdown_it.token import Token

from dlightrag.engine.answer.markdown import answer_markdown

TargetKind = Literal["artifact", "evidence", "external", "embedded", "unsupported"]
_MARKDOWN = answer_markdown()
_RASTER_DATA = re.compile(
    r"data:image/(?:gif|jpeg|jpg|png|webp);base64,[a-z0-9+/=]+", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class Reference:
    target: str
    label: str
    image: bool


@dataclass(frozen=True, slots=True)
class InlineReference(Reference):
    """One reference's half-open slice of an inline token's children."""

    start: int
    end: int
    link_span: tuple[int, int] | None = None


def inline_references(children: Sequence[Token]) -> Iterator[InlineReference]:
    """Visit actual links and images, identifying images inside link labels.

    A linked image remains a real image, but its placement must preserve the
    containing link's interaction. Image alt text is never a second set of
    links. CommonMark prohibits nested links, so the next close belongs to the
    current opening token.
    """
    index = 0
    while index < len(children):
        token = children[index]
        if token.type == "image":
            yield InlineReference(
                target=str(token.attrGet("src") or ""),
                label=token.content,
                image=True,
                start=index,
                end=index + 1,
            )
        elif token.type == "link_open":
            end = index + 1
            while end < len(children) and children[end].type != "link_close":
                end += 1
            if end < len(children):
                yield InlineReference(
                    target=str(token.attrGet("href") or ""),
                    label="".join(child.content for child in children[index + 1 : end]),
                    image=False,
                    start=index,
                    end=end + 1,
                )
                for child_index in range(index + 1, end):
                    child = children[child_index]
                    if child.type == "image":
                        yield InlineReference(
                            target=str(child.attrGet("src") or ""),
                            label=child.content,
                            image=True,
                            start=child_index,
                            end=child_index + 1,
                            link_span=(index, end + 1),
                        )
                index = end
        index += 1


def markdown_references(text: str) -> tuple[Reference, ...]:
    """Return actual link/image occurrences, including repeats, in reading order."""
    return tuple(
        Reference(reference.target, reference.label, reference.image)
        for block in _MARKDOWN.parse(text)
        if block.type == "inline"
        for reference in inline_references(block.children or ())
    )


class _HTMLReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[Reference] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTMLParser keeps comments, text and raw script/style content out of
        # this callback, and decodes character references in attribute values.
        for name, value in attrs:
            if name in {"href", "src"} and value is not None:
                self.references.append(Reference(value, "", tag == "img" and name == "src"))


def html_references(text: str) -> tuple[Reference, ...]:
    """Read HTML URL attributes without interpreting source text as markup."""
    parser = _HTMLReferences()
    parser.feed(text)
    parser.close()
    return tuple(parser.references)


def _absolute_http(target: str) -> bool:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in target
    ):
        return False
    try:
        parsed = urlsplit(target)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return False
        if "\\" in parsed.netloc:
            return False
        # Accessing port also validates malformed or out-of-range ports.
        _ = parsed.port
        return True
    except ValueError:
        return False


def classify_target(target: str, *, image: bool = False) -> TargetKind:
    """Classify parsed destinations; every unresolved local address is unsupported."""
    scheme, separator, payload = target.partition(":")
    scheme = scheme.lower()
    if separator and scheme in {"artifact", "evidence"}:
        return "artifact" if scheme == "artifact" else "evidence"
    if _absolute_http(target):
        return "external"
    if image and (
        _RASTER_DATA.fullmatch(target)
        or scheme == "blob"
        and separator
        and payload
        and _absolute_http(payload)
    ):
        return "embedded"
    return "unsupported"


def resolve_artifact_target(target: str, bindings: Mapping[str, str]) -> str | None:
    """Resolve a settled binding or an explicit id; callers check the descriptor."""
    if target in bindings:
        return bindings[target]
    scheme, separator, payload = target.partition(":")
    return payload if separator and scheme.lower() == "artifact" else None
