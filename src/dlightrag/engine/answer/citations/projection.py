# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project validated citations onto the public URL of their source.

A published Markdown Artifact leaves this application, where an internal marker
means nothing: ``[9-1]`` names one admitted excerpt of one Run's evidence. Only
*after* the Citation Processor has validated every marker, a marker whose source
has a public HTTP(S) URL is rewritten as a self-describing link, so the file
stays citable anywhere it is copied to.

Deliberately narrow:

- public HTTP(S) sources only -- private corpus paths and Resource Handles keep
  their marker, because they have no destination outside the application, and
  the in-app Source panel stays the way to reach them;
- the marker itself is the link text (``[9-1](url "Title")``), so the published
  prose reads exactly as the model wrote it while the URL travels with it, and
  the source title rides along as the link's title (visible on hover, and in the
  raw file). A URL cannot address one excerpt, so the link lands the reader on
  the source page while the text still says which excerpt supported the claim,
  and the in-app badge -> Source panel keeps the exact-excerpt view;
- never a new citation dialect: the model still writes ``[n]``/``[n-m]`` and the
  citation index stays the single authority. This is a rendering projection over
  already-validated sources, not a second parser.

Spans whose text is data are masked before substitution (see ``_MASKED_SPANS``)
so a marker inside code or inside an existing link never gains a link of its own.
Two limitations are deliberate and safe in the direction they fail: a citation
inside a masked span keeps its marker (it is simply not projected), and a text
reference link (``[a][9]``) is not masked because two adjacent citations
(``[9-1][10-1]``) are lexically identical to it.
"""

import re
from collections.abc import Iterable
from typing import Final

from dlightrag.engine.public_http import validate_public_web_url

from .contracts import CITATION_TRAILING_BOUNDARY, SourceReference

# Document-level ``[n]`` and excerpt-level ``[n-m]`` markers as the model writes
# them. The closing bracket keeps ``[9-1]`` and ``[9abc]`` out of the doc-level
# form; the trailing boundary is the Citation Contract's own, so ``[9]x`` is
# never a marker. The negative lookbehind skips the marker this module produces
# inside ``[[9-1] Title](url)``, which makes the projection idempotent.
# Non-numeric references (``att-1``) are not matched: they carry no public URL.
_MARKER: Final = re.compile(rf"(?<!\[)\[([0-9]+)(?:-([0-9]+))?\]{CITATION_TRAILING_BOUNDARY}")

_TITLE_LIMIT: Final = 80
# A link title is a quoted string, so only its delimiter, its escape character,
# and the table-cell separator need escaping. A pipe matters most: a citation
# inside a table cell would otherwise split the row.
_TITLE_ESCAPE: Final = str.maketrans({"\\": r"\\", '"': r"\"", "|": r"\|"})

# Spans whose text is data, never a citation: code keeps its array indices, and
# an existing link keeps its label and destination. A marker rewritten inside
# one of these would nest links (which Markdown cannot represent) or truncate an
# href at a balanced-paren parse, so they are masked before substitution and
# restored afterwards. Masking the destination also keeps the projection
# idempotent for a URL that itself contains a bracket-digit run.
_FENCED_CODE: Final = re.compile(r"(?:```|~~~).*?(?:```|~~~|$)", re.DOTALL)
# Indented code needs its preceding blank line; without that check, a nested
# list's continuation text would be masked and lose its links.
_INDENTED_CODE: Final = re.compile(r"(?<=\n\n)(?:(?: {4}|\t)[^\n]*(?:\n|$))+")
# Backtick runs of the same width open and close the span.
_INLINE_CODE: Final = re.compile(r"(`+)(?:[^`]|`(?!\1))*?\1")
# A link label may itself contain balanced brackets (CommonMark counts depth), so
# ``[see [9]](url)`` is one link whose label holds a marker-shaped run.
_LINK_LABEL: Final = r"\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\]"
_LINK_OR_IMAGE: Final = re.compile(
    rf"!?{_LINK_LABEL}\(\s*(?:<[^>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)"
    rf"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\n()]*\)))?\s*\)"
)
# Only the image reference form is masked: ``[9-1][10-1]`` -- two adjacent
# citations, which the model writes often -- is lexically identical to a text
# reference link, so text reference links are left to the marker pass. A real one
# needs a definition to render at all, and the Citation Contract forbids a
# bibliography section.
_IMAGE_REFERENCE: Final = re.compile(rf"!{_LINK_LABEL}{_LINK_LABEL}")
_LINK_DEFINITION: Final = re.compile(r"^ {0,3}\[[^\]\n]+\]:.*$", re.MULTILINE)
_AUTOLINK: Final = re.compile(r"<[a-zA-Z][a-zA-Z0-9+.\-]*:[^<>\s]*>")

_MASKED_SPANS: Final = (
    _FENCED_CODE,
    _INDENTED_CODE,
    _INLINE_CODE,
    _LINK_OR_IMAGE,
    _IMAGE_REFERENCE,
    _LINK_DEFINITION,
    _AUTOLINK,
)


def link_public_citations(
    answer: str,
    sources: Iterable[SourceReference],
) -> str:
    """Rewrite public citations as ``[[n[-m]] title](url)`` links.

    The marker stays visible, so prose that refers to a citation still reads the
    same and the reader keeps the excerpt number, while the title makes the
    downloaded file self-sufficient.
    """
    public: dict[str, tuple[str, str]] = {}
    for source in sources:
        try:
            url = validate_public_web_url(source.source_uri)
        except ValueError:
            continue
        public[str(source.id)] = (title_for(source), url)
    if not public or not answer:
        return answer

    answer, protected = _mask_markdown(answer)

    def _replace(match: re.Match[str]) -> str:
        ref_id, chunk_idx = match.group(1), match.group(2)
        entry = public.get(ref_id)
        if entry is None:
            return match.group(0)
        title, url = entry
        marker = ref_id if chunk_idx is None else f"{ref_id}-{chunk_idx}"
        return f'[{marker}](<{url}> "{title}")'

    projected = _MARKER.sub(_replace, answer)
    for index in range(len(protected) - 1, -1, -1):
        projected = projected.replace(f"\x00P{index}\x00", protected[index])
    return projected


def _mask_markdown(answer: str) -> tuple[str, list[str]]:
    """Replace code, link, and autolink spans with placeholders, in order."""
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        index = len(protected)
        protected.append(match.group(0))
        return f"\x00P{index}\x00"

    for pattern in _MASKED_SPANS:
        answer = pattern.sub(replace, answer)
    return answer, protected


def title_for(source: SourceReference) -> str:
    """Return one single-line link title for a source.

    Whitespace is collapsed because a raw newline would end the inline link, and
    the title is truncated before escaping so a cut cannot leave a dangling
    escape.
    """
    title = " ".join(str(source.title or "").split())
    if not title:
        return f"Source {source.id}"
    if len(title) > _TITLE_LIMIT:
        title = title[: _TITLE_LIMIT - 1].rstrip() + "…"
    return title.translate(_TITLE_ESCAPE)


__all__ = ["link_public_citations", "title_for"]
