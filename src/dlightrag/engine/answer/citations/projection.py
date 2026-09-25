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

The shared Markdown citation tokens identify precisely the spans eligible for
projection. Code, math, escapes, reference definitions, links and image alt text
keep their original source bytes.
"""

from collections.abc import Iterable
from typing import Final

from dlightrag.engine.public_http import validate_public_web_url

from .contracts import SourceReference
from .syntax import Citation, parse_citations

_TITLE_LIMIT: Final = 80
# A link title is a quoted string, so only its delimiter, its escape character,
# and the table-cell separator need escaping. A pipe matters most: a citation
# inside a table cell would otherwise split the row.
_TITLE_ESCAPE: Final = str.maketrans({"\\": r"\\", '"': r"\"", "|": r"\|"})


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

    def replacement(citation: Citation) -> str:
        entry = public.get(citation.ref_id)
        if entry is None:
            return citation.marker
        title, url = entry
        return f'[{citation.key}](<{url}> "{title}")'

    return parse_citations(answer).rewrite(replacement)


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
