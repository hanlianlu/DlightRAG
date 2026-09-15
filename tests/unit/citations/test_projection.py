# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for projecting public document citations onto their URL."""

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.projection import link_public_citations


def _source(ref_id: str, *, uri: str, title: str | None = None) -> SourceReference:
    return SourceReference(
        id=ref_id,
        title=title,
        source_uri=uri,
        workspace="default",
        download_locator=uri,
    )


def test_public_document_citation_becomes_a_self_describing_link() -> None:
    sources = [
        _source(
            "9", uri="https://www.cls.cn/detail/2041214", title="沃尔沃汽车将在全球裁员近3000人"
        )
    ]

    projected = link_public_citations("沃尔沃将裁员近 3000 人 [9]。", sources)

    assert projected == (
        "沃尔沃将裁员近 3000 人 [[9] 沃尔沃汽车将在全球裁员近3000人]"
        "(<https://www.cls.cn/detail/2041214>)。"
    )


def test_excerpt_marker_projects_and_keeps_its_excerpt_number() -> None:
    """A URL cannot address one excerpt, so the number stays in the label."""
    sources = [_source("9", uri="https://example.com/report", title="Report")]

    assert link_public_citations("Fact [9-1] only.", sources) == (
        "Fact [[9-1] Report](<https://example.com/report>) only."
    )


def test_private_sources_keep_their_marker() -> None:
    sources = [
        _source("1", uri="local://default/report.pdf", title="Local report"),
        _source("2", uri="res-opaque-handle", title="Handle"),
    ]

    assert link_public_citations("Facts [1] and [2] and [1-2].", sources) == (
        "Facts [1] and [2] and [1-2]."
    )


def test_unknown_reference_is_left_alone() -> None:
    sources = [_source("1", uri="https://example.com/a", title="A")]

    assert link_public_citations("Known [1], copied [7].", sources) == (
        "Known [[1] A](<https://example.com/a>), copied [7]."
    )


def test_label_drops_brackets_and_collapses_whitespace() -> None:
    sources = [_source("3", uri="https://example.com/b", title="Head [draft]\n second line")]

    projected = link_public_citations("See [3].", sources)

    assert projected == "See [[3] Head (draft) second line](<https://example.com/b>)."


def test_label_falls_back_and_truncates() -> None:
    long_title = "题" * 120
    sources = [
        SourceReference(
            id="4",
            title=None,
            source_uri="https://example.com/c",
            workspace="default",
            download_locator="https://example.com/c",
        ),
        _source("5", uri="https://example.com/d", title=long_title),
    ]

    projected = link_public_citations("See [4] and [5].", sources)

    assert "[[4] Source 4](<https://example.com/c>)" in projected
    label = projected.split("[[5] ", 1)[1].split("](<", 1)[0]
    assert len(label) == 80 and label.endswith("…")


def test_projection_is_idempotent() -> None:
    sources = [_source("9", uri="https://example.com/report", title="Report")]
    once = link_public_citations("Fact [9] and [9-2].", sources)

    assert link_public_citations(once, sources) == once


def test_url_with_parentheses_survives_the_link_destination() -> None:
    sources = [_source("2", uri="https://example.com/wiki/Foo_(a)_(b)", title="Foo")]

    projected = link_public_citations("See [2] and [2-3].", sources)

    assert projected == (
        "See [[2] Foo](<https://example.com/wiki/Foo_(a)_(b)>) and"
        " [[2-3] Foo](<https://example.com/wiki/Foo_(a)_(b)>)."
    )


def test_code_keeps_its_text_while_prose_projects() -> None:
    sources = [_source("1", uri="https://example.com/a", title="A")]
    answer = "Array index ```py\narr[1] = 2\n``` and inline `arr[1]`, fact [1]."

    projected = link_public_citations(answer, sources)

    assert "```py\narr[1] = 2\n```" in projected
    assert "`arr[1]`" in projected
    assert projected.endswith("fact [[1] A](<https://example.com/a>).")


def test_label_escapes_markdown_emphasis_and_code_punctuation() -> None:
    sources = [_source("7", uri="https://example.com/c", title="Q1* results with `code` and _x_")]

    projected = link_public_citations("See [7].", sources)

    assert projected == (
        "See [[7] Q1\\* results with \\`code\\` and \\_x\\_](<https://example.com/c>)."
    )


def test_label_escapes_a_pipe_so_a_table_row_is_not_split() -> None:
    sources = [
        _source("10", uri="https://example.com/wardsauto", title="Volvo cost saving | WardsAuto")
    ]

    projected = link_public_citations(
        "| scenario | evidence |\n| --- | --- |\n| base | see [10] |",
        sources,
    )

    assert "Volvo cost saving \\| WardsAuto" in projected


def test_existing_markdown_structures_are_left_alone() -> None:
    """A marker inside a link, image, definition, or autolink is data, not a citation.

    A *text* reference link (``[a][9]``) is deliberately not protected: two
    adjacent citations (``[9-1][10-1]``) are lexically identical, and citations win.
    """
    sources = [_source("9", uri="https://example.com/a", title="T")]

    for text in (
        "[see [9]](https://example.com/x)",
        "[x](https://example.com/path/[9])",
        "![9](img.png)",
        "[9]: https://example.com",
        "![alt][9]",
        "<https://example.com/[9]>",
    ):
        assert link_public_citations(text, sources) == text


def test_projection_stays_idempotent_over_a_bracketed_destination() -> None:
    sources = [_source("9", uri="https://example.com/a", title="T")]
    once = link_public_citations("fact [9].", sources)
    bracketed = once.replace("example.com/a", "example.com/wiki/Foo[9]")

    assert link_public_citations(bracketed, sources) == bracketed


def test_double_backtick_and_indented_code_keep_their_markers() -> None:
    sources = [_source("1", uri="https://example.com/a", title="A")]
    text = "span ``arr[1]`` end\n\n    indented arr[1]\n\nfact [1]."

    projected = link_public_citations(text, sources)

    assert "``arr[1]``" in projected
    assert "    indented arr[1]" in projected
    assert projected.endswith("fact [[1] A](<https://example.com/a>).")


def test_label_truncation_cannot_leave_a_dangling_escape() -> None:
    sources = [_source("5", uri="https://example.com/d", title="x" * 79 + "*y")]

    projected = link_public_citations("See [5].", sources)

    label = projected.split("[[5] ", 1)[1].split("](<", 1)[0]
    assert label.endswith("…")
    assert not label.endswith("\\")
