"""Citation syntax belongs to the answer grammar, with exact source edits."""

import pytest

from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.highlight import extract_all_citing_sentences
from dlightrag.engine.answer.citations.syntax import parse_citations


@pytest.mark.parametrize(
    "text",
    [
        "`values[1]`",
        "```python\nvalues[1]\n## References\nkeep()\n```\nAfter.",
        "[官网][1]\n\n[1]: https://example.com",
        "[name [1]](artifact:report[2].md)",
        "![alt [1]](https://example.com/image.png)",
        r"Escaped \[1] and &#91;2] and $x[3]$ and \(x[4]\).",
        "Body.\n\n## References\nOrdinary prose.\n\n## Appendix\nKeep this too.",
    ],
)
def test_finalization_preserves_non_citation_syntax_exactly(text: str) -> None:
    assert finalize_answer(text, {}).answer == text
    assert extract_all_citing_sentences(text) == {}


@pytest.mark.parametrize(
    "text",
    [
        "Claim [1].",
        "> - Claim [1].\n>   Other [2].",
        "# Title [1] #\n\nHeading [2]\n===",
        "| A | B |\n|---|---|\n| [1] \\| x | **[1]** [2] |",
        "> - | A | B |\r\n>   |---|---|\r\n>   | [1] \\| [2] | `code[9]` |",
        "1. 1 [1]\n\n   Continuation [2].",
        "Claim [1].\r\n\r\nClaim [2].\x00",
    ],
)
def test_source_spans_target_exact_markers_and_preserve_surroundings(text: str) -> None:
    document = parse_citations(text)
    assert document.citations
    for citation in document.citations:
        assert text[citation.start : citation.end] == citation.marker
    assert document.rewrite(lambda citation: citation.marker) == text
    assert document.rewrite(lambda _: "") == text.replace("[1]", "").replace("[2]", "")


def test_many_escaped_pipes_keep_exact_cell_positions() -> None:
    text = "| A | B |\n|---|---|\n| " + "word \\| " * 4096 + "[1] | [2] |"
    document = parse_citations(text)
    assert [citation.marker for citation in document.citations] == ["[1]", "[2]"]
    assert document.rewrite(lambda _: "") == text.replace("[1]", "").replace("[2]", "")


def test_real_reference_links_win_over_citation_shaped_labels() -> None:
    text = "Claim [1]. [official][2] [3-1].\n\n[1]: https://one.test\n[2]: https://two.test"
    assert [citation.marker for citation in parse_citations(text).citations] == ["[3-1]"]


def test_adjacent_citations_remain_distinct_without_reference_definitions() -> None:
    assert [c.marker for c in parse_citations("Claim [1-1][2-1].").citations] == [
        "[1-1]",
        "[2-1]",
    ]


def test_highlighting_only_collects_real_citing_sentences() -> None:
    result = extract_all_citing_sentences(
        "`code[2]`\n\nClaim [1-1].\n\n[x][3]\n\n[3]: https://x.test"
    )
    assert result == {"1-1": ["Claim [1-1]."]}


def test_unterminated_empty_quoted_table_row_is_safe_on_all_surfaces() -> None:
    from dlightrag.adapters.http.browser.presentation import render_answer_html
    from dlightrag.engine.answer.citations.contracts import SourceReference
    from dlightrag.engine.answer.citations.projection import link_public_citations

    text = "> | A | B |\n> |---|---|\n>  "
    source = SourceReference(
        id="1", source_uri="https://example.com", workspace="test", download_locator="source"
    )
    assert finalize_answer(text, {}).answer == text
    assert parse_citations(text).rewrite(lambda citation: citation.marker) == text
    assert link_public_citations(text, [source]) == text
    rendered = render_answer_html(text, known_sources={})
    assert "<table>" in rendered
    assert rendered == render_answer_html(text + "\n", known_sources={})


@pytest.mark.parametrize(
    "text",
    [
        "```\nunterminated fence [1]",
        "> ```\n> nested unterminated fence [1]",
        "    indented code [1]",
        "Paragraph [1].  ",
        "| A | B |\n|---|---|\n| [1] | B |",
        "> - nested [1]\r\n>   continued",
        "## Heading [1]",
    ],
)
def test_parser_terminator_does_not_change_original_markdown_html(text: str) -> None:
    from markdown_it import MarkdownIt

    from dlightrag.engine.answer.markdown import answer_markdown

    assert answer_markdown().render(text) == MarkdownIt("gfm-like", {"html": False}).render(text)
