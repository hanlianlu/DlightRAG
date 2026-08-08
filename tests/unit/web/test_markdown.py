"""Tests for Markdown rendering in web UI."""


def test_render_markdown_bold():
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("**bold text**")
    assert "<strong>bold text</strong>" in result


def test_render_markdown_gfm_table():
    from dlightrag.web.markdown import render_markdown

    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = render_markdown(md)
    assert "<table>" in result
    assert "<td>1</td>" in result
    assert "<td>2</td>" in result


def test_render_markdown_fenced_code_highlighted():
    from dlightrag.web.markdown import render_markdown

    md = "```python\nprint('hello')\n```"
    result = render_markdown(md)
    # Pygments output is wrapped in <pre class="highlight">
    assert 'class="highlight"' in result
    assert "print" in result


def test_render_markdown_no_double_pre_wrapping():
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("```python\nx = 1\n```")
    # Should not have double <pre> wrapping
    assert result.count("<pre") == 1


def test_render_markdown_fenced_code_unknown_lang():
    from dlightrag.web.markdown import render_markdown

    md = "```unknownlang\nfoo bar\n```"
    result = render_markdown(md)
    # Falls back to plain <pre><code>
    assert "<code>" in result
    assert "foo bar" in result


def test_render_markdown_fenced_code_no_lang():
    from dlightrag.web.markdown import render_markdown

    md = "```\nplain code\n```"
    result = render_markdown(md)
    assert "<code>" in result
    assert "plain code" in result


def test_render_markdown_mermaid_fence_marked():
    """A mermaid fence is emitted as a marked source block for client upgrade."""
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("```mermaid\ngraph TD\n  A-->B\n```")
    assert 'class="mermaid-source"' in result
    assert 'data-lang="mermaid"' in result
    assert "graph TD" in result


def test_render_markdown_mermaid_source_escaped():
    """Markup inside a mermaid fence stays escaped (no HTML smuggling)."""
    from dlightrag.web.markdown import render_markdown

    result = render_markdown('```mermaid\ngraph TD\n  A["<script>alert(1)</script>"]\n```')
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert 'class="mermaid-source"' in result


def test_mermaid_marker_survives_nh3():
    """The mermaid marker must survive server-side nh3 sanitization."""
    from dlightrag.web.safe_html import safe_answer_preview

    result = safe_answer_preview("```mermaid\ngraph TD\n  A-->B\n```")
    assert 'class="mermaid-source"' in result
    assert 'data-lang="mermaid"' in result
    assert "graph TD" in result


def test_render_markdown_inline_code():
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("use `foo()` here")
    assert "<code>foo()</code>" in result


def test_render_markdown_latex_passthrough():
    """Dollar-sign math should pass through as literal text (MathJax handles client-side)."""
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("The formula $E=mc^2$ is famous.")
    assert "$E=mc^2$" in result


def test_render_markdown_display_latex_passthrough():
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("$$\\sum_{i=1}^n x_i$$")
    assert "$$" in result


def test_render_markdown_xss_script_escaped():
    """Raw HTML <script> must be escaped, not passed through."""
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_render_markdown_lists():
    from dlightrag.web.markdown import render_markdown

    md = "- item 1\n- item 2\n"
    result = render_markdown(md)
    assert "<li>" in result
    assert "item 1" in result


def test_reference_label_helpers_are_shared_across_citation_surfaces():
    from dlightrag.web.deps import _reference_aria_label, _reference_label, templates

    assert _reference_label("1") == "1"
    assert _reference_label("1", "2") == "1-2"
    assert _reference_aria_label("1") == "Source 1"
    assert _reference_aria_label("1", "2") == "Source 1, chunk 2"
    assert templates.env.filters["reference_label"] is _reference_label


def test_citation_badges_basic():
    """[1-2] in plain text becomes a citation badge."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("See [1-2] for details."))
    assert 'class="citation-badge"' in result
    assert 'data-ref="1"' in result
    assert 'data-chunk="2"' in result
    assert 'aria-label="Source 1, chunk 2"' in result
    assert ">1-2</span>" in result
    assert "[1-2]</span>" not in result


def test_citation_badges_doc_level():
    """[3] doc-level citation becomes a badge."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("See [3] for details."))
    assert 'class="citation-badge"' in result
    assert 'data-ref="3"' in result
    assert 'aria-label="Source 3"' in result
    assert ">3</span>" in result
    assert "[3]</span>" not in result


def test_citation_badges_in_inline_code_skipped():
    """[1-2] inside inline code must NOT become a badge."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("Use `array[1-2]` in code."))
    # The [1-2] is inside <code>, should not be a badge
    assert "<code>" in result
    assert result.count('class="citation-badge"') == 0


def test_citation_badges_in_fenced_code_skipped():
    """[1-2] inside fenced code must NOT become a badge."""
    from dlightrag.web.deps import _citation_badges

    md = "```\narray[1-2] = value\n```\n\nSee [1-2] for info."
    result = str(_citation_badges(md))
    # Only the [1-2] outside code should be a badge
    assert result.count('class="citation-badge"') == 1


def test_citation_badges_in_table():
    """[1-2] in a table cell should become a badge."""
    from dlightrag.web.deps import _citation_badges

    md = "| Source | Note |\n|---|---|\n| [1-2] | data |"
    result = str(_citation_badges(md))
    assert 'class="citation-badge"' in result


def test_citation_badges_markdown_rendering():
    """Verify markdown is actually rendered (not just escaped)."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("**bold** text [1-1]"))
    assert "<strong>bold</strong>" in result
    assert 'class="citation-badge"' in result


def test_render_chunk_content_html_table_passthrough():
    """HTML tables in chunk content should pass through (not be escaped)."""
    from dlightrag.web.markdown import render_chunk_content

    html = "<table><tr><th>Name</th></tr><tr><td>Alice</td></tr></table>"
    result = render_chunk_content(html)
    assert "<table>" in result
    assert "<td>Alice</td>" in result


def test_render_chunk_content_markdown_formatting():
    """Markdown formatting in chunk content should be rendered."""
    from dlightrag.web.markdown import render_chunk_content

    result = render_chunk_content("**bold** and *italic*")
    assert "<strong>bold</strong>" in result
    assert "<em>italic</em>" in result


def test_render_chunk_content_mixed_html_and_markdown():
    """Chunk with both markdown text and HTML table."""
    from dlightrag.web.markdown import render_chunk_content

    content = "## Summary\n\nKey findings:\n\n<table><tr><td>Revenue</td><td>$1M</td></tr></table>"
    result = render_chunk_content(content)
    assert "<h2>" in result
    assert "<table>" in result
    assert "<td>Revenue</td>" in result


def test_render_markdown_still_escapes_html():
    """Existing render_markdown must still escape HTML (answer safety)."""
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("<table><tr><td>test</td></tr></table>")
    assert "<table>" not in result
    assert "&lt;table&gt;" in result


def test_highlight_content_renders_html_table():
    """HTML table in chunk content should render, not show raw tags."""
    from dlightrag.web.deps import _highlight_content

    html = "<table><tr><td>Support</td><td>Zoe</td></tr></table>"
    result = str(_highlight_content(html))
    assert "<table>" in result
    assert "<td>Support</td>" in result
    assert "&lt;table&gt;" not in result


def test_highlight_content_xss_stripped():
    """Script tags must be stripped by nh3 sanitization."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content('<script>alert("xss")</script>Normal text'))
    assert "<script>" not in result
    assert "Normal text" in result


def test_highlight_content_phrase_in_table():
    """Highlight phrase inside a table cell should work."""
    from dlightrag.web.deps import _highlight_content

    html = "<table><tr><td>Revenue grew 15%</td></tr></table>"
    result = str(_highlight_content(html, ["Revenue grew 15%"]))
    assert '<span class="highlight">' in result
    assert "Revenue grew 15%" in result


def test_highlight_content_phrase_skips_tag_attrs():
    """Highlight should not match text inside HTML tag attributes."""
    from dlightrag.web.deps import _highlight_content

    html = '<a href="class-info">class info link</a>'
    result = str(_highlight_content(html, ["class info"]))
    assert 'href="class-info"' in result
    assert '<span class="highlight">class info</span>' in result


def test_highlight_content_plain_text():
    """Plain text (no HTML, no markdown) still renders correctly."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("Just a simple text chunk."))
    assert "Just a simple text chunk." in result


def test_highlight_content_markdown_formatting():
    """Markdown in chunk content should be rendered."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("**bold** text"))
    assert "<strong>bold</strong>" in result


def test_highlight_content_phrase_with_quotes():
    """Apostrophes and double quotes must not defeat phrase matching."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("The company's revenue rose.", ["company's revenue"]))
    assert '<span class="highlight">company\'s revenue</span>' in result


def test_highlight_content_phrase_with_escaped_entity():
    """Phrases containing characters that render as entities still match."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("Total was 5 < 10 always", ["5 < 10"]))
    assert '<span class="highlight">5 &lt; 10</span>' in result


def test_highlight_content_overlapping_phrases_not_nested():
    """Overlapping phrases must not produce nested highlight spans."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("Revenue grew 15% last year", ["Revenue grew", "grew 15%"]))
    assert result.count('<span class="highlight">') == 1


def test_highlight_content_phrase_carrying_markdown_syntax():
    """Phrases quoted verbatim from raw Markdown still match the rendered text."""
    from dlightrag.web.deps import _highlight_content

    emphasis = str(_highlight_content("**Revenue** grew 15% in 2024", ["**Revenue** grew 15%"]))
    assert '<strong><span class="highlight">Revenue</span></strong>' in emphasis
    assert '<span class="highlight"> grew 15%</span>' in emphasis

    heading = str(_highlight_content("## Key findings\nBody.", ["## Key findings"]))
    assert '<h2><span class="highlight">Key findings</span></h2>' in heading

    table = str(
        _highlight_content("| Region | Sales |\n| --- | --- |\n| EMEA | 12% |", ["| EMEA | 12% |"])
    )
    assert '<td><span class="highlight">EMEA</span></td>' in table
    assert '<td><span class="highlight">12%</span></td>' in table
    # No stray span in the whitespace between cells.
    assert "</td>\n<span" not in table


def test_highlight_content_phrase_inside_code_span():
    """A phrase crossing a code span keeps the identifier intact."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("Use the `render_chunk` helper", ["`render_chunk` helper"]))
    assert '<code><span class="highlight">render_chunk</span></code>' in result


def test_highlight_content_anchors_the_cited_occurrence():
    """Positional anchoring highlights the occurrence the phrase came from."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("cost 5%. Later the cost 5% again.", ["cost 5%"]))
    assert result.startswith('<p><span class="highlight">cost 5%</span>. Later')


def test_highlight_content_ignores_phrase_only_present_in_markup():
    """A phrase that resolves to non-visible source text must not be highlighted."""
    from dlightrag.web.deps import _highlight_content

    result = str(_highlight_content("See [report](https://example.com/q3) now", ["example.com"]))
    assert 'class="highlight"' not in result


def test_markdown_after_a_one_line_table_still_renders():
    """CommonMark ends an HTML block at a blank line, and parsers emit neither."""
    from dlightrag.web.markdown import render_chunk_content

    content = "<table><tr><td>Name</td></tr></table>\n**Answer:** " + "\\_" * 4
    result = render_chunk_content(content)

    assert "<td>Name</td>" in result
    assert "<strong>Answer:</strong>" in result
    assert "\\_" not in result
    assert "____" in result


def test_chunk_without_a_block_tag_is_untouched():
    from dlightrag.web.markdown import separate_html_blocks

    content = "**Bold** line\n<equation>x = 1</equation>\n<sup>2</sup> trailing"

    assert separate_html_blocks(content) == content


def test_a_table_already_followed_by_a_blank_line_is_untouched():
    from dlightrag.web.markdown import separate_html_blocks

    content = "<table><tr><td>a</td></tr></table>\n\n**After**"

    assert separate_html_blocks(content) == content
