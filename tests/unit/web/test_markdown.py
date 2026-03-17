"""Tests for Markdown rendering in web UI."""

from __future__ import annotations


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


def test_render_markdown_inline_code():
    from dlightrag.web.markdown import render_markdown

    result = render_markdown("use `foo()` here")
    assert "<code>foo()</code>" in result


def test_render_markdown_latex_passthrough():
    """Dollar-sign math should pass through as literal text (KaTeX handles client-side)."""
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


def test_citation_badges_basic():
    """[1-2] in plain text becomes a citation badge."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("See [1-2] for details."))
    assert 'class="citation-badge"' in result
    assert 'data-ref="1"' in result
    assert 'data-chunk="2"' in result


def test_citation_badges_doc_level():
    """[3] doc-level citation becomes a badge."""
    from dlightrag.web.deps import _citation_badges

    result = str(_citation_badges("See [3] for details."))
    assert 'class="citation-badge"' in result
    assert 'data-ref="3"' in result


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
