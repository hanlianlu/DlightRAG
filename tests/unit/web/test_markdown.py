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
    # Pygments wraps in div.highlight
    assert 'class="highlight"' in result
    assert "print" in result


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
