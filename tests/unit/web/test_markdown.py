"""Tests for Markdown rendering in web UI."""


def test_render_markdown_bold():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("**bold text**")
    assert "<strong>bold text</strong>" in result


def test_render_markdown_gfm_table():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = render_markdown(md)
    assert "<table>" in result
    assert "<td>1</td>" in result
    assert "<td>2</td>" in result


def test_render_markdown_fenced_code_highlighted():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    md = "```python\nprint('hello')\n```"
    result = render_markdown(md)
    # Pygments output is wrapped in <pre class="highlight">
    assert 'class="highlight"' in result
    assert "print" in result


def test_render_markdown_no_double_pre_wrapping():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("```python\nx = 1\n```")
    # Should not have double <pre> wrapping
    assert result.count("<pre") == 1


def test_render_markdown_fenced_code_unknown_lang():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    md = "```unknownlang\nfoo bar\n```"
    result = render_markdown(md)
    # Falls back to plain <pre><code>
    assert "<code>" in result
    assert "foo bar" in result


def test_render_markdown_fenced_code_no_lang():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    md = "```\nplain code\n```"
    result = render_markdown(md)
    assert "<code>" in result
    assert "plain code" in result


def test_render_markdown_mermaid_fence_marked():
    """A mermaid fence is emitted as a marked source block for client upgrade."""
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("```mermaid\ngraph TD\n  A-->B\n```")
    assert 'class="mermaid-source"' in result
    assert 'data-lang="mermaid"' in result
    assert "graph TD" in result


def test_render_markdown_mermaid_source_escaped():
    """Markup inside a mermaid fence stays escaped (no HTML smuggling)."""
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown('```mermaid\ngraph TD\n  A["<script>alert(1)</script>"]\n```')
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert 'class="mermaid-source"' in result


def test_mermaid_marker_survives_nh3():
    """The mermaid marker must survive server-side nh3 sanitization."""
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation

    result = (
        build_answer_presentation(
            answer="```mermaid\ngraph TD\n  A-->B\n```", sources=[], evidence_images=[]
        )
        .parts[0]
        .html
    )
    assert 'class="mermaid-source"' in result
    assert 'data-lang="mermaid"' in result
    assert "graph TD" in result


def test_render_markdown_inline_code():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("use `foo()` here")
    assert "<code>foo()</code>" in result


def test_render_markdown_latex_passthrough():
    """Dollar-sign math should pass through as literal text (MathJax handles client-side)."""
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("The formula $E=mc^2$ is famous.")
    assert "$E=mc^2$" in result


def test_render_markdown_display_latex_passthrough():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("$$\\sum_{i=1}^n x_i$$")
    assert "$$" in result


def test_render_markdown_xss_script_escaped():
    """Raw HTML <script> must be escaped, not passed through."""
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_render_markdown_lists():
    from dlightrag.adapters.http.browser.markdown import render_markdown

    md = "- item 1\n- item 2\n"
    result = render_markdown(md)
    assert "<li>" in result
    assert "item 1" in result


def test_reference_label_helpers_are_shared_across_citation_surfaces():
    from dlightrag.adapters.http.browser.presentation import _reference_aria_label, _reference_label

    assert _reference_label("1") == "1"
    assert _reference_label("1", "2") == "1-2"
    assert _reference_aria_label("1") == "Source 1"
    assert _reference_aria_label("1", "2") == "Source 1, chunk 2"


def test_citation_badges_basic():
    """[1-2] in plain text becomes a citation badge."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("See [1-2] for details.", known_sources={"1": "One"})
    assert 'class="citation-badge"' in result
    assert 'data-ref="1"' in result
    assert 'data-chunk="2"' in result
    assert 'aria-label="Source 1, chunk 2"' in result
    assert ">1-2</cite>" in result
    assert "[1-2]</cite>" not in result


def test_citation_badges_doc_level():
    """[3] doc-level citation becomes a badge."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("See [3] for details.", known_sources={"3": "Three"})
    assert 'class="citation-badge"' in result
    assert 'data-ref="3"' in result
    assert 'aria-label="Source 3"' in result
    assert ">3</cite>" in result
    assert "[3]</cite>" not in result


def test_citation_badges_in_inline_code_skipped():
    """[1-2] inside inline code must NOT become a badge."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("Use `array[1-2]` in code.", known_sources={"1": "One"})
    # The [1-2] is inside <code>, should not be a badge
    assert "<code>" in result
    assert result.count('class="citation-badge"') == 0


def test_citation_badges_in_fenced_code_skipped():
    """[1-2] inside fenced code must NOT become a badge."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    md = "```\narray[1-2] = value\n```\n\nSee [1-2] for info."
    result = render_answer_html(md, known_sources={"1": "One"})
    # Only the [1-2] outside code should be a badge
    assert result.count('class="citation-badge"') == 1


def test_citation_badges_in_table():
    """[1-2] in a table cell should become a badge."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    md = "| Source | Note |\n|---|---|\n| [1-2] | data |"
    result = render_answer_html(md, known_sources={"1": "One"})
    assert 'class="citation-badge"' in result


def test_citation_badges_markdown_rendering():
    """Verify markdown is actually rendered (not just escaped)."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("**bold** text [1-1]", known_sources={"1": "One"})
    assert "<strong>bold</strong>" in result
    assert 'class="citation-badge"' in result


def test_render_chunk_content_html_table_passthrough():
    """HTML tables in chunk content should pass through (not be escaped)."""
    from dlightrag.adapters.http.browser.markdown import render_chunk_content

    html = "<table><tr><th>Name</th></tr><tr><td>Alice</td></tr></table>"
    result = render_chunk_content(html)
    assert "<table>" in result
    assert "<td>Alice</td>" in result


def test_render_chunk_content_markdown_formatting():
    """Markdown formatting in chunk content should be rendered."""
    from dlightrag.adapters.http.browser.markdown import render_chunk_content

    result = render_chunk_content("**bold** and *italic*")
    assert "<strong>bold</strong>" in result
    assert "<em>italic</em>" in result


def test_render_chunk_content_mixed_html_and_markdown():
    """Chunk with both markdown text and HTML table."""
    from dlightrag.adapters.http.browser.markdown import render_chunk_content

    content = "## Summary\n\nKey findings:\n\n<table><tr><td>Revenue</td><td>$1M</td></tr></table>"
    result = render_chunk_content(content)
    assert "<h2>" in result
    assert "<table>" in result
    assert "<td>Revenue</td>" in result


def test_render_markdown_still_escapes_html():
    """Existing render_markdown must still escape HTML (answer safety)."""
    from dlightrag.adapters.http.browser.markdown import render_markdown

    result = render_markdown("<table><tr><td>test</td></tr></table>")
    assert "<table>" not in result
    assert "&lt;table&gt;" in result


def test_highlight_content_renders_html_table():
    """HTML table in chunk content should render, not show raw tags."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    html = "<table><tr><td>Support</td><td>Zoe</td></tr></table>"
    result = render_source_chunk_html(html)
    assert "<table>" in result
    assert "<td>Support</td>" in result
    assert "&lt;table&gt;" not in result


def test_highlight_content_xss_stripped():
    """Script tags must be stripped by nh3 sanitization."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html('<script>alert("xss")</script>Normal text')
    assert "<script>" not in result
    assert "Normal text" in result


def test_highlight_content_phrase_in_table():
    """Highlight phrase inside a table cell should work."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    html = "<table><tr><td>Revenue grew 15%</td></tr></table>"
    result = render_source_chunk_html(html, ["Revenue grew 15%"])
    assert '<span class="highlight">' in result
    assert "Revenue grew 15%" in result


def test_highlight_content_phrase_skips_tag_attrs():
    """Highlight should not match text inside HTML tag attributes."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    html = '<a href="class-info">class info link</a>'
    result = render_source_chunk_html(html, ["class info"])
    assert 'href="class-info"' in result
    assert '<span class="highlight">class info</span>' in result


def test_highlight_content_plain_text():
    """Plain text (no HTML, no markdown) still renders correctly."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("Just a simple text chunk.")
    assert "Just a simple text chunk." in result


def test_highlight_content_markdown_formatting():
    """Markdown in chunk content should be rendered."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("**bold** text")
    assert "<strong>bold</strong>" in result


def test_highlight_content_phrase_with_quotes():
    """Apostrophes and double quotes must not defeat phrase matching."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("The company's revenue rose.", ["company's revenue"])
    assert '<span class="highlight">company\'s revenue</span>' in result


def test_highlight_content_phrase_with_escaped_entity():
    """Phrases containing characters that render as entities still match."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("Total was 5 < 10 always", ["5 < 10"])
    assert '<span class="highlight">5 &lt; 10</span>' in result


def test_highlight_content_overlapping_phrases_not_nested():
    """Overlapping phrases must not produce nested highlight spans."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("Revenue grew 15% last year", ["Revenue grew", "grew 15%"])
    assert result.count('<span class="highlight">') == 1


def test_highlight_content_phrase_carrying_markdown_syntax():
    """Phrases quoted verbatim from raw Markdown still match the rendered text."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    emphasis = render_source_chunk_html("**Revenue** grew 15% in 2024", ["**Revenue** grew 15%"])
    assert '<strong><span class="highlight">Revenue</span></strong>' in emphasis
    assert '<span class="highlight"> grew 15%</span>' in emphasis

    heading = render_source_chunk_html("## Key findings\nBody.", ["## Key findings"])
    assert '<h2><span class="highlight">Key findings</span></h2>' in heading

    table = render_source_chunk_html(
        "| Region | Sales |\n| --- | --- |\n| EMEA | 12% |",
        ["| EMEA | 12% |"],
    )
    assert '<td><span class="highlight">EMEA</span></td>' in table
    assert '<td><span class="highlight">12%</span></td>' in table
    # No stray span in the whitespace between cells.
    assert "</td>\n<span" not in table


def test_highlight_content_phrase_inside_code_span():
    """A phrase crossing a code span keeps the identifier intact."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("Use the `render_chunk` helper", ["`render_chunk` helper"])
    assert '<code><span class="highlight">render_chunk</span></code>' in result


def test_highlight_content_anchors_the_cited_occurrence():
    """Positional anchoring highlights the occurrence the phrase came from."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("cost 5%. Later the cost 5% again.", ["cost 5%"])
    assert result.startswith('<p><span class="highlight">cost 5%</span>. Later')


def test_highlight_content_ignores_phrase_only_present_in_markup():
    """A phrase that resolves to non-visible source text must not be highlighted."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("See [report](https://example.com/q3) now", ["example.com"])
    assert 'class="highlight"' not in result


def test_markdown_after_a_one_line_table_still_renders():
    """CommonMark ends an HTML block at a blank line, and parsers emit neither."""
    from dlightrag.adapters.http.browser.markdown import render_chunk_content

    content = "<table><tr><td>Name</td></tr></table>\n**Answer:** " + "\\_" * 4
    result = render_chunk_content(content)

    assert "<td>Name</td>" in result
    assert "<strong>Answer:</strong>" in result
    assert "\\_" not in result
    assert "____" in result


def test_chunk_without_a_block_tag_is_untouched():
    from dlightrag.adapters.http.browser.markdown import separate_html_blocks

    content = "**Bold** line\n<equation>x = 1</equation>\n<sup>2</sup> trailing"

    assert separate_html_blocks(content) == content


def test_a_table_already_followed_by_a_blank_line_is_untouched():
    from dlightrag.adapters.http.browser.markdown import separate_html_blocks

    content = "<table><tr><td>a</td></tr></table>\n\n**After**"

    assert separate_html_blocks(content) == content


def test_bold_pseudo_items_each_keep_their_line():
    """Bold is inline, so a parser numbering items as **1.** merges them all."""
    from dlightrag.adapters.http.browser.markdown import render_chunk_content

    result = render_chunk_content("**1.** first\n**Answer:** ____\n**2.** second")

    assert result.count("<br />") == 2
    assert result.count("<p>") == 1


def test_a_real_list_is_left_for_markdown_to_break():
    """Inserting breaks into a list Markdown already understands loosens it."""
    from dlightrag.adapters.http.browser.markdown import (
        normalize_chunk_source,
        render_chunk_content,
    )

    for source in ("1. one\n2. two", "- alpha\n- beta", "## Heading\nBody"):
        assert normalize_chunk_source(source) == source

    assert "<p>" not in render_chunk_content("1. one\n2. two")


def test_a_wrapped_sentence_is_not_broken():
    from dlightrag.adapters.http.browser.markdown import normalize_chunk_source

    prose = "output fell gradually from\n1978 to 2007"

    assert normalize_chunk_source(prose) == prose


def test_a_highlight_never_cuts_a_formula_in_half():
    from dlightrag.adapters.http.browser.markdown import inject_highlights, render_chunk_content

    body = "i _ {t} = i ^ {*} + a (\\pi_ {t} - \\pi^ {*})"
    source = f"Taylor argued the bank should use this rule:\n$$\n{body}\n$$\nwhere a is positive."
    html = render_chunk_content(source)

    out = inject_highlights(html, source, [body])

    formula = out[out.index("$$") : out.rindex("$$") + 2]
    assert "<" not in formula
    assert out.count('<span class="highlight">') == 1


def test_a_highlight_outside_a_formula_still_marks_only_itself():
    from dlightrag.adapters.http.browser.markdown import inject_highlights, render_chunk_content

    source = "The rule matters.\n$$\nx = y\n$$\nIt was named after Taylor."
    html = render_chunk_content(source)

    out = inject_highlights(html, source, ["The rule matters."])

    assert '<span class="highlight">The rule matters.</span>' in out
    assert "$$\nx = y\n$$" in out


def test_unresolved_marker_stays_literal_text_instead_of_a_badge():
    """A marker outside the published sources must not promise a source."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html(
        "Copied claim [9-1] and doc claim [9], kept claim [1-1].",
        known_sources={"1": "One"},
    )

    assert 'data-ref="1"' in result
    assert 'data-ref="9"' not in result
    assert "[9-1]" in result
    assert "[9]" in result


def test_external_links_open_in_a_new_tab():
    """A cited public URL must not navigate the running application away."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html(
        "See [Report](<https://example.com/report>).",
        known_sources={},
    )

    assert 'href="https://example.com/report"' in result
    assert 'target="_blank"' in result


def test_internal_links_keep_their_own_navigation():
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("See [Files](/web/files).", known_sources={})

    assert 'href="/web/files"' in result
    assert "target=" not in result


def test_marker_inside_an_existing_link_label_is_not_badged():
    """A link label is link text; badging inside it would nest a control."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html(
        "Fact [see [9]](<https://example.com/x>) and [9].",
        known_sources={"9": "Nine"},
    )

    assert 'href="https://example.com/x"' in result
    assert 'target="_blank"' in result
    assert ">see [9]</a>" in result
    # The standalone marker still badges, and never inside the link.
    assert '<cite class="citation-badge"' in result
    assert "</cite></a>" not in result


def test_markdown_artifact_presentation_badges_nothing_for_a_projected_citation():
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation

    presentation = build_answer_presentation(
        answer='Fact [9](<https://example.com/x> "Title").',
        sources=[
            {
                "id": "9",
                "title": "Title",
                "type": "web",
                "source_uri": "https://example.com/x",
            }
        ],
        evidence_images=[],
    )

    html = presentation.parts[0].html
    assert 'href="https://example.com/x"' in html
    assert 'target="_blank"' in html
    assert 'title="Title"' in html and ">9</a>" in html
    assert "<cite" not in html


def test_projected_citation_inside_a_table_cell_keeps_its_cell():
    """A table row must survive a source title that contains a pipe."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html
    from dlightrag.engine.answer.citations.contracts import SourceReference
    from dlightrag.engine.answer.citations.projection import link_public_citations

    source = SourceReference(
        id="10",
        title="Volvo cost saving | WardsAuto",
        source_uri="https://example.com/wardsauto",
        workspace="default",
        download_locator="https://example.com/wardsauto",
    )
    projected = link_public_citations(
        "| scenario | evidence |\n| --- | --- |\n| base | see [10] |",
        [source],
    )

    html = render_answer_html(projected, known_sources={"10": "Ten"})

    assert html.count("<td>") == 2
    assert 'href="https://example.com/wardsauto"' in html
    assert 'title="Volvo cost saving | WardsAuto"' in html and ">10</a>" in html


def test_citation_badge_carries_its_source_title_as_a_tooltip():
    """Every published source names itself on hover, private corpus included."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html(
        "Web fact [9-1] and local fact [2-1] and unknown [7].",
        known_sources={"9": "沃尔沃将裁员3000人 | 财联社", "2": "romer.pdf"},
    )

    assert (
        'data-ref="9" data-chunk="1" role="button" tabindex="0" title="沃尔沃将裁员3000人 | 财联社"'
        in result
    )
    assert 'data-ref="2" data-chunk="1" role="button" tabindex="0" title="romer.pdf"' in result
    # An unpublished ref stays literal: no badge, so no tooltip promise either.
    assert "[7]" in result and 'data-ref="7"' not in result


def test_badge_tooltip_escapes_markup_from_a_source_title():
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html(
        "Fact [1-1].",
        known_sources={"1": 'A "quoted" & <b>bold</b> title'},
    )

    assert 'title="A &quot;quoted&quot; &amp; &lt;b&gt;bold&lt;/b&gt; title"' in result
    assert "<b>" not in result


def test_a_bare_url_in_an_answer_becomes_a_followable_link():
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("See https://example.com/report now", known_sources={})

    assert '<a href="https://example.com/report" target="_blank"' in result
    assert ">https://example.com/report</a>" in result


def test_autolinking_leaves_a_chinese_sentence_outside_the_href():
    """linkify reads to whitespace, so a Chinese sentence would become an address."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    for text, href in [
        ("参考 https://example.com/x。", "https://example.com/x"),
        ("参考https://example.com/x，然后继续", "https://example.com/x"),
        ("见 https://example.com/report（详见文档）", "https://example.com/report"),
    ]:
        result = render_answer_html(text, known_sources={})

        assert f'href="{href}"' in result, text
        # The punctuation the sentence owns stays visible text, not an escape.
        assert "%E3%80%82" not in result and "%EF%BC%8C" not in result, text
        assert result.rstrip().endswith(("</p>",)), text


def test_an_address_a_chinese_word_run_continues_is_left_as_text():
    """The same characters end a Chinese path, so the renderer never guesses."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    result = render_answer_html("见https://example.com/report即可", known_sources={})

    assert "<a " not in result
    assert "见https://example.com/report即可" in result


def test_a_chinese_path_or_query_component_belongs_to_the_address():
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    path = render_answer_html("https://example.com/wiki/中文条目", known_sources={})
    query = render_answer_html("https://example.com/a?id=中文", known_sources={})

    assert 'href="https://example.com/wiki/%E4%B8%AD%E6%96%87%E6%9D%A1%E7%9B%AE"' in path
    assert 'href="https://example.com/a?id=%E4%B8%AD%E6%96%87"' in query


def test_autolinking_never_guesses_a_domain_or_an_address_from_text():
    """`report.md`, `build.sh`, and `clip.mov` end in real top-level domains."""
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    for text in [
        "The analysis is in report.md",
        "Run build.sh then clip.mov",
        "Unpack archive.zip",
        "See www.example.com/report",
        "See example.com/report",
        "Mail dev@example.com now",
    ]:
        result = render_answer_html(text, known_sources={})

        assert "<a " not in result, text
        assert text in result, text


def test_autolinking_leaves_code_and_explicit_links_alone():
    from dlightrag.adapters.http.browser.presentation import render_answer_html

    code = render_answer_html(
        "Inline `https://example.com` and:\n\n```\nhttps://example.com\n```", known_sources={}
    )
    explicit = render_answer_html("[Report](https://example.com/x)", known_sources={})

    assert "<code>https://example.com</code>" in code
    assert "<pre><code>https://example.com\n</code></pre>" in code
    assert explicit.count("<a ") == 1


def test_a_bare_url_in_a_quoted_source_chunk_stays_text():
    """A chunk quotes a document; its own address is not this product's link."""
    from dlightrag.adapters.http.browser.presentation import render_source_chunk_html

    result = render_source_chunk_html("See https://example.com/report now")

    assert "<a " not in result
    assert "https://example.com/report" in result
