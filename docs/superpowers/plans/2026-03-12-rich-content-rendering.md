# Rich Content Rendering Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Markdown, table, code highlighting, and LaTeX math rendering to DlightRAG's Web UI answer display.

**Architecture:** Server-side rendering via markdown-it-py + Pygments replaces `Markup.escape()` in the citation_badges filter. Client-side KaTeX auto-render handles LaTeX math after the `done` SSE event. Streaming phase remains plain text.

**Tech Stack:** markdown-it-py (gfm-like preset), Pygments, KaTeX CDN

**Spec:** `docs/superpowers/specs/2026-03-12-rich-content-rendering-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/dlightrag/web/markdown.py` | markdown-it-py + Pygments renderer, single `render_markdown()` function |
| Create | `src/dlightrag/web/static/pygments.css` | Monokai dark theme CSS for syntax highlighting |
| Create | `tests/unit/web/test_markdown.py` | Unit tests for `render_markdown()` and `_citation_badges` with markdown |
| Modify | `pyproject.toml` | Add `markdown-it-py[plugins]` and `pygments` dependencies |
| Modify | `src/dlightrag/web/deps.py` | Replace `Markup.escape()` with `render_markdown()` + code block protection |
| Modify | `src/dlightrag/web/templates/base.html` | Add KaTeX CDN links and `pygments.css` stylesheet |
| Modify | `src/dlightrag/web/static/style.css` | Add scoped `.ai-message-content` Markdown element styles |
| Modify | `src/dlightrag/web/static/citation.js` | Add KaTeX `renderMathInElement()` call in `done` handler |

---

## Chunk 1: Core Implementation

### Task 1: Add Python dependencies

**Files:**
- Modify: `pyproject.toml:39-43`

- [ ] **Step 1: Add markdown-it-py and pygments to pyproject.toml**

In `pyproject.toml`, find the `# Web frontend` comment block and add after the `jinja2` line:

```toml
    # Web frontend
    "jinja2>=3.1",
    "markdown-it-py[plugins]>=3.0",
    "pygments>=2.17",
```

- [ ] **Step 2: Run uv sync to install**

Run: `uv sync`
Expected: successful resolution and install of markdown-it-py, mdit-py-plugins, pygments

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "from markdown_it import MarkdownIt; from pygments import highlight; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add markdown-it-py and pygments for rich content rendering"
```

---

### Task 2: Create render_markdown() with TDD

**Files:**
- Create: `src/dlightrag/web/markdown.py`
- Create: `tests/unit/web/test_markdown.py`

- [ ] **Step 1: Write failing tests for render_markdown**

Create `tests/unit/web/test_markdown.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/web/test_markdown.py -v`
Expected: all tests FAIL with `ModuleNotFoundError: No module named 'dlightrag.web.markdown'`

- [ ] **Step 3: Implement render_markdown**

Create `src/dlightrag/web/markdown.py`:

```python
"""Markdown-to-HTML renderer for Web UI answers.

Uses markdown-it-py (GFM-like preset) for Markdown/tables/lists and
Pygments for fenced code block syntax highlighting. LaTeX $...$ passes
through as literal text for client-side KaTeX rendering.
"""

from __future__ import annotations

from markdown_it import MarkdownIt
from pygments import highlight as pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

_FORMATTER = HtmlFormatter(cssclass="highlight", wrapcode=True)


def _highlight_fn(code: str, lang: str, _attrs: str) -> str:
    """Pygments highlight callback for markdown-it-py fenced code blocks.

    Returns highlighted HTML if language is known, empty string to fall back
    to the default ``<pre><code>`` wrapper.
    """
    if not lang:
        return ""
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        return ""
    return pygments_highlight(code, lexer, _FORMATTER)


_md = MarkdownIt("gfm-like", {"html": False, "highlight": _highlight_fn}).disable("linkify")


def render_markdown(text: str) -> str:
    """Convert Markdown text to HTML with syntax-highlighted code blocks."""
    return _md.render(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/test_markdown.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/web/markdown.py tests/unit/web/test_markdown.py
git commit -m "feat(web): add Markdown renderer with Pygments code highlighting"
```

---

### Task 3: Update citation_badges with code block protection

**Files:**
- Modify: `src/dlightrag/web/deps.py:1-58`
- Modify: `tests/unit/web/test_markdown.py` (append tests)

- [ ] **Step 1: Write failing tests for citation badges with markdown**

Append to `tests/unit/web/test_markdown.py`:

```python
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
```

- [ ] **Step 2: Run all 6 new tests to verify they fail**

Run: `uv run pytest tests/unit/web/test_markdown.py -k "citation_badges" -v`
Expected: FAIL — `test_citation_badges_markdown_rendering` and `test_citation_badges_in_inline_code_skipped` fail because current code uses `Markup.escape()` (no `<strong>` or `<code>` in output). `test_citation_badges_in_fenced_code_skipped` fails because no code block protection. Basic badge tests may pass since the regex still works on escaped text.

- [ ] **Step 3: Implement code block protection and update _citation_badges**

Edit `src/dlightrag/web/deps.py`:

**a)** Add import after the existing imports (after `from dlightrag.citations.parser import CITATION_PATTERN`):

```python
from dlightrag.web.markdown import render_markdown
```

**b)** Add these two helper functions before `_citation_badges` (note: `re` is already imported in deps.py):

```python
def _protect_code_blocks(html: str) -> tuple[str, list[str]]:
    """Replace <pre>...</pre> and <code>...</code> with indexed placeholders.

    Protects code content from citation regex replacement.
    """
    protected: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"\x00CODE{idx}\x00"

    # Protect <pre> blocks first (they contain <code>)
    html = re.sub(r"<pre[^>]*>.*?</pre>", _replace, html, flags=re.DOTALL)
    # Then protect remaining inline <code>
    html = re.sub(r"<code[^>]*>.*?</code>", _replace, html, flags=re.DOTALL)
    return html, protected


def _restore_code_blocks(html: str, protected: list[str]) -> str:
    """Restore code block placeholders with original content."""
    for idx, original in enumerate(protected):
        html = html.replace(f"\x00CODE{idx}\x00", original)
    return html
```

**c)** Replace `_citation_badges` with this complete function:

```python
def _citation_badges(text: str) -> Markup:
    """Replace citation patterns with clickable badge HTML.

    Handles both [ref_id-chunk_idx] and [n] doc-level formats.
    Renders Markdown first, then injects badges while protecting code blocks.
    """
    from dlightrag.citations.parser import DOC_CITATION_PATTERN

    # Render Markdown to HTML (replaces Markup.escape)
    html = render_markdown(str(text))

    # Protect code blocks from citation regex
    html, protected = _protect_code_blocks(html)

    # First pass: replace chunk-level [ref-chunk] with badges
    def _chunk_badge(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = m.group(2)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'data-chunk="{chunk_idx}" onclick="filterSource(this)">'
            f"[{ref_id}-{chunk_idx}]</span>"
        )

    result = CITATION_PATTERN.sub(_chunk_badge, html)

    # Second pass: replace doc-level [n] with badges
    def _doc_badge(m: re.Match) -> str:
        ref_id = m.group(1)
        return (
            f'<span class="citation-badge" data-ref="{ref_id}" '
            f'onclick="filterSource(this)">'
            f"[{ref_id}]</span>"
        )

    result = DOC_CITATION_PATTERN.sub(_doc_badge, result)

    # Restore code blocks
    result = _restore_code_blocks(result, protected)

    return Markup(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/web/test_markdown.py -v`
Expected: all 16 tests PASS

- [ ] **Step 5: Run existing web tests to verify no regressions**

Run: `uv run pytest tests/unit/web/ -v`
Expected: all tests PASS (existing `test_routes.py` unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/web/deps.py tests/unit/web/test_markdown.py
git commit -m "feat(web): render Markdown in citation badges with code block protection"
```

---

### Task 4: Add frontend assets and templates

**Files:**
- Create: `src/dlightrag/web/static/pygments.css`
- Modify: `src/dlightrag/web/templates/base.html:1-14`
- Modify: `src/dlightrag/web/static/style.css` (append at end)
- Modify: `src/dlightrag/web/static/citation.js:386-405`

- [ ] **Step 1: Generate Pygments CSS (scoped to avoid collision)**

The existing `style.css` has a `.highlight` class for semantic text highlighting in source chunks.
Pygments must be scoped to `.ai-message-content .highlight` to avoid overriding it.

Run: `uv run pygmentize -S monokai -f html -a '.ai-message-content .highlight' > src/dlightrag/web/static/pygments.css`

Verify: `head -5 src/dlightrag/web/static/pygments.css`
Expected: CSS rules starting with `.ai-message-content .highlight .hll { ... }`

- [ ] **Step 2: Update base.html with KaTeX CDN and pygments.css**

Replace the full content of `src/dlightrag/web/templates/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DlightRAG</title>
    <script src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>
    <link rel="stylesheet" href="/static/style.css">

    <!-- KaTeX math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/contrib/auto-render.min.js"></script>

    <!-- Pygments code highlighting theme -->
    <link rel="stylesheet" href="/static/pygments.css">
</head>
<body>
    {% block content %}{% endblock %}
    <script src="/static/citation.js"></script>
</body>
</html>
```

- [ ] **Step 3: Add Markdown content styles to style.css**

Append to the end of `src/dlightrag/web/static/style.css`:

```css
/* === Markdown Content (scoped to AI answer) === */

.ai-message-content h1,
.ai-message-content h2,
.ai-message-content h3,
.ai-message-content h4 {
    color: var(--text-200);
    margin: 1em 0 0.5em;
    line-height: 1.3;
}

.ai-message-content h1 { font-size: 1.4em; font-weight: 600; }
.ai-message-content h2 { font-size: 1.2em; font-weight: 600; }
.ai-message-content h3 { font-size: 1.05em; font-weight: 600; }
.ai-message-content h4 { font-size: 1em; font-weight: 500; }

.ai-message-content table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 13px;
}

.ai-message-content th,
.ai-message-content td {
    border: 1px solid rgba(120, 113, 108, 0.2);
    padding: 8px 12px;
    text-align: left;
}

.ai-message-content th {
    background: var(--bg-elevated);
    color: var(--text-200);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.ai-message-content tr:nth-child(even) {
    background: rgba(28, 25, 23, 0.5);
}

.ai-message-content pre {
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    overflow-x: auto;
    margin: 1em 0;
    font-size: 13px;
    line-height: 1.5;
}

.ai-message-content code {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.9em;
}

.ai-message-content :not(pre) > code {
    background: var(--bg-elevated);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--gold-100);
}

.ai-message-content ul,
.ai-message-content ol {
    margin: 0.5em 0;
    padding-left: 1.5em;
}

.ai-message-content li {
    margin: 0.25em 0;
}

.ai-message-content blockquote {
    border-left: 3px solid var(--gold-500);
    padding-left: 14px;
    margin: 1em 0;
    color: var(--text-400);
}

.ai-message-content hr {
    border: none;
    border-top: 1px solid rgba(120, 113, 108, 0.15);
    margin: 1.5em 0;
}

.ai-message-content p {
    margin: 0.5em 0;
}

.ai-message-content a {
    color: var(--gold-200);
    text-decoration: none;
}

.ai-message-content a:hover {
    text-decoration: underline;
}

/* Pygments highlight container inside answer */
.ai-message-content .highlight {
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
    margin: 1em 0;
    overflow-x: auto;
}

.ai-message-content .highlight pre {
    margin: 0;
    background: transparent;
}
```

- [ ] **Step 4: Add KaTeX trigger to citation.js**

In `src/dlightrag/web/static/citation.js`, find the `handleSSEData` function's `else if (eventType === 'done')` block. Inside that block, locate the `if (sourceData)` section (line ~398-404) that ends with `aiDiv.appendChild(sourceData);` and a closing `}`. Add the KaTeX call **after** that `if (sourceData) { ... }` block but **before** the closing of the `else if (eventType === 'done')` block (i.e., before the next `} else if`):

```javascript
        // Render LaTeX math formulas via KaTeX (client-side)
        if (window.renderMathInElement) {
            renderMathInElement(contentDiv, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                ],
                throwOnError: false,
            });
        }
```

- [ ] **Step 5: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/web/static/pygments.css src/dlightrag/web/templates/base.html src/dlightrag/web/static/style.css src/dlightrag/web/static/citation.js
git commit -m "feat(web): add KaTeX, Pygments CSS, and Markdown styles to frontend"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/unit/ -v`
Expected: all tests PASS, no regressions

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/dlightrag/web/markdown.py src/dlightrag/web/deps.py tests/unit/web/test_markdown.py --fix && uv run ruff format src/dlightrag/web/markdown.py src/dlightrag/web/deps.py tests/unit/web/test_markdown.py`
Expected: clean or auto-fixed

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -u
git commit -m "style: lint fixes for rich content rendering"
```

(Skip if no changes.)

- [ ] **Step 4: Manual verification (if server is available)**

Start the dev server (`dlightrag-api --env-file .env`) and open `http://localhost:8100/web/`. Test with queries that produce:
- **Tables**: "Compare X and Y in a table"
- **Code blocks**: ask for a code example in Python
- **Math**: "Show the formula for $E=mc^2$ and $$\sum_{i=1}^n x_i$$"
- **Mixed citations + code**: verify `[1-2]` inside code blocks is NOT badged

Check that the existing source-chunk `.highlight` class (gold underline in the side panel) still works correctly — this verifies no CSS collision from Pygments.
