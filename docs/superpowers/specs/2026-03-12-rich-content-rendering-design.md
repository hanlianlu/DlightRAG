# Rich Content Rendering for Web UI

## Goal

Add Markdown, table, code highlighting, and LaTeX math rendering to DlightRAG's Web UI answer display, following htmx + Jinja2 server-side rendering best practices.

## Architecture

Server-side rendering for Markdown, tables, and code highlighting via markdown-it-py + Pygments. Client-side KaTeX auto-render for LaTeX math — the only client-side exception, justified by the lack of a production-quality pure-Python LaTeX-to-HTML solution.

Streaming phase remains plain text. The `done` SSE event delivers fully rendered HTML from the server, then KaTeX processes math formulas client-side.

## Rendering Pipeline

### Current

```
LLM answer (raw text)
  → Markup.escape()
  → citation badge injection ([1-2] → badge HTML)
  → client innerHTML replacement
```

### New

```
LLM answer (raw text)
  → markdown-it-py render()        # Markdown → HTML (tables, lists, bold, code blocks)
  → Pygments highlight              # Code block syntax highlighting (via highlight callback)
  → citation badge injection        # Find [1-2] patterns in rendered HTML, inject badges
  → client innerHTML replacement
  → KaTeX renderMathInElement()    # Scan $...$ and $$...$$ for math rendering
```

### Key Decisions

- **`html: False` explicitly set** — the `gfm-like` preset defaults to `html: True`, which would pass raw HTML from LLM output through unescaped (XSS risk). We override with `html: False` so `<script>` etc. are escaped to `&lt;script&gt;`.
- **`linkify` disabled** — `gfm-like` enables `linkify` which requires the `linkify-it-py` package. We disable it to avoid the extra dependency. Bare URLs in LLM output stay as plain text.
- **Citation badges skip `<code>` blocks** — after markdown rendering, the citation regex runs on the full HTML. Without protection, `[1-2]` inside fenced code blocks or inline code would get badge HTML injected, breaking code display. The `_citation_badges` function must temporarily extract `<code>...</code>` and `<pre>...</pre>` content before regex replacement, then restore it.
- **dollarmath plugin NOT enabled** — `$...$` passes through markdown-it-py as literal text, then KaTeX auto-render handles it client-side. This avoids double-processing.
- **Streaming unchanged** — plain text preview via `span.textContent`, fully replaced by rich HTML on `done` event.

## Python Dependencies

Add to `pyproject.toml`:

```
markdown-it-py[plugins]>=3.0    # Markdown rendering + mdit-py-plugins
pygments>=2.17                   # Code syntax highlighting
```

Note: `linkify-it-py` is NOT added — we disable the `linkify` rule instead.

Both are pure Python, zero C extensions. Widely used by MkDocs, Jupyter, Sphinx.

## File Changes

### New Files

**`src/dlightrag/web/markdown.py`**

Initializes markdown-it-py with GFM-like preset and Pygments highlight callback. Exposes a single function:

```python
def render_markdown(text: str) -> str:
    """Convert Markdown text to HTML with syntax-highlighted code blocks."""
```

- Uses `MarkdownIt("gfm-like", {"html": False, "highlight": _highlight_fn}).disable("linkify")`
- `gfm-like` preset already includes GFM tables — no `.enable("table")` needed
- `_highlight_fn` calls Pygments `highlight()` with `HtmlFormatter(cssclass="highlight", wrapcode=True)`
- Falls back to unhighlighted `<pre><code>` if Pygments cannot identify the language

**`src/dlightrag/web/static/pygments.css`**

Static CSS file generated once via:

```bash
pygmentize -S monokai -f html -a '.highlight' > pygments.css
```

Monokai dark theme matches the existing deep-dark UI (`--bg-base: #0c0a09`).

### Modified Files

**`src/dlightrag/web/deps.py`**

`_citation_badges` filter changes:

```python
def _citation_badges(text: str) -> Markup:
    # Before: escaped = str(Markup.escape(str(text)))
    # After:
    html = render_markdown(str(text))
    # Protect <code> and <pre> blocks from citation regex
    html, protected = _protect_code_blocks(html)
    result = CITATION_PATTERN.sub(_chunk_badge, html)
    result = DOC_CITATION_PATTERN.sub(_doc_badge, result)
    result = _restore_code_blocks(result, protected)
    return Markup(result)
```

`_protect_code_blocks` replaces `<code>...</code>` and `<pre>...</pre>` content with placeholders before citation regex runs, then `_restore_code_blocks` puts them back. This prevents `[1-2]` inside code examples from becoming citation badges.

No new Jinja2 filters. The `answer_done.html` template (`{{ answer | citation_badges }}`) remains unchanged.

**`src/dlightrag/web/templates/base.html`**

Add KaTeX CDN and Pygments CSS:

```html
<head>
    <script src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>
    <link rel="stylesheet" href="/static/style.css">

    <!-- KaTeX math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/contrib/auto-render.min.js"></script>

    <!-- Pygments code highlighting theme -->
    <link rel="stylesheet" href="/static/pygments.css">
</head>
```

**`src/dlightrag/web/static/style.css`**

Add scoped Markdown content styles under `.ai-message-content`:

- Headings (`h1`–`h4`): size, weight, margin
- Tables: border, padding, alternating row background matching dark theme
- Code blocks (`pre`): background, border-radius, overflow-x scroll
- Inline code: background pill style
- Lists (`ul`, `ol`): proper indentation and bullet styles
- Blockquotes: left border + muted color
- Horizontal rules

All scoped to `.ai-message-content` to avoid affecting other UI elements.

**`src/dlightrag/web/static/citation.js`**

One addition at the end of the `done` event handler:

```javascript
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

### Unchanged Files

- `answer_done.html` — template interface unchanged
- `routes.py` — routing logic unchanged
- All backend code (service, retriever, config, etc.) — zero impact

## Streaming Behavior

No change. Streaming phase remains plain text (`span.textContent = data`).

User experience: raw Markdown text appears token-by-token during streaming → `done` event instantly replaces with rich HTML (formatted tables, highlighted code, rendered math). This matches common htmx community patterns for LLM chat interfaces.

## Testing

- **Unit test `render_markdown()`**: verify tables, fenced code blocks, inline code, LaTeX `$...$` passthrough (not consumed by markdown-it-py), XSS escaping (`<script>` in input)
- **Unit test `_citation_badges`**: verify citation injection still works after markdown rendering — `[1-2]` in plain text, in code blocks (should NOT become badge), in tables
- **Manual verification**: dark theme visual check for tables, code blocks, math formulas in the Web UI
