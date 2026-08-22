# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""High-signal static checks for browser-side modules.

These tests protect public browser behavior and served asset boundaries. They
avoid pinning exact visual token values or module decomposition details.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_UI = FRONTEND / "ui"
FRONTEND_STYLES = FRONTEND / "styles"


def _css_rule(path: Path, selector: str) -> str:
    css = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.S)
    assert match is not None
    return match.group("body")


def test_lit_app_owns_the_final_conversation_shell() -> None:
    app = (FRONTEND_UI / "app.ts").read_text(encoding="utf-8")

    for selector in (
        'id="chat-sidebar"',
        'id="new-conversation-btn"',
        'id="conversation-list"',
        'id="conversation-sidebar-toggle"',
        'id="conversation-sidebar-open"',
        'id="delete-conversation-dialog"',
        'id="delete-all-conversations-btn"',
        'id="delete-all-conversations-dialog"',
        'id="discard-draft-dialog"',
    ):
        assert selector in app
    assert 'aria-label="Conversations"' in app
    assert not (ROOT / "src/dlightrag/web/templates/index.html").exists()
    assert not (ROOT / "src/dlightrag/web/templates/base.html").exists()


def test_lit_app_projects_one_unified_attachment_policy() -> None:
    app = (FRONTEND_UI / "app.ts").read_text(encoding="utf-8")

    for field in (
        "data-attachment-count-limit",
        "data-attachment-image-max-bytes",
        "data-attachment-document-max-bytes",
        "data-attachment-extensions",
        "data-attachment-image-capability",
        "data-attachment-image-limit",
        "attachments.accept",
    ):
        assert field in app
    for stale in (
        "data-effective-current-upload-limit",
        "data-document-current-upload-limit",
        "data-max-upload-bytes",
        "data-answer-image-capability",
    ):
        assert stale not in app


def test_bootstrap_advertises_exact_backend_attachment_limits() -> None:
    bootstrap_source = (ROOT / "src/dlightrag/web/routes/bootstrap.py").read_text(encoding="utf-8")

    assert "count_limit=application.config.answer.generation.max_attachments" in bootstrap_source
    assert (
        "attachment_limit = application.config.answer.generation.max_attachment_bytes"
        in bootstrap_source
    )
    assert "image_max_bytes=attachment_limit" in bootstrap_source
    assert "document_max_bytes=attachment_limit" in bootstrap_source


def test_frontend_submits_only_the_unified_attachments_part() -> None:
    request_builder = (FRONTEND / "lib" / "answer_request.ts").read_text(encoding="utf-8")

    assert "form.append('attachments', file, file.name)" in request_builder
    for source in ("lib/answer_request.ts", "ui/chat.ts"):
        text = (FRONTEND / source).read_text(encoding="utf-8")
        assert "append('images'" not in text
        assert "append('documents'" not in text


def test_vite_html_has_no_external_script_or_unresolved_theme_placeholder() -> None:
    for name in ("index.html", "login.html"):
        source = (FRONTEND / name).read_text(encoding="utf-8")
        built = (ROOT / "src/dlightrag/web/static/app" / name).read_text(encoding="utf-8")
        assert 'src="https://' not in source
        assert "__THEME_INIT__" not in built
        assert re.search(r'/static/app/assets/theme-init-[^"/]+\.js', built)


def test_web_shell_bootstraps_theme_preference_before_app_assets() -> None:
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    theme = (FRONTEND / "theme-init.ts").read_text(encoding="utf-8")
    built = (ROOT / "src/dlightrag/web/static/app/index.html").read_text(encoding="utf-8")

    html_open = re.search(r"<html\b[^>]*>", index)
    assert html_open is not None
    assert 'lang="en"' in html_open.group(0)
    assert 'data-theme="system"' in html_open.group(0)
    assert 'data-color-mode="dark"' in html_open.group(0)
    assert '<meta name="color-scheme" content="dark light">' in index
    assert "'dlightrag-theme'" in theme
    assert "localStorage.getItem" in theme
    assert "matchMedia('(prefers-color-scheme: dark)')" in theme

    theme_script = built.index("/assets/theme-init-")
    app_script = built.index("/assets/app-")
    stylesheet = built.index('<link rel="stylesheet"')
    assert theme_script < stylesheet
    assert theme_script < app_script


def test_web_static_css_build_keeps_only_served_bundles() -> None:
    static_root = ROOT / "src/dlightrag/web/static"
    assets = static_root / "app" / "assets"

    assert {path.name for path in static_root.glob("*.css")} == {"pygments.css"}
    styles = [path.name for path in assets.glob("style-*.css")]
    assert len(styles) == 1


def test_pygments_css_matches_generator() -> None:
    generator_path = ROOT / "scripts" / "generate_pygments_css.py"
    spec = importlib.util.spec_from_file_location("generate_pygments_css", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    css = (ROOT / "src/dlightrag/web/static/pygments.css").read_text(encoding="utf-8")
    assert generator.generate_css() == css


def test_web_static_js_build_has_no_orphan_chunks() -> None:
    app_root = ROOT / "src/dlightrag/web/static/app"
    assets = app_root / "assets"
    import_pattern = re.compile(
        r"""(?:import\(`\./([^`]+\.js)`\)|import\(["']\./([^"']+\.js)["']\)|from["']\./([^"']+\.js)["'])"""
    )
    html = "\n".join(
        (app_root / filename).read_text(encoding="utf-8")
        for filename in ("index.html", "login.html")
    )
    roots = set(re.findall(r'/static/app/assets/([^"/]+\.js)', html))
    expected = {path.name for path in assets.glob("*.js")}
    seen: set[str] = set()
    stack = list(roots)

    while stack:
        filename = stack.pop()
        if filename in seen:
            continue
        seen.add(filename)
        content = (assets / filename).read_text(encoding="utf-8")
        for match in import_pattern.finditer(content):
            child = next(part for part in match.groups() if part)
            if child not in seen:
                stack.append(child)

    assert expected == seen


def test_chat_message_bubbles_wrap_unbroken_queries() -> None:
    user_message = _css_rule(FRONTEND_STYLES / "chat.module.css", ".userMessage")
    wrapper = _css_rule(FRONTEND_STYLES / "chat.module.css", ".userMessageWrapper")
    ai_message = _css_rule(FRONTEND_STYLES / "chat.module.css", ".aiMessageContent")

    assert "width: fit-content;" not in user_message
    assert "max-width: 100%;" in user_message
    assert "min-width: 0;" in user_message
    assert "overflow-wrap: anywhere;" in user_message
    assert "white-space: pre-wrap;" in user_message
    assert "min-width: 0;" in wrapper
    assert "overflow-wrap: anywhere;" in ai_message


def _presentation_source(*, source_uri: str, download_url: str | None = None):
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.presentation import build_answer_presentation

    source = SourceReferencePayload(
        id="1",
        title=None,
        source_uri=source_uri,
        download_url=download_url,
        chunks=[],
    )
    return build_answer_presentation(
        answer="Cited [1].",
        sources=[source],
        answer_images=[],
    ).sources[0]


def test_presentation_preserves_authorized_download_without_nesting_markup() -> None:
    source = _presentation_source(
        source_uri="local://default/notes.md",
        download_url="/web/api/files/raw/doc-notes?workspace=default",
    )
    source_view = (FRONTEND_UI / "source_panel_view.ts").read_text(encoding="utf-8")

    assert source.download_url == "/web/api/files/raw/doc-notes?workspace=default"
    assert source.title == "Source"
    assert "safeSameOriginHref(source.download_url)" in source_view
    assert 'aria-label="Download source"' in source_view
    assert "<a href=${download}" in source_view


def test_presentation_hides_download_without_caller_permission() -> None:
    source = _presentation_source(source_uri="local://default/notes.md")

    assert source.download_url is None


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://exa.ai/library/weather/gothenburg-sweden?latitude=57.7052&longitude=11.9737",
        "http://www.sgas.ruc.edu.cn/xwgg/yjyxw/f1a3ff59a5894391b7b0db77951c08b4.htm",
    ],
)
def test_presentation_projects_public_web_provenance(source_uri: str) -> None:
    source = _presentation_source(source_uri=source_uri)

    assert source.source_url == source_uri


def test_presentation_rejects_non_public_provenance() -> None:
    for value in (
        "local://default/report.pdf",
        "https://127.0.0.1/private",
        "res-opaque",
    ):
        assert _presentation_source(source_uri=value).source_url is None


def test_answer_presentation_uses_semantic_citations_and_no_legacy_paths() -> None:
    from dlightrag.web.presentation import build_answer_presentation

    presentation = build_answer_presentation(
        answer="Answer [1].",
        sources=[],
        answer_images=[],
    )
    source_view = (FRONTEND_UI / "source_panel_view.ts").read_text(encoding="utf-8")
    answer_view = (FRONTEND_UI / "answer_presentation.ts").read_text(encoding="utf-8")

    assert '<cite class="citation-badge"' in presentation.answer_html
    assert "src.path" not in source_view
    assert "answer-ref-item" in answer_view
    assert "source-doc-badge" in source_view
    assert "{{" not in source_view + answer_view


def test_source_anchor_allowlist_rejects_unsafe_attributes_and_targets() -> None:
    from dlightrag.web.safe_html import sanitize_html_fragment

    html = sanitize_html_fragment(
        '<a href="/web/api/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_self">Download</a>'
    )

    assert 'aria-label="Download source"' in html
    assert "onclick" not in html
    assert "style=" not in html
    assert "target=" not in html


def test_panel_action_icons_are_accessible_svg_buttons() -> None:
    file_panel = (FRONTEND_UI / "files-panel.ts").read_text(encoding="utf-8")
    source_panel = (FRONTEND_UI / "source_panel_view.ts").read_text(encoding="utf-8")

    assert "&#10005;" not in file_panel
    assert "&#x2B07;" not in source_panel
    assert "aria-label=${`Delete ${file.file_name}`}" in file_panel
    assert 'class="file-delete-icon"' in file_panel
    assert 'class="source-action-icon-svg"' in source_panel
    assert 'stroke="currentColor"' in source_panel


def test_history_images_are_lazy_async_thumbnails_with_on_demand_originals() -> None:
    images_source = (FRONTEND_UI / "images.ts").read_text(encoding="utf-8")

    assert "imgEl.loading = 'lazy'" in images_source
    assert "imgEl.decoding = 'async'" in images_source
    assert "imgEl.src = thumbnailSrc" in images_source
    assert "imageButton.setAttribute('data-full-src', fullSrc)" in images_source


def test_split_panel_adapter_preserves_cancel_and_compact_guards() -> None:
    split_panel = (FRONTEND_UI / "split_panel.ts").read_text(encoding="utf-8")

    assert not (FRONTEND_UI / "resize.ts").exists()
    assert "document.dispatchEvent(new Event('pointerup'))" in split_panel
    assert "['pointercancel', 'touchcancel', 'blur']" in split_panel
    assert "['pointerup', 'mouseup', 'touchend']" in split_panel
    assert "if (state.split.disabled) return" in split_panel
    assert "event.stopImmediatePropagation()" in split_panel


def _css_blocks() -> list[tuple[str, str]]:
    """Every `selector { declarations }` pair across the served stylesheets."""
    blocks: list[tuple[str, str]] = []
    for sheet in sorted(FRONTEND_STYLES.rglob("*.css")):
        css = re.sub(r"/\*.*?\*/", "", sheet.read_text(encoding="utf-8"), flags=re.S)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            blocks.append((selector.strip(), body))
    return blocks


def _declarations(body: str) -> dict[str, str]:
    decls: dict[str, str] = {}
    for line in body.split(";"):
        name, _, value = line.partition(":")
        name, value = name.strip().lower(), value.strip()
        if not name or not value:
            continue
        decls[name] = value
        if name == "border":
            # A base `border: 1px solid X` is what a hover `border-color` must beat.
            decls.setdefault("border-color", value.split()[-1])
    return decls


def test_jinja_template_tree_is_deleted() -> None:
    assert not (ROOT / "src/dlightrag/web/templates").exists()


def test_production_web_sources_have_no_htmx_contract() -> None:
    sources = [
        *FRONTEND.rglob("*.ts"),
        *FRONTEND.rglob("*.html"),
        *(ROOT / "src/dlightrag/web").rglob("*.py"),
    ]
    for path in sources:
        if "node_modules" in path.parts:
            continue
        source = path.read_text(encoding="utf-8").lower()
        assert "htmx" not in source
        assert not re.search(r"\bhx-[a-z]", source)


def test_webawesome_adoption_is_limited_to_split_panel_without_default_theme() -> None:
    imports = [
        (path.relative_to(FRONTEND), line.strip())
        for path in FRONTEND.rglob("*.ts")
        if "node_modules" not in path.parts
        for line in path.read_text(encoding="utf-8").splitlines()
        if "@awesome.me/webawesome" in line
    ]
    assert imports == [
        (
            Path("ui/split_panel.ts"),
            "import WaSplitPanel from "
            "'@awesome.me/webawesome/dist/components/split-panel/split-panel.js';",
        )
    ]

    production_css = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (FRONTEND_STYLES, FRONTEND / "tokens")
        for path in root.glob("*.css")
    )
    assert "@awesome.me/webawesome" not in production_css
    assert "webawesome/dist/styles" not in production_css


def test_button_hover_rules_change_something() -> None:
    """A hover that restates the base is the same as having no hover at all."""
    blocks = _css_blocks()
    base = {sel: _declarations(body) for sel, body in blocks if ":hover" not in sel}

    for selector, body in blocks:
        if ":hover" not in selector:
            continue
        hover = _declarations(body)
        if not hover:
            continue
        for part in selector.split(","):
            root = part.strip().split(":hover")[0].strip()
            if root not in base:
                continue
            changed = any(base[root].get(prop) != value for prop, value in hover.items())
            assert changed, f"{part.strip()} restates {root} and renders no feedback"
