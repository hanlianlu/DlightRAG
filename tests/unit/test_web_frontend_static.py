# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""High-signal static checks for browser-side modules.

These tests protect public browser behavior and served asset boundaries. They
avoid pinning exact visual token values or module decomposition details.
"""

import importlib.util
import re
from pathlib import Path
from typing import get_type_hints

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


def test_index_has_final_conversation_shell() -> None:
    template = (ROOT / "src" / "dlightrag" / "web" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )

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
        assert selector in template
    assert 'aria-label="Conversations"' in template


def test_index_advertises_one_unified_attachment_policy() -> None:
    from dlightrag.web.deps import templates

    html = templates.env.get_template("index.html").render(
        request=None,
        workspace="default",
        workspaces=[],
        primary_workspace="default",
        active_workspaces=["default"],
        query_attachment_count_limit=6,
        query_attachment_image_max_bytes=15728640,
        query_attachment_document_max_bytes=104857600,
        query_attachment_extensions=["md", "pdf"],
        query_attachment_image_capability="supported",
        query_attachment_image_limit=3,
        query_attachment_accept="image/*,.md,.pdf",
    )

    # One collection: unified count + per-item byte limits + formats + image capability.
    assert 'data-attachment-count-limit="6"' in html
    assert 'data-attachment-image-max-bytes="15728640"' in html
    assert 'data-attachment-document-max-bytes="104857600"' in html
    assert 'data-attachment-image-capability="supported"' in html
    assert 'data-attachment-image-limit="3"' in html
    assert '"md"' in html and '"pdf"' in html
    assert 'accept="image/*,.md,.pdf"' in html

    # The split image/document admission surface is gone.
    for stale in (
        "data-effective-current-upload-limit",
        "data-document-current-upload-limit",
        "data-max-upload-bytes",
        "data-answer-image-capability",
    ):
        assert stale not in html


def test_index_route_advertises_exact_backend_attachment_limits() -> None:
    chat_source = (ROOT / "src/dlightrag/web/routes/chat.py").read_text(encoding="utf-8")

    assert '"query_attachment_count_limit": manager.config.answer.max_attachments,' in chat_source
    assert chat_source.count("manager.config.answer.max_attachment_bytes,") >= 2


def test_frontend_submits_only_the_unified_attachments_part() -> None:
    frontend = ROOT / "frontend"
    request_builder = (frontend / "lib" / "answer_request.ts").read_text(encoding="utf-8")

    assert "form.append('attachments', file, file.name)" in request_builder
    # No split image/document parts in the submission path.
    for source in ("lib/answer_request.ts", "ui/chat.ts"):
        text = (frontend / source).read_text(encoding="utf-8")
        assert "append('images'" not in text
        assert "append('documents'" not in text


def test_web_shell_does_not_block_on_external_cdn_scripts() -> None:
    web_root = ROOT / "src/dlightrag/web"
    base_html = (web_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert 'src="https://' not in base_html


def test_web_shell_bootstraps_theme_preference_before_stylesheet() -> None:
    base_html = (ROOT / "src" / "dlightrag" / "web" / "templates" / "base.html").read_text(
        encoding="utf-8"
    )

    html_open = re.search(r"<html\b[^>]*>", base_html)
    assert html_open is not None
    html_open_tag = html_open.group(0)
    assert 'lang="en"' in html_open_tag
    assert 'data-theme="system"' in html_open_tag
    assert 'data-color-mode="dark"' in html_open_tag
    assert '<meta name="color-scheme" content="dark light">' in base_html
    assert "'dlightrag-theme'" in base_html
    assert "localStorage.getItem" in base_html
    assert "matchMedia('(prefers-color-scheme: dark)')" in base_html

    bootstrap = base_html.index("localStorage.getItem")
    first_stylesheet = base_html.index('<link rel="stylesheet" href="/static/generated/style.css')
    assert bootstrap < first_stylesheet


def test_web_static_css_build_keeps_only_served_bundles() -> None:
    static_root = ROOT / "src/dlightrag/web/static"
    generated_root = static_root / "generated"

    css_files = {path.name for path in static_root.glob("*.css")}
    generated_css_files = {path.name for path in generated_root.glob("*.css")}

    assert css_files == {"pygments.css"}
    assert generated_css_files == {"style.css"}


def test_pygments_css_matches_generator() -> None:
    generator_path = ROOT / "scripts" / "generate_pygments_css.py"
    spec = importlib.util.spec_from_file_location("generate_pygments_css", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    css = (ROOT / "src/dlightrag/web/static/pygments.css").read_text(encoding="utf-8")
    assert generator.generate_css() == css


def test_web_static_js_build_has_no_orphan_chunks() -> None:
    static_js = ROOT / "src/dlightrag/web/static/generated/js"
    import_pattern = re.compile(r"""(?:import\(`\./([^`]+\.js)`\)|from"\./([^"]+\.js)")""")

    expected = {path.name for path in static_js.glob("*.js")}
    seen: set[str] = set()
    stack = ["main.js"]

    while stack:
        filename = stack.pop()
        if filename in seen:
            continue
        seen.add(filename)
        content = (static_js / filename).read_text(encoding="utf-8")
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


def test_source_panel_does_not_nest_download_links_inside_toggle_buttons() -> None:
    source_panel = (ROOT / "src/dlightrag/web/templates/partials/source_panel.html").read_text(
        encoding="utf-8"
    )

    button_start = source_panel.index('data-action="toggle-doc"')
    download_link = source_panel.index('class="source-action-icon"')
    button_end = source_panel.index("</button>", button_start)

    assert not button_start < download_link < button_end


def test_source_panel_requires_download_for_every_authorized_source() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.deps import templates

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url="/web/files/raw/doc-notes?workspace=default",
        chunks=[],
    )

    html = templates.env.get_template("partials/source_panel.html").render(sources=[source])
    source_panel_text = (ROOT / "src/dlightrag/web/templates/partials/source_panel.html").read_text(
        encoding="utf-8"
    )

    assert html.count('class="source-action-icon"') == 1
    assert 'href="/web/files/raw/doc-notes?workspace=default"' in html
    assert "{% if src.download_url %}" in source_panel_text


def test_source_panel_hides_download_without_caller_permission() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.deps import templates

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url=None,
        chunks=[],
    )

    html = templates.env.get_template("partials/source_panel.html").render(sources=[source])

    assert 'class="source-action-icon"' not in html


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://exa.ai/library/weather/gothenburg-sweden?latitude=57.7052&longitude=11.9737",
        "http://www.sgas.ruc.edu.cn/xwgg/yjyxw/f1a3ff59a5894391b7b0db77951c08b4.htm",
    ],
)
def test_source_panel_links_public_web_provenance_without_download_permission(
    source_uri: str,
) -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_answer_done

    source = SourceReferencePayload(
        id="1",
        title="Gothenburg weather",
        source_uri=source_uri,
        download_url=None,
        chunks=[],
    )

    html = safe_answer_done(answer="Cited.", sources=[source], answer_images=[])

    assert 'aria-label="Open source"' in html
    assert source_uri.split("?", 1)[0] in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    assert " download" not in html


def test_source_panel_does_not_link_non_public_provenance() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_answer_done

    sources = [
        SourceReferencePayload(id="1", source_uri="local://default/report.pdf", chunks=[]),
        SourceReferencePayload(id="2", source_uri="https://127.0.0.1/private", chunks=[]),
        SourceReferencePayload(id="3", source_uri="res-opaque", chunks=[]),
    ]

    html = safe_answer_done(answer="Cited.", sources=sources, answer_images=[])

    assert 'aria-label="Open source"' not in html
    assert "https://127.0.0.1/private" not in html


def test_source_templates_use_the_public_source_contract() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_answer_done

    partials = ROOT / "src/dlightrag/web/templates/partials"
    template_text = "\n".join(
        (partials / name).read_text(encoding="utf-8")
        for name in ("source_panel.html", "answer_done.html")
    )

    assert "src.url" not in template_text
    assert "src.path" not in template_text
    assert "src.download_url" in template_text
    assert "src.source_uri" in template_text
    assert get_type_hints(safe_answer_done)["sources"] == list[SourceReferencePayload]


def test_sanitized_source_download_preserves_accessible_name() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_answer_done

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url="/web/files/raw/doc-notes?workspace=default",
        chunks=[],
    )

    html = safe_answer_done(answer="Cited.", sources=[source], answer_images=[])

    assert 'aria-label="Download source"' in html
    assert 'download=""' in html or " download" in html


def test_source_titles_fall_back_without_legacy_paths() -> None:
    from dlightrag.answer.citations.schemas import SourceReferencePayload
    from dlightrag.web.deps import templates

    source = SourceReferencePayload(
        id="1",
        source_uri="local://default/notes.md",
        download_url="/web/files/raw/doc-notes?workspace=default",
        chunks=[],
    )
    partials = ROOT / "src/dlightrag/web/templates/partials"

    answer_html = templates.env.get_template("partials/answer_done.html").render(
        answer="Answer [1].",
        sources=[source],
        answer_images=[],
    )
    source_html = templates.env.get_template("partials/source_panel.html").render(sources=[source])

    assert '<span class="answer-ref-title">Source</span>' in answer_html
    assert '<span class="source-doc-title">Source</span>' in source_html
    assert "src.path" not in (partials / "answer_done.html").read_text(encoding="utf-8")
    assert "src.path" not in (partials / "source_panel.html").read_text(encoding="utf-8")


def test_source_anchor_allowlist_rejects_unsafe_attributes_and_targets() -> None:
    from dlightrag.web.safe_html import sanitize_html_fragment

    html = sanitize_html_fragment(
        '<a href="/web/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_self">Download</a>'
    )

    assert 'aria-label="Download source"' in html
    assert "onclick" not in html
    assert "style=" not in html
    assert "target=" not in html


def test_panel_action_icons_are_accessible_svg_buttons() -> None:
    file_list = (ROOT / "src/dlightrag/web/templates/partials/file_list.html").read_text(
        encoding="utf-8"
    )
    source_panel = (ROOT / "src/dlightrag/web/templates/partials/source_panel.html").read_text(
        encoding="utf-8"
    )

    assert "&#10005;" not in file_list
    assert "&#x2B07;" not in source_panel
    assert 'aria-label="Delete {{ file.file_name }}"' in file_list
    assert 'class="file-delete-icon"' in file_list
    assert 'class="source-action-icon-svg"' in source_panel
    assert 'stroke="currentColor"' in source_panel


def test_reference_labels_do_not_render_square_brackets() -> None:
    partials = ROOT / "src/dlightrag/web/templates/partials"
    answer_done = (partials / "answer_done.html").read_text(encoding="utf-8")
    source_panel = (partials / "source_panel.html").read_text(encoding="utf-8")

    assert "[{{ src.id }}]" not in answer_done
    assert "[{{ src.id }}]" not in source_panel
    assert 'class="answer-ref-item"' in answer_done
    assert 'data-action="open-ref-source"' in answer_done
    assert 'role="button"' in answer_done
    assert 'tabindex="0"' in answer_done
    assert '<span class="answer-ref-id">{{ src.id | reference_label }}</span>' in answer_done
    assert '<span class="source-doc-badge">{{ src.id | reference_label }}</span>' in source_panel


def test_history_images_are_lazy_async_thumbnails_with_on_demand_originals() -> None:
    images_source = (FRONTEND_UI / "images.ts").read_text(encoding="utf-8")

    assert "imgEl.loading = 'lazy'" in images_source
    assert "imgEl.decoding = 'async'" in images_source
    assert "imgEl.src = thumbnailSrc" in images_source
    assert "imageButton.setAttribute('data-full-src', fullSrc)" in images_source


def test_panel_resize_uses_pointer_capture_and_cancel_cleanup() -> None:
    resize_js = (FRONTEND_UI / "resize.ts").read_text(encoding="utf-8")

    assert ".setPointerCapture(event.pointerId)" in resize_js
    assert ".releasePointerCapture(activePointerId)" in resize_js
    assert "'pointerId' in e" in resize_js
    assert "pointercancel" in resize_js
    assert "window.addEventListener('blur', finishDrag)" in resize_js


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


def test_every_template_button_has_a_hover_rule() -> None:
    """A control with no hover feedback reads as inert."""
    templates = ROOT / "src" / "dlightrag" / "web" / "templates"
    hover_selectors = [sel for sel, _ in _css_blocks() if ":hover" in sel]

    for template in sorted(templates.rglob("*.html")):
        for tag in re.findall(r"<button\b[^>]*>", template.read_text(encoding="utf-8")):
            classes = re.search(r'class="([^"]*)"', tag)
            names = classes.group(1).split() if classes else []
            covered = any(
                f".{name}" in selector for name in names for selector in hover_selectors
            ) or any("button:hover" in selector for selector in hover_selectors)
            assert covered, f"{template.name}: button {tag} has no hover rule"


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
