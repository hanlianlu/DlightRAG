# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""High-signal static checks for browser-side modules.

These tests protect public browser behavior and served asset boundaries. They
avoid pinning exact visual token values or module decomposition details.
"""

import re
from pathlib import Path
from typing import get_type_hints

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_UI = FRONTEND / "ui"
FRONTEND_STYLES = FRONTEND / "styles"


def _css_rule(path: Path, selector: str) -> str:
    css = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.S)
    assert match is not None
    return match.group("body")


def test_web_answer_frontend_has_no_legacy_session_or_history_payload() -> None:
    chat_source = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "conversation_id" in chat_source
    assert "conversation_history" not in chat_source
    assert "session_id" not in chat_source
    assert not (FRONTEND / "stores" / "sessionStore.ts").exists()
    assert not (FRONTEND_UI / "clearHistory.ts").exists()


def test_conversation_bootstrap_cannot_block_independent_ui_initializers() -> None:
    main_source = (FRONTEND_UI / "main.ts").read_text(encoding="utf-8")
    bootstrap = main_source.index("void conversationStore.initialize()")

    for initializer in (
        "initWorkspaces();",
        "setupPanel();",
        "setupSourcePanel();",
        "setupFilesPanel();",
        "setupPanelResize();",
        "setupHtmxInteractions();",
        "setupImageInputs();",
        "setupQueryForm();",
    ):
        assert main_source.index(initializer) < bootstrap


def test_answer_submission_fails_locally_without_active_conversation() -> None:
    chat_source = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")
    unavailable_guard = chat_source.index("if (!conversationId)")
    answer_fetch = chat_source.index("fetch('/web/answer'")

    assert unavailable_guard < answer_fetch
    guard_body = chat_source[unavailable_guard:answer_fetch]
    assert "conversationStore.errorMessage" in guard_body
    assert "setAnswerError(" in guard_body
    assert "return;" in guard_body


def test_dead_history_and_image_id_frontend_surfaces_are_removed() -> None:
    bus_source = (FRONTEND / "events" / "bus.ts").read_text(encoding="utf-8")
    chat_css = (FRONTEND_STYLES / "chat.module.css").read_text(encoding="utf-8")
    renderer = (FRONTEND / "lib" / "chat_renderer.ts").read_text(encoding="utf-8")

    for legacy_event in ("chatExchangeComplete", "chatHistoryRestored", "chatHistoryCleared"):
        assert legacy_event not in bus_source
    assert ".outOfContext" not in chat_css
    assert ".contextDivider" not in chat_css
    assert "current_image_ids" not in renderer
    assert "imageIds" not in renderer


def test_composer_enter_shortcut_respects_ime_composition() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert ".addEventListener('beforeinput'" in chat_js
    assert "e.inputType === 'insertLineBreak'" in chat_js
    assert "e.isComposing === true" in chat_js
    assert "submitComposerForm(queryForm);" in chat_js
    assert "let textareaIsComposing = false;" not in chat_js
    assert "compositionJustEnded" not in chat_js

    keydown = chat_js.index(".addEventListener('keydown'")
    keydown_block = chat_js[keydown : chat_js.index(".addEventListener('beforeinput'", keydown)]
    assert "submitComposerForm(" not in keydown_block
    assert "form.dispatchEvent(new Event('submit'))" not in keydown_block


def test_web_shell_does_not_block_on_external_cdn_scripts() -> None:
    web_root = ROOT / "src/dlightrag/web"
    base_html = (web_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert 'src="https://' not in base_html
    assert 'src="/static/vendor/htmx.min.js' in base_html
    assert (web_root / "static" / "vendor" / "htmx.min.js").is_file()


def test_web_static_css_build_keeps_only_served_bundles() -> None:
    static_root = ROOT / "src/dlightrag/web/static"
    generated_root = static_root / "generated"

    css_files = {path.name for path in static_root.glob("*.css")}
    generated_css_files = {path.name for path in generated_root.glob("*.css")}

    assert css_files == {"pygments.css"}
    assert generated_css_files == {"style.css"}


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
    download_link = source_panel.index('class="source-dl-icon"')
    button_end = source_panel.index("</button>", button_start)

    assert not button_start < download_link < button_end


def test_source_panel_requires_download_for_every_authorized_source() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
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

    assert html.count('class="source-dl-icon"') == 1
    assert 'href="/web/files/raw/doc-notes?workspace=default"' in html
    assert "{% if src.download_url %}" in source_panel_text


def test_source_panel_hides_download_without_caller_permission() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web.deps import templates

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url=None,
        chunks=[],
    )

    html = templates.env.get_template("partials/source_panel.html").render(sources=[source])

    assert 'class="source-dl-icon"' not in html


def test_source_templates_use_only_the_public_download_contract() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web.answer_events import _AnswerPayload
    from dlightrag.web.safe_html import safe_answer_done, safe_source_panel

    partials = ROOT / "src/dlightrag/web/templates/partials"
    template_text = "\n".join(
        (partials / name).read_text(encoding="utf-8")
        for name in ("source_panel.html", "answer_done.html")
    )

    assert "src.url" not in template_text
    assert "src.path" not in template_text
    assert "src.download_url" in template_text
    assert get_type_hints(_AnswerPayload)["sources"] == list[SourceReferencePayload]
    assert get_type_hints(safe_answer_done)["sources"] == list[SourceReferencePayload]
    assert get_type_hints(safe_source_panel)["sources"] == list[SourceReferencePayload]


def test_active_source_code_has_no_removed_path_fallbacks() -> None:
    engine = (ROOT / "src/dlightrag/core/ingestion/engine.py").read_text(encoding="utf-8")
    answer_media = (ROOT / "src/dlightrag/core/answer_media.py").read_text(encoding="utf-8")

    assert "metadata_path" not in engine
    assert 'getattr(source, "path"' not in answer_media


def test_sanitized_source_download_preserves_accessible_name() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web.safe_html import safe_source_panel

    source = SourceReferencePayload(
        id="1",
        title="notes.md",
        source_uri="local://default/notes.md",
        download_url="/web/files/raw/doc-notes?workspace=default",
        chunks=[],
    )

    html = safe_source_panel(sources=[source])

    assert 'aria-label="Download source"' in html
    assert 'download=""' in html or " download" in html


def test_source_titles_fall_back_without_legacy_paths() -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
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


def test_source_download_aria_allowlist_does_not_allow_unsafe_anchor_attributes() -> None:
    from dlightrag.web.safe_html import sanitize_html_fragment

    html = sanitize_html_fragment(
        '<a href="/web/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_blank">Download</a>'
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
    assert 'class="source-dl-icon-svg"' in source_panel
    assert 'stroke="currentColor"' in source_panel


def test_files_panel_requests_are_controller_owned() -> None:
    index = (ROOT / "src/dlightrag/web/templates/index.html").read_text(encoding="utf-8")
    file_list = (ROOT / "src/dlightrag/web/templates/partials/file_list.html").read_text(
        encoding="utf-8"
    )
    progress = (ROOT / "src/dlightrag/web/templates/partials/ingest_progress.html").read_text(
        encoding="utf-8"
    )

    assert 'id="files-btn"' in index
    assert 'hx-get="/web/files"' not in index

    assert 'hx-post="/web/files/upload"' not in file_list
    assert 'hx-delete="/web/files"' not in file_list
    assert 'data-action="delete-file"' in file_list
    assert "data-file-path=" in file_list
    assert "source-dl-icon" not in file_list
    assert " download" not in file_list

    assert 'id="ingest-progress"' in progress
    assert "hx-get" not in progress
    assert "hx-trigger" not in progress
    assert "hx-sync" not in progress


def test_panel_modules_have_separate_shell_source_and_files_controllers() -> None:
    main_js = (FRONTEND_UI / "main.ts").read_text(encoding="utf-8")
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")
    htmx_js = (FRONTEND_UI / "htmx.ts").read_text(encoding="utf-8")
    files_panel_js = (FRONTEND_UI / "files-panel.ts").read_text(encoding="utf-8")
    source_panel_js = (FRONTEND_UI / "source-panel.ts").read_text(encoding="utf-8")

    assert "setupPanel" in main_js
    assert "setupFilesPanel" in main_js
    assert "setupSourcePanel" in main_js

    for forbidden in (
        "/web/files",
        "upload-form",
        "filterSource",
        "ingestStore",
        "createWorkspace",
    ):
        assert forbidden not in panel_js

    assert "refreshFilePanel" in files_panel_js
    assert "uploadFilesToWorkspace" in files_panel_js
    assert "startIngestPolling" in files_panel_js
    assert "AbortController" in files_panel_js
    assert "/web/files/upload" in files_panel_js
    assert "/web/ingest-status" in files_panel_js

    assert "filterSource" in source_panel_js
    assert "openRefSource" in source_panel_js
    assert "showAllSources" in source_panel_js

    assert "/web/files" not in htmx_js
    assert "isStaleFilePanelResponse" not in htmx_js
    assert "ingestStore" not in htmx_js


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


def test_answer_image_strip_uses_shared_thumbnail_field() -> None:
    answer_done = (ROOT / "src/dlightrag/web/templates/partials/answer_done.html").read_text(
        encoding="utf-8"
    )

    assert "img.thumbnail_url" in answer_done
    assert "img.thumb_url" not in answer_done


def test_panel_resize_uses_pointer_capture_and_cancel_cleanup() -> None:
    resize_js = (FRONTEND_UI / "resize.ts").read_text(encoding="utf-8")

    assert ".setPointerCapture(event.pointerId)" in resize_js
    assert ".releasePointerCapture(activePointerId)" in resize_js
    assert "'pointerId' in e" in resize_js
    assert "pointercancel" in resize_js
    assert "window.addEventListener('blur', finishDrag)" in resize_js
