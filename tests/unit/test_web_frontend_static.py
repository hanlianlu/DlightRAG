# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""High-signal static checks for browser-side modules.

These tests protect public browser behavior and served asset boundaries. They
avoid pinning exact visual token values or module decomposition details.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_UI = FRONTEND / "ui"
FRONTEND_STYLES = FRONTEND / "styles"


def _css_rule(path: Path, selector: str) -> str:
    css = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.S)
    assert match is not None
    return match.group("body")


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
