# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""High-signal static checks for browser-side modules.

These tests protect public browser behavior and served asset boundaries. They
avoid pinning exact visual token values or module decomposition details.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_UI = FRONTEND / "ui"
FRONTEND_LIB = FRONTEND / "lib"
FRONTEND_STORES = FRONTEND / "stores"
FRONTEND_STYLES = FRONTEND / "styles"


def _css_rule(path: Path, selector: str) -> str:
    css = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.S)
    assert match is not None
    return match.group("body")


def test_chat_submit_does_not_auto_close_open_panel() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "closePanel();" not in chat_js
    assert "import {closePanel} from './panel.ts';" not in chat_js


def test_composer_enter_shortcut_respects_ime_composition() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "textarea.addEventListener('beforeinput'" in chat_js
    assert "e.inputType === 'insertLineBreak'" in chat_js
    assert "e.isComposing === true" in chat_js
    assert "submitComposerForm(form);" in chat_js
    assert "let textareaIsComposing = false;" not in chat_js
    assert "compositionJustEnded" not in chat_js

    keydown = chat_js.index("textarea.addEventListener('keydown'")
    keydown_block = chat_js[
        keydown : chat_js.index("textarea.addEventListener('beforeinput'", keydown)
    ]
    assert "submitComposerForm(form)" not in keydown_block
    assert "form.dispatchEvent(new Event('submit'))" not in keydown_block


def test_web_shell_does_not_block_on_external_cdn_scripts() -> None:
    web_root = ROOT / "src/dlightrag/web"
    base_html = (web_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert 'src="https://' not in base_html
    assert 'src="/static/vendor/htmx.min.js' in base_html
    assert (web_root / "static" / "vendor" / "htmx.min.js").is_file()


def test_web_static_css_build_keeps_only_served_bundles() -> None:
    static_root = ROOT / "src/dlightrag/web/static"

    css_files = {path.name for path in static_root.glob("*.css")}

    assert css_files == {"pygments.css", "style.css"}


def test_web_static_js_build_has_no_orphan_chunks() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
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


def test_unused_ingest_progress_frontend_contract_is_removed() -> None:
    web_root = ROOT / "src/dlightrag/web"
    checked = [
        web_root / "routes" / "files.py",
        *(FRONTEND_UI).glob("*.ts"),
        *(FRONTEND_LIB).glob("*.ts"),
        *(FRONTEND_STORES).glob("*.ts"),
        *(FRONTEND).glob("vite.config.ts"),
        *(FRONTEND).glob("types.d.ts"),
        *(web_root / "templates").rglob("*.html"),
    ]

    offenders = [path for path in checked if "ingest/progress" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_workspace_management_uses_topbar_selector_not_side_panel() -> None:
    web_root = ROOT / "src/dlightrag/web"
    index_html = (web_root / "templates" / "index.html").read_text(encoding="utf-8")
    workspaces_js = (FRONTEND_UI / "workspaces.ts").read_text(encoding="utf-8")

    assert 'id="workspace-selector"' in index_html
    assert "workspace-chips" not in index_html
    assert not (web_root / "templates" / "partials" / "workspace_list.html").exists()
    assert "workspaceStore" in workspaces_js
    assert "data-all" in workspaces_js


def test_workspace_delete_removes_canonical_workspace_and_ingest_target() -> None:
    workspaces_js = (FRONTEND_UI / "workspaces.ts").read_text(encoding="utf-8")
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")

    assert "removeWorkspace(workspace, detail.next_workspace)" in workspaces_js
    assert "workspaceStore.remove(workspace" in workspaces_js
    assert "workspaceStore.active.indexOf(" in workspaces_js
    assert "workspaceDeleted" in panel_js
    assert "ingestStore.set(detail.next_workspace || workspaceStore.primary)" in panel_js


def test_panel_auto_dismiss_keeps_composer_interactive() -> None:
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")

    assert "PANEL_DISMISS_EXEMPT_SELECTOR" in panel_js
    assert "#composer" in panel_js
    assert "#files-btn" in panel_js
    assert ".panel" in panel_js
    assert "shouldDismissPanelOnOutsideClick(e.target)" in panel_js


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


def test_reference_labels_do_not_render_square_brackets() -> None:
    partials = ROOT / "src/dlightrag/web/templates/partials"
    answer_done = (partials / "answer_done.html").read_text(encoding="utf-8")
    source_panel = (partials / "source_panel.html").read_text(encoding="utf-8")

    assert "[{{ src.id }}]" not in answer_done
    assert "[{{ src.id }}]" not in source_panel
    assert '<span class="answer-ref-id">{{ src.id | reference_label }}</span>' in answer_done
    assert '<span class="source-doc-badge">{{ src.id | reference_label }}</span>' in source_panel


def test_composer_autoresize_measures_content_after_height_reset() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    height_reset = chat_js.index("textarea.style.height = 'auto';")
    content_measure = chat_js.index("const contentHeight = textarea.scrollHeight;")
    multiline_update = chat_js.index("form.classList.toggle('multiline'", content_measure)
    height_apply = chat_js.index(
        "textarea.style.height = Math.min(contentHeight, maxHeight) + 'px';"
    )

    assert height_reset < content_measure < multiline_update < height_apply


def test_panel_resize_uses_pointer_capture_and_cancel_cleanup() -> None:
    resize_js = (FRONTEND_UI / "resize.ts").read_text(encoding="utf-8")

    assert "handle.setPointerCapture(e.pointerId)" in resize_js
    assert "handle.releasePointerCapture(activePointerId)" in resize_js
    assert "'pointerId' in e" in resize_js
    assert "pointercancel" in resize_js
    assert "window.addEventListener('blur', finishDrag)" in resize_js
