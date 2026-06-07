# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Static regression checks for browser-side modules."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_chat_module_imports_close_panel_before_using_it() -> None:
    chat_js = (ROOT / "src/dlightrag/web/static/js/chat.js").read_text(encoding="utf-8")

    assert "closePanel();" in chat_js
    assert "import {closePanel} from './panel.js';" in chat_js


def test_chat_streaming_is_split_into_transport_and_renderer_modules() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
    chat_js = (static_js / "chat.js").read_text(encoding="utf-8")

    assert (static_js / "sse.js").is_file()
    assert (static_js / "chat_renderer.js").is_file()
    assert "from './sse.js'" in chat_js
    assert "from './chat_renderer.js'" in chat_js
    assert "response.body.getReader()" not in chat_js
    assert "contentDiv.innerHTML" not in chat_js


def test_composer_enter_shortcut_respects_ime_composition() -> None:
    chat_js = (ROOT / "src/dlightrag/web/static/js/chat.js").read_text(encoding="utf-8")

    assert "function isImeCompositionKeyEvent(e)" in chat_js
    assert "e.isComposing === true" in chat_js
    assert "e.keyCode === 229" in chat_js
    assert "let textareaIsComposing = false;" in chat_js
    assert "textarea.addEventListener('compositionstart'" in chat_js
    assert "textarea.addEventListener('compositionend'" in chat_js

    enter_check = chat_js.index("if (e.key === 'Enter' && !e.shiftKey)")
    ime_guard = chat_js.index(
        "if (textareaIsComposing || isImeCompositionKeyEvent(e))", enter_check
    )
    submit_dispatch = chat_js.index("form.dispatchEvent(new Event('submit'))", enter_check)
    assert enter_check < ime_guard < submit_dispatch


def test_htmx_behavior_lives_in_dedicated_module() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
    main_js = (static_js / "main.js").read_text(encoding="utf-8")
    panel_js = (static_js / "panel.js").read_text(encoding="utf-8")

    assert (static_js / "htmx.js").is_file()
    assert "setupHtmxInteractions" in main_js
    assert "htmx:afterRequest" not in panel_js


def test_web_shell_does_not_block_on_external_cdn_scripts() -> None:
    web_root = ROOT / "src/dlightrag/web"
    base_html = (web_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert 'src="https://' not in base_html
    assert 'src="/static/vendor/htmx.min.js' in base_html
    assert (web_root / "static" / "vendor" / "htmx.min.js").is_file()


def test_vendored_htmx_version_matches_license_metadata() -> None:
    vendor_root = ROOT / "src/dlightrag/web/static/vendor"
    htmx_js = (vendor_root / "htmx.min.js").read_text(encoding="utf-8")
    licenses = (vendor_root / "LICENSES.md").read_text(encoding="utf-8")

    version = re.search(r'version:"([^"]+)"', htmx_js)

    assert version is not None
    assert f"## htmx {version.group(1)}" in licenses
    assert f"https://cdn.jsdelivr.net/npm/htmx.org@{version.group(1)}/dist/htmx.min.js" in licenses


def test_unused_ingest_progress_frontend_contract_is_removed() -> None:
    web_root = ROOT / "src/dlightrag/web"
    checked = [
        web_root / "routes" / "files.py",
        *(web_root / "static" / "js").glob("*.js"),
        *(web_root / "templates").rglob("*.html"),
    ]

    offenders = [path for path in checked if "ingest/progress" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_workspace_management_uses_topbar_selector_not_side_panel() -> None:
    web_root = ROOT / "src/dlightrag/web"
    index_html = (web_root / "templates" / "index.html").read_text(encoding="utf-8")
    workspaces_js = (web_root / "static" / "js" / "workspaces.js").read_text(encoding="utf-8")

    assert 'id="workspace-selector"' in index_html
    assert "workspace-chips" not in index_html
    assert not (web_root / "templates" / "partials" / "workspace_list.html").exists()
    assert "dlightrag_workspace_ids" in workspaces_js
    assert "document.cookie" in workspaces_js


def test_workspace_delete_removes_canonical_workspace_and_ingest_target() -> None:
    web_root = ROOT / "src/dlightrag/web"
    static_js = web_root / "static" / "js"
    workspaces_js = (static_js / "workspaces.js").read_text(encoding="utf-8")
    panel_js = (static_js / "panel.js").read_text(encoding="utf-8")

    assert "removeWorkspace(workspace, detail.next_workspace)" in workspaces_js
    assert "item.workspace !== workspace" in workspaces_js
    assert "active !== workspace" in workspaces_js

    assert "workspaceDeleted" in panel_js
    assert "setIngestWorkspace(detail.next_workspace || getPrimaryWorkspace())" in panel_js


def test_panel_auto_dismiss_keeps_composer_interactive() -> None:
    panel_js = (ROOT / "src/dlightrag/web/static/js/panel.js").read_text(encoding="utf-8")

    assert "PANEL_DISMISS_EXEMPT_SELECTOR" in panel_js
    assert "#composer" in panel_js
    assert "#files-btn" in panel_js
    assert ".panel" in panel_js
    assert "shouldDismissPanelOnOutsideClick(e.target)" in panel_js


def test_manual_folder_upload_panel_swap_processes_htmx_fragments() -> None:
    panel_js = (ROOT / "src/dlightrag/web/static/js/panel.js").read_text(encoding="utf-8")
    folder_upload_js = (ROOT / "src/dlightrag/web/static/js/folder-upload.js").read_text(
        encoding="utf-8"
    )

    assert "export function applyPanelHtml(html)" in panel_js
    assert "htmx.process(panelContent)" in panel_js
    assert "import {applyPanelHtml} from './panel.js';" in folder_upload_js
    assert "applyPanelHtml(html)" in folder_upload_js
    assert "getElementById('panel-content')" not in folder_upload_js
    assert "innerHTML = html" not in folder_upload_js
    assert "htmx.process" not in folder_upload_js


def test_source_panel_math_rendering_is_lazy_and_scoped() -> None:
    panel_js = (ROOT / "src/dlightrag/web/static/js/panel.js").read_text(encoding="utf-8")
    chat_renderer_js = (ROOT / "src/dlightrag/web/static/js/chat_renderer.js").read_text(
        encoding="utf-8"
    )

    assert "import {renderMath} from './mathjax.js';" in panel_js
    assert "from './chat_renderer.js'" not in panel_js
    assert "from './images.js'" not in panel_js
    assert "renderExpandedSourceMath" in panel_js
    assert "renderMath(panelContent)" not in panel_js
    assert "renderMath(sourceData)" not in chat_renderer_js


def test_lightbox_click_delegation_lives_in_images_module() -> None:
    static_js = ROOT / "src/dlightrag/web/static/js"
    images_js = (static_js / "images.js").read_text(encoding="utf-8")
    panel_js = (static_js / "panel.js").read_text(encoding="utf-8")
    chat_renderer_js = (static_js / "chat_renderer.js").read_text(encoding="utf-8")

    assert 'data-action="open-lightbox"' in images_js
    assert "getAttribute('data-full-src')" in images_js
    assert "openLightbox" not in panel_js
    assert "openLightbox" not in chat_renderer_js


def test_source_panel_does_not_nest_download_links_inside_toggle_buttons() -> None:
    source_panel = (ROOT / "src/dlightrag/web/templates/partials/source_panel.html").read_text(
        encoding="utf-8"
    )

    button_start = source_panel.index('data-action="toggle-doc"')
    download_link = source_panel.index('class="source-dl-icon"')
    button_end = source_panel.index("</button>", button_start)

    assert not button_start < download_link < button_end


def test_files_panel_controls_use_shared_utility_typography() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    assert ".topbar-btn," in css
    assert ".upload-btn {" in css
    assert ".show-all-btn" in css
    shared_block = css.split(".topbar-btn,", 1)[1].split("}", 1)[0]
    assert ".upload-btn" in shared_block
    assert ".show-all-btn" in shared_block
    assert "font-size: var(--font-size-caption);" in shared_block
    assert "font-weight: 600;" in shared_block
    assert "letter-spacing: 0.04em;" in shared_block
    assert "text-transform: uppercase;" in shared_block


def test_mobile_panel_is_full_width_overlay_not_squeezed_layout() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    assert "@media (max-width: 640px)" in css
    mobile_block = css.split("@media (max-width: 640px)", 1)[1]
    assert "width: 100vw;" in mobile_block
    assert "max-width: none;" in mobile_block
    assert "min-width: 0;" in mobile_block
    assert "margin-right: 0;" in mobile_block
    assert ".panel-resize-handle" in mobile_block
    assert "display: none;" in mobile_block


def test_topbar_controls_have_stable_mobile_flex_behavior() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    workspace_block = css.split(".workspace-selector {", 1)[1].split("}", 1)[0]
    topbar_button_block = css.split(".topbar-btn {", 1)[1].split("}", 1)[0]

    assert "min-width: 0;" in workspace_block
    assert "flex: 0 1 auto;" in workspace_block
    assert "flex-shrink: 0;" in topbar_button_block


def test_panel_action_icons_are_svg_buttons_not_text_glyphs() -> None:
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


def test_source_panel_default_gold_accents_are_restrained() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    source_header_hover = css.split(".source-doc-header:hover {", 1)[1].split("}", 1)[0]
    source_chunk_block = css.split(".source-chunk {", 1)[1].split("}", 1)[0]

    assert "background: var(--color-bg-hover);" in source_header_hover
    assert "border-left: 3px solid rgba(210, 182, 97, 0.24);" in source_chunk_block
    assert "border-left: 3px solid var(--color-gold-400);" not in css


def test_source_panel_active_chunk_uses_subtle_selection_not_gold_fill() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    active_chunk_block = css.split(".source-chunk.active {", 1)[1].split("}", 1)[0]

    assert "border-left-color: rgba(210, 182, 97, 0.42);" in active_chunk_block
    assert "box-shadow: inset 1px 0 0 rgba(210, 182, 97, 0.18);" in active_chunk_block
    assert "background: rgba(87, 74, 36, 0.15);" not in active_chunk_block
    assert "var(--color-gold-200)" not in active_chunk_block


def test_source_panel_phrase_highlight_is_scoped_and_quieter_than_global_highlight() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    global_highlight = css.split(".highlight {", 1)[1].split("}", 1)[0]
    source_highlight = css.split(".source-chunk-content .highlight {", 1)[1].split("}", 1)[0]

    assert "background: rgba(210, 182, 97, 0.12);" in global_highlight
    assert "background: rgba(210, 182, 97, 0.10);" in source_highlight
    assert "color: var(--color-text-secondary);" in source_highlight
    assert "border-bottom" not in source_highlight


def test_source_download_icon_is_visible_without_hover() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    icon_block = css.split(".source-dl-icon {", 1)[1].split("}", 1)[0]
    hover_block = css.split(".source-dl-icon:hover {", 1)[1].split("}", 1)[0]

    assert "border: 1px solid var(--color-border-subtle);" in icon_block
    assert "color: var(--color-text-secondary);" in icon_block
    assert "opacity: 1;" in icon_block
    assert "background: rgba(41, 37, 36, 0.38);" in icon_block
    assert "background: var(--color-accent-hover);" in hover_block


def test_visual_tokens_separate_neutral_and_accent_hover_states() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    assert "--color-bg-hover: rgba(120, 113, 108, 0.10);" in css
    assert "--color-bg-hover-strong: rgba(120, 113, 108, 0.16);" in css
    assert "--color-accent-hover: rgba(210, 182, 97, 0.08);" in css
    assert "--color-accent-hover-strong: rgba(210, 182, 97, 0.14);" in css


def test_type_scale_uses_corrected_compact_utopia_coefficients() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")
    root_block = css.split(":root {", 1)[1].split("body {", 1)[0]
    type_block = root_block.split("/* ── Type scale ── */", 1)[1].split("/* ── Space scale", 1)[0]

    assert "Viewport: 390px → 1440px | Base: 15px → 16px | Ratio: 1.2" in root_block
    assert "Base: 16px → 19px" not in root_block
    assert "--step-0: clamp(0.93750rem, 0.095238vi + 0.914286rem, 1.00000rem);" in type_block
    assert "0.00285714vi" not in type_block
    assert "16.00px → 19.00px" not in type_block


def test_chat_typography_uses_semantic_line_height_tokens() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")
    root_block = css.split(":root {", 1)[1].split("body {", 1)[0]
    composer_input = css.split(".composer-input {", 1)[1].split("}", 1)[0]
    user_message = css.split(".user-message {", 1)[1].split("}", 1)[0]
    ai_message = css.split(".ai-message {", 1)[1].split("}", 1)[0]
    ai_content = css.split(".ai-message-content {", 1)[1].split("}", 1)[0]

    assert "--line-height-control: 1.55;" in root_block
    assert "--line-height-message: 1.65;" in root_block
    assert "line-height: var(--line-height-control);" in composer_input
    assert "line-height: var(--line-height-control);" in user_message
    assert "line-height: var(--line-height-message);" in ai_message
    assert "line-height: var(--line-height-message);" in ai_content


def test_radius_tokens_are_static_and_have_a_clear_component_scale() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    assert "--radius-xs: 0.375rem;" in css
    assert "--radius-sm: 0.5rem;" in css
    assert "--radius-md: 0.625rem;" in css
    assert "--radius-lg: 0.75rem;" in css


def test_radius_usage_matches_visual_hierarchy() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    workspace_selector = css.split(".workspace-selector {", 1)[1].split("}", 1)[0]
    workspace_popover = css.split(".workspace-popover {", 1)[1].split("}", 1)[0]
    workspace_check = css.split(".workspace-popover-check {", 1)[1].split("}", 1)[0]
    composer = css.split(".composer-form {", 1)[1].split("}", 1)[0]
    user_message = css.split(".user-message {", 1)[1].split("}", 1)[0]
    source_chunk = css.split(".source-chunk {", 1)[1].split("}", 1)[0]
    file_item = css.split(".file-item {", 1)[1].split("}", 1)[0]
    upload_zone = css.split(".upload-zone {", 1)[1].split("}", 1)[0]

    assert "border-radius: var(--radius-sm);" in workspace_selector
    assert "border-radius: var(--radius-md);" in workspace_popover
    assert "border-radius: var(--radius-xs);" in workspace_check
    assert "border-radius: var(--radius-lg);" in composer
    assert (
        "border-radius: var(--radius-lg) var(--radius-lg) var(--radius-sm) var(--radius-lg);"
        in user_message
    )
    assert "border-radius: 0 var(--radius-md) var(--radius-md) 0;" in source_chunk
    assert "border-radius: var(--radius-sm);" in file_item
    assert "border-radius: var(--radius-md);" in upload_zone


def test_workspace_selector_is_neutral_by_default_with_subtle_accent_states() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    selector_block = css.split(".workspace-selector {", 1)[1].split("}", 1)[0]
    state_block = css.split(".workspace-selector:hover,\n.workspace-selector.open {", 1)[1].split(
        "}",
        1,
    )[0]

    assert "background: rgba(41, 37, 36, 0.56);" in selector_block
    assert "border: 1px solid var(--color-border-subtle);" in selector_block
    assert "color: var(--color-text-secondary);" in selector_block
    assert "background: rgba(87, 74, 36, 0.18);" not in selector_block
    assert "color: var(--color-gold-100);" not in selector_block

    assert "background: var(--color-bg-hover);" in state_block
    assert "border-color: rgba(210, 182, 97, 0.22);" in state_block


def test_lightbox_controls_match_dark_ui_not_bright_white_buttons() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    close_block = css.split(".image-lightbox-close {", 1)[1].split("}", 1)[0]
    nav_block = css.split(".image-lightbox-prev,\n.image-lightbox-next {", 1)[1].split(
        "}",
        1,
    )[0]
    hover_block = css.split(".image-lightbox-prev:hover,\n.image-lightbox-next:hover {", 1)[
        1
    ].split("}", 1)[0]

    for block in (close_block, nav_block):
        assert "background: rgba(28, 25, 23, 0.84);" in block
        assert "border: 1px solid rgba(245, 245, 244, 0.12);" in block
        assert "color: var(--color-text-secondary);" in block
        assert "background: rgba(255, 255, 255, 0.92);" not in block
        assert "color: #0f172a;" not in block

    assert "background: rgba(41, 37, 36, 0.92);" in hover_block
    assert "border-color: rgba(245, 245, 244, 0.18);" in hover_block


def test_drop_overlay_uses_neutral_boundary_with_subtle_text() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    overlay_block = css.split(".drop-overlay {", 1)[1].split("}", 1)[0]
    content_block = css.split(".drop-overlay-content {", 1)[1].split("}", 1)[0]

    assert "background: rgba(12, 10, 9, 0.62);" in overlay_block
    assert "border: 1px dashed rgba(168, 162, 158, 0.34);" in overlay_block
    assert "border: 2px dashed var(--color-gold-400);" not in overlay_block
    assert "color: var(--color-text-secondary);" in content_block
    assert "color: var(--color-gold-200);" not in content_block


def test_panel_resize_handle_does_not_paint_full_height_drag_block() -> None:
    css = (ROOT / "src/dlightrag/web/static/style.css").read_text(encoding="utf-8")

    handle_block = css.split(".panel-resize-handle {", 1)[1].split("}", 1)[0]
    handle_state_block = css.split(".panel-resize-handle:hover,", 1)[1].split("}", 1)[0]

    assert "touch-action: none;" in handle_block
    assert "background: transparent;" in handle_block
    assert "background: var(--color-bg-hover);" not in handle_state_block
    assert "background: transparent;" in handle_state_block


def test_panel_resize_uses_pointer_capture_and_cancel_cleanup() -> None:
    resize_js = (ROOT / "src/dlightrag/web/static/js/resize.js").read_text(encoding="utf-8")

    assert "handle.setPointerCapture(e.pointerId)" in resize_js
    assert "handle.releasePointerCapture(activePointerId)" in resize_js
    assert "'pointerId' in e" in resize_js
    assert "pointercancel" in resize_js
    assert "window.addEventListener('blur', finishDrag)" in resize_js
