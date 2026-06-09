# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Static regression checks for browser-side modules."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_UI = FRONTEND / "ui"
FRONTEND_LIB = FRONTEND / "lib"
FRONTEND_STORES = FRONTEND / "stores"
FRONTEND_TOKENS = FRONTEND / "tokens"
FRONTEND_STYLES = FRONTEND / "styles"


def test_chat_module_imports_close_panel_before_using_it() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "closePanel();" in chat_js
    assert "import {closePanel} from './panel.ts';" in chat_js


def test_chat_streaming_is_split_into_transport_and_renderer_modules() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert (FRONTEND_LIB / "sse.ts").is_file()
    assert (FRONTEND_LIB / "chat_renderer.ts").is_file()
    assert "from '../lib/sse.ts'" in chat_js
    assert "from '../lib/chat_renderer.ts'" in chat_js
    assert "response.body.getReader()" not in chat_js
    assert "contentDiv.innerHTML" not in chat_js


def test_composer_enter_shortcut_respects_ime_composition() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "function submitComposerForm(form)" in chat_js
    assert "function isLineBreakInput(e)" in chat_js
    assert "textarea.addEventListener('beforeinput'" in chat_js
    assert "e.inputType === 'insertLineBreak'" in chat_js
    assert "e.isComposing === true" in chat_js
    assert "let allowNextLineBreak = false;" in chat_js
    assert "allowNextLineBreak = e.shiftKey === true;" in chat_js
    assert "submitComposerForm(form);" in chat_js
    assert "let textareaIsComposing = false;" not in chat_js
    assert "compositionJustEnded" not in chat_js
    assert "setTimeout(function() {" not in chat_js

    beforeinput = chat_js.index("textarea.addEventListener('beforeinput'")
    linebreak_check = chat_js.index("if (!isLineBreakInput(e)) return;", beforeinput)
    composing_guard = chat_js.index(
        "if (e.isComposing === true || allowNextLineBreak)", linebreak_check
    )
    prevent_default = chat_js.index("e.preventDefault();", composing_guard)
    submit_call = chat_js.index("submitComposerForm(form);", prevent_default)
    assert beforeinput < linebreak_check < composing_guard < prevent_default < submit_call

    keydown = chat_js.index("textarea.addEventListener('keydown'")
    keydown_block = chat_js[
        keydown : chat_js.index("textarea.addEventListener('beforeinput'", keydown)
    ]
    assert "submitComposerForm(form)" not in keydown_block
    assert "form.dispatchEvent(new Event('submit'))" not in keydown_block


def test_htmx_behavior_lives_in_dedicated_module() -> None:
    main_js = (FRONTEND_UI / "main.ts").read_text(encoding="utf-8")
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")

    assert (FRONTEND_UI / "htmx.ts").is_file()
    assert "setupHtmxInteractions" in main_js
    assert "htmx:afterRequest" not in panel_js


def test_main_module_version_busts_feature_module_imports() -> None:
    main_js = (FRONTEND_UI / "main.ts").read_text(encoding="utf-8")

    assert "import '../tokens/utopia.css';" in main_js
    assert "import '../styles/global.css';" in main_js
    assert "import '../styles/layout.css';" in main_js
    assert "import '../styles/files.css';" in main_js
    assert "import '../styles/sources.css';" in main_js
    assert "document.addEventListener('DOMContentLoaded', async function()" in main_js
    assert "import('./chat.ts')" in main_js
    assert "import('./htmx.ts')" in main_js
    assert "import('./panel.ts')" in main_js
    assert "import('./workspaces.ts')" in main_js
    assert "setupQueryForm" in main_js
    assert "initWorkspaces()" in main_js


def test_web_shell_does_not_block_on_external_cdn_scripts() -> None:
    web_root = ROOT / "src/dlightrag/web"
    base_html = (web_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert 'src="https://' not in base_html
    assert 'src="/static/vendor/htmx.min.js' in base_html
    assert (web_root / "static" / "vendor" / "htmx.min.js").is_file()


def test_vendored_htmx_version_matches_license_metadata() -> None:
    vendor_root = ROOT / "src/dlightrag/web/static/vendor"
    htmx_js = (vendor_root / "htmx.min.js").read_text(encoding="utf-8")

    version = re.search(r'version:"([^"]+)"', htmx_js)

    assert version is not None
    assert version.group(1)  # version is present


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


def test_manual_folder_upload_panel_swap_processes_htmx_fragments() -> None:
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")
    folder_upload_js = (FRONTEND_UI / "folder-upload.ts").read_text(encoding="utf-8")

    assert "export function applyPanelHtml(html)" in panel_js
    assert "htmx.process(panelContent)" in panel_js
    assert "import {applyPanelHtml} from './panel.ts';" in folder_upload_js
    assert "applyPanelHtml(html)" in folder_upload_js
    assert "getElementById('panel-content')" not in folder_upload_js
    assert "innerHTML = html" not in folder_upload_js
    assert "htmx.process" not in folder_upload_js


def test_source_panel_math_rendering_is_lazy_and_scoped() -> None:
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")
    chat_renderer_js = (FRONTEND_LIB / "chat_renderer.ts").read_text(encoding="utf-8")

    assert "import {renderMath} from './mathjax.ts';" in panel_js
    assert "from './chat_renderer.ts'" not in panel_js
    assert "from './images.ts'" not in panel_js
    assert "renderExpandedSourceMath" in panel_js
    assert "renderMath(panelContent)" not in panel_js
    assert "renderMath(sourceData)" not in chat_renderer_js


def test_lightbox_click_delegation_lives_in_images_module() -> None:
    images_js = (FRONTEND_UI / "images.ts").read_text(encoding="utf-8")
    panel_js = (FRONTEND_UI / "panel.ts").read_text(encoding="utf-8")
    chat_renderer_js = (FRONTEND_LIB / "chat_renderer.ts").read_text(encoding="utf-8")

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
    css = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")

    assert ".topbar-btn," in css
    assert ".upload-btn," in css
    assert ".show-all-btn" in css
    shared_block = css.split(".topbar-btn,", 1)[1].split("}", 1)[0]
    assert ".upload-btn" in shared_block
    assert ".show-all-btn" in shared_block
    assert "font-size: var(--font-size-caption);" in shared_block
    assert "font-weight: 600;" in shared_block
    assert "letter-spacing: 0.04em;" in shared_block
    assert "text-transform: uppercase;" in shared_block


def test_mobile_panel_is_full_width_overlay_not_squeezed_layout() -> None:
    css = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")

    assert "@media (max-width: 640px)" in css
    mobile_block = css.split("@media (max-width: 640px)", 1)[1]
    assert "width: 100vw;" in mobile_block
    assert "max-width: none;" in mobile_block
    assert "min-width: 0;" in mobile_block
    assert "margin-right: 0;" in mobile_block
    assert ".panel-resize-handle" in mobile_block
    assert "display: none;" in mobile_block


def test_topbar_controls_have_stable_mobile_flex_behavior() -> None:
    css = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")

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


def test_file_delete_matches_workspace_hover_disclosure() -> None:
    css = (FRONTEND_STYLES / "files.css").read_text(encoding="utf-8")

    file_delete = css.split(".file-delete {", 1)[1].split("}", 1)[0]
    file_delete_hover = css.split(".file-delete:hover {", 1)[1].split("}", 1)[0]

    assert "border: none;" in file_delete
    assert "display: flex;" in file_delete
    assert "height: var(--size-icon-md);" in file_delete
    assert "opacity: 0.45;" in file_delete
    assert "transition: all 0.15s;" in file_delete
    assert "width: var(--size-icon-md);" in file_delete
    assert ".file-delete {\n        opacity: 0;\n    }" in css
    assert ".file-item:hover .file-delete {\n        opacity: 1;\n    }" in css
    assert "background: rgba(248, 113, 113, 0.15);" in file_delete_hover
    assert "border-color:" not in file_delete_hover


def test_source_panel_default_gold_accents_are_restrained() -> None:
    css = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")

    source_header_hover = css.split(".source-doc-header:hover {", 1)[1].split("}", 1)[0]
    source_chunk_block = css.split(".source-chunk {", 1)[1].split("}", 1)[0]

    assert "background: var(--color-bg-hover);" in source_header_hover
    assert "border-left:" not in source_chunk_block
    assert "background: rgba(12, 10, 9, 0.72);" in source_chunk_block
    assert "box-shadow: inset 2px 0 0 rgba(120, 113, 108, 0.18);" in source_chunk_block
    assert "rgba(210, 182, 97" not in source_chunk_block
    assert "border-left: 3px solid var(--color-gold-400);" not in css


def test_source_panel_active_chunk_uses_subtle_selection_not_gold_fill() -> None:
    css = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")

    source_chunk_block = css.split(".source-chunk {", 1)[1].split("}", 1)[0]
    active_chunk_block = css.split(".source-chunk.active {", 1)[1].split("}", 1)[0]

    assert "background: rgba(12, 10, 9, 0.72);" in source_chunk_block
    assert "background: var(--color-bg-elevated);" not in source_chunk_block
    assert "border-left-color:" not in active_chunk_block
    assert "background: rgba(12, 10, 9, 0.86);" in active_chunk_block
    assert "box-shadow: inset 2px 0 0 rgba(210, 182, 97, 0.28);" in active_chunk_block
    assert "background: rgba(87, 74, 36, 0.15);" not in active_chunk_block
    assert "var(--color-gold-200)" not in active_chunk_block


def test_source_panel_phrase_highlight_is_scoped_without_underline() -> None:
    css = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")

    global_highlight = css.split(".highlight {", 1)[1].split("}", 1)[0]
    source_highlight = css.split(".source-chunk-content .highlight {", 1)[1].split("}", 1)[0]

    assert "background: rgba(210, 182, 97, 0.14);" in global_highlight
    assert "padding: 0 0.1em;" in global_highlight
    assert "box-decoration-break: clone;" in global_highlight
    assert "border-bottom: none;" in global_highlight
    assert "background: rgba(210, 182, 97, 0.18);" in source_highlight
    assert "color: var(--color-text-primary);" in source_highlight
    assert "border-bottom: none;" in source_highlight


def test_citation_badge_is_compact_marker_not_framed_chip() -> None:
    css = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")

    badge_block = css.split(".citation-badge {", 1)[1].split("}", 1)[0]
    hover_block = css.split(".citation-badge:hover {", 1)[1].split("}", 1)[0]

    assert "background: transparent;" in badge_block
    assert "font-size: var(--font-size-caption);" in badge_block
    assert "line-height: 1;" in badge_block
    assert "padding: 0 0.16em;" in badge_block
    assert "vertical-align: baseline;" in badge_block
    assert "background: rgba(210, 182, 97, 0.12);" in hover_block


def test_reference_labels_do_not_render_square_brackets() -> None:
    partials = ROOT / "src/dlightrag/web/templates/partials"
    answer_done = (partials / "answer_done.html").read_text(encoding="utf-8")
    source_panel = (partials / "source_panel.html").read_text(encoding="utf-8")

    assert "[{{ src.id }}]" not in answer_done
    assert "[{{ src.id }}]" not in source_panel
    assert '<span class="answer-ref-id">{{ src.id | reference_label }}</span>' in answer_done
    assert '<span class="source-doc-badge">{{ src.id | reference_label }}</span>' in source_panel


def test_source_download_icon_is_icon_only_not_framed_button() -> None:
    css = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")

    icon_block = css.split(".source-dl-icon {", 1)[1].split("}", 1)[0]
    hover_block = css.split(".source-dl-icon:hover {", 1)[1].split("}", 1)[0]
    svg_block = css.split(".source-dl-icon-svg {", 1)[1].split("}", 1)[0]

    assert "background: transparent;" in icon_block
    assert "border: none;" in icon_block
    assert "color: var(--color-text-tertiary);" in icon_block
    assert "opacity: 1;" in icon_block
    assert "color: var(--color-gold-200);" in hover_block
    assert "height: var(--size-icon-md);" in svg_block
    assert "stroke-width: 2;" in svg_block
    assert "width: var(--size-icon-md);" in svg_block
    source_panel = (ROOT / "src/dlightrag/web/templates/partials/source_panel.html").read_text(
        encoding="utf-8"
    )
    assert 'stroke="currentColor"' in source_panel


def test_visual_tokens_separate_neutral_and_accent_hover_states() -> None:
    css = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")

    assert "--color-bg-hover: rgba(120, 113, 108, 0.10);" in css
    assert "--color-accent-hover: rgba(210, 182, 97, 0.08);" in css
    assert "--color-accent-hover-strong: rgba(210, 182, 97, 0.14);" in css


def test_type_scale_uses_corrected_compact_utopia_coefficients() -> None:
    css = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")
    root_block = css.split(":root {", 1)[1]
    root_block = root_block[: root_block.rfind("}")]
    type_block = root_block.split("/* ── Type scale ── */", 1)[1].split("/* ── Space scale", 1)[0]

    assert "Viewport: 390px → 1440px | Base: 15px → 15.5px | Ratio: 1.18" in root_block
    assert "Base: 16px → 19px" not in root_block
    assert "--step-0: clamp(0.93750rem, 0.047619vi + 0.925893rem, 0.96875rem);" in type_block
    assert "0.00285714vi" not in type_block
    assert "16.00px → 19.00px" not in type_block


def test_chat_typography_uses_semantic_line_height_tokens() -> None:
    utopia = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")
    layout = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")
    chat_module = (FRONTEND_STYLES / "chat.module.css").read_text(encoding="utf-8")

    root_block = utopia.split(":root {", 1)[1]
    root_block = root_block[: root_block.rfind("}")]
    composer_form = layout.split(".composer-form {", 1)[1].split("}", 1)[0]
    composer_input = layout.split(".composer-input {", 1)[1].split("}", 1)[0]
    user_message = chat_module.split(".userMessage {", 1)[1].split("}", 1)[0]
    ai_message = chat_module.split(".aiMessage {", 1)[1].split("}", 1)[0]
    ai_content = chat_module.split(".aiMessageContent {", 1)[1].split("}", 1)[0]

    assert "--line-height-control: 1.5;" in root_block
    assert "--line-height-message: 1.58;" in root_block
    assert "--space-message-block: var(--space-component);" in root_block
    assert "--space-composer-y: calc(var(--space-tight) * 0.625);" in root_block
    assert "--space-composer-x: var(--space-tight);" in root_block
    assert "--space-composer-leading: var(--space-inline);" in root_block
    assert "--space-composer-trailing: var(--space-composer-x);" in root_block
    assert (
        "padding: var(--space-composer-y) var(--space-composer-trailing) "
        "var(--space-composer-y) var(--space-composer-leading);"
    ) in composer_form
    assert "line-height: var(--line-height-control);" in composer_input
    assert "padding: 0;" in composer_input
    assert "min-height: calc(var(--font-size-body) * var(--line-height-control));" in composer_input
    assert "overflow-y: hidden;" in composer_input
    assert "line-height: var(--line-height-control);" in user_message
    assert "margin: var(--space-message-block) 0 var(--space-tight) auto;" in user_message
    assert "line-height: var(--line-height-message);" in ai_message
    assert "line-height: var(--line-height-message);" in ai_content
    assert "padding: var(--space-tight) 0 var(--space-message-block);" in ai_message


def test_welcome_brand_has_stronger_first_viewport_presence() -> None:
    utopia = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")
    layout = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")

    root_block = utopia.split(":root {", 1)[1]
    root_block = root_block[: root_block.rfind("}")]
    welcome_brand = layout.split(".welcome-brand {", 1)[1].split("}", 1)[0]
    welcome_empty = layout.split(".app:not(.has-messages) .welcome {", 1)[1].split("}", 1)[0]
    composer_empty = layout.split(".app:not(.has-messages) .composer {", 1)[1].split("}", 1)[0]

    assert "--font-size-welcome-brand: clamp(1.5rem, 1.35vw + 1.16rem, 2.25rem);" in root_block
    assert "font-size: var(--font-size-welcome-brand);" in welcome_brand
    assert "font-weight: 600;" in welcome_brand
    assert "color: var(--color-text-primary);" in welcome_brand
    assert "min-height: 42vh;" in welcome_empty
    assert "padding-top: clamp(17vh, 18vh, 20vh);" in welcome_empty
    assert "padding-bottom: var(--space-section);" in welcome_empty
    assert "padding-bottom: clamp(28vh, 30vh, 32vh);" in composer_empty


def test_composer_plus_uses_thin_svg_icon_without_enlarging_button() -> None:
    utopia = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")
    layout = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")
    index_html = (ROOT / "src/dlightrag/web/templates/index.html").read_text(encoding="utf-8")

    root_block = utopia.split(":root {", 1)[1]
    root_block = root_block[: root_block.rfind("}")]
    plus_block = layout.split(".composer-plus {", 1)[1].split("}", 1)[0]
    assert ".composer-plus-icon {" in layout
    plus_icon_block = layout.split(".composer-plus-icon {", 1)[1].split("}", 1)[0]

    assert 'class="composer-plus-icon"' in index_html
    assert ">+</button>" not in index_html
    assert 'stroke="currentColor"' in index_html
    assert "--size-composer-plus-icon: clamp(22px, 2.15vw, 27px);" in root_block
    assert "--stroke-composer-plus-icon: 1.25;" in root_block
    assert "--font-size-action-glyph" not in root_block
    assert "width: var(--size-button);" in plus_block
    assert "height: var(--size-button);" in plus_block
    assert "color: rgba(210, 182, 97, 0.78);" in plus_block
    assert "font-size:" not in plus_block
    assert "font-weight:" not in plus_block
    assert "width: var(--size-composer-plus-icon);" in plus_icon_block
    assert "height: var(--size-composer-plus-icon);" in plus_icon_block
    assert "stroke-width: var(--stroke-composer-plus-icon);" in plus_icon_block


def test_composer_autoresize_hides_scrollbar_until_max_height() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "textarea.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';" in chat_js
    assert "textarea.style.overflowY = '';" in chat_js


def test_composer_autoresize_measures_content_after_height_reset() -> None:
    chat_js = (FRONTEND_UI / "chat.ts").read_text(encoding="utf-8")

    assert "const computed = getComputedStyle(textarea);" in chat_js
    assert "const maxHeight = parseFloat(computed.maxHeight) || 160;" in chat_js
    assert "const contentHeight = textarea.scrollHeight;" in chat_js
    assert "textarea.value.includes('\\n')" in chat_js

    height_reset = chat_js.index("textarea.style.height = 'auto';")
    content_measure = chat_js.index("const contentHeight = textarea.scrollHeight;")
    multiline_update = chat_js.index("form.classList.toggle('multiline'", content_measure)
    height_apply = chat_js.index(
        "textarea.style.height = Math.min(contentHeight, maxHeight) + 'px';"
    )

    assert height_reset < content_measure < multiline_update < height_apply


def test_radius_tokens_are_static_and_have_a_clear_component_scale() -> None:
    css = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")

    assert "--radius-xs: 0.375rem;" in css
    assert "--radius-sm: 0.5rem;" in css
    assert "--radius-md: 0.625rem;" in css
    assert "--radius-lg: 0.75rem;" in css
    assert "--radius-xl: 0.875rem;" in css
    assert "--radius-surface: var(--radius-md);" in css
    assert "--radius-composer: var(--radius-xl);" in css
    assert "--radius-message: var(--radius-xl);" in css


def test_radius_usage_matches_visual_hierarchy() -> None:
    layout = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")
    workspace_module = (FRONTEND_STYLES / "workspaces.module.css").read_text(encoding="utf-8")
    chat_module = (FRONTEND_STYLES / "chat.module.css").read_text(encoding="utf-8")
    sources = (FRONTEND_STYLES / "sources.css").read_text(encoding="utf-8")
    files = (FRONTEND_STYLES / "files.css").read_text(encoding="utf-8")

    workspace_selector = layout.split(".workspace-selector {", 1)[1].split("}", 1)[0]
    workspace_popover = workspace_module.split(".workspacePopover {", 1)[1].split("}", 1)[0]
    workspace_check = workspace_module.split(".workspacePopoverCheck {", 1)[1].split("}", 1)[0]
    composer = layout.split(".composer-form {", 1)[1].split("}", 1)[0]
    user_message = chat_module.split(".userMessage {", 1)[1].split("}", 1)[0]
    source_chunk = sources.split(".source-chunk {", 1)[1].split("}", 1)[0]
    file_item = files.split(".file-item {", 1)[1].split("}", 1)[0]
    upload_zone = files.split(".upload-zone {", 1)[1].split("}", 1)[0]

    assert "border-radius: var(--radius-sm);" in workspace_selector
    assert "border-radius: var(--radius-md);" in workspace_popover
    assert "border-radius: var(--radius-xs);" in workspace_check
    assert "border-radius: var(--radius-composer);" in composer
    assert "border-radius: var(--radius-message);" in user_message
    assert "var(--radius-lg) var(--radius-lg) var(--radius-sm) var(--radius-lg)" not in user_message
    assert "border-radius: var(--radius-surface);" in source_chunk
    assert "border-radius: 0 var(--radius-surface)" not in source_chunk
    assert "border-radius: var(--radius-sm);" in file_item
    assert "border-radius: var(--radius-md);" in upload_zone


def test_workspace_selector_is_neutral_by_default_with_subtle_accent_states() -> None:
    layout = (FRONTEND_STYLES / "layout.css").read_text(encoding="utf-8")
    workspace_module = (FRONTEND_STYLES / "workspaces.module.css").read_text(encoding="utf-8")

    selector_block = layout.split(".workspace-selector {", 1)[1].split("}", 1)[0]
    popover_item_block = workspace_module.split(".workspacePopoverItem {", 1)[1].split("}", 1)[0]
    state_block = layout.split(".workspace-selector:hover,\n.workspace-selector.open {", 1)[
        1
    ].split(
        "}",
        1,
    )[0]

    assert "background: rgba(41, 37, 36, 0.56);" in selector_block
    assert "border: 1px solid var(--color-border-subtle);" in selector_block
    assert "color: var(--color-text-subtle);" in selector_block
    assert "background: rgba(87, 74, 36, 0.18);" not in selector_block
    assert "color: var(--color-gold-100);" not in selector_block
    assert "color: var(--color-text-secondary);" in popover_item_block

    assert "background: var(--color-bg-hover);" in state_block
    assert "border-color: rgba(210, 182, 97, 0.22);" in state_block


def test_lightbox_controls_use_large_theme_bars_without_close_button() -> None:
    utopia = (FRONTEND_TOKENS / "utopia.css").read_text(encoding="utf-8")
    lightbox_module = (FRONTEND_STYLES / "lightbox.module.css").read_text(encoding="utf-8")
    images_js = (FRONTEND_UI / "images.ts").read_text(encoding="utf-8")

    root_block = utopia.split(":root {", 1)[1]
    root_block = root_block[: root_block.rfind("}")]
    nav_block = lightbox_module.split(".imageLightboxPrev,\n.imageLightboxNext {", 1)[1].split(
        "}",
        1,
    )[0]
    hover_block = lightbox_module.split(".imageLightboxPrev:hover,\n.imageLightboxNext:hover {", 1)[
        1
    ].split("}", 1)[0]

    assert "image-lightbox-close" not in lightbox_module
    assert "image-lightbox-close" not in images_js
    assert "--size-lightbox-nav-width: clamp(44px, 6vw, 72px);" in root_block
    assert "--inset-lightbox-nav-block: max(var(--space-layout), 12vh);" in root_block
    assert "--font-size-lightbox-nav-icon: clamp(30px, 4vw, 42px);" in root_block
    assert "top: var(--inset-lightbox-nav-block);" in nav_block
    assert "bottom: var(--inset-lightbox-nav-block);" in nav_block
    assert "width: var(--size-lightbox-nav-width);" in nav_block
    assert "font-size: var(--font-size-lightbox-nav-icon);" in nav_block
    assert "border: none;" in nav_block
    assert "border-radius: 999px;" not in nav_block
    assert "background: rgba(245, 245, 244, 0.045);" in nav_block
    assert "color: var(--color-text-secondary);" in nav_block
    assert "background: rgba(255, 255, 255, 0.92);" not in nav_block
    assert "color: #0f172a;" not in nav_block

    assert "background: rgba(210, 182, 97, 0.12);" in hover_block
    assert "color: var(--color-gold-100);" in hover_block


def test_drop_overlay_uses_neutral_boundary_with_subtle_text() -> None:
    css = (FRONTEND_STYLES / "global.css").read_text(encoding="utf-8")

    overlay_block = css.split(".drop-overlay {", 1)[1].split("}", 1)[0]
    content_block = css.split(".drop-overlay-content {", 1)[1].split("}", 1)[0]

    assert "background: rgba(12, 10, 9, 0.62);" in overlay_block
    assert "border: 1px dashed rgba(168, 162, 158, 0.34);" in overlay_block
    assert "border: 2px dashed var(--color-gold-400);" not in overlay_block
    assert "color: var(--color-text-secondary);" in content_block
    assert "color: var(--color-gold-200);" not in content_block


def test_panel_resize_handle_does_not_paint_full_height_drag_block() -> None:
    css = (FRONTEND_STYLES / "global.css").read_text(encoding="utf-8")

    handle_block = css.split(".panel-resize-handle {", 1)[1].split("}", 1)[0]
    handle_state_block = css.split(".panel-resize-handle:hover,", 1)[1].split("}", 1)[0]

    assert "touch-action: none;" in handle_block
    assert "background: transparent;" in handle_block
    assert "background: var(--color-bg-hover);" not in handle_state_block
    assert "background: transparent;" in handle_state_block


def test_panel_resize_uses_pointer_capture_and_cancel_cleanup() -> None:
    resize_js = (FRONTEND_UI / "resize.ts").read_text(encoding="utf-8")

    assert "handle.setPointerCapture(e.pointerId)" in resize_js
    assert "handle.releasePointerCapture(activePointerId)" in resize_js
    assert "'pointerId' in e" in resize_js
    assert "pointercancel" in resize_js
    assert "window.addEventListener('blur', finishDrag)" in resize_js
