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


def test_bootstrap_advertises_exact_backend_attachment_limits() -> None:
    bootstrap_source = (ROOT / "src/dlightrag/adapters/http/browser/routes/bootstrap.py").read_text(
        encoding="utf-8"
    )

    assert "count_limit=application.config.answer.generation.max_attachments" in bootstrap_source
    assert (
        "attachment_limit = application.config.answer.generation.max_attachment_bytes"
        in bootstrap_source
    )
    assert "image_max_bytes=attachment_limit" in bootstrap_source
    assert "document_max_bytes=attachment_limit" in bootstrap_source


def test_frontend_submits_only_the_unified_attachments_part() -> None:
    request_builder = (FRONTEND / "lib" / "answer-request.ts").read_text(encoding="utf-8")

    assert "form.append('attachments', file, file.name)" in request_builder
    assert "append('images'" not in request_builder
    assert "append('documents'" not in request_builder


def test_vite_html_has_no_external_script_or_unresolved_theme_placeholder() -> None:
    for name in ("index.html", "login.html", "design-system.html", "product-showcase.html"):
        source = (FRONTEND / name).read_text(encoding="utf-8")
        built = (ROOT / "src/dlightrag/adapters/http/browser/static/app" / name).read_text(
            encoding="utf-8"
        )
        assert 'src="https://' not in source
        assert "__THEME_INIT__" not in built
        assert re.search(r'/static/app/assets/theme-init-[^"/]+\.js', built)


def test_web_shell_bootstraps_theme_preference_before_app_assets() -> None:
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    theme = (FRONTEND / "theme-init.ts").read_text(encoding="utf-8")
    built = (ROOT / "src/dlightrag/adapters/http/browser/static/app/index.html").read_text(
        encoding="utf-8"
    )

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
    static_root = ROOT / "src/dlightrag/adapters/http/browser/static"
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

    css = (ROOT / "src/dlightrag/adapters/http/browser/static/pygments.css").read_text(
        encoding="utf-8"
    )
    assert generator.generate_css() == css


def test_web_static_js_build_has_no_orphan_chunks() -> None:
    app_root = ROOT / "src/dlightrag/adapters/http/browser/static/app"
    assets = app_root / "assets"
    import_pattern = re.compile(
        r"""(?:import\(`\./([^`]+\.js)`\)|import\(["']\./([^"']+\.js)["']\)|from["']\./([^"']+\.js)["'])"""
    )
    html = "\n".join(
        (app_root / filename).read_text(encoding="utf-8")
        for filename in (
            "index.html",
            "login.html",
            "design-system.html",
            "product-showcase.html",
        )
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
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation
    from dlightrag.application.answer_runs.citations import SourceReferencePayload

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
        evidence_images=[],
    ).sources[0]


def test_presentation_preserves_authorized_download_without_nesting_markup() -> None:
    source = _presentation_source(
        source_uri="local://default/notes.md",
        download_url="/web/api/files/raw/doc-notes?workspace=default",
    )
    source_view = (FRONTEND_UI / "inspector-sources.ts").read_text(encoding="utf-8")

    assert source.download_url == "/web/api/files/raw/doc-notes?workspace=default"
    assert source.title == "Source"
    assert "safeSameOriginHref(source.downloadUrl)" in source_view
    assert "msg('Download source', {id: 'inspectorSources.downloadSource'})" in source_view
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
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation

    presentation = build_answer_presentation(
        answer="Answer [1].",
        sources=[],
        evidence_images=[],
    )
    source_view = (FRONTEND_UI / "inspector-sources.ts").read_text(encoding="utf-8")
    answer_view = (FRONTEND_UI / "answer-presentation.ts").read_text(encoding="utf-8")

    assert '<cite class="citation-badge"' in presentation.parts[0].html
    assert "answer_images" not in presentation.model_dump()
    assert "src.path" not in source_view
    assert "answer-ref-item" in answer_view
    assert "source-doc-badge" in source_view
    assert "{{" not in source_view + answer_view


def test_source_anchor_allowlist_rejects_unsafe_attributes_and_targets() -> None:
    from dlightrag.adapters.http.browser.safe_html import sanitize_html_fragment

    html = sanitize_html_fragment(
        '<a href="/web/api/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_self">Download</a>'
    )

    assert 'aria-label="Download source"' in html
    assert "onclick" not in html
    assert "style=" not in html
    assert "target=" not in html


def test_inspector_cutover_removes_legacy_setup_and_universal_panel_surface() -> None:
    app = (FRONTEND_UI / "app.ts").read_text(encoding="utf-8")
    inspector = (FRONTEND_UI / "inspector.ts").read_text(encoding="utf-8")

    for replaced in ("panel.ts", "source-panel.ts", "source_panel_view.ts", "files-panel.ts"):
        assert not (FRONTEND_UI / replaced).exists()
    for legacy in ("setupPanel", "setupSourcePanel", "setupFilesPanel", "panelOpening"):
        assert legacy not in app + inspector
    assert "PanelController" not in "".join(
        path.read_text(encoding="utf-8") for path in FRONTEND_UI.glob("*.ts")
    )
    assert "<dl-inspector" in app
    assert "customElements.define('dl-inspector'" in inspector


def test_panel_action_icons_use_the_accessible_semantic_registry() -> None:
    file_panel = (FRONTEND_UI / "inspector-files.ts").read_text(encoding="utf-8")
    source_panel = (FRONTEND_UI / "inspector-sources.ts").read_text(encoding="utf-8")

    assert "aria-label=${msg(str`Delete ${file.fileName}`," in file_panel
    assert "'inspectorFiles.deleteFileAria'" in file_panel
    assert "icon('close', {size: 'sm', className: fileStyles['file-delete-icon']})" in file_panel
    assert "icon('download', {size: 'sm', className: s['source-action-icon-svg']})" in source_panel
    assert (
        "icon('open-external', {size: 'sm', className: s['source-action-icon-svg']})"
        in source_panel
    )
    assert "<svg" not in file_panel + source_panel


def test_rich_content_pipeline_has_one_owner_and_two_call_sites() -> None:
    pipeline = (FRONTEND_UI / "rich-rendering.ts").read_text(encoding="utf-8")
    answer_view = (FRONTEND_UI / "answer-presentation.ts").read_text(encoding="utf-8")
    source_view = (FRONTEND_UI / "inspector-sources.ts").read_text(encoding="utf-8")

    # The pipeline owns every rendering stage; the surfaces never touch them
    # directly, so adding a stage in one place cannot be forgotten in another.
    for stage in ("setSanitizedLlmHtml", "renderMath", "renderDiagrams", "secureExternalLinks"):
        assert stage in pipeline
    for stage in ("setSanitizedLlmHtml", "renderMath", "renderDiagrams", "secureExternalLinks"):
        assert stage not in answer_view + source_view

    # Both surfaces converge on the same two narrow entries.
    for entry in ("mountRichHtml", "typesetRichContent"):
        assert entry in answer_view
        assert entry in source_view


def test_split_layout_separates_behavior_from_product_state() -> None:
    split_adapter = (FRONTEND_UI / "split-panel.ts").read_text(encoding="utf-8")
    split_element = (FRONTEND / "design-system" / "elements" / "split-layout.ts").read_text(
        encoding="utf-8"
    )

    assert not (FRONTEND_UI / "resize.ts").exists()
    assert "COMPACT_SHELL_MEDIA" in split_adapter
    assert "dlightrag-panel-width" in split_adapter
    assert "dlightrag-artifact-canvas-width" in split_adapter
    assert "addEventListener('dl-split-change'" in split_adapter
    assert "savePreferred(state)" in split_adapter
    assert "dl-split-input" in split_element
    assert "dl-split-change" in split_element
    assert 'role="separator"' in split_element
    assert "hasPointerCapture" in split_element


def _css_blocks() -> list[tuple[str, str]]:
    """Every `selector { declarations }` pair across the served stylesheets."""
    blocks: list[tuple[str, str]] = []
    sheets = [
        *FRONTEND_STYLES.rglob("*.css"),
        *(FRONTEND / "design-system").rglob("*.css"),
    ]
    for sheet in sorted(sheets):
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
    assert not (ROOT / "src/dlightrag/adapters/http/browser/templates").exists()


def test_production_web_sources_have_no_htmx_contract() -> None:
    sources = [
        *FRONTEND.rglob("*.ts"),
        *FRONTEND.rglob("*.html"),
        *(ROOT / "src/dlightrag/adapters/http/browser").rglob("*.py"),
    ]
    for path in sources:
        if "node_modules" in path.parts:
            continue
        source = path.read_text(encoding="utf-8").lower()
        assert "htmx" not in source
        assert not re.search(r"\bhx-[a-z]", source)


def test_webawesome_is_absent_from_production_and_dependencies() -> None:
    production_sources = [
        path
        for suffix in ("*.ts", "*.css", "*.html")
        for path in FRONTEND.rglob(suffix)
        if "node_modules" not in path.parts and not path.name.endswith(".test.ts")
    ]
    assert all(
        "@awesome.me/webawesome" not in path.read_text(encoding="utf-8")
        for path in production_sources
    )
    package = (FRONTEND / "package.json").read_text(encoding="utf-8")
    assert "@awesome.me/webawesome" not in package


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


def test_lit_shell_completion_has_one_owner_and_no_compatibility_layer() -> None:
    app = (FRONTEND_UI / "app.ts").read_text(encoding="utf-8")
    main = (FRONTEND_UI / "main.ts").read_text(encoding="utf-8")
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in FRONTEND_UI.glob("*.ts")
        if not path.name.endswith((".test.ts", ".browser.test.ts"))
    )

    for replaced in ("images.ts", "memory.ts", "workspaces.ts", "workspace_events.ts"):
        assert not (FRONTEND_UI / replaced).exists()
    assert not (FRONTEND / "events").exists()
    for adapter in (
        "setupSettingsAdapter",
        "setupChatMemoryOperationAdapter",
        "setupImageLightbox",
        "setupNotifications",
        "setupTheme",
        "initWorkspaces",
        "syncShellInert",
    ):
        assert adapter not in app + main + production

    definitions = re.findall(r"customElements\.define\(['\"]([^'\"]+)", production)
    assert definitions
    assert all(name.startswith("dl-") for name in definitions)
    assert "customElements.define('workspace-scope'" not in production
    assert "customElements.define('workspace-create'" not in production
    assert "customElements.define('ingest-target'" not in production

    assert "@dl-chat-memory-operation" in app
    assert "@dl-chat-background-click" in app
    assert "#chat-area" not in app and "open-artifact" not in app
    assert "@dl-image-open" in app
    assert "@dl-toast-request" in app
    assert "@dl-modal-state-change" in app
    assert "@dl-artifact-canvas-state-change" in app
    assert "document.getElementById" not in app
    assert "document.querySelector" not in app
    browser_adapters = (FRONTEND_UI / "browser-adapters.ts").read_text(encoding="utf-8")
    assert "initializeBrowserAdapters(app)" in main
    assert "document.readyState === 'loading'" in main
    assert "setupPanelSplits();" in browser_adapters
    assert "setupMathRendering();" in browser_adapters
    for retired in (
        "workspaceDeleted",
        "workspaceToggled",
        "ingestWorkspaceChanged",
        "bus.emit",
        "panelOpening",
    ):
        assert retired not in app + main + production
    assert "'events/bus" not in app + main + production
    workspace_create = (FRONTEND_UI / "workspace-create.ts").read_text(encoding="utf-8")
    workspace_scope = (FRONTEND_UI / "workspace-scope.ts").read_text(encoding="utf-8")
    assert "this.handles.ingest.set(created.workspace)" in workspace_create
    assert (
        "this.handles.ingest.set(deleted.nextWorkspace || this.handles.workspaces.primary)"
        in workspace_scope
    )
    assert "nanoevents" not in (FRONTEND / "package.json").read_text(encoding="utf-8")

    toast = (FRONTEND_UI / "toast.ts").read_text(encoding="utf-8")
    assert "toastListeners" not in toast and "pendingToast" not in toast
    assert "export function showToast" not in toast
    assert "export function showActionToast" not in toast
    assert "nextRequestId" not in toast and "id: request.id" not in toast

    design_system = (FRONTEND_UI / "design-system.ts").read_text(encoding="utf-8")
    product_showcase = (FRONTEND_UI / "product-showcase.ts").read_text(encoding="utf-8")
    for product_import in ("./notifications.ts", "./theme.ts", "./toast.ts"):
        assert product_import not in design_system
        assert product_import in product_showcase
    assert "ICON_REGISTRY" in design_system
    assert "<dl-split-layout" in design_system
    assert "<dl-notification-offer" not in design_system
    assert "<dl-notification-offer" in product_showcase
