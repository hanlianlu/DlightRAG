# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Server-side static checks for the served browser assets.

Frontend source contracts live in frontend/ui/*.structure.test.ts.
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_STYLES = FRONTEND / "styles"


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
    app_root = static_root / "app"
    assets = app_root / "assets"
    served = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*app_root.glob("*.html"), *assets.glob("*.js"))
    )

    assert {path.name for path in static_root.glob("*.css")} == {"pygments.css"}
    stylesheets = {path.name for path in assets.glob("*.css")}
    assert stylesheets
    assert {name for name in stylesheets if f"assets/{name}" not in served} == set()


def test_web_static_catalog_stylesheet_reaches_only_catalog_pages() -> None:
    """The catalog's page rules, such as its `body`, never reach the application."""
    app_root = ROOT / "src/dlightrag/adapters/http/browser/static/app"
    assets = app_root / "assets"
    catalog = [
        path.name
        for path in assets.glob("*.css")
        if ".ds-shell" in path.read_text(encoding="utf-8")
    ]
    referrers = sorted(
        path.name
        for path in (*app_root.glob("*.html"), *assets.glob("*.js"))
        if any(name in path.read_text(encoding="utf-8") for name in catalog)
    )

    assert len(catalog) == 1
    assert referrers == ["design-system.html", "product-showcase.html"]


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
