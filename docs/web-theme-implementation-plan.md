# Web Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-frame-correct `System / Light / Dark` appearance preference with Mineral Light, the existing dark identity, readable syntax highlighting, and an accessible topbar menu.

**Architecture:** Store only the browser preference in local storage and project it onto `<html data-theme data-color-mode>`. A pre-CSS inline bootstrap resolves the first frame; a small runtime module owns system changes, cross-tab synchronization, and the appearance menu. CSS components consume semantic tokens, while the token layer and root-scoped Pygments rules own both palettes.

**Tech Stack:** Jinja2, TypeScript 7, CSS custom properties, Vite 8, Pygments, HTMX, node:test, pytest, Playwright.

---

## File Map

- Create `frontend/lib/theme.ts`: pure preference validation and effective-mode resolution.
- Create `frontend/lib/theme.test.ts`: focused unit evidence for the pure state rules.
- Create `frontend/ui/theme.ts`: DOM attributes, persistence, media/storage listeners, menu lifecycle, and ARIA synchronization.
- Modify `frontend/package.json`: include focused `lib/*.test.ts` files in the existing node:test command.
- Modify `frontend/lib/listbox.ts`: generalize the existing roving arrow navigation for listbox options and radio-menu items.
- Modify `frontend/ui/workspaces.ts`: use the renamed generic navigation helper.
- Modify `frontend/ui/main.ts`: initialize the theme control with the rest of the shell.
- Modify `src/dlightrag/web/templates/base.html`: safe root defaults, native color-scheme declaration, and pre-CSS bootstrap.
- Modify `src/dlightrag/web/templates/index.html`: static dynamic trigger icons and three-state radio menu.
- Modify `frontend/tokens/utopia.css`: semantic accent/elevation/surface tokens and Mineral Light overrides.
- Modify `frontend/styles/global.css`: semantic hardcode cleanup, native color-scheme behavior, tooltip support, and MathJax color inheritance.
- Modify `frontend/styles/layout.css`: theme trigger and menu layout plus semantic hardcode cleanup.
- Modify `frontend/styles/chat.module.css`: semantic accent/raw-color migration and themed rich content.
- Modify `frontend/styles/conversations.module.css`: semantic selection, shadow, danger, and focus colors.
- Modify `frontend/styles/files.css`: semantic upload/progress colors.
- Modify `frontend/styles/sources.css`: themed source surfaces, highlights, and captions.
- Modify `frontend/styles/lightbox.module.css`: semantic scrim, controls, and shadows.
- Modify `frontend/styles/workspaces.module.css`: semantic accent/action colors.
- Replace `src/dlightrag/web/static/pygments.css`: generated scoped Friendly and GitHub Dark token rules.
- Modify `tests/unit/test_web_frontend_static.py`: retain only meaningful first-frame and dual-Pygments static contracts.
- Create `tests/e2e/test_web_theme.py`: user-observable preference, system, persistence, keyboard, and degraded-storage behavior.
- Modify `NOTICE`: record the Lucide icon license used by the embedded SVG geometry.
- Regenerate `src/dlightrag/web/static/generated/`: production Vite output; remove superseded hashed bundles.

## Task 1: Pure Theme State and First-Frame Bootstrap

**Files:**
- Create: `frontend/lib/theme.ts`
- Create: `frontend/lib/theme.test.ts`
- Modify: `frontend/package.json`
- Modify: `src/dlightrag/web/templates/base.html`
- Modify: `tests/unit/test_web_frontend_static.py`

- [ ] **Step 1: Add failing pure state tests**

Create `frontend/lib/theme.test.ts`:

```ts
import assert from 'node:assert/strict';
import test from 'node:test';

import {parseThemePreference, resolveColorMode} from './theme.ts';

test('theme preference accepts only the three product choices', () => {
    assert.equal(parseThemePreference('system'), 'system');
    assert.equal(parseThemePreference('light'), 'light');
    assert.equal(parseThemePreference('dark'), 'dark');
    assert.equal(parseThemePreference(null), 'system');
    assert.equal(parseThemePreference('sepia'), 'system');
});

test('system resolves from the media preference while explicit choices win', () => {
    assert.equal(resolveColorMode('system', true), 'dark');
    assert.equal(resolveColorMode('system', false), 'light');
    assert.equal(resolveColorMode('light', true), 'light');
    assert.equal(resolveColorMode('dark', false), 'dark');
});
```

Update the frontend test script:

```json
"test": "node --experimental-strip-types --test stores/*.test.ts lib/*.test.ts"
```

- [ ] **Step 2: Add failing first-frame static contracts**

Add focused assertions to `tests/unit/test_web_frontend_static.py`:

```python
def test_theme_bootstrap_precedes_stylesheet() -> None:
    base_html = (ROOT / "src/dlightrag/web/templates/base.html").read_text(encoding="utf-8")

    assert '<html lang="en" data-theme="system" data-color-mode="dark">' in base_html
    assert '<meta name="color-scheme" content="dark light">' in base_html
    assert 'dlightrag-theme' in base_html
    assert "prefers-color-scheme: dark" in base_html
    assert base_html.index("dlightrag-theme") < base_html.index('rel="stylesheet"')
```

Do not assert the full bootstrap source or internal local-variable names.

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
npm --prefix frontend test
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py::test_theme_bootstrap_precedes_stylesheet
```

Expected: the TypeScript test fails because `theme.ts` does not exist, and the Python test fails because the root attributes/bootstrap are absent.

- [ ] **Step 4: Implement the minimal pure model**

Create `frontend/lib/theme.ts`:

```ts
export const THEME_STORAGE_KEY = 'dlightrag-theme';

export type ThemePreference = 'system' | 'light' | 'dark';
export type ColorMode = 'light' | 'dark';

export function parseThemePreference(value: string | null): ThemePreference {
    return value === 'light' || value === 'dark' || value === 'system' ? value : 'system';
}

export function resolveColorMode(
    preference: ThemePreference,
    prefersDark: boolean,
): ColorMode {
    return preference === 'system' ? (prefersDark ? 'dark' : 'light') : preference;
}
```

- [ ] **Step 5: Add the pre-CSS bootstrap**

Change the root and `<head>` in `base.html`:

```html
<html lang="en" data-theme="system" data-color-mode="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="dark light">
    <title>DlightRAG</title>
    <meta name="htmx-config" content='{"allowEval": false, "allowScriptTags": false}'>
    <script>
        (() => {
            const root = document.documentElement;
            const key = 'dlightrag-theme';
            let preference = 'system';
            try {
                const saved = localStorage.getItem(key);
                if (saved === 'system' || saved === 'light' || saved === 'dark') {
                    preference = saved;
                } else if (saved !== null) {
                    localStorage.removeItem(key);
                }
            } catch (_) { /* Storage is optional for an appearance preference. */ }
            const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true;
            const mode = preference === 'system' ? (prefersDark ? 'dark' : 'light') : preference;
            root.dataset.theme = preference;
            root.dataset.colorMode = mode;
            root.style.colorScheme = mode;
        })();
    </script>
    <link rel="stylesheet" href="/static/generated/style.css?v={{ static_version() }}">
```

Keep HTMX script evaluation disabled; this bootstrap is part of the initial full document, not an HTMX fragment.

- [ ] **Step 6: Run focused validation**

Run:

```bash
npm --prefix frontend test
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py
npm --prefix frontend run typecheck
```

Expected: all commands pass.

- [ ] **Step 7: Commit the state boundary**

```bash
git add frontend/lib/theme.ts frontend/lib/theme.test.ts frontend/package.json \
  src/dlightrag/web/templates/base.html tests/unit/test_web_frontend_static.py
git commit -m "feat(web): bootstrap theme preference"
```

## Task 2: Semantic Tokens and Mineral Light

**Files:**
- Modify: `frontend/tokens/utopia.css`
- Modify: `frontend/styles/global.css`
- Modify: `frontend/styles/layout.css`
- Modify: `frontend/styles/chat.module.css`
- Modify: `frontend/styles/conversations.module.css`
- Modify: `frontend/styles/files.css`
- Modify: `frontend/styles/sources.css`
- Modify: `frontend/styles/lightbox.module.css`
- Modify: `frontend/styles/workspaces.module.css`

- [ ] **Step 1: Add semantic theme tokens**

Keep the existing primitive type, spacing, radii, and dark colors. Add purpose-based aliases to the dark root:

```css
:root {
    color-scheme: dark;
    --color-accent-action: #d2b661;
    --color-accent-emphasis: #f7de9d;
    --color-accent-muted: #a7904c;
    --color-action-bg: #574a24;
    --color-action-bg-hover: #7e6c37;
    --color-on-action: #f7de9d;
    --color-danger-strong: #b91c1c;
    --color-on-danger: #f5f5f4;
    --color-bg-control: rgb(41 37 36 / 56%);
    --color-bg-selected: rgb(120 113 108 / 13%);
    --color-bg-row-alt: rgb(28 25 23 / 50%);
    --color-bg-source: rgb(12 10 9 / 72%);
    --color-bg-source-active: rgb(12 10 9 / 86%);
    --color-bg-hover-faint: rgb(255 255 255 / 2%);
    --shadow-elevation-sm: 0 4px 16px rgb(0 0 0 / 30%);
    --shadow-elevation-md: 0 12px 28px rgb(0 0 0 / 42%);
    --shadow-elevation-lg: 0 12px 34px rgb(0 0 0 / 46%);
    --shadow-sidebar: 12px 0 36px rgb(0 0 0 / 18%);
    --shadow-panel: -8px 0 32px rgb(0 0 0 / 30%);
}
```

Add the approved effective light override:

```css
:root[data-color-mode='light'] {
    color-scheme: light;
    --color-text-primary: #24231f;
    --color-text-secondary: #35342f;
    --color-text-tertiary: #45443f;
    --color-text-muted: #66645d;
    --color-text-subtle: #6f6d65;
    --color-text-dim: #726f67;
    --color-bg-base: #f5f5f2;
    --color-bg-surface: #ffffff;
    --color-bg-elevated: #e9e8e2;
    --color-bg-subtle: #efefeb;
    --color-bg-hover: rgb(65 63 56 / 8%);
    --color-bg-control: #ffffff;
    --color-bg-selected: #e4e1d5;
    --color-bg-row-alt: rgb(69 68 63 / 4%);
    --color-bg-source: #e9e8e2;
    --color-bg-source-active: #dedcd3;
    --color-bg-hover-faint: rgb(69 68 63 / 4%);
    --color-border-subtle: rgb(69 68 63 / 15%);
    --color-border-strong: rgb(69 68 63 / 26%);
    --color-divider: rgb(69 68 63 / 12%);
    --color-danger: #b42318;
    --color-danger-strong: #b42318;
    --color-danger-surface: rgb(180 35 24 / 10%);
    --color-on-danger: #ffffff;
    --color-accent-action: #806719;
    --color-accent-emphasis: #574a24;
    --color-accent-muted: #806719;
    --color-action-bg: #806719;
    --color-action-bg-hover: #6d5715;
    --color-on-action: #ffffff;
    --color-accent-text: #806719;
    --color-accent-indicator: #806719;
    --color-accent-border-strong: rgb(128 103 25 / 56%);
    --color-accent-border: rgb(128 103 25 / 46%);
    --color-accent-border-soft: rgb(128 103 25 / 36%);
    --color-accent-border-faint: rgb(128 103 25 / 24%);
    --color-accent-surface: rgb(128 103 25 / 12%);
    --color-accent-hover: rgb(128 103 25 / 8%);
    --color-accent-hover-strong: rgb(128 103 25 / 14%);
    --color-accent-ghost: rgb(128 103 25 / 6%);
    --color-scrim: rgb(0 0 0 / 38%);
    --color-scrim-strong: rgb(0 0 0 / 48%);
    --shadow-elevation-sm: 0 4px 16px rgb(44 42 34 / 8%);
    --shadow-elevation-md: 0 12px 28px rgb(44 42 34 / 12%);
    --shadow-elevation-lg: 0 12px 34px rgb(44 42 34 / 14%);
    --shadow-sidebar: 12px 0 36px rgb(44 42 34 / 8%);
    --shadow-panel: -8px 0 32px rgb(44 42 34 / 10%);
    --shadow-overlay: 0 18px 48px rgb(44 42 34 / 18%);
}
```

Adjust individual values during browser contrast validation only when the computed role/background pair fails WCAG AA or the approved Mineral Light reference.

- [ ] **Step 2: Migrate direct primitive accent usage**

Use this purpose-based replacement table across component styles:

| Existing use | Semantic replacement |
|---|---|
| foreground/link `var(--color-gold-200)` | `var(--color-accent-action)` |
| emphasized foreground `var(--color-gold-100)` | `var(--color-accent-emphasis)` |
| subdued reference `var(--color-gold-300)` | `var(--color-accent-muted)` |
| button fill `var(--color-gold-500)` | `var(--color-action-bg)` |
| button hover `var(--color-gold-400)` | `var(--color-action-bg-hover)` |
| text on filled accent | `var(--color-on-action)` |

Where one declaration currently uses a gold primitive for a structural rule rather than these roles, add one narrowly named semantic token instead of forcing an incorrect mapping.

- [ ] **Step 3: Replace component theme literals with existing or new semantic tokens**

Migrate only color and shadow literals. Examples:

```css
.aiMessageContent tr:nth-child(even) {
    background: var(--color-bg-row-alt);
}

:global(.conversation-row[aria-current='page']) {
    background: var(--color-bg-selected);
}

.source-chunk {
    background: var(--color-bg-source);
}

.source-chunk.active {
    background: var(--color-bg-source-active);
}
```

Use semantic danger/on-danger and shared elevation tokens for dialogs, popovers, sidebar, panel, and toast. Keep dark image-isolation scrims dark in both modes.

Do not alter layout declarations, class names, spacing, typography, or transition timing.

- [ ] **Step 4: Run CSS and production-build validation**

Run:

```bash
npm --prefix frontend run lint:css
npm --prefix frontend run typecheck
npm --prefix frontend run build
rg -n 'var\(--color-gold-' frontend/styles
rg -n '#[0-9a-fA-F]{3,8}\b|rgba?\(|rgb\(' frontend/styles --glob '*.css'
```

Expected: lint/typecheck/build pass; the first audit has no direct component gold references; the second contains only justified literals that are not theme-bearing or have an explicit reason to remain.

- [ ] **Step 5: Commit the palette**

```bash
git add frontend/tokens/utopia.css frontend/styles src/dlightrag/web/static/generated
git commit -m "feat(web): add mineral light palette"
```

## Task 3: Dual Syntax Highlighting and Math Color

**Files:**
- Replace: `src/dlightrag/web/static/pygments.css`
- Modify: `frontend/styles/global.css`
- Modify: `tests/unit/test_web_frontend_static.py`

- [ ] **Step 1: Add failing dual-mode Pygments contracts**

Add one focused test:

```python
def test_pygments_css_is_scoped_to_both_effective_color_modes() -> None:
    css = (ROOT / "src/dlightrag/web/static/pygments.css").read_text(encoding="utf-8")

    assert '[data-color-mode="light"] .highlight' in css
    assert '[data-color-mode="dark"] .highlight' in css
    assert "friendly" in css.lower()
    assert "github-dark" in css.lower()
```

Do not assert individual Pygments token colors.

- [ ] **Step 2: Run the contract and verify it fails**

Run:

```bash
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py::test_pygments_css_is_scoped_to_both_effective_color_modes
```

Expected: FAIL because the existing file is one unscoped default palette.

- [ ] **Step 3: Generate the two scoped palettes**

Generate from the installed Pygments version:

```bash
{
  printf '%s\n' '/* Generated from Pygments Friendly (light) and GitHub Dark (dark).' \
    '   Regenerate with HtmlFormatter(style=...).get_style_defs(root selector).' \
    '   Container backgrounds are owned by DlightRAG semantic tokens. */'
    PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python - <<'PY'
import re

from pygments.formatters import HtmlFormatter

for mode, style in (("light", "friendly"), ("dark", "github-dark")):
    root = f'[data-color-mode="{mode}"] .highlight'
    print(f"/* {style} */")
    for line in HtmlFormatter(style=style).get_style_defs(root).splitlines():
                if not line.startswith(root):
                        continue
                line = re.sub(r"background(?:-color)?:\s*[^;}]+;?\s*", "", line)
                if not re.search(r"\{\s*\}", line):
                        print(line)
PY
} > src/dlightrag/web/static/pygments.css
```

Inspect the file to ensure every retained selector begins with a color-mode root and no Pygments container background remains.

- [ ] **Step 4: Make MathJax inherit the themed text color**

Add to `global.css`:

```css
mjx-container {
    color: inherit;
}
```

- [ ] **Step 5: Run focused validation**

Run:

```bash
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py
npm --prefix frontend run lint:css
npm --prefix frontend run build
```

Expected: all commands pass.

- [ ] **Step 6: Commit rich-content theming**

```bash
git add src/dlightrag/web/static/pygments.css frontend/styles/global.css \
  tests/unit/test_web_frontend_static.py src/dlightrag/web/static/generated
git commit -m "fix(web): theme rich content colors"
```

## Task 4: Accessible Appearance Menu

**Files:**
- Create: `frontend/ui/theme.ts`
- Modify: `frontend/lib/listbox.ts`
- Modify: `frontend/ui/workspaces.ts`
- Modify: `frontend/ui/main.ts`
- Modify: `src/dlightrag/web/templates/index.html`
- Modify: `frontend/styles/layout.css`
- Modify: `NOTICE`
- Create: `tests/e2e/test_web_theme.py`

- [ ] **Step 1: Add failing browser behavior tests**

Create `tests/e2e/test_web_theme.py` with product-level cases:

```python
from playwright.sync_api import Page


def test_system_theme_follows_browser_and_explicit_choice_persists(page: Page) -> None:
    page.emulate_media(color_scheme="light")
    page.goto("/web/")

    root = page.locator("html")
    assert root.get_attribute("data-theme") == "system"
    assert root.get_attribute("data-color-mode") == "light"

    page.get_by_role("button", name="Appearance").click()
    page.get_by_role("menuitemradio", name="Dark").click()
    assert root.get_attribute("data-theme") == "dark"
    assert root.get_attribute("data-color-mode") == "dark"

    page.reload()
    assert root.get_attribute("data-theme") == "dark"
    assert root.get_attribute("data-color-mode") == "dark"


def test_system_theme_tracks_runtime_media_changes(page: Page) -> None:
    page.emulate_media(color_scheme="dark")
    page.goto("/web/")
    root = page.locator("html")
    assert root.get_attribute("data-color-mode") == "dark"

    page.emulate_media(color_scheme="light")
    assert root.get_attribute("data-theme") == "system"
    assert root.get_attribute("data-color-mode") == "light"


def test_theme_menu_keyboard_and_focus_contract(page: Page) -> None:
    page.goto("/web/")
    trigger = page.get_by_role("button", name="Appearance")
    trigger.focus()
    trigger.press("ArrowDown")

    system = page.get_by_role("menuitemradio", name="System")
    assert system.is_focused()
    assert system.get_attribute("aria-checked") == "true"
    system.press("End")
    assert page.get_by_role("menuitemradio", name="Dark").is_focused()
    page.keyboard.press("Escape")
    assert trigger.is_focused()
    assert trigger.get_attribute("aria-expanded") == "false"


def test_theme_switching_survives_unavailable_storage(page: Page) -> None:
    page.add_init_script(
        """
        Storage.prototype.getItem = () => { throw new DOMException('blocked'); };
        Storage.prototype.setItem = () => { throw new DOMException('blocked'); };
        """
    )
    page.goto("/web/")
    page.get_by_role("button", name="Appearance").click()
    page.get_by_role("menuitemradio", name="Light").click()
    assert page.locator("html").get_attribute("data-color-mode") == "light"


def test_theme_preference_synchronizes_to_another_tab(page: Page) -> None:
    page.goto("/web/")
    other_page = page.context.new_page()
    try:
        other_page.goto("/web/")
        page.get_by_role("button", name="Appearance").click()
        page.get_by_role("menuitemradio", name="Light").click()
        assert other_page.locator("html").get_attribute("data-theme") == "light"
        assert other_page.locator("html").get_attribute("data-color-mode") == "light"
    finally:
        other_page.close()


def test_theme_survives_files_panel_partial_updates(page: Page) -> None:
    page.goto("/web/")
    page.get_by_role("button", name="Appearance").click()
    page.get_by_role("menuitemradio", name="Light").click()
    page.get_by_role("button", name="Files").click()
    page.locator("#panel.open").wait_for()
    assert page.locator("html").get_attribute("data-theme") == "light"
    assert page.locator("html").get_attribute("data-color-mode") == "light"
```

- [ ] **Step 2: Run the browser file and verify it fails**

Run:

```bash
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/e2e/test_web_theme.py
```

Expected: FAIL because the Appearance trigger and runtime module do not exist.

- [ ] **Step 3: Generalize roving navigation without a compatibility shell**

Replace `installListboxArrowNavigation` with:

```ts
export function installRovingArrowNavigation(
    container: HTMLElement,
    itemSelector: string,
): void {
    container.addEventListener('keydown', (event) => {
        if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
        const items = Array.from(container.querySelectorAll<HTMLElement>(itemSelector));
        if (items.length === 0) return;
        event.preventDefault();
        const current = items.indexOf(document.activeElement as HTMLElement);
        const next = event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? items.length - 1
              : current < 0
                ? event.key === 'ArrowDown' ? 0 : items.length - 1
                : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        items[next]?.focus();
    });
}
```

Update the workspace popover call to:

```ts
installRovingArrowNavigation(popover, '[role="option"]');
```

- [ ] **Step 4: Add static theme control markup**

Add a `theme-control` wrapper after `Files` in the topbar. Use these stable hooks and semantics:

```html
<div class="theme-control" id="theme-control">
    <button class="theme-trigger" id="theme-trigger" type="button"
            aria-label="Appearance" aria-haspopup="menu"
            aria-controls="theme-menu" aria-expanded="false">
        <!-- Lucide Moon, class theme-trigger-icon--dark, aria-hidden=true -->
        <!-- Lucide Sun, class theme-trigger-icon--light, aria-hidden=true -->
        <span class="theme-tooltip" role="tooltip">Appearance</span>
    </button>
    <div class="theme-menu" id="theme-menu" role="menu" aria-label="Appearance" hidden>
        <button type="button" role="menuitemradio" data-theme-value="system" aria-checked="true">
            <!-- Monitor icon --> <span>System</span><span class="theme-menu-check" aria-hidden="true">✓</span>
        </button>
        <button type="button" role="menuitemradio" data-theme-value="light" aria-checked="false">
            <!-- Sun icon --> <span>Light</span><span class="theme-menu-check" aria-hidden="true">✓</span>
        </button>
        <button type="button" role="menuitemradio" data-theme-value="dark" aria-checked="false">
            <!-- Moon icon --> <span>Dark</span><span class="theme-menu-check" aria-hidden="true">✓</span>
        </button>
    </div>
</div>
```

Copy the exact Lucide paths recorded in the approved design research. Do not add an icon package.

- [ ] **Step 5: Implement runtime ownership**

Create `frontend/ui/theme.ts` around the pure functions and existing helpers. Its public entry point is:

```ts
export function setupTheme(): void;
```

The module must:

```ts
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
const root = document.documentElement;

function applyPreference(preference: ThemePreference, persist: boolean): void {
    const mode = resolveColorMode(preference, mediaQuery.matches);
    root.dataset.theme = preference;
    root.dataset.colorMode = mode;
    root.style.colorScheme = mode;
    if (persist) {
        try { localStorage.setItem(THEME_STORAGE_KEY, preference); } catch (_) {}
    }
    syncMenuState(preference);
    syncSystemListener(preference === 'system');
}
```

Use `createAutoDismiss` for outside-click/Escape and `installRovingArrowNavigation(menu, '[role="menuitemradio"]')` for movement. Dismissal behavior must preserve modern pointer focus semantics: Escape restores trigger focus; outside pointer click closes while preserving the clicked target's natural focus. Handle `storage` events whose key is `THEME_STORAGE_KEY` or `null` by parsing the new value and applying without writing it back.

Add `setupTheme` to the existing dynamic-import tuple in `frontend/ui/main.ts` and call it before feature modules that do not depend on it.

- [ ] **Step 6: Style the trigger and menu**

In `layout.css`:

- size the trigger from `--size-button` and icons at 17px;
- select `.theme-trigger-icon--dark` or `--light` from `data-color-mode`;
- use semantic hover, focus, surface, border, shadow, and accent tokens;
- keep the tooltip hidden until hover/focus-visible;
- position the menu from the topbar control without affecting topbar geometry;
- keep labels visible at mobile widths and constrain the menu inside the viewport;
- honor the existing reduced-motion rule without adding global theme transitions.

- [ ] **Step 7: Record Lucide attribution**

Append a concise entry to `NOTICE` naming Lucide Icons, its ISC license, and the embedded Sun/Moon/Monitor SVG geometry. Do not add source headers or generated signatures.

- [ ] **Step 8: Run focused implementation validation**

Run:

```bash
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run lint:css
npm --prefix frontend run build
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py tests/e2e/test_web_theme.py
```

Expected: all commands pass.

- [ ] **Step 9: Commit the control**

```bash
git add frontend/lib/listbox.ts frontend/ui/theme.ts frontend/ui/workspaces.ts \
  frontend/ui/main.ts frontend/styles/layout.css src/dlightrag/web/templates/index.html \
  tests/e2e/test_web_theme.py NOTICE src/dlightrag/web/static/generated
git commit -m "feat(web): add appearance menu"
```

## Task 5: Browser Polish and Evidence-Driven Corrections

**Files:**
- Modify only files with a demonstrated visual, contrast, or interaction defect from Tasks 2-4.
- Modify or delete tests only when they encode superseded implementation details rather than product behavior.

- [ ] **Step 1: Start the isolated Web application**

Use the existing E2E app fixture for deterministic UI behavior first:

```bash
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/e2e/test_web_theme.py --headed
```

For manual browser inspection, start the project-supported local server from the worktree on an unused port and record its URL. Do not reuse the temporary theme mockup server.

- [ ] **Step 2: Capture desktop and mobile screenshots**

Use Playwright for effective dark and Mineral Light at:

- desktop: `1440 x 900`;
- mobile: `390 x 844`.

Exercise and inspect:

- base chat and composer;
- conversation sidebar;
- Files panel and upload state;
- Sources panel with citation highlight;
- Markdown table, inline code, fenced code, and MathJax;
- confirmation dialog, toast, and image lightbox;
- open appearance menu.

Confirm that dynamic content does not resize the topbar, composer, menu, or fixed panels.

- [ ] **Step 3: Check computed contrast for owned text roles**

Use a Playwright page evaluation to read computed foreground/background colors for:

- body text on base;
- primary and muted text on surface;
- accent links/buttons on their backgrounds;
- danger actions and their on-danger text.

Calculate WCAG contrast and require at least `4.5:1` for normal text and `3:1` for large text/icons. Adjust semantic token values, not component overrides.

- [ ] **Step 4: Check first-frame behavior**

Create fresh browser contexts for system dark and system light with no local storage. Capture the earliest post-navigation screenshot and root attributes. Confirm the intended effective mode is present before any interactive module initializes and no opposite-theme frame appears.

- [ ] **Step 5: Correct only observed defects**

For each observed issue:

1. name the failing surface and semantic role;
2. change the smallest owning token or component rule;
3. rerun the affected screenshot/contrast check;
4. add or revise a permanent test only if it protects a durable user-observable contract.

Do not add palette knobs, animation systems, CSS frameworks, generalized preference stores, or extra test abstractions.

- [ ] **Step 6: Re-run focused gates and commit polish**

Run:

```bash
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run lint:css
npm --prefix frontend run build
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py tests/e2e/test_web_theme.py
git diff --check
```

If corrections were needed:

```bash
git add frontend src/dlightrag/web/static src/dlightrag/web/templates tests NOTICE
git commit -m "fix(web): polish light theme surfaces"
```

If no corrections were needed, do not create an empty or evidence-only commit.

## Task 6: Full Validation, Rebuild, Integration, and Cleanup

**Files:**
- Generated output from `npm --prefix frontend run build`
- No additional source files unless a gate exposes a theme regression.

- [ ] **Step 1: Audit final scope and stale paths**

Run:

```bash
git status --short
git diff main...HEAD --stat
git diff main...HEAD --check
rg -n 'var\(--color-gold-' frontend/styles || true
rg -n 'installListboxArrowNavigation' frontend || true
```

Expected: only theme/spec/plan/generated changes; no stale helper name; no direct component gold primitives; no temporary files.

- [ ] **Step 2: Run the frontend release gates**

```bash
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run lint:css
npm --prefix frontend run build
npm --prefix frontend audit --omit=dev
```

Expected: all commands pass. If audit reports an existing lockfile advisory without a theme-related dependency change, report it rather than running an unrelated automated upgrade.

- [ ] **Step 3: Run focused and full repository CI**

```bash
PYTHONPATH=src /Users/hanlianlyu/Github/DlightRAG/.venv/bin/python -m pytest -q \
  tests/unit/test_web_frontend_static.py tests/e2e/test_web_theme.py
make ci
```

Expected: focused tests and repository CI pass. If a pre-existing unrelated test is red, preserve its evidence, identify it explicitly, and do not change theme code to satisfy it.

- [ ] **Step 4: Review implementation against the approved spec**

Confirm each design completion condition in `docs/web-theme-design.md`, then run a code review focused on behavior regressions, unnecessary abstractions, stale raw theme paths, accessibility, and test value. Fix only concrete findings and rerun the narrowest falsifying check after each fix.

- [ ] **Step 5: Ensure the feature branch is committed and clean**

```bash
git status --short
git --no-pager log --oneline main..HEAD
```

If source changes remain after validation:

```bash
git add -A
git commit -m "feat(web): complete theme support"
```

Expected: clean feature worktree with a small sequence of plain commits and no generated orphan bundles.

- [ ] **Step 6: Merge to main without losing concurrent user work**

From the main checkout, verify it is clean and inspect whether `main` advanced during implementation. If it advanced, rebase the feature branch onto current `main` inside the worktree and rerun focused validation. Then fast-forward merge:

```bash
cd /Users/hanlianlyu/Github/DlightRAG
git status --short --branch
git merge --ff-only feat/web-theme
```

Do not reset, checkout over, or discard concurrent changes.

- [ ] **Step 7: Rebuild and verify on main**

```bash
npm --prefix frontend ci
npm --prefix frontend run build
make ci
git status --short --branch
```

Expected: rebuilt generated assets are committed or identical, CI passes, and `main` is clean. If rebuilding changes tracked bundles, commit them on `main` with a focused build commit before pushing.

- [ ] **Step 8: Push main with explicit non-force semantics**

```bash
git push origin main
```

Expected: remote `main` advances to the validated local `main`. Never force-push for this feature.

- [ ] **Step 9: Remove the temporary worktree and branch**

```bash
git worktree remove .worktrees/web-theme
git branch -d feat/web-theme
git worktree list
git branch --list
```

Expected: only the primary checkout remains for this feature, and `feat/web-theme` no longer exists.

- [ ] **Step 10: Report final evidence**

Report:

- user-visible theme behavior and selected Mineral Light direction;
- exact focused tests, frontend gates, `make ci`, and browser viewport checks run;
- local URL used for final inspection;
- pushed main commit(s);
- confirmation that the worktree and feature branch were removed;
- any residual advisory that was demonstrably pre-existing and unrelated.