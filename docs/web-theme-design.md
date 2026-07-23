# Web Theme Design

**Status:** Approved for implementation  
**Date:** 2026-07-23

## Purpose

Add a polished `System / Light / Dark` appearance preference to the DlightRAG Web UI. Keep the existing dark theme's identity, add the approved warm-neutral **Mineral Light** palette, and make the correct theme visible on the first painted frame.

This work is limited to appearance. It does not add internationalization, Mermaid, custom palettes, account-level preferences, or server-side preference storage.

## Product Decisions

- New users default to `System` and follow the operating-system preference.
- Explicit `Light` or `Dark` selection overrides the system preference.
- `System` responds to operating-system changes while the page is open.
- The light palette is Mineral Light: warm stone surfaces, graphite text, and a darker accessible gold accent.
- The control is an icon button to the right of `Files` in the topbar.
- The trigger shows a Lucide `Moon` in the effective dark mode and `Sun` in the effective light mode.
- The menu uses Lucide `Monitor`, `Sun`, and `Moon` icons for `System`, `Light`, and `Dark`.
- The existing dark appearance remains visually stable except for correcting code-highlight contrast.

## Engineering Constraints

- Keep the implementation small and dependency-free. Do not add a theme framework, state-management layer, or icon runtime for this feature.
- Prefer semantic CSS tokens over component-specific light-mode overrides.
- Reuse the existing popover dismissal and keyboard-navigation infrastructure.
- Avoid unrelated layout, typography, or component refactors.
- Direct frontend dependencies must use current stable release floors with caret ranges for compatible onward updates; major upgrades must stay compatible with the CI Node baseline and pass frontend gates.
- Tests are evidence and regression records, not sources of product requirements. Update or remove tests that encode obsolete, overly granular, or low-value behavior instead of preserving that behavior in production code.
- The final implementation must be developed in an isolated worktree, rebuilt, validated, merged to `main`, committed and pushed, then leave no temporary worktree or branch behind.

## State Model

The root element carries both the stored preference and the effective color mode:

```html
<html data-theme="system" data-color-mode="dark">
```

- `data-theme`: `system | light | dark`; this is the user preference.
- `data-color-mode`: `light | dark`; this is the currently rendered appearance.

The preference is stored in local storage under `dlightrag-theme`. Theme persistence is a browser-only presentation concern; no API endpoint, cookie, database column, or HTMX request state is required.

An inline bootstrap in `<head>` runs before the stylesheet link. It validates the saved preference, resolves `System` with `matchMedia('(prefers-color-scheme: dark)')`, and updates both root attributes. The HTML defaults to `system + dark`, so any bootstrap failure preserves today's safe dark appearance.

The document also declares native `color-scheme` support. The effective mode controls form controls, scrollbars, and browser-owned UI consistently with the page.

## Runtime Module

A focused `frontend/ui/theme.ts` module owns runtime behavior:

- parse and validate the stored preference;
- resolve the effective color mode;
- update root attributes and native `color-scheme`;
- persist an explicit selection when storage is available;
- update trigger icons, menu selection, and ARIA state;
- listen for system color-scheme changes only while `System` is selected;
- listen for the browser `storage` event to synchronize other tabs;
- degrade to an in-memory selection when local storage is unavailable.

Pure preference parsing and color-mode resolution remain separate from DOM wiring so the meaningful state rules are easy to test without a simulated browser.

No nanoevents bus or dedicated store is needed: theme state has one owner and only mutates `<html>` plus its own control.

## Theme Control

The topbar trigger is a square, borderless icon button sized with `--size-button`. Both decorative SVGs are present in the server-rendered template; CSS selects the correct one from `data-color-mode`, so the first frame never shows an empty or stale icon.

The trigger has:

- accessible name and tooltip `Appearance`;
- `aria-haspopup="menu"`;
- synchronized `aria-expanded`.

The popover uses:

- `role="menu"`;
- three `role="menuitemradio"` choices;
- synchronized `aria-checked` and a visual checkmark;
- `Monitor System`, `Sun Light`, and `Moon Dark` rows.

Interaction behavior:

- click, Enter, or Space opens the menu;
- ArrowDown opens and focuses the active choice;
- ArrowUp, ArrowDown, Home, and End provide roving navigation;
- Enter or Space applies a choice and closes the menu;
- Escape closes and restores trigger focus;
- outside pointer click closes while preserving the clicked target's natural focus;
- a choice applies immediately without reload or HTMX activity.

The existing generic popover dismissal helper remains the lifecycle owner. The current listbox arrow-navigation helper may be renamed and generalized to a role-selector-based roving helper, with the workspace selector updated to the new name. No compatibility wrapper should remain if all callers migrate together.

Lucide SVG geometry is embedded statically with `currentColor`, a 24px viewBox, 17px rendered size, round caps/joins, and a quieter DlightRAG stroke. The repository NOTICE records the applicable Lucide license; no package dependency is added.

## Color System

Theme-specific values live only in the token layer. Components consume semantic aliases.

Core palette:

| Semantic role | Dark | Mineral Light |
|---|---|---|
| Page background | `#0c0a09` | `#f5f5f2` |
| Primary surface | `#1c1917` | `#ffffff` |
| Elevated surface | `#292524` | `#e9e8e2` |
| Primary text | `#f5f5f4` | `#24231f` |
| Body text | `#d6d3d1` | `#45443f` |
| Muted text | `#a8a29e` | `#66645d` |
| Primary accent | `#d2b661` | `#806719` |
| Danger | `#f87171` | `#b42318` |

The implementation will define the complete light values needed by existing semantic roles, including hover, active, border, source surface, overlays, selection, shadow, and on-accent text.

Components currently referencing `gold-100/200/...` directly will move to purpose-based aliases such as action accent, strong accent, muted accent, and on-accent text. Component-level raw colors that encode theme behavior will move to semantic tokens. Primitive palette values remain allowed inside the token file.

This is a color-boundary cleanup, not a general CSS rewrite. Existing spacing, typography, geometry, layout, and motion remain unchanged.

## Rich Content

### Pygments

The checked-in `pygments.css` contains two generated, root-scoped palettes:

- Pygments `friendly` for `data-color-mode="light"`;
- Pygments `github-dark` for `data-color-mode="dark"`.

Generated selectors are scoped to the effective color mode. Pygments-owned container backgrounds are removed so code blocks continue to use DlightRAG surface tokens. The file includes its deterministic regeneration commands.

This also fixes the current mismatch where default light-syntax colors are displayed on a dark code background.

### MathJax

MathJax output explicitly inherits `currentColor`. Theme changes do not trigger re-typesetting and do not disturb frozen streaming blocks.

### Images and Overlays

Lightbox scrims remain dark in both modes because their purpose is image isolation. Caption, border, shadow, and panel colors use semantic tokens. Uploaded images and source page images are not recolored.

## Failure Handling

- Missing storage value resolves to `System`.
- Invalid storage value is ignored, removed when possible, and resolves to `System`.
- Storage read/write failures do not prevent in-page switching.
- Missing `matchMedia` or bootstrap failure preserves the dark fallback.
- System listeners are detached when an explicit preference is selected.
- Runtime system changes update only the effective mode, not the stored preference.
- HTMX partial replacement cannot reset the theme because the state lives on `<html>` outside all swap targets.

## Verification Strategy

Verification scales with behavior and risk. It does not preserve obsolete implementation details.

### Focused unit evidence

- valid, missing, and invalid preference parsing;
- preference plus system state resolving to effective mode;
- small pure helpers only, without a browser mock framework.

### Browser behavior evidence

Playwright verifies:

- first visit follows simulated system light and dark preferences;
- explicit selection overrides the system and persists across reload;
- System responds to a runtime media-query change;
- local storage failure still permits in-page switching;
- menu keyboard navigation and radio semantics;
- Escape dismissal restores trigger focus, while outside pointer dismissal preserves clicked-target focus;
- theme survives representative HTMX partial updates;
- desktop and mobile geometry remain stable.

### Visual and accessibility evidence

Inspect screenshots for both effective modes across the chat, Files, Sources, code, tables, MathJax, dialogs, toast, and lightbox surfaces. Check computed contrast for primary text, muted text, interactive gold, and danger states against their actual backgrounds. Verify the first screenshot frame already has the intended theme.

### Repository gates

- frontend unit tests, typecheck, stylelint, and Vite production build;
- focused Python/static and Playwright tests;
- repository `make ci` when feasible;
- production image rebuild and local Web smoke test before completion.

If an existing test conflicts solely because it asserts a superseded selector, bundle name, raw color, or implementation detail, revise or delete that assertion and retain only behaviorally meaningful coverage.

## Completion State

The feature is complete when:

- all three preferences work with no incorrect first frame;
- Mineral Light and the existing Dark theme are coherent across all owned surfaces;
- syntax highlighting is readable in both modes;
- the menu is keyboard and screen-reader operable;
- no new runtime dependency or server-side preference path exists;
- final build and CI evidence pass;
- the rebuilt application is available for local inspection;
- implementation is merged, committed, and pushed on `main`;
- temporary worktree and feature branch are removed.