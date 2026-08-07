# Web Theme Design

**Status:** Implemented
**Date:** 2026-07-23

## Purpose

The DlightRAG Web UI provides a polished `System / Light / Dark` appearance preference. It keeps the dark theme's identity, adds the warm-neutral **Mineral Light** palette, and makes the correct theme visible on the first painted frame.

This work is limited to appearance. It does not add internationalization, custom palettes, account-level preferences, or server-side preference storage.

## Product Decisions

- New users default to `System` and follow the operating-system preference.
- Explicit `Light` or `Dark` selection overrides the system preference.
- `System` responds to operating-system changes while the page is open.
- The light palette is Mineral Light: warm stone surfaces, graphite text, and a darker accessible gold accent.
- The control is an icon button to the right of `Files` in the topbar.
- The trigger shows a Lucide `Moon` in the effective dark mode and `Sun` in the effective light mode.
- The menu uses Lucide `Monitor`, `Sun`, and `Moon` icons for `System`, `Light`, and `Dark`.
- The existing dark appearance remains visually stable except for correcting code-highlight contrast.

## Architecture Constraints

- Keep the implementation small and dependency-free. Do not add a theme framework, state-management layer, or icon runtime for this feature.
- Prefer semantic CSS tokens over component-specific light-mode overrides.
- Reuse the existing popover dismissal and keyboard-navigation infrastructure.
- Avoid unrelated layout, typography, or component refactors.

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

The generic popover dismissal helper owns the menu lifecycle. The shared
`installRovingArrowNavigation` helper accepts a role selector and provides the
same keyboard model to theme, workspace, and file menus.

Lucide SVG geometry is embedded statically with `currentColor`, a 24px viewBox, 17px rendered size, round caps/joins, and a quieter DlightRAG stroke. The repository NOTICE records the applicable Lucide license; no package dependency is added.

## Color System

Theme-specific values live only in the token layer. Components consume semantic aliases.

Core palette:

Both themes are Tailwind stone, mirrored by step, so a value's place in the ramp
can be read rather than measured.

| Semantic role | Dark | Mineral Light |
|---|---|---|
| Page background | `#0c0a09` stone-950 | `#fafaf9` stone-50 |
| Primary surface | `#1c1917` stone-900 | `#e7e5e4` stone-200 |
| Elevated surface | `#292524` stone-800 | `#d6d3d1` stone-300 |
| Primary text | `#f5f5f4` stone-100 | `#1c1917` stone-900 |
| Body text | `#d6d3d1` stone-300 | `#44403c` stone-700 |
| Muted text | `#a8a29e` stone-400 | `#57534e` stone-600 |
| Primary accent | `#d2b661` gold-200 | `#7e6c37` gold-400 |
| Danger | `#f87171` | `#b42318` |

The light surfaces skip stone-100: measured in OKLCH it sits only 1.5 L\* from
stone-50, a quarter of dark's first step. Stone-50/200/300 steps by 6.2 and 5.4
against dark's 6.9 and 5.2, so both themes separate their surfaces by the same
perceived amount.

The gold ramp is perceptually even by construction — 12.33 L\* per step, hue
held within 2° — so each theme takes the step that clears its canvas rather than
a value tuned by hand. Borders, dividers and row tints are stone at low alpha:
stone-500 in dark, stone-700 in light. Scrims and overlay shadows are stone-950
and stone-900 respectively.

Each ramp steps away from the page background as elevation rises, so the
conversation always holds the strongest contrast against its text: lighter in
dark, darker in light. `frontend/tokens/ramp.test.ts` asserts that direction and
rejects any colour that is not a member of a declared ramp.

Full-height drawers are edges rather than floating cards, so they are separated
by that tone step plus a hairline border. Neither theme casts a drawer shadow:
in light the surface sits below the canvas, and in dark a black shadow over
`#0c0a09` resolves to `rgb(8,7,6)`. Only elements that overlap the conversation
— popovers, menus, dialogs, toasts — carry one.

The token layer defines the complete light values needed by existing semantic roles, including hover, active, border, source surface, overlays, selection, shadow, and on-accent text.

Components consume purpose-based aliases such as action accent, strong accent,
muted accent, and on-accent text. Primitive palette values remain internal to
the token file.

This is a color-boundary cleanup, not a general CSS rewrite. Existing spacing, typography, geometry, layout, and motion remain unchanged.

## Rich Content

### Pygments

The checked-in `pygments.css` contains two generated, root-scoped palettes:

- Pygments `xcode` for `data-color-mode="light"`;
- Pygments `github-dark` for `data-color-mode="dark"`.

Generated selectors are scoped to the effective color mode. Pygments-owned container backgrounds are removed so code blocks continue to use DlightRAG surface tokens. Two low-contrast upstream foregrounds are replaced with fixed accessible values. The file includes its deterministic regeneration command.

This prevents light-syntax colors from being displayed on a dark code
background before or after an appearance change.

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
