# Web Theme Design

**Status:** Implemented — Mineral themes, Soft geometry, and owned Split Layouts
**Date:** 2026-07-23
**Updated:** 2026-08-31

## Purpose

The DlightRAG Web UI provides a polished `System / Light / Dark` appearance preference. It keeps the dark theme's identity, adds the warm-neutral **Mineral Light** palette, and makes the correct theme visible on the first painted frame. Its current appearance also includes role-based Soft geometry and token-bridged split panels.

This design does not add custom palettes, account-level preferences, or server-side preference storage. Internationalization is a separate product concern with its own plan; it is not a theme-design constraint.

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

- Keep theme state dependency-free. Do not add a theme framework, state-management layer, or external icon runtime.
- Prefer semantic CSS tokens over component-specific light-mode overrides.
- Reuse the existing popover dismissal and keyboard-navigation infrastructure.
- The package-owned design system's runtime CSS remains authoritative and is projected deterministically for design tooling.
- Native Drawer/Dialog behavior remains product-owned; split behavior belongs to the package-owned design system.

## State Model

The root element carries both the stored preference and the effective color mode:

```html
<html data-theme="system" data-color-mode="dark">
```

- `data-theme`: `system | light | dark`; this is the user preference.
- `data-color-mode`: `light | dark`; this is the currently rendered appearance.

The preference is stored in local storage under `dlightrag-theme`. Theme persistence is a browser-only presentation concern; no API endpoint, cookie, database column, or server request state is required.

Vite emits `frontend/theme-init.ts` as a dedicated hashed classic script. The static `<head>` loads it before any stylesheet or application module. It validates the saved preference, resolves `System` with `matchMedia('(prefers-color-scheme: dark)')`, and updates both root attributes. The HTML defaults to `system + dark`, so any bootstrap failure preserves today's safe dark appearance.

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

The topbar trigger is a square, borderless icon button sized with `--size-button`. Both decorative SVGs are present in the Vite/Lit application shell; CSS selects the correct one from `data-color-mode`, so the first frame never shows an empty or stale icon.

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
- a choice applies immediately without reload or a server request.

The generic popover dismissal helper owns the menu lifecycle. The shared
`installRovingArrowNavigation` helper accepts a role selector and provides the
same keyboard model to theme, workspace, and file menus.

Semantic icon geometry is generated from the pinned, generation-only `lucide-static` package into the checked-in design-system registry. Production bundles retain only selected geometry rendered with `currentColor`; controls own accessible names and the repository NOTICE records the Lucide license. No runtime icon dependency is loaded.

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
| Danger | `#f87171` | `#b91c1c` |

The mirrored stone ramps keep perceptual surface steps comparable; the gold
ramp supplies an accessible accent in each mode. Borders and row tints use
low-alpha stone values. `frontend/tokens/ramp.test.ts` enforces ramp membership
and elevation direction.

Docked panels use tone plus a hairline border, not shadows. Only overlapping
popovers, menus, dialogs, and toasts cast shadows. Components consume semantic
aliases; primitive palette values remain private to the token layer.

Spacing and typography remain unchanged by the theme. Geometry and panel behavior follow the separate role rules below.

## Geometry And Panels

Geometry follows surface role rather than component size. Controls use 10px,
cards and rich-content containers 16px, popovers 18px, dialogs 22px, and the
composer 24px. Pills remain `999px` and circles remain 50%. Full-viewport app
shells, docked sidebars and panels, structural sections, and internal seams stay
square at every viewport. This keeps Soft contained surfaces from rounding the
application silhouette or opening dark corner wedges.

Inspector and Artifact Canvas use nested local `dl-split-layout` elements on
wide screens. The design-system element owns axis layout, pointer and keyboard
input, and separator ARIA; the app adapter owns open state, breakpoints,
clamping, and persistence. Inspector and Artifact Canvas persist separate
preferred pixel widths; clamping for the conversation sidebar and minimum chat
width never overwrites those preferences. A single token-backed hairline has an
invisible 12px hit area, so the previous double-border seam is not possible.

Opening an Artifact citation is Shell-mediated. On desktop, Side remains Side,
Wide remains Wide, and Fullscreen reduces only to Wide while Sources opens. On
compact screens, the Canvas closes and Sources opens in the Inspector drawer.

Below 1200px resizing is disabled and the panel is an overlay: the primary app
remains full viewport width under the scrim, while modal focus, inert state,
Escape, and focus restoration remain native DlightRAG behavior. At phone widths
the active panel becomes full bleed. External Drawer and Dialog components remain
rejected; existing native overlays keep these geometry rules.

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
- Lit application rerenders cannot reset the theme because the state lives on `<html>` outside `<dl-app>`.
