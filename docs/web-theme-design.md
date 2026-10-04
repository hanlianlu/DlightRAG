# Web Theme Design

The Web UI offers a `System / Light / Dark` appearance preference: the dark theme
and the warm-neutral **Mineral Light** palette, correct on the first painted
frame, with role-based Soft geometry and token-bridged split panels. There are no
custom palettes, account-level preferences, or server-side preference storage.

## Product Decisions

- New users default to `System` and follow the operating-system preference.
- Explicit `Light` or `Dark` selection overrides the system preference.
- `System` responds to operating-system changes while the page is open.
- The light palette is Mineral Light: warm stone surfaces, graphite text, and a darker accessible gold accent.
- The control is an icon button to the right of `Files` in the topbar.
- The trigger shows a Lucide `Moon` in the effective dark mode and `Sun` in the effective light mode.
- The menu uses Lucide `Monitor`, `Sun`, and `Moon` icons for `System`, `Light`, and `Dark`.

## Architecture Constraints

- Keep theme state dependency-free. Do not add a theme framework, state-management layer, or external icon runtime.
- Prefer semantic CSS tokens over component-specific light-mode overrides.
- Reuse the existing popover dismissal and keyboard-navigation infrastructure.
- The package-owned design system's runtime CSS remains authoritative and is projected deterministically for design tooling.
- Every page loads only the stylesheets its own entry imports, so page-level rules such as the design-system catalog's `body` never reach the application, and the application's never reach the catalog.
- Native Drawer/Dialog behavior remains product-owned; split behavior belongs to the package-owned design system.

## State Model

The root element carries both the stored preference and the effective color mode:

```html
<html data-theme="system" data-color-mode="dark">
```

- `data-theme`: `system | light | dark`; this is the user preference.
- `data-color-mode`: `light | dark`; this is the currently rendered appearance.

The preference is stored in local storage under `dlightrag-theme`. Theme persistence is a browser-only presentation concern; no API endpoint, cookie, database column, or server request state is required.

Vite emits `frontend/theme-init.ts` as a dedicated hashed classic script. The static `<head>` loads it before any stylesheet or application module. It validates the saved preference, resolves `System` with `matchMedia('(prefers-color-scheme: dark)')`, and updates both root attributes. The HTML defaults to `system + dark`, so any bootstrap failure keeps the safe dark appearance.

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

Interaction behavior (the product-wide menu contract):

- click, ArrowDown, Enter, or Space opens the menu on the first choice;
  ArrowUp opens it on the last;
- ArrowUp and ArrowDown move with wrapping, Home and End jump to the ends, and
  a typed letter moves to the next choice whose label starts with it;
- Enter or Space applies a choice and closes the menu;
- Escape closes and restores trigger focus;
- outside pointer click closes while preserving the clicked target's natural focus;
- a choice applies immediately without reload or a server request.

The generic popover dismissal helper owns the menu lifecycle. The design
system's `dl-menu` element applies the in-menu keys, and its
`menuButtonFocus()` maps the button's keys, so the theme, answer mode, agent
effort, and conversation actions menus behave identically. The workspace and
file pickers are dialogs, but reuse the same `rovingFocusKeydown()` step.

Semantic icon geometry is generated from the pinned, generation-only `lucide-static` package into the checked-in design-system registry. Production bundles retain only selected geometry rendered with `currentColor`; controls own accessible names and the repository NOTICE records the Lucide license. No runtime icon dependency is loaded.

A semantic icon whose state varies keeps one glyph per state (the theme control's `Monitor`/`Sun`/`Moon`). Geometry a state ladder needs and Lucide does not ship is authored in the same registry as a derived `dlightrag` entry that keeps the Lucide shape: the agent effort ladder is Lucide's `gauge` dial with the needle at each level's stop, and `status-dot` is a product coin. Those entries are generated, reviewed, and tested like the selected Lucide ones, and no state-bearing glyph ever carries accessible meaning on its own.

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
| Subtle marks | `#78716c` stone-500 | `#78716c` stone-500 |
| Dim, disabled only | `#57534e` stone-600 | `#a8a29e` stone-400 |
| Primary accent | `#d2b661` gold-200 | `#7e6c37` gold-400 |
| Accent text | `#a7904c` gold-300 | `#574a24` gold-500 |
| Focus ring | gold-200 at 64% | gold-500 at 80% |
| Danger | `#f87171` | `#b91c1c` |

Muted is the faintest text role that clears WCAG AA, 4.5:1, on every surface in
both modes. Subtle (`--color-text-subtle`) is the ramp's midpoint, so the mirror
leaves it stone-500 in both modes, and it reaches only about 3.2:1 on an
elevated surface. It colours non-text marks held to 3:1, such as ghost icon
buttons at rest and unselected radio rings, and never text: a caption or label
that would sit at Subtle takes Muted and keeps its rank through size, weight, or
case.

Dim (`--color-text-dim`) sits one step below Subtle and clears neither floor:
1.99:1 on an elevated surface in dark and 1.69:1 in Mineral Light. WCAG sets no
contrast for an inactive control, so Dim colours only a disabled control, such as
the Send button's glyph while there is nothing to send. Placeholder and ghost
text are still text and take Muted, a clear step fainter than the Primary text a
person types. A mark that shows a state, such as an idle status light, a
disclosure chevron, or an unchecked switch thumb, takes Subtle.

Accent text (`--color-accent-text`) is the gold for small text, such as the Mine
badge in the skill menu and the Load more and Try again buttons in
failed-document recovery. It clears 4.5:1 on every surface in both modes:
gold-300 in dark, one step quieter than the Primary accent (4.87:1 at worst),
and gold-500 in Mineral Light, one step darker, because gold-400 reads only
3.45:1 on an elevated surface where gold-500 reads 5.85:1. The composer's attach
glyph takes it too, well above the 3:1 a mark needs.

The focus ring (`--color-control-ring`, aliased as `--focus-ring-color`) is a
2px outline drawn 2px outside its control, so it reads against the surface or
row tint around the control, not the control's own fill. It is held to 3:1 on
every surface and row tint in both modes (3.68:1 at worst in dark, 3.42:1 in
Mineral Light), which covers both its contrast with adjacent colours and its
change from the unfocused state. `frontend/ui/geometry.browser.test.ts`
enforces that floor.

The mirrored stone ramps keep perceptual surface steps comparable; the gold
ramp supplies an accessible accent in each mode. Borders and row tints use
low-alpha stone values. `frontend/tokens/ramp.test.ts` enforces ramp membership,
elevation direction, both contrast floors, and that only a disabled control
takes Dim.

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
invisible 12px hit area.

Opening an Artifact citation is Shell-mediated. On desktop, Side remains Side,
Wide remains Wide, and Fullscreen reduces only to Wide while Sources opens. On
compact screens, the Canvas closes and Sources opens in the Inspector drawer.

Below 1200px resizing is disabled and the panel is an overlay: the primary app
remains full viewport width under the scrim, while modal focus, inert state,
Escape, and focus restoration remain native DlightRAG behavior. At phone widths
the active panel becomes full bleed. External Drawer and Dialog components remain
rejected; existing native overlays keep these geometry rules.

## Settings

Settings is one native modal, `<dialog id="settings-dialog">`, owned by `dl-settings-dialog`: a navigation
column beside the one page it opens. The dialog owns what every page shares (opening and closing, focus,
which page shows, and the title, description and notices around it); each page is an element of its own
that owns its data and reports a short summary for its navigation row.

- **Geometry.** Desktop is `min(880px, 100vw - 2 * --space-layout)` wide and
  `min(640px, 100dvh - 2 * --space-layout)` high. The height is fixed, so no page resizes the dialog.
  It takes `--radius-dialog`, a hairline `--color-border-subtle` border, `--shadow-overlay`,
  `--color-bg-surface` and the shared scrim. The navigation column is 15rem (240px) wide, wide enough
  for "Conversation Sessions" and a count on one line, and a hairline sets it off from the page.
  Docked surfaces inside the dialog (the navigation, the pages, their cards) cast no shadow: the dialog
  is the one overlapping surface, and the notice is the one thing that floats over it, as a toast.
- **Navigation.** `<nav aria-label="Settings">` holds three labelled groups: Agent (Connections, Agent
  Accounts, Profile Memory), Data (Conversation Sessions), and General (Language). Each item is a native
  `.dl-nav-item` button with an icon, its label, and a short status (`1/2`, `3`, `5`, `18`; none for
  Language or for Memory while it is off). The page that is showing has `aria-current="page"` and the
  `--color-selected-row` tint; a dialog that is closed marks none. ArrowUp, ArrowDown, Home and End move
  focus among the items through `rovingFocusKeydown`, and Enter or Space opens one. The page pane is a
  region named by its heading, and a page that is not showing is `hidden`, not unmounted.
- **Pages.** A page is a column of cards: `--radius-card`, a 1px `--color-border-subtle` border, rows
  separated by the same hairline. A switch card is its switch's label, so a tap anywhere on it turns the
  switch. A destructive action is a `.dl-btn.dl-btn-danger-text` button in its own card. A notice is the
  app's toast (`.toast`, with its shadow and its Undo) in a region the dialog owns, because the shell's
  region would sit under the scrim. It sits outside the page pane, so a Memory change shows wherever the
  reader is, including a phone's section list. Agent Accounts is a table of websites where its page is at
  least 36rem wide and three-line rows (the website, how it signs in, when it last did) where it is
  narrower; the page measures its own width, so a narrow window and a phone get the same rows.
- **Text.** Small text is `--color-text-muted` or stronger: `--color-text-subtle` reads at 3.65 to 1 on the
  dark surface and 3.82 to 1 on the light one, below the 4.5 to 1 that small text needs, so group labels,
  statuses and column headings keep their rank through size and weight instead. A browser check measures
  every word of every page against what is painted under it, in both themes.
- **Controls.** Icon buttons are `dl-icon-button`. Inside the dialog the control ladder sets
  `--control-hit-target: var(--size-button)`, so they are compact beside a pointer and 44px under 1200px.
  Switches are `dl-switch--dense`: the compact track on desktop and the regular 40 by 24 on a phone,
  with a hit area of the same ladder size either way.
- **Phone.** At `(width <= 720px), (height <= 480px)` the dialog fills the screen with square corners and
  becomes two levels: a section list (each row shows its icon, name, a status line such as "MCP · 1 of 2
  enabled", and a disclosure), then one page whose header holds Back, the page title and Close. The list
  is the first level unless a page is named, and Back returns focus to the row of the page it left. Every
  control a finger meets is at least 44px.
- **Focus.** Opening focuses the current navigation item (the first list row on a phone). Closing returns
  focus to the control that opened Settings. Escape closes, as for any native dialog.

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
