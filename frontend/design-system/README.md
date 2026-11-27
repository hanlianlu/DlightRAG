# DlightRAG Design System

`frontend/design-system/` is the package-ready internal module for shared visual foundations, semantic icons, native-first primitives, and behavior-heavy UI elements. Product Features remain in `frontend/ui/` and may consume only the public entries `design-system/index.ts` and `design-system/index.css`.

## Boundary

The module must not import API clients, stores, the router, domain/event buses, XState, Mermaid, DOMPurify, or product message IDs. `design-system/testing/architecture.test.ts` enforces that boundary, forbids deep feature imports, and rejects legacy `.ui-*`, Web Awesome, raw feature SVG, and text-glyph icons. Its explicit raw-graphics allowlist is limited to vendor-generated Mermaid/MathJax output; URL-backed content images remain `<img>` content, never control icons.

CSS uses the fixed layer order:

```css
@layer reset, foundations, primitives, components, features, utilities;
```

Application and catalog entries both load `design-system/index.css`. Product global styles and CSS Modules enter only the `components` and `features` layers.

The reset layer also sets body text: Body text on the page background, in `--font-body` at body size, antialiased. A page adds only its own chrome, such as the application's clipped viewport or the catalog's 390px minimum width. Below its 1200px layout breakpoint the product raises `--size-button` to the hit target; that rule stays with the product, so the catalog shows controls at the design system's own sizes.

## Foundations and tokens

Runtime CSS is authoritative. Foundation sources are separated by concern in `foundations/`: scale, color, type, geometry, motion, and roles.

## Icons

Callers render only semantic names:

```ts
icon('add', {size: 'sm'})
```

Sizes are fixed to `xs/sm/md/lg = 12/16/20/24px`; the stroke weight is `1.5` in the glyph's own 24-unit viewBox — a quarter lighter than Lucide's `2` — and the 12px step keeps a `2`-unit floor, so the four sizes carry 1.00/1.00/1.25/1.50px of ink. One 24-unit master geometry serves the whole ladder: the weight is a function of the size and no step ships a second optical drawing, because one grid and one weight rule are what keep every mark in a single design language. 16/20px therefore render at a fractional scale (2/3, 5/6) and take the engine's antialiasing; that is the accepted cost of scaling one master grid, not a defect to patch per size. Every icon is decorative (`aria-hidden="true"`, `focusable="false"`), so the parent control owns its accessible name. Source geometry and optical metadata live in the generated registry without path rewriting.

`icons/selection.json` is the governed selection. Lucide is primary; any Iconoir, Phosphor, or DlightRAG custom entry must record its source and receive an optical/license review. Regenerate and drift-check with:

```bash
npm run generate:icons
npm run check:icons
```

The pinned `lucide-static` package is generation-only. Production bundles contain only the checked-in selected geometry.

## Primitives and elements

Prefer native `button`, `input`, `radio`, `checkbox`, and `dialog` with `.dl-*` classes. Add a custom `dl-*` element only when behavior and accessibility state justify it. Classes have no compatibility aliases.

`.dl-input` is the field the reader types into, and `.dl-dialog-input` is that same field with the spacing a dialog row needs. `.dl-note` is one quiet line that says what is happening or what is missing, on a `p` or any other element, and it keeps its face inside a `.confirm-dialog`, whose own paragraph face is for the dialog's sentences. `.dl-mark` marks the words a search found.

`.dl-text-action` is the quiet control whose face is its own words: caption type, the accent text role, one line, the WCAG 2.2 target-size floor, the focus ring, and a hover that underlines where the accent cannot get darker. A feature places it and says what it says; `dl-text-action--disclosure` turns the trailing `.dl-icon` with the `aria-expanded` its owner states.

`.dl-icon-button--dense` composes with `.dl-icon-button` for a control drawn inside a line of words: the button is drawn at the size `--dl-icon-button-size` states — the mark's own size, so the ink stands where the box does and the line's measures hold — rather than at the control ladder's square. Its stroke needs no statement of its own: the weight is a measure of the glyph's viewBox, and the 12px floor keeps a mark in a line of words at one CSS pixel of ink. The reader still presses `--control-hit-target`, which reaches out invisibly around the box through the same positioned pseudo-element `dl-switch` uses. Take it where the control ladder's own square would tower over the line; a control among neighbours of its own size takes the plain class.

`.dl-action-row` is the row of inline affordances and short statuses parted by one hairline — the row under an answer, and any row that parts its things the same way. It lays its children out with `--space-tight` between them, draws the line before every child but the first, and stands each side of that line off by that same measure, so every divider in the row is one distance from what it parts and as tall as the part it opens. A part that is no box of its own — one that composes `display: contents`, so its own children join the row's line, as the reader's thumbs do — declares `data-action-part` on the part it hoists, and the row draws that part's line, sets it off, and takes its height from the row's line. A part is its own words or marks and nothing else: the row resets a native control's own padding, because a feature stylesheet that zeroed it instead would take the clearance from the line with it. No feature draws a divider for such a row.

`.dl-chips` is the wrapping row of `.dl-chip`, a native `label` around a radio (one choice among the row) or a checkbox (any of them) for a handful of short options laid out. The input stays in the tab order and the accessibility tree but is not drawn: the chip is the pill, its chosen look comes from `:has(input:checked)`, and the focus ring from `:has(input:focus-visible)`. The owner names the group, each value, and what a change does.

`.dl-switch` is a native `button` with `role="switch"` and `aria-checked`. Its hit area is `--control-hit-target` (the control ladder's size) however small the track is drawn: an invisible extension of the button itself, so a tap beside the track still turns it. `dl-switch--dense` composes with it for rows beside a pointer: the compact track there, and the regular one on the phone layout (`(width <= 720px), (height <= 480px)`), where a finger is the pointer.

`.dl-nav-item` is the navigation row: a native `button` inside a `nav` with an icon (`.dl-nav-item-icon`), a label (`.dl-nav-item-label`), and a short trailing status (`.dl-nav-item-status`), where `aria-current="page"` marks the page that is showing and a disabled row dims, with no hover. Its `.dl-nav-item-detail` is the full status line under the label: it is the row's description in every form, and only the list form shows it. `dl-nav-item--list` is the form for a bordered list of entries that each lead on to something (a phone's section list, the Schedules dialog's list): the row is square and sits on a hairline in a `.dl-list-card` that owns the border and the clip, it shows the detail line and a `.dl-nav-item-disclosure` instead of the short status. `.dl-nav-group-label` is the quiet label over a run of entries in a `nav` column (the sidebar's Recents): it takes a nav item's inline padding, so in the column it starts where a row's icon starts. `.dl-dialog-checkbox--row` is the choice row for a radio or checkbox inside a bordered card, whose own spacing and clip it relies on.

`.dl-list-card` is the bordered list: a `ul` of `li`s, or a run of rows in a `nav` or `div`, with the outer border and rounded corners, the entries clipped to them, and one hairline before every entry but the first. It clips with `overflow: clip`, not `hidden`, so a list in a dialog's fixed-height grid is never cut short.

`.dl-dialog-primary` is the dialog's one affirmative button, written inside `.dl-dialog-actions` beside the plain ones (and beside `.dl-dialog-danger` where it is destructive): the action fill on their geometry, dimmed and without hover while disabled, with the focus ring on its outside.

`@keyframes dl-pulse` is the one pulse a work-in-progress mark shares. A mark states its own depth with `--pulse-start-opacity`, `--pulse-mid-opacity`, and `--pulse-mid-scale` (unstated, it is the status dot's), and a CSS Module names it `global(dl-pulse)`, because a module renames every animation name it uses. Durations are `--duration-*` tokens, and a loop's period is the one kind that reduced motion does not zero, so each loop states what it does instead of moving.

Under a touch pointer the utilities layer holds every text-entry control at `--font-size-touch-entry` (16px), the smallest size iOS Safari leaves a focused field at. No feature repeats that rule.

`dl-menu` owns the one menu keyboard contract: ArrowDown and ArrowUp move with wrapping, Home and End jump to the ends, a typed letter moves to the next item whose label starts with it, and Enter and Space stay with each item. Items stay out of the tab order (`tabindex="-1"`), and an `aria-disabled` item still takes focus but never activates. Each step scrolls the item wholly into view with any `scroll-margin` it sets, rather than leaving that to the engine's focus scrolling, which differs between engines. The menu asks its owner to close it with `dl-menu-dismiss`: `detail.restoreFocus` is true for Escape, so the owner returns focus to the menu button, and false for Tab or focus moving to an element other than the menu or the button that controls it (`aria-controls`), so focus stays where it went. Focus that lands on no element (WebKit does not focus a pressed button, and a window can lose focus) raises nothing: the owner's outside-click dismissal closes the menu then. Its `focusItem('first' | 'last')` pairs with `menuButtonFocus(event)`, which maps a menu button's ArrowDown, Enter, and Space to the first item and ArrowUp to the last. Pickers that are not menus reuse the roving step through `rovingFocusKeydown(event, items)`, which returns at once for a key of an IME composition, so the arrow that picks a candidate in a field inside a picker does not carry focus out of it.

Every popover and menu that opens from a trigger takes its placement from `.dl-anchored`: below and start-aligned inside the positioned trigger wrapper, `.dl-anchored--end` for end alignment, `.dl-anchored--above` to open upward, and `--anchored-gap` for the distance (negative to overlap the wrapper). Start and end follow the writing direction. The surface keeps its own look, layer, and `[hidden]` rule, and may inset itself from the edge it aligns with.

`dl-split-layout` owns axis layout, pointer/keyboard resizing, and separator ARIA. Its pixel interface is `size`, `min`, `max`, `primary=start|end`, and `orientation=horizontal|vertical`; its owner names the separator with `label`. It emits `dl-split-input` while resizing and `dl-split-change` when committed, both with `{position}` in normalized pixels. `Enter` or `Space` on the separator collapses the pane to `0` (pressed again, it restores the width the pane had), as does a drag past `min` by `collapse-threshold` (half of `min` by default), and a double-click raises `dl-split-reset`, which asks the owner for the pane's default width. During a pointer drag, Escape is taken in the document's capture phase to cancel it: `dl-split-input` puts the start width back and no `dl-split-change` follows, since nothing was committed. Product adapters own breakpoints, open/close meaning, and persistence. When a product overlay is trapped by the split's isolated panes, the owning adapter may raise that pane with `--split-start-layer` or `--split-end-layer`; the default for both is `0`.

`isImeKey(event)` says that a key press belongs to an IME composition and not to the page: the Enter or Escape that confirms or cancels a candidate is the composition's. Chromium and Firefox mark that keydown `isComposing`; WebKit delivers it after `compositionend` with `keyCode` 229, so the helper checks both. A handler for Enter, Escape, or a menu key in a field the reader types into returns before it acts when this is true, and no handler carries a test of its own for it.

Element modules have no registration side effects. Entrypoints explicitly call the idempotent `defineDesignSystemElements()`.

## Catalog and verification

- `design-system.html`: isolated foundations/primitives catalog; no product components.
- `npm test`: boundaries, generated drift, and structural rules.
- `npm run test:browser`: default Chromium behavior suite, which loads no product CSS. Its `product-styles` group runs `ui/a11y.browser.test.ts`, `ui/geometry.browser.test.ts`, and every `ui/*.layout.browser.test.ts` against the stylesheets and class names one production build ships to the application page: a test that reads what the stylesheets do belongs there, and axe judges WCAG 2.0, 2.1 and 2.2 level A and AA in dark and light. Only `design-system/testing/design-system.browser.test.ts` links a stylesheet by hand, the design system's own, because it tests the primitives without the product.
- `npm run test:browser:cross-engine`: Chromium, Firefox, and WebKit contract suite.

Catalog coverage includes dark/light modes, 390/1440px specimens, focus/disabled states, and forced-colors rules.
