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

Runtime CSS is authoritative. Foundation sources are separated by concern in `foundations/`: scale, color, type, geometry, motion, and roles. Run:

```bash
npm run generate:tokens
npm run check:tokens
```

The deterministic `foundations/tokens.generated.json` is a DTCG-style projection for documentation and design tooling; it is not a runtime input.

## Icons

Callers render only semantic names:

```ts
icon('add', {size: 'sm'})
```

Sizes are fixed to `xs/sm/md/lg = 12/16/20/24px`; stroke defaults to `1.75`. Every icon is decorative (`aria-hidden="true"`, `focusable="false"`), so the parent control owns its accessible name. Source geometry and optical metadata live in the generated registry without path rewriting.

`icons/selection.json` is the governed selection. Lucide is primary; any Iconoir, Phosphor, or DlightRAG custom entry must record its source and receive an optical/license review. Regenerate and drift-check with:

```bash
npm run generate:icons
npm run check:icons
```

The pinned `lucide-static` package is generation-only. Production bundles contain only the checked-in selected geometry.

## Primitives and elements

Prefer native `button`, `input`, `radio`, `checkbox`, and `dialog` with `.dl-*` classes. Add a custom `dl-*` element only when behavior and accessibility state justify it. Classes have no compatibility aliases.

`.dl-switch` is a native `button` with `role="switch"` and `aria-checked`. Its hit area is `--control-hit-target` (the control ladder's size) however small the track is drawn: an invisible extension of the button itself, so a tap beside the track still turns it. `dl-switch--dense` composes with it for rows beside a pointer: the compact track there, and the regular one on the phone layout (`(width <= 720px), (height <= 480px)`), where a finger is the pointer.

`.dl-nav-item` is the navigation row: a native `button` inside a `nav` with an icon (`.dl-nav-item-icon`), a label (`.dl-nav-item-label`), and a short trailing status (`.dl-nav-item-status`), where `aria-current="page"` marks the page that is showing. Its `.dl-nav-item-detail` is the full status line under the label: it is the row's description in every form, and only the list form shows it. `dl-nav-item--list` is the form for a screen that is the navigation (a phone's section list): the row is square and sits on a hairline in a bordered list that owns the border and the clip, it shows the detail line and a `.dl-nav-item-disclosure` instead of the short status. `.dl-dialog-checkbox--row` is the choice row for a radio or checkbox inside a bordered card, whose own spacing and clip it relies on.

`dl-menu` owns the one menu keyboard contract: ArrowDown and ArrowUp move with wrapping, Home and End jump to the ends, a typed letter moves to the next item whose label starts with it, and Enter and Space stay with each item. Items stay out of the tab order (`tabindex="-1"`), and an `aria-disabled` item still takes focus but never activates. Each step scrolls the item wholly into view with any `scroll-margin` it sets, rather than leaving that to the engine's focus scrolling, which differs between engines. The menu asks its owner to close it with `dl-menu-dismiss`: `detail.restoreFocus` is true for Escape, so the owner returns focus to the menu button, and false for Tab or focus moving to an element other than the menu or the button that controls it (`aria-controls`), so focus stays where it went. Focus that lands on no element (WebKit does not focus a pressed button, and a window can lose focus) raises nothing: the owner's outside-click dismissal closes the menu then. Its `focusItem('first' | 'last')` pairs with `menuButtonFocus(event)`, which maps a menu button's ArrowDown, Enter, and Space to the first item and ArrowUp to the last. Pickers that are not menus reuse the roving step through `rovingFocusKeydown(event, items)`.

Every popover and menu that opens from a trigger takes its placement from `.dl-anchored`: below and start-aligned inside the positioned trigger wrapper, `.dl-anchored--end` for end alignment, `.dl-anchored--above` to open upward, and `--anchored-gap` for the distance (negative to overlap the wrapper). Start and end follow the writing direction. The surface keeps its own look, layer, and `[hidden]` rule, and may inset itself from the edge it aligns with.

`dl-split-layout` owns axis layout, pointer/keyboard resizing, and separator ARIA. Its pixel interface is `size`, `min`, `max`, `primary=start|end`, and `orientation=horizontal|vertical`; its owner names the separator with `label`. It emits `dl-split-input` while resizing and `dl-split-change` when committed, both with `{position}` in normalized pixels. Product adapters own breakpoints, open/close meaning, and persistence. When a product overlay is trapped by the split's isolated panes, the owning adapter may raise that pane with `--split-start-layer` or `--split-end-layer`; the default for both is `0`.

Element modules have no registration side effects. Entrypoints explicitly call the idempotent `defineDesignSystemElements()`.

## Catalog and verification

- `design-system.html`: isolated foundations/primitives catalog; no product components.
- `product-showcase.html`: feature composition showcase.
- `npm test`: boundaries, generated drift, and structural rules.
- `npm run test:browser`: default Chromium behavior suite. Its `product-styles` group runs `ui/a11y.browser.test.ts` against the stylesheets and class names one production build ships to the application page, in dark and light.
- `npm run test:browser:cross-engine`: Chromium, Firefox, and WebKit contract suite.

Catalog coverage includes dark/light modes, 390/1440px specimens, focus/disabled states, and forced-colors rules.
