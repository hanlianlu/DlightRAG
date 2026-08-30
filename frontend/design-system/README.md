# DlightRAG Design System

`frontend/design-system/` is the package-ready internal module for shared visual foundations, semantic icons, native-first primitives, and behavior-heavy UI elements. Product Features remain in `frontend/ui/` and may consume only the public entries `design-system/index.ts` and `design-system/index.css`.

## Boundary

The module must not import API clients, stores, the router, domain/event buses, XState, Mermaid, DOMPurify, or product message IDs. `design-system/testing/architecture.test.ts` enforces that boundary, forbids deep feature imports, and rejects legacy `.ui-*`, Web Awesome, raw feature SVG, and text-glyph icons. Its explicit raw-graphics allowlist is limited to vendor-generated Mermaid/MathJax output; URL-backed content images remain `<img>` content, never control icons.

CSS uses the fixed layer order:

```css
@layer reset, foundations, primitives, components, features, utilities;
```

Application and catalog entries both load `design-system/index.css`. Product global styles and CSS Modules enter only the `components` and `features` layers.

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

`dl-split-layout` owns axis layout, pointer/keyboard resizing, and separator ARIA. Its pixel interface is `size`, `min`, `max`, `primary=start|end`, and `orientation=horizontal|vertical`. It emits `dl-split-input` while resizing and `dl-split-change` when committed, both with `{position}` in normalized pixels. Product adapters own breakpoints, open/close meaning, and persistence.

Element modules have no registration side effects. Entrypoints explicitly call the idempotent `defineDesignSystemElements()`.

## Catalog and verification

- `design-system.html`: isolated foundations/primitives catalog; no product components.
- `product-showcase.html`: feature composition showcase.
- `npm test`: boundaries, generated drift, and structural rules.
- `npm run test:browser`: default Chromium behavior suite.
- `npm run test:browser:cross-engine`: Chromium, Firefox, and WebKit contract suite.

Catalog coverage includes dark/light modes, 390/1440px specimens, focus/disabled states, and forced-colors rules.
