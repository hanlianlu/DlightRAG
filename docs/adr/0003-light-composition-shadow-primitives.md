# Light composition and Shadow primitives

The Web application layer composes in Light DOM so MathJax, Mermaid, and sanitized HTML stay in the document. Reusable design-system primitives that carry no domain state use open Shadow DOM, following `dl-split-layout`. Rich HTML is never typeset inside a shadow root; a primitive may only slot Light content. Features receive stores through an explicit `AppHandles` bag from the Shell.

## Status

Accepted. AppHandles and the Light-composition contract have landed. Shadow primitives beyond `dl-split-layout` (`dl-icon-button`, `dl-menu`) are decided and not yet shipped.

## Context

Lit does not require the whole product to live in Shadow DOM. DlightRAG already mixed the two: Features use `LightElement` (render root is the host) so token cascade and typesetting see real layout, while `dl-split-layout` attaches an open shadow tree for chrome that has a public attribute/event API. Leaving that split unnamed produced two failure modes: global CSS files next to CSS Modules, and the temptation to wrap Feature-shaped UI (lists, cards, source chunks) in capsules.

All-Shadow would force `::part` holes through the rich-content pipeline. All-Light leaves primitive chrome in the same document contract as Chat. `@lit/context` would hide store dependencies behind ancestors, against Feature ownership and against the existing constructor-injection seams (`ConversationStore`, `RunController`).

## Decision

- **Composition (Light DOM):** Shell, Web Features, lists, and any host of `mountRichHtml` / `typesetRichContent`. Application styles are Feature-owned CSS Modules in `@layer features`. `layout` CSS belongs to the Shell, not Inspector.
- **Encapsulation (open Shadow DOM):** design-system primitives with no domain stores. `dl-split-layout` exists. `dl-icon-button` and `dl-menu` are the only additional Shadow primitives this decision authorizes. Tokens pierce via CSS variables, not per-control `::part`. Native `<dialog>` stays in Light DOM (focus and `::backdrop`).
- **Rich HTML host:** sanitized HTML, MathJax, and Mermaid run only in Light DOM. A future card that needs rich rendering is a Feature, or a primitive that *slots* Light children. Typesetting inside `shadowRoot` requires a new ADR.
- **Handles:** `main` / `dl-app` construct stores once and pass an `AppHandles` object. Primitives never receive it. Features do not import store singletons.

## Considered Options

- All Light DOM — keeps typesetting, leaves primitive chrome unencapsulated.
- All Shadow DOM — breaks the rich-content pipeline.
- `@lit/context` for stores — implicit interface; tests must mount providers; Feature ownership becomes skippable.
