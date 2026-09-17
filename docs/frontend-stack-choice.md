# Frontend stack choice: Lit design system with a Svelte app shell?

**Repository:** DlightRAG at `55828f1a`
**Assessed:** 2026-09-17
**Scope:** frontend layering and framework choice (design-system implementation, application-shell framework, whether the two should be different frameworks). Backend and run-time protocol are out of scope.
**Status:** an assessment snapshot, **not a decision record**. It binds nothing; a decision taken from it belongs in an ADR, and §8 lists what such a decision would have to change.
**Method:** primary external sources (§10) plus measurements of this repository (§4, reproducible with the commands in Appendix A).

## Executive summary

1. **The pattern is real and has a name**: a framework-agnostic design system in Web Components (Lit) consumed by an application built in a framework. It is recommended in the industry — but on **criteria** that are about reusing the design system, not about application-level developer experience: one UI library consumed by several frameworks, several applications, or vanilla JavaScript; embeddable surfaces that need strict style isolation; insurance against rebuilding the UI when the application's framework changes (§1.1).
2. **None of those criteria hold in DlightRAG today.** The design system is `private: true`, publishes no `exports`/`files`, consists of three primitives, and has exactly one consumer: this Web application (§4). Splitting the layers today pays the full boundary tax without collecting the benefit the pattern exists for.
3. **Recommendation: do not migrate the whole application; keep the Lit design system.** The reason is not that Lit is better. It is that **decision A — implementing the design system as Web Components — has already bought the insurance worth having**: it is what makes the application shell replacable at any time, incrementally, instead of in one rewrite (§5).
4. Three decisions are routinely conflated and should be kept apart (§2): **A** whether the design system is Web Components (answered: yes, for good reasons), **B** which framework the application shell uses (open, and now cheap to change), **C** whether the two layers are different frameworks (a consequence of A, not a separate decision).
5. If B is ever changed: **Svelte is a first-class consumer of Lit elements** (100% on the interop suite, §1.2). The friction sits on the **authoring and encapsulation** side — style and token scoping, context that cannot cross a custom element, SSR, property-vs-attribute timing — which is the side **you do not intend to keep** (§3). That is precisely what makes the proposed shape architecturally coherent; what remains open is only whether the whole-application rewrite is worth paying for today.

## 1. The external picture (2026-09)

### 1.1 What the pattern is actually for

| Criterion | Why it matters |
|---|---|
| One UI library across frameworks or applications | The library is a product with several consumers; its cost is amortized by reuse |
| Strict style/behaviour isolation | Micro-frontends, third-party embedding, widgets |
| Framework-change insurance | Rebuilding the UI on a framework change is unacceptable |
| Split ownership | A separate team or an external vendor owns the design system |

Conversely, for **one application and one team** the conventional advice is one framework across both layers (a framework application with that framework's own components), because Web Component boundaries carry the tax described in §3, and only multiple consumers amortize it.

### 1.2 The ecosystem state that bears on changing B

- **The consuming side has converged.** `custom-elements-everywhere.com` scores **Svelte at 100%** (Basic 16/16, Advanced 16/16): data is passed using the "property if the instance already defines it, otherwise attribute" heuristic, and every event-name style (lowercase, camelCase, kebab-case, PascalCase) is supported.
- **Wrapper packages are the historical norm, not a Svelte requirement.** Lit publishes `@lit/react` for React because React needs it; Svelte, Vue and Angular consume custom elements without one.
- **Property-versus-attribute timing is the real hazard.** Svelte only assigns a property when the element instance already defines it, so element-upgrade order relative to first render becomes a bug source.
- **The SSR limitation belongs to the authoring framework.** Svelte's own documentation states that custom elements are "not generally suitable for server-side rendering" because a shadow root is invisible until JavaScript loads; Declarative Shadow DOM has been Baseline (newly available) since February 2024 and Lit offers `@lit-labs/ssr`. This application is a SPA whose routing and conversations are owned by the server, so SSR is not a requirement here at all.

## 2. Keep the three decisions apart

| Decision | State | Relationship |
|---|---|---|
| A. Is the design system Web Components? | **Answered yes.** Three primitives expose attribute/event APIs; the light-composition contract is recorded in ADR 0003, whose reason is that MathJax, Mermaid and sanitized HTML must stay in the main document | A is what makes B and C cheap to revisit |
| B. Which framework is the application shell? | **Open.** Today Lit, the same stack as the design system | Independent of A; A is what makes it replaceable incrementally |
| C. Are the layers different frameworks? | Today there are two *layers* (design system, features), not two *frameworks* | A consequence of A, not a separate choice |

## 3. Boundary friction, with its landing place here

| Friction | Primary source | Landing place in this repository | Severity |
|---|---|---|---|
| Styles are **encapsulated**, not merely scoped: global CSS and `:global()` do not apply | Svelte docs, *Caveats and limitations* | If Svelte authored the design system, this repository's token cascade and global CSS contract (ADR 0003) would stop applying. A Svelte *application* consuming Lit primitives is unaffected | High (it decides which layer uses what) |
| Component styles are inlined into JavaScript strings instead of extracted `.css` | Svelte docs, *Caveats* | Conflicts with the repository's `styles/**/*.css` linting, CSS Modules and CSS structure tests, if Svelte owned the design system | Medium |
| Context does **not** cross a custom element (`setContext`/`getContext`) | Svelte docs, *Caveats* | The repository already injects an explicit `AppHandles` bag, which is the standard workaround; a Svelte↔Lit boundary must still pass data as properties | Medium (already mitigated) |
| Slotted content renders eagerly in the DOM; `let:` has no effect | Svelte docs, *Caveats* | Only relevant if a feature is decomposed into multi-slot composition | Low |
| A property whose name starts with `on` is treated as an event listener | Svelte docs, *Caveats* | A naming convention; a naming structure test could hold it (the existing one only asserts kebab-case filenames) | Low |
| Properties must be **listed explicitly** to be exposed as element properties | Svelte docs, *Caveats* | The boundary API must be declared; consistent with the repository's habit of locking contracts with structure tests | Low |
| Upgrade timing makes property-versus-attribute ambiguous | custom-elements-everywhere | First paint versus `customElements.define` order needs a convention (low risk with the current single Vite entry) | Low |

The negative friction is concentrated in **authoring the design system (or any component library) in Svelte**. Consuming Lit primitives from a Svelte application is essentially clean (§1.2). That is the part of the proposal that is right; the open question is only whether a whole-application rewrite is worth paying for today.

## 4. Measured baseline (2026-09-17; commands in Appendix A)

| Dimension | Measurement |
|---|---|
| Hand-written frontend | ≈ **22.7k lines** (ui 11.5k, styles 4.3k, lib 2.0k, api 1.6k, design-system 1.4k, stores 1.3k, i18n 0.6k, tokens 10 lines; tests and generated files excluded) |
| Components | **27 custom elements**; largest files `run-dialogs.ts` 966 lines, `chat-feature.ts` 855, `chat-composer.ts` 763 |
| Manual reactivity | 97 `{state: true}` fields, 35 `attribute: false` props, 21 manual `requestUpdate` calls |
| Test coupling | **187** unit/structure tests, **233** browser tests, 13 e2e files; `updateComplete` appears **348 times, 300 of them in tests**; tests construct elements directly **134 times** |
| Dependencies | lit, @lit/localize, xstate, valibot, dompurify, mermaid — already restrained |
| Bundle | own app chunk **316 kB raw**; the weight is diagram vendors (elk 1.4 MB, cytoscape 428 kB, katex 256 kB, the Mermaid family; lazily loaded). The framework itself is roughly **6–8 kB gzip** |
| Design system packaging | `package.json`: `private: true`, no `exports`/`files` — **not a distributable artifact** |

## 5. Criteria against the repository

| Criterion (§1.1) | Repository state | Holds? |
|---|---|---|
| Several frameworks or applications consume the design system | A single consumer | No |
| Embedding or isolation scenario | Main document only (light composition) | No (today) |
| Insurance against rebuilding the UI on a framework change | Three primitives plus tokens and icons | Yes, but small |
| A need for framework-level DX or optimization (animation, ecosystem, forms) | None today; the most complex flows already run in XState, which is framework-agnostic | No |

## 6. Recommendation

1. **Do not migrate the whole application now.**
2. **Keep the Lit design system.** It already provides the property of the pattern that is worth having — B being replaceable incrementally — and replacing it today would trade insurance for a rewrite.
3. **Do the cheap in-stack improvements first** (roughly a tenth of the cost of an island trial): split the largest components (`run-dialogs.ts` at 966 lines first), collapse the duplicated prop declarations, and make "control state must be reactive" explicit as a convention or check. Today it is caught by the browser test each new control receives, which works but is implicit.
4. **When a trigger in §7 appears, prove it with an island, not a rewrite.** An equivalent whole-application rewrite pays three bills at once: components, tests, and the contract documents.

## 7. Triggers that would change this conclusion

**Favouring a change or a trial**
1. The product gains an **embeddable form** — a third-party site embedding a DlightRAG answer or evidence card. Web Components stop being a frontier practice and become the requirement, and the "several consumers" criterion becomes true at the same time. This is the most realistic trigger here.
2. A **second frontend** appears (desktop shell, mobile shell, admin console).
3. The design system is **published** for outside consumption (the Shoelace route).
4. Collaborators with a different stack join, or a specific new feature is materially cheaper in Svelte (animation- or transition-heavy work such as an artifact-canvas editor or an inspector visualization).

**Favouring the status quo**
- The largest component stays under ~1.5k lines and the element count under ~45.
- Reactive-state defects keep being caught before merge by the existing tests and review (a recent example: a non-reactive control field was caught by the browser test written with that control).
- Diagram vendors keep dominating the bundle, where a framework change would save noise.

## 8. If B is ever changed, this must change with it

- **ADRs.** ADR 0002 (browser wire validation: valibot schemas need an equivalent on the Svelte side), ADR 0003 (the light-composition and shadow-primitive contract still holds, but *who may host a Feature* must be restated).
- **Architecture documentation.** The Web frontend ownership section of `docs/architecture.md` and the caption of `architecture-frontend.svg`, whose phrase **"collective light-DOM Lit Feature owners" must be rewritten**.
- **Design documentation.** `docs/web-theme-design.md`: its icon-geometry rule and its requirement to "update trigger icons, menu selection, and ARIA state" still apply, but its Lit wording (`Vite/Lit application shell`, "Lit application rerenders …") must change.
- **Tooling.** `svelte-check` alongside or instead of `tsc`; the `web-test-runner` browser-test infrastructure can be reused, but the Lit coupling in those 233 tests must be rewritten; the i18n catalog moves off lit-localize; the structure tests (naming, architecture, i18n catalog, interaction emphasis) need Svelte equivalents.
- **Transition cost.** Two mental models and two test configurations coexisting for the duration, which belongs in the decision's cost.

## 9. Known limits of this assessment

- The external coverage is limited (three primary documents, one industry comparison, two index sites). No cross-framework migration experiment was run; §1's ecosystem and DX claims rest on vendor documentation, with industry commentary only as background.
- The §4 size figures come from one local build, not from multi-browser or multi-network performance measurements.
- Alternatives to Svelte (Vue, Solid, React) were not assessed; that belongs to the layer *after* decision B and is deliberately out of scope.

## 10. Primary sources (accessed 2026-09-17)

- Svelte docs — *Custom elements*, including *Caveats and limitations* (style encapsulation, SSR, context not crossing elements, `on`-prefixed property names, properties listed explicitly): https://svelte.dev/docs/svelte/custom-elements
- Custom Elements Everywhere (Svelte at 100%, including the property/attribute heuristic and supported event-name styles): https://custom-elements-everywhere.com/
- lit.dev — *React* integration (the `@lit/react` wrapper and why it exists): https://lit.dev/docs/frameworks/react/
- Smashing Magazine (2025-03) — *Web Components vs. Framework Components* (reuse, isolation, ecosystem and DX trade-offs): https://www.smashingmagazine.com/2025/03/web-components-vs-framework-components/
- MDN — *Using shadow DOM* (encapsulation and CSS-custom-property penetration): https://developer.mozilla.org/en-US/docs/Web/API/Web_components/Using_shadow_DOM
- webstatus.dev — *Declarative shadow DOM* (Baseline, newly available 2024-02-20): https://webstatus.dev/features/declarative-shadow-dom
- web.dev — *Declarative Shadow DOM* (the mechanism and its limits): https://web.dev/articles/declarative-shadow-dom

## Appendix A: commands used

```bash
cd frontend
# size, excluding tests and generated output
for d in ui lib api stores tokens design-system styles i18n; do
  find $d -name '*.ts' -o -name '*.css' | grep -v test | grep -v generated | xargs wc -l | tail -1
done
# components and largest files
grep -roE 'customElements\.define\(' ui/ | wc -l ; wc -l ui/*.ts | sort -rn | head
# manual reactivity and test coupling
grep -roE '[a-zA-Z]+: \{state: true\}' ui/ | wc -l
grep -c updateComplete ui/*.ts ui/*.test.ts | awk -F: '{s+=$2} END {print s}'
# dependencies and packaging state
python3 -c "import json;d=json.load(open('package.json'));print(d['private'], d.get('exports'), sorted(d['dependencies']))"
# bundle size, from the deployed image
docker exec dlightrag-dlightrag-api-1 sh -c 'cd /app/.venv/lib/python3.14/site-packages/dlightrag/adapters/http/browser/static/app/assets && du -k *.js *.css | sort -rn | head'
```
