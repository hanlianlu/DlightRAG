// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {LOCALE_STATUS_EVENT, msg} from '@lit/localize';
import {sanitizeSvg} from '../lib/safe-html.ts';
import type {ColorMode} from '../lib/theme.ts';

// Mermaid fences arrive as <pre class="mermaid-source">…escaped source…</pre>
// (see src/dlightrag/adapters/http/browser/markdown.py). Mermaid itself is imported lazily on the
// first diagram, so answers without diagrams never pay for the chunk. Each
// completed source block is upgraded to an isolated <img> diagram; any failure
// leaves the readable source in place.
//
// A diagram is drawn in the page's color mode (`data-color-mode` on <html>, see
// docs/web-theme-design.md). A mode change redraws the diagrams already on the page from their
// source, so they follow the theme; a language change describes them again, in the new language.

const MERMAID_SELECTOR = 'pre.mermaid-source';
const FIGURE_SELECTOR = 'figure.mermaid-figure';
const MAX_CACHE = 60;

type MermaidModule = typeof import('mermaid').default;

/** The part of Mermaid this module uses. Injected so a test can hold a load that fails. */
export type MermaidApi = {
  initialize: MermaidModule['initialize'];
  render(id: string, source: string): Promise<{svg: string; diagramType?: string}>;
};

export type MermaidLoader = () => Promise<MermaidApi>;

const importMermaid: MermaidLoader = async () => (await import('mermaid')).default;

function colorMode(): ColorMode {
  return document.documentElement.dataset.colorMode === 'dark' ? 'dark' : 'light';
}

// Isolated diagrams cannot inherit page CSS, so the font is read from the
// --font-body token (design-system/foundations/type.css) — the same source the
// body uses — keeping diagrams visually consistent with the answer text.
function bodyFontStack(): string {
  const token = getComputedStyle(document.documentElement).getPropertyValue('--font-body');
  return token.trim() || 'sans-serif';
}

function mermaidConfig(mode: ColorMode): Parameters<MermaidApi['initialize']>[0] {
  return {
    startOnLoad: false,
    securityLevel: 'strict', // strips scripts/handlers and disables click binding
    theme: mode === 'dark' ? 'dark' : 'default',
    fontFamily: bodyFontStack(),
    flowchart: {htmlLabels: false}, // SVG-native <text> labels, no <foreignObject>
  };
}

/** What a reader who cannot see the diagram is told: its own title, else what kind of diagram it is. */
function diagramLabel(source: string, diagramType: string): string {
  const title = /^\s*accTitle\s*:\s*(.+?)\s*$/m.exec(source)?.[1];
  if (title) return title;
  switch (diagramType) {
    case 'flowchart':
    case 'flowchart-v2':
    case 'flowchart-elk':
      return msg('Flowchart', {id: 'mermaid.flowchart'});
    case 'sequence':
      return msg('Sequence diagram', {id: 'mermaid.sequence'});
    case 'classDiagram':
      return msg('Class diagram', {id: 'mermaid.class'});
    case 'stateDiagram':
      return msg('State diagram', {id: 'mermaid.state'});
    case 'er':
      return msg('Entity relationship diagram', {id: 'mermaid.er'});
    case 'gantt':
      return msg('Gantt chart', {id: 'mermaid.gantt'});
    case 'pie':
      return msg('Pie chart', {id: 'mermaid.pie'});
    case 'mindmap':
      return msg('Mind map', {id: 'mermaid.mindmap'});
    case 'timeline':
      return msg('Timeline', {id: 'mermaid.timeline'});
    case 'journey':
      return msg('User journey', {id: 'mermaid.journey'});
    case 'gitGraph':
      return msg('Git graph', {id: 'mermaid.gitGraph'});
    default:
      return msg('Diagram', {id: 'mermaid.diagram'});
  }
}

interface Drawing {
  svg: string;
  diagramType: string;
  /** The color mode the SVG was drawn in. */
  mode: ColorMode;
}

/** What a mounted figure was drawn from, so it can be drawn or described again. */
interface Mounted {
  source: string;
  diagramType: string;
  mode: ColorMode;
}

let idSeq = 0;

// An <img> loaded from a Blob URL renders the SVG in the browser's secure static
// mode (no scripts, no external fetches), so a crafted diagram cannot touch the
// surrounding answer DOM even before our own sanitization is considered.
function showDiagram(img: HTMLImageElement, source: string, drawing: Drawing): void {
  const url = URL.createObjectURL(new Blob([drawing.svg], {type: 'image/svg+xml'}));
  const release = () => URL.revokeObjectURL(url);
  img.onload = release;
  img.onerror = release;
  img.alt = diagramLabel(source, drawing.diagramType);
  img.src = url;
}

function diagramFigure(source: string, drawing: Drawing): HTMLElement {
  const figure = document.createElement('figure');
  figure.className = 'mermaid-figure';
  const img = document.createElement('img');
  img.decoding = 'async';
  showDiagram(img, source, drawing);
  figure.appendChild(img);
  return figure;
}

/** Upgrades mermaid source blocks to diagrams and keeps them in the page's color mode. */
export class DiagramRenderer {
  readonly #load: MermaidLoader;
  #mermaid: Promise<MermaidApi> | null = null;
  #initializedFor: ColorMode | null = null;
  #queue: Promise<unknown> = Promise.resolve();
  #observer: MutationObserver | null = null;
  // Identical source in the same mode -> identical SVG, so the done-event re-render and history
  // restore reuse the result instead of re-parsing.
  readonly #cache = new Map<string, Omit<Drawing, 'mode'>>();
  readonly #figures = new WeakMap<HTMLElement, Mounted>();

  constructor(load: MermaidLoader = importMermaid) {
    this.#load = load;
  }

  /**
   * Presence-gated, non-blocking upgrade of completed mermaid source blocks within a container.
   * With no diagram present, the Mermaid chunk is never fetched.
   */
  render(container: Element): void {
    const blocks = Array.from(container.querySelectorAll<HTMLElement>(MERMAID_SELECTOR));
    if (blocks.length === 0) return;
    void this.#upgradeAll(blocks);
  }

  // A failed load is not remembered: the next diagram tries again.
  #ensureMermaid(): Promise<MermaidApi> {
    this.#mermaid ??= this.#load().catch((error: unknown) => {
      this.#mermaid = null;
      throw error;
    });
    return this.#mermaid;
  }

  // Mermaid keeps one global config, so the mode is chosen, applied and drawn in one step that no
  // other draw interleaves with.
  #draw(source: string): Promise<Drawing> {
    const task = async (): Promise<Drawing> => {
      const mermaid = await this.#ensureMermaid();
      const mode = colorMode();
      const key = `${mode}\n${source}`;
      let drawn = this.#cache.get(key);
      if (drawn === undefined) {
        if (this.#initializedFor !== mode) {
          mermaid.initialize(mermaidConfig(mode));
          this.#initializedFor = mode;
        }
        idSeq += 1;
        const result = await mermaid.render(`mermaid-${idSeq}`, source);
        drawn = {svg: sanitizeSvg(result.svg), diagramType: result.diagramType ?? ''};
        if (this.#cache.size >= MAX_CACHE) this.#cache.clear();
        this.#cache.set(key, drawn);
      }
      return {...drawn, mode};
    };
    const drawing = this.#queue.then(task);
    this.#queue = drawing.catch(() => undefined);
    return drawing;
  }

  async #upgradeAll(blocks: HTMLElement[]): Promise<void> {
    try {
      await this.#ensureMermaid();
    } catch {
      return; // chunk failed to load -> keep source everywhere
    }
    for (const pre of blocks) {
      try {
        await this.#upgradeBlock(pre);
      } catch {
        // parse error / unsupported diagram: leave the readable source in place
      }
    }
  }

  async #upgradeBlock(pre: HTMLElement): Promise<void> {
    const source = (pre.textContent ?? '').trim();
    if (!source) return;
    const drawing = await this.#draw(source);
    // The block-freeze rebuild may have removed this node while we awaited.
    if (!pre.isConnected) return;
    const figure = diagramFigure(source, drawing);
    this.#figures.set(figure, {source, diagramType: drawing.diagramType, mode: drawing.mode});
    pre.replaceWith(figure);
    this.#followPage();
    // The mode may have changed after this was drawn and before it was mounted.
    if (drawing.mode !== colorMode()) this.#redrawStale();
  }

  /** From the first figure on, the diagrams follow the page: its color mode, and its language, which is
   *  what tells a reader who cannot see a diagram what it is. */
  #followPage(): void {
    if (this.#observer) return;
    this.#observer = new MutationObserver(() => { this.#redrawStale(); });
    this.#observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-color-mode'],
    });
    window.addEventListener(LOCALE_STATUS_EVENT, (event) => {
      if (event.detail.status === 'ready') this.#describeAgain();
    });
  }

  #describeAgain(): void {
    for (const figure of document.querySelectorAll<HTMLElement>(FIGURE_SELECTOR)) {
      const drawn = this.#figures.get(figure);
      const img = figure.querySelector('img');
      if (drawn && img) img.alt = diagramLabel(drawn.source, drawn.diagramType);
    }
  }

  #redrawStale(): void {
    const mode = colorMode();
    const stale = Array.from(document.querySelectorAll<HTMLElement>(FIGURE_SELECTOR))
      .filter((figure) => {
        const drawn = this.#figures.get(figure);
        return drawn !== undefined && drawn.mode !== mode;
      });
    if (stale.length > 0) void this.#redraw(stale);
  }

  async #redraw(figures: HTMLElement[]): Promise<void> {
    for (const figure of figures) {
      const drawn = this.#figures.get(figure);
      const img = figure.querySelector('img');
      if (!drawn || !img) continue;
      try {
        const drawing = await this.#draw(drawn.source);
        if (!figure.isConnected) continue;
        showDiagram(img, drawn.source, drawing);
        this.#figures.set(figure, {source: drawn.source, diagramType: drawing.diagramType, mode: drawing.mode});
      } catch {
        // keep the diagram it has: readable, if in the other theme's colors
      }
    }
  }
}

const diagrams = new DiagramRenderer();

export function renderDiagrams(container: Element): void {
  diagrams.render(container);
}

/** Whether `container` still holds a diagram as its source: not drawn yet, or left so by a load that failed. */
export function holdsDiagramSource(container: Element): boolean {
  return container.querySelector(MERMAID_SELECTOR) !== null;
}
