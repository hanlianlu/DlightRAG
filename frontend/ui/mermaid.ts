// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {sanitizeSvg} from '../lib/safe_html.ts';

// Mermaid fences arrive as <pre class="mermaid-source">…escaped source…</pre>
// (see src/dlightrag/web/markdown.py). Mermaid itself is imported lazily on the
// first diagram, so answers without diagrams never pay for the chunk. Each
// completed source block is upgraded to an isolated <img> diagram; any failure
// leaves the readable source in place.

const MERMAID_SELECTOR = 'pre.mermaid-source';
const MAX_CACHE = 60;
// Match the app body font (frontend/styles/global.css) so isolated diagrams,
// which cannot inherit page CSS, stay visually consistent with the answer text.
const FONT_STACK =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif';

// Identical source -> identical SVG, so the done-event re-render and history
// restore reuse the result instead of re-parsing.
const svgCache = new Map<string, string>();

async function loadMermaid() {
  const {default: mermaid} = await import('mermaid');
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict', // strips scripts/handlers and disables click binding
    theme: document.documentElement.dataset.colorMode === 'dark' ? 'dark' : 'default',
    fontFamily: FONT_STACK,
    flowchart: {htmlLabels: false}, // SVG-native <text> labels, no <foreignObject>
  });
  return mermaid;
}

let mermaidReady: ReturnType<typeof loadMermaid> | null = null;
let idSeq = 0;

function ensureMermaid(): ReturnType<typeof loadMermaid> {
  if (!mermaidReady) mermaidReady = loadMermaid();
  return mermaidReady;
}

function rememberSvg(source: string, svg: string): void {
  if (svgCache.size >= MAX_CACHE) svgCache.clear();
  svgCache.set(source, svg);
}

// An <img> loaded from a Blob URL renders the SVG in the browser's secure static
// mode (no scripts, no external fetches), so a crafted diagram cannot touch the
// surrounding answer DOM even before our own sanitization is considered.
function diagramFigure(svg: string): HTMLElement {
  const figure = document.createElement('figure');
  figure.className = 'mermaid-figure';
  const img = document.createElement('img');
  const url = URL.createObjectURL(new Blob([svg], {type: 'image/svg+xml'}));
  const release = () => URL.revokeObjectURL(url);
  img.onload = release;
  img.onerror = release;
  img.alt = 'Diagram';
  img.decoding = 'async';
  img.src = url;
  figure.appendChild(img);
  return figure;
}

async function upgradeBlock(
  mermaid: Awaited<ReturnType<typeof loadMermaid>>,
  pre: HTMLElement,
): Promise<void> {
  const source = (pre.textContent ?? '').trim();
  if (!source) return;
  let svg = svgCache.get(source);
  if (svg === undefined) {
    const {svg: raw} = await mermaid.render(`mermaid-${(idSeq += 1)}`, source);
    svg = sanitizeSvg(raw);
    rememberSvg(source, svg);
  }
  // The block-freeze rebuild may have removed this node while we awaited.
  if (pre.isConnected) pre.replaceWith(diagramFigure(svg));
}

async function upgradeAll(blocks: HTMLElement[]): Promise<void> {
  let mermaid: Awaited<ReturnType<typeof loadMermaid>>;
  try {
    mermaid = await ensureMermaid();
  } catch {
    return; // chunk failed to load -> keep source everywhere
  }
  for (const pre of blocks) {
    try {
      await upgradeBlock(mermaid, pre);
    } catch {
      // parse error / unsupported diagram: leave the readable source in place
    }
  }
}

// Presence-gated, non-blocking upgrade of completed mermaid source blocks within
// a container. With no diagram present, the Mermaid chunk is never fetched.
export function renderDiagrams(container: Element): void {
  const blocks = Array.from(container.querySelectorAll<HTMLElement>(MERMAID_SELECTOR));
  if (blocks.length === 0) return;
  void upgradeAll(blocks);
}
