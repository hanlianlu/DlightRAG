// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** axe-core for the browser tests: loaded once per page, and run to the accessibility bar the product holds. */

interface AxeNode {
  target: string[];
  any: {data?: {fgColor?: string; bgColor?: string; contrastRatio?: number} | null}[];
}

interface Axe {
  run: (
    root: HTMLElement,
    options: {
      preload?: boolean;
      runOnly: {type: string; values: string[]};
      checks?: Record<string, {options: Record<string, unknown>}>;
    },
  ) => Promise<{violations: {id: string; impact?: string | null; nodes: AxeNode[]}[]}>;
}

/** The area a reader can press: the box a control is drawn in, plus however far an invisible hit-area
 *  extension reaches past it. The design system draws some controls smaller than the target they are
 *  pressed at (`dl-switch`, the thumbs in the action row), and the extension is a positioned pseudo-element of
 *  the control itself, so it is measured from there. Only a pseudo-element that is generated, taken out of the
 *  flow, and open to the pointer is one: a decorative `::before` is not a place to press. */
export function pressArea(button: HTMLElement): {
  left: number; top: number; right: number; bottom: number; width: number; height: number;
} {
  const rect = button.getBoundingClientRect();
  const extension = getComputedStyle(button, '::before');
  const pressable = extension.content !== 'none' && extension.content !== 'normal'
    && extension.position === 'absolute' && extension.pointerEvents !== 'none';
  const reaches = (side: string): number => {
    const value = Number.parseFloat(extension.getPropertyValue(side));
    return pressable && Number.isFinite(value) ? Math.max(0, -value) : 0;
  };
  const left = rect.left - reaches('left');
  const top = rect.top - reaches('top');
  const right = rect.right + reaches('right');
  const bottom = rect.bottom + reaches('bottom');
  return {left, top, right, bottom, width: right - left, height: bottom - top};
}

/** WCAG 2.2's smallest target, in CSS pixels. */
const MINIMUM_TARGET = 24;

/** axe measures a target by the box of its element, and cannot see an extension drawn in a pseudo-element, so
 *  a control pressed at 24 pixels each way through one is the same control to a reader and passes. */
function reachesMinimumTarget(node: AxeNode): boolean {
  const control = node.target.length === 1 ? document.querySelector<HTMLElement>(node.target[0]!) : null;
  if (!control) return false;
  const area = pressArea(control);
  return area.width >= MINIMUM_TARGET - 0.5 && area.height >= MINIMUM_TARGET - 0.5;
}

/** WCAG 2.0, 2.1 and 2.2 at levels A and AA: the 2.1 and 2.2 tags add rules, they do not include 2.0's. */
const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

async function loadAxe(): Promise<Axe> {
  const loaded = (): Axe | undefined => (window as unknown as {axe?: Axe}).axe;
  if (!loaded()?.run) {
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = new URL('../node_modules/axe-core/axe.min.js', import.meta.url).href;
      script.addEventListener('load', () => { resolve(); }, {once: true});
      script.addEventListener('error', () => { reject(new Error('axe-core failed to load')); }, {once: true});
      document.head.append(script);
    });
  }
  const axe = loaded();
  if (!axe?.run) throw new Error('axe-core run() is unavailable');
  return axe;
}

/** Finish running transitions, so axe reads each color at its settled value. */
function settleTransitions(): void {
  for (const animation of document.getAnimations()) {
    if (animation instanceof CSSTransition) animation.finish();
  }
}

function describeViolation(colorMode: string, rule: string, node: AxeNode): string {
  const contrast = node.any.find((check) => check.data?.contrastRatio)?.data;
  const measured = contrast ? ` (${contrast.fgColor} on ${contrast.bgColor}, ${contrast.contrastRatio}:1)` : '';
  return `${colorMode}: ${rule} ${node.target.join(' ')}${measured}`;
}

/** Serious and critical WCAG A and AA violations under `root`, judged in each color mode (both by default).
 *  `preload: false` keeps axe from fetching the page's stylesheets again, which a page that imports its
 *  sheets from other sheets cannot answer. */
export async function seriousViolations(
  root: HTMLElement,
  {colorModes = ['dark', 'light'], preload}: {colorModes?: readonly string[]; preload?: boolean} = {},
): Promise<string[]> {
  const axe = await loadAxe();
  const found: string[] = [];
  const original = document.documentElement.dataset.colorMode;
  try {
    for (const colorMode of colorModes) {
      document.documentElement.dataset.colorMode = colorMode;
      settleTransitions();
      const results = await axe.run(root, {
        preload,
        runOnly: {type: 'tag', values: WCAG_TAGS},
        // By default axe leaves failing one-character text, such as a reference id, incomplete. It is
        // text, so it is held to the same contrast.
        checks: {'color-contrast': {options: {ignoreLength: true}}},
      });
      for (const violation of results.violations) {
        if (violation.impact !== 'serious' && violation.impact !== 'critical') continue;
        const nodes = violation.id === 'target-size'
          ? violation.nodes.filter((node) => !reachesMinimumTarget(node))
          : violation.nodes;
        found.push(...nodes.map((node) => describeViolation(colorMode, violation.id, node)));
      }
    }
  } finally {
    if (original === undefined) delete document.documentElement.dataset.colorMode;
    else document.documentElement.dataset.colorMode = original;
  }
  return found;
}
